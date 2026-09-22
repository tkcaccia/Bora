"""Topology-safe multicore conversion of a pyramidal label mask to GeoJSON."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import numpy as np
import rasterio
import tifffile
from rasterio.features import shapes
from shapely import coverage_simplify, from_wkb, make_valid, to_wkb
from shapely.geometry import mapping, shape
from shapely.ops import unary_union


def _spatial_shape(shape_):
    shape_ = tuple(map(int, shape_))
    return shape_ if len(shape_) == 2 else shape_[-2:]


def _union_group(item):
    value, geometries = item
    repaired = [geometry if geometry.is_valid else make_valid(geometry)
                for geometry in geometries]
    return value, unary_union(repaired)


def _polygonize_native_strip(item):
    mask_path, y0, y1 = item
    import pyvips
    image = pyvips.Image.new_from_file(str(mask_path), access="random")
    crop = image.crop(0, int(y0), image.width, int(y1-y0))
    dtype = {"uchar":np.uint8, "ushort":np.uint16, "uint":np.uint32}[crop.format]
    data = np.ndarray(buffer=crop.write_to_memory(), dtype=dtype,
                      shape=(crop.height, crop.width)).copy()
    raster = data.astype(np.uint16 if int(data.max()) <= 65535 else np.int32,
                         copy=False)
    transform = rasterio.Affine(1, 0, 0, 0, 1, int(y0))
    result = []
    for geometry, raw_value in shapes(
            raster, mask=raster > 0, transform=transform, connectivity=8):
        candidate = shape(geometry)
        if not candidate.is_valid:
            candidate = make_valid(candidate)
        result.append((int(raw_value), to_wkb(candidate)))
    return result


def convert(mask_path, output_path, *, page=2, max_page_side=12000,
            min_area=500.0, simplify=8.0, workers=4, group_prefix="group_",
            striped_native=True):
    """Convert labels using coverage-aware simplification and parallel unions."""
    started = perf_counter()
    with tifffile.TiffFile(mask_path) as tif:
        levels = list(tif.series[0].levels)
        full_h, full_w = _spatial_shape(levels[0].shape)
        if page < 0:
            page = len(levels) - 1
            for index, level in enumerate(levels):
                if max(_spatial_shape(level.shape)) <= int(max_page_side):
                    page = index
                    break
        data = None if page == 0 and striped_native and workers > 1 else np.asarray(levels[page].asarray())
        page_h, page_w = _spatial_shape(levels[page].shape)
    scale_x, scale_y = full_w / page_w, full_h / page_h

    geometries, values = [], []
    used_stripes = data is None
    polygon_workers = 1
    if used_stripes:
        edges = np.linspace(0, full_h, min(int(workers), full_h) + 1, dtype=int)
        tasks = [(str(mask_path), int(edges[i]), int(edges[i+1]))
                 for i in range(len(edges)-1)]
        polygon_workers = len(tasks)
        with ProcessPoolExecutor(max_workers=len(tasks),
                                 mp_context=get_context("spawn")) as pool:
            for result in pool.map(_polygonize_native_strip, tasks):
                for value, encoded in result:
                    candidate = from_wkb(encoded)
                    if candidate.area >= float(min_area):
                        values.append(value)
                        geometries.append(candidate)
    else:
        transform = rasterio.Affine(scale_x, 0, 0, 0, scale_y, 0)
        raster = data.astype(np.uint16 if int(data.max()) <= 65535 else np.int32,
                             copy=False)
        for geometry, raw_value in shapes(
                raster, mask=raster > 0, transform=transform, connectivity=8):
            value = int(raw_value)
            candidate = shape(geometry)
            if not candidate.is_valid:
                candidate = make_valid(candidate)
            if candidate.area >= float(min_area):
                geometries.append(candidate)
                values.append(value)

    if simplify > 0 and geometries:
        # Unlike independent per-label simplify, this retains coincident shared
        # edges and cannot introduce inter-class gaps or overlaps.
        geometries = list(coverage_simplify(
            geometries, tolerance=float(simplify), simplify_boundary=True))

    buckets = {}
    for value, geometry in zip(values, geometries):
        if not geometry.is_empty:
            buckets.setdefault(value, []).append(geometry)
    items = sorted(buckets.items())
    if len(items) > 1 and workers > 1:
        with ProcessPoolExecutor(max_workers=min(int(workers), len(items)),
                                 mp_context=get_context("spawn")) as pool:
            dissolved = list(pool.map(_union_group, items))
    else:
        dissolved = [_union_group(item) for item in items]

    features = []
    for value, geometry in dissolved:
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        features.append({
            "type": "Feature", "geometry": mapping(geometry),
            "properties": {"value": int(value),
                           "classification": f"{group_prefix}{value}"},
        })
    result = {"type": "FeatureCollection", "features": features}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(result), encoding="utf-8")
    return {
        "features": len(features), "source_components": len(geometries),
        "resolved_page": int(page), "working_shape_yx": [page_h, page_w],
        "scale_xy": [scale_x, scale_y], "runtime_seconds": perf_counter() - started,
        "workers_requested": int(workers),
        "polygon_workers": polygon_workers,
        "union_workers": min(int(workers), max(1, len(items))),
        "striped_native": used_stripes,
    }
