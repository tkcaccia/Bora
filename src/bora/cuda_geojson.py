"""CUDA-assisted conversion of categorical TIFF masks to GeoJSON.

CUDA performs connected-component labeling and area filtering. Raster-to-vector
polygonization and GEOS topology operations remain on CPU because neither
Rasterio nor Shapely provides CUDA kernels.
"""
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
from shapely import coverage_simplify, make_valid
from shapely.geometry import mapping, shape

from .multicpu_geojson import _spatial_shape, _union_group


class CudaGeoJSONError(RuntimeError):
    pass


def _cuda_preflight(device):
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() <= int(device):
            raise CudaGeoJSONError(f"CUDA device {device} is unavailable")
        cp.cuda.Device(int(device)).use()
    except CudaGeoJSONError:
        raise
    except Exception as exc:
        raise CudaGeoJSONError(
            "CUDA GeoJSON requires a working CuPy installation and CUDA device") from exc


def _cuda_filter_components(data, min_pixels, device=0):
    try:
        import cupy as cp
        from cupyx.scipy import ndimage as cndi
    except Exception as exc:
        raise CudaGeoJSONError(
            "CUDA GeoJSON requires CuPy (install Bora's cuda-geojson extra)") from exc
    try:
        cp.cuda.Device(int(device)).use()
        gpu = cp.asarray(data)
        output = cp.zeros_like(gpu)
        labels = cp.asnumpy(cp.unique(gpu))
        structure = cp.ones((3, 3), dtype=cp.uint8)
        component_count = 0
        removed_components = 0
        for raw_value in labels:
            value = int(raw_value)
            if value <= 0:
                continue
            components, count = cndi.label(gpu == value, structure=structure)
            count = int(count)
            component_count += count
            if not count:
                continue
            sizes = cp.bincount(components.ravel(), minlength=count + 1)
            keep = sizes >= int(max(1, min_pixels))
            keep[0] = False
            removed_components += int(cp.count_nonzero(~keep[1:]).get())
            output[keep[components]] = value
        result = cp.asnumpy(output)
        cp.cuda.Stream.null.synchronize()
        free, total = cp.cuda.runtime.memGetInfo()
        device_name = cp.cuda.runtime.getDeviceProperties(int(device))["name"]
        if isinstance(device_name, bytes):
            device_name = device_name.decode()
        return result, {
            "cuda_device": int(device),
            "cuda_device_name": str(device_name),
            "cuda_memory_total": int(total),
            "cuda_memory_free_after_filter": int(free),
            "components_before_filter": component_count,
            "components_removed": removed_components,
        }
    except CudaGeoJSONError:
        raise
    except Exception as exc:
        raise CudaGeoJSONError(f"CUDA component filtering failed: {exc}") from exc


def convert(mask_path, output_path, *, page=2, max_page_side=12000,
            min_area=500.0, simplify=8.0, workers=4, group_prefix="group_",
            device=0, fallback=True):
    """Convert a label mask with CUDA component filtering and CPU vectorization."""
    started = perf_counter()
    try:
        _cuda_preflight(device)
    except CudaGeoJSONError as exc:
        if not fallback:
            raise
        from .multicpu_geojson import convert as cpu_convert
        report = cpu_convert(
            mask_path, output_path, page=page, max_page_side=max_page_side,
            min_area=min_area, simplify=simplify, workers=workers,
            group_prefix=group_prefix, striped_native=True)
        report.update({"accelerator": "cpu-fallback", "cuda_device": int(device),
                       "cuda_error": str(exc)})
        return report
    with tifffile.TiffFile(mask_path) as tif:
        levels = list(tif.series[0].levels)
        full_h, full_w = _spatial_shape(levels[0].shape)
        if page < 0:
            page = len(levels) - 1
            for index, level in enumerate(levels):
                if max(_spatial_shape(level.shape)) <= int(max_page_side):
                    page = index
                    break
        data = np.asarray(levels[page].asarray())
        page_h, page_w = _spatial_shape(levels[page].shape)
    scale_x, scale_y = full_w / page_w, full_h / page_h
    min_pixels = max(1, int(np.ceil(float(min_area) / (scale_x * scale_y))))

    try:
        filtered, cuda_report = _cuda_filter_components(data, min_pixels, device)
    except CudaGeoJSONError as exc:
        if not fallback:
            raise
        del data
        from .multicpu_geojson import convert as cpu_convert
        report = cpu_convert(
            mask_path, output_path, page=page, max_page_side=max_page_side,
            min_area=min_area, simplify=simplify, workers=workers,
            group_prefix=group_prefix, striped_native=True)
        report.update({"accelerator": "cpu-fallback", "cuda_device": int(device),
                       "cuda_error": str(exc)})
        return report

    transform = rasterio.Affine(scale_x, 0, 0, 0, scale_y, 0)
    raster = filtered.astype(
        np.uint16 if int(filtered.max(initial=0)) <= 65535 else np.int32,
        copy=False)
    geometries, values = [], []
    for geometry, raw_value in shapes(
            raster, mask=raster > 0, transform=transform, connectivity=8):
        candidate = shape(geometry)
        if not candidate.is_valid:
            candidate = make_valid(candidate)
        if not candidate.is_empty and candidate.area >= float(min_area):
            geometries.append(candidate)
            values.append(int(raw_value))

    if simplify > 0 and geometries:
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
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(
        {"type": "FeatureCollection", "features": features}), encoding="utf-8")
    return {
        "features": len(features), "source_components": len(geometries),
        "resolved_page": int(page), "working_shape_yx": [page_h, page_w],
        "scale_xy": [scale_x, scale_y], "runtime_seconds": perf_counter() - started,
        "workers_requested": int(workers),
        "union_workers": min(int(workers), max(1, len(items))),
        "min_component_pixels": min_pixels, "accelerator": "cuda",
        **cuda_report,
    }
