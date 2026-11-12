"""Repair only unoccupied vector pixel gaps after global polygonization."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rasterio.features import rasterize, shapes
from rasterio.transform import Affine
from shapely.geometry import Polygon, mapping, shape
from shapely.ops import unary_union

from .supervised_wsi import atomic


def patched_working_geojson(working_path: Path, old: np.memmap,
                            cleaned: np.memmap, destination: Path,
                            mpp_xy: tuple[float, float]) -> dict:
    """Assign empty vector slivers in changed pixel cells without cutting tissue.

    A pixel-cell patch that subtracts *all* existing polygons can split a thin
    legitimate polygon into a subthreshold part (observed in failed v31).
    Intersect each changed region with the old coverage and add only its
    previously unoccupied remainder to the adjacent predicted class.
    """
    data = json.loads(working_path.read_text())
    grouped = {}
    templates = {}
    for feature in data["features"]:
        value = int(feature["properties"]["value"])
        grouped.setdefault(value, []).append(shape(feature["geometry"]))
        templates.setdefault(value, feature)
    geometries = {value: unary_union(parts) for value, parts in grouped.items()}
    patch_groups = {}
    for y in range(0, old.shape[0], 512):
        previous = np.asarray(old[y:y+512])
        updated = np.asarray(cleaned[y:y+512])
        changed = previous != updated
        if not changed.any():
            continue
        for raw, value in shapes(updated.astype(np.int16), mask=changed,
                                 connectivity=4, transform=Affine(1, 0, 0, 0, 1, y)):
            patch_groups.setdefault(int(value), []).append(shape(raw))
    if not patch_groups:
        raise ValueError("Recovery found no changed raster pixels")
    residuals = {}
    patch_count = 0
    for value, regions in patch_groups.items():
        if value == 0 or value not in geometries:
            raise ValueError("Tiny-hole repair must assign an existing tissue class")
        for region in regions:
            occupied = unary_union([g.intersection(region) for g in geometries.values()
                                    if g.intersects(region)])
            empty = region.difference(occupied)
            if empty.is_empty:
                raise ValueError("Changed raster cell has no unoccupied vector gap")
            residuals.setdefault(value, []).append(empty)
            patch_count += 1
    for value, gaps in residuals.items():
        geometries[value] = geometries[value].union(unary_union(gaps))
    area = mpp_xy[0] * mpp_xy[1]
    minimum = float("inf")
    repaired_features = []
    for value, geometry in sorted(geometries.items()):
        feature = templates[value]
        if not geometry.is_valid:
            raise ValueError("Patch produced invalid class geometry")
        for polygon in ([geometry] if geometry.geom_type == "Polygon" else geometry.geoms):
            minimum = min(minimum, polygon.area * area)
            if polygon.area * area < 2000 - 1e-4:
                raise ValueError("Patch created subthreshold polygon")
            for ring in polygon.interiors:
                if Polygon(ring).area * area < 2000 - 1e-4:
                    raise ValueError("Patch created subthreshold polygon hole")
        feature["geometry"] = mapping(geometry)
        feature["properties"]["area_pixels"] = float(geometry.area)
        feature["properties"]["area_um2"] = float(geometry.area * area)
        feature["properties"]["component_id"] = f"supervised_class_{value:02d}_global"
        feature["properties"]["component_index"] = 0
        repaired_features.append(feature)
    data["features"] = repaired_features
    data["raster_tie_repair"] = {
        "source": "exact global vector", "patches": patch_count,
        "minimum_area_um2": 2000,
        "method": "union only previously unoccupied geometry within changed pixel cells"}
    atomic(destination, data)
    return {"patches": patch_count, "minimum_polygon_um2": minimum,
            "feature_count": len(data["features"])}


def audit_working_raster(working: Path, expected: np.memmap, temporary: Path) -> int:
    data = json.loads(working.read_text())
    pairs = [(feature["geometry"], int(feature["properties"]["value"]))
             for feature in data["features"]]
    rendered = np.memmap(temporary, mode="w+", dtype=np.uint8, shape=expected.shape)
    rasterize(pairs, out=rendered, transform=Affine.identity(), fill=0,
              dtype=np.uint8, all_touched=False)
    mismatch = sum(int(np.count_nonzero(rendered[y:y+512] != expected[y:y+512]))
                   for y in range(0, expected.shape[0], 512))
    del rendered
    temporary.unlink()
    if mismatch:
        raise ValueError(f"Patched working vector raster differs by {mismatch} pixels")
    return mismatch

