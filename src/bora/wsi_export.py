"""Bounded whole-slide mask/GeoJSON export in native-pixel windows.

Each non-overlapping window is fitted as a planar coverage and its *final*
geometry is rasterized into the OME mask. This keeps the two deliverables in
agreement without ever polygonizing the whole slide in memory. The published
Bora vector cleanup enforces both continuous and raster area >=20 pixels.
"""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
from time import monotonic

import numpy as np
import shapely
from rasterio.features import rasterize, shapes
from rasterio.transform import Affine
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

from bora.compact_geojson import boundary_stats, geometry_agreement
from bora.mask_cleanup import finalize_vector_regions


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _ring_area(ring):
    vertices = np.asarray(ring, dtype=np.float64)
    return abs(float(np.dot(vertices[:-1, 0], vertices[1:, 1]) -
                     np.dot(vertices[:-1, 1], vertices[1:, 0]))) / 2


def sparse_lossless_window(tile, x, y, report, reason):
    """Vectorize sparse or fragmented foreground without expensive global unions.

    Native pixel contours already have exact topology. Removing polygon parts
    and holes below 20 pixels here matches the published final cleanup, while
    avoiding an O(number-of-components) union/coverage simplification on glass.
    """
    h, w = tile.shape
    mpp = report["native_mpp_xy"]
    names = {int(key): value for key, value in report["class_names"].items()}
    parts = defaultdict(list)
    source_segments = final_segments = removed_parts = filled_holes = 0
    for geometry, value in shapes(tile, mask=tile > 0, connectivity=4):
        if geometry["type"] != "Polygon":
            raise ValueError("Expected native pixel polygons")
        rings = geometry["coordinates"]
        source_segments += sum(len(ring) - 1 for ring in rings)
        original_area = _ring_area(rings[0]) - sum(_ring_area(ring) for ring in rings[1:])
        if original_area < 20:
            removed_parts += 1
            continue
        kept = [rings[0]]
        for ring in rings[1:]:
            if _ring_area(ring) < 20:
                filled_holes += 1
            else:
                kept.append(ring)
        final_segments += sum(len(ring) - 1 for ring in kept)
        parts[int(value)].append(kept)
    local_features = []
    native_features = []
    for value, polygons in sorted(parts.items()):
        name = names.get(value, f"class_{value}")
        local_geometry = {"type": "MultiPolygon", "coordinates": polygons}
        local_features.append((local_geometry, value))
        native_geometry = {"type": "MultiPolygon", "coordinates": [
            [[(px + x, py + y) for px, py in ring] for ring in polygon]
            for polygon in polygons]}
        area = sum(_ring_area(polygon[0]) - sum(_ring_area(ring) for ring in polygon[1:])
                   for polygon in polygons)
        native_features.append({"type": "Feature", "geometry": native_geometry,
            "properties": {"value": value, "name": name,
                "classification": {"name": name}, "predicted_tissue_id": value,
                "predicted_tissue": name, "objectType": "annotation",
                "tissue_identity_source": "Neighbor context, native fastPLS",
                "area_pixels": area, "area_um2": area*mpp[0]*mpp[1],
                "window_core_xywh": [x, y, w, h]}})
    rendered = rasterize(local_features, out_shape=(h, w), fill=0, dtype=np.uint8)
    audit = {"background": not native_features, "sparse_lossless_contours": True,
        "fast_lossless_reason": reason,
        "tolerance_pixels": 0.0, "tolerance_um": 0.0,
        "changed_pixels": int(np.count_nonzero(tile != rendered)),
        "source_segments": source_segments, "final_segments": final_segments,
        "diagonal_segments": 0, "minimum_class_iou": 1.0,
        "changed_tissue_area_fraction": 0.0,
        "small_area_cleanup": {"minimum_region_area_pixels": 20,
            "strictly_less_than_threshold": True,
            "positive_polygon_parts_below_threshold": 0,
            "polygon_holes_below_threshold": 0,
            "isolated_regions_filled_with_background": removed_parts,
            "filled_holes": filled_holes}}
    return native_features, rendered, audit


def fit_window(tile, x, y, report, tolerance=0.75):
    """Return native-coordinate features, an exactly matching tile, and audit."""
    h, w = tile.shape
    if np.count_nonzero(tile) < tile.size * .05:
        return sparse_lossless_window(tile, x, y, report, "sparse_foreground")
    transitions = (np.count_nonzero(tile[:, 1:] != tile[:, :-1]) +
                   np.count_nonzero(tile[1:, :] != tile[:-1, :]))
    if transitions > tile.size * .05:
        return sparse_lossless_window(tile, x, y, report, "fragmented_contours")
    by_class = defaultdict(list)
    for geometry, value in shapes(tile, mask=tile > 0, connectivity=4):
        by_class[int(value)].append(shape(geometry))
    names = ({int(key): value for key, value in report["class_names"].items()}
             if "class_names" in report else
             {int(report["group_class_ids"][int(group)-1]): name
              for group, name in report["group_class_names"].items()})
    mpp = report["native_mpp_xy"]
    data = {"type": "FeatureCollection", "native_mpp_xy": mpp, "features": []}
    for value, polygons in sorted(by_class.items()):
        merged = unary_union(polygons)
        if not merged.is_valid:
            raise ValueError(f"Invalid input geometry for class {value} at {x},{y}")
        name = names.get(value, f"class_{value}")
        data["features"].append({"type": "Feature", "geometry": mapping(merged),
            "properties": {"value": value, "name": name,
                "classification": {"name": name}, "predicted_tissue_id": value,
                "predicted_tissue": name, "objectType": "annotation",
                "tissue_identity_source": "Neighbor context, native fastPLS"}})
    if not data["features"]:
        return [], np.zeros_like(tile), {"background": True, "changed_pixels": 0,
                                           "source_segments": 0, "final_segments": 0}
    original = json.loads(json.dumps(data))
    fitted = json.loads(json.dumps(data))
    source = [shape(feature["geometry"]) for feature in data["features"]]
    selected = 0.0
    try:
        candidate = shapely.coverage_simplify(source, tolerance)
        for feature, geometry in zip(fitted["features"], candidate):
            feature["geometry"] = mapping(geometry)
        fitted = finalize_vector_regions(fitted, (h, w), 20)
        agreement = geometry_agreement(original, fitted)
        if (agreement["all_classes_preserved"] and agreement["all_polygons_valid"]
                and agreement["interclass_overlap_area_pixels"] < 1e-7
                and agreement["minimum_class_iou"] >= .97
                and agreement["changed_tissue_area_fraction"] <= .015):
            selected = tolerance
        else:
            fitted = finalize_vector_regions(original, (h, w), 20)
            agreement = geometry_agreement(original, fitted)
    except (ValueError, shapely.GEOSException):
        fitted = finalize_vector_regions(original, (h, w), 20)
        agreement = geometry_agreement(original, fitted)
    cleanup = fitted["final_native_pixel_region_cleanup"]
    if (cleanup["positive_polygon_parts_below_threshold"] or
            cleanup["polygon_holes_below_threshold"] or
            not agreement["all_polygons_valid"] or
            agreement["interclass_overlap_area_pixels"] >= 1e-7):
        raise ValueError(f"Invalid final vector coverage at {x},{y}")
    local = rasterize(((f["geometry"], int(f["properties"]["value"]))
                       for f in fitted["features"]), out_shape=(h, w),
                      fill=0, dtype=np.uint8)
    before, after = boundary_stats(original), boundary_stats(fitted)
    features = []
    for feature in fitted["features"]:
        geometry = shape(feature["geometry"])
        if geometry.is_empty:
            continue
        feature["geometry"] = mapping(shapely.affinity.translate(geometry, xoff=x, yoff=y))
        feature["properties"]["area_pixels"] = float(geometry.area)
        feature["properties"]["area_um2"] = float(geometry.area * mpp[0] * mpp[1])
        feature["properties"]["window_core_xywh"] = [x, y, w, h]
        features.append(feature)
    audit = {"background": False, "tolerance_pixels": selected,
             "tolerance_um": selected * (mpp[0] + mpp[1]) / 2,
             "changed_pixels": int(np.count_nonzero(tile != local)),
             "source_segments": before["boundary_segments"],
             "final_segments": after["boundary_segments"],
             "diagonal_segments": after["diagonal_segments"],
             "minimum_class_iou": agreement["minimum_class_iou"],
             "changed_tissue_area_fraction": agreement["changed_tissue_area_fraction"],
             "small_area_cleanup": cleanup}
    return features, local, audit


def _export_one(args):
    output, report, shape, source_name, fit_core, index, x, y, cw, ch = args
    output = Path(output)
    source = np.memmap(output / source_name, mode="r", dtype=np.uint8,
                       shape=shape)
    final = np.memmap(output / "classified.final.dat", mode="r+", dtype=np.uint8,
                      shape=shape)
    chunks = output / "vector-windows"
    tile = np.asarray(source[y:y+ch, x:x+cw])
    begin = monotonic()
    features = []
    rendered = np.zeros_like(tile)
    subaudits = []
    for dy in range(0, ch, fit_core):
        for dx in range(0, cw, fit_core):
            sy, sx = min(ch, dy + fit_core), min(cw, dx + fit_core)
            subfeatures, subrendered, subaudit = fit_window(
                tile[dy:sy, dx:sx], x + dx, y + dy, report)
            rendered[dy:sy, dx:sx] = subrendered
            features.extend(subfeatures)
            subaudits.append(subaudit)
    active = [a for a in subaudits if not a["background"]]
    audit = {"background": not active, "fit_core_pixels": fit_core,
             "fit_subwindows": len(subaudits),
             "fitted_subwindows": sum(a.get("tolerance_pixels", 0) > 0 for a in active),
             "lossless_subwindows": sum(a.get("tolerance_pixels", 0) == 0 for a in active),
             "tolerance_pixels": min((a.get("tolerance_pixels", 0) for a in active), default=0),
             "minimum_class_iou": min((a.get("minimum_class_iou", 1) for a in active), default=1),
             "changed_tissue_area_fraction": max(
                 (a.get("changed_tissue_area_fraction", 0) for a in active), default=0),
             "changed_pixels": sum(a["changed_pixels"] for a in subaudits),
             "source_segments": sum(a["source_segments"] for a in subaudits),
             "final_segments": sum(a["final_segments"] for a in subaudits),
             "diagonal_segments": sum(a.get("diagonal_segments", 0) for a in subaudits),
             "small_area_cleanup": {
                 "positive_polygon_parts_below_threshold": sum(
                     a.get("small_area_cleanup", {}).get("positive_polygon_parts_below_threshold", 0)
                     for a in subaudits),
                 "polygon_holes_below_threshold": sum(
                     a.get("small_area_cleanup", {}).get("polygon_holes_below_threshold", 0)
                     for a in subaudits)},
             "subwindow_audits": subaudits}
    final[y:y+ch, x:x+cw] = rendered
    final.flush()
    feature_path = chunks / f"{index:03d}.geojson"
    temporary = feature_path.with_suffix(".geojson.tmp")
    with temporary.open("w") as stream:
        json.dump(features, stream, separators=(",", ":"), allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, feature_path)
    audit.update(index=index, core_xywh=[x, y, cw, ch],
                 seconds=monotonic() - begin, feature_count=len(features))
    atomic_json(chunks / f"{index:03d}.json", audit)
    return audit


def export(output, *, core=4096, fit_core=None, workers=3):
    output = Path(output)
    if (output / "VECTOR_COMPLETE.json").exists():
        raise FileExistsError("Completed vector export already exists")
    report = json.loads((output / "CLASSIFIED_COMPLETE.json").read_text())
    if not (output / "CLEANED_COMPLETE.json").exists():
        raise FileNotFoundError("Raster cleanup is incomplete")
    source_name=("classified.background.dat" if (output/"BACKGROUND_COMPLETE.json").exists()
                 else "classified.cleaned.dat")
    if not (output/source_name).exists():
        raise FileNotFoundError("Finalized native raster is missing")
    h, w = report["native_shape_yx"]
    if core < 1:
        raise ValueError("Positive native core size required")
    if fit_core is None:
        fit_core = min(core, 2048)
    if fit_core < 1 or fit_core > core:
        raise ValueError("Fit core must be positive and no larger than the native export core")
    if workers < 1:
        raise ValueError("At least one CPU vector worker is required")
    raster_path = output / "classified.final.dat"
    if raster_path.exists():
        expected = h*w
        if raster_path.stat().st_size != expected:
            raise ValueError("Partial final raster has the wrong byte count")
    else:
        with raster_path.open("wb") as stream:
            stream.truncate(h*w)
    chunks = output / "vector-windows"
    chunks.mkdir(exist_ok=True)
    totals = defaultdict(int)
    started = monotonic()
    count = 0
    tasks = []
    for y in range(0, h, core):
        for x in range(0, w, core):
            y1, x1 = min(h, y + core), min(w, x + core)
            index = count
            count += 1
            checkpoint = chunks / f"{index:03d}.json"
            feature_path = chunks / f"{index:03d}.geojson"
            if checkpoint.exists():
                if not feature_path.exists():
                    raise ValueError(f"Missing vector features for checkpoint {index}")
                audit = json.loads(checkpoint.read_text())
                if audit["core_xywh"] != [x, y, x1-x, y1-y]:
                    raise ValueError(f"Mismatched vector checkpoint {index}")
                for key in ("changed_pixels", "source_segments", "final_segments",
                            "diagonal_segments", "feature_count"):
                    totals[key] += audit[key]
                totals["fitted_windows" if audit.get("tolerance_pixels",0)>0 else "unfitted_windows"] += 1
                continue
            tasks.append((str(output), report, (h, w), source_name, fit_core,
                          index, x, y, x1-x, y1-y))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_export_one, task) for task in tasks]
        try:
            for future in as_completed(futures):
                audit = future.result()
                index = audit["index"]
                for key in ("changed_pixels", "source_segments", "final_segments",
                            "diagonal_segments", "feature_count"):
                    totals[key] += audit[key]
                totals["fitted_windows" if audit.get("tolerance_pixels",0)>0 else "unfitted_windows"] += 1
                print(json.dumps({"vector_window": index, "total": count,
                                  "seconds": round(audit["seconds"], 2),
                                  "tolerance_pixels": audit.get("tolerance_pixels", 0)}), flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    geo = output / "mask.geojson"
    geo_temporary = output / "mask.geojson.tmp"
    with geo_temporary.open("w") as stream:
        stream.write('{"type":"FeatureCollection","coordinate_space":"native_image_pixel_edges",')
        stream.write('"origin":"top_left","native_shape_yx":'+json.dumps([h,w])+',')
        stream.write('"native_mpp_xy":'+json.dumps(report["native_mpp_xy"])+',"features":[')
        first = True
        for index in range(count):
            features = json.loads((chunks / f"{index:03d}.geojson").read_text())
            for feature in features:
                if not first:
                    stream.write(",")
                json.dump(feature, stream, separators=(",", ":"), allow_nan=False)
                first = False
        stream.write('],"vector_export":{"method":"per-window topology-preserving coverage fit",')
        stream.write('"minimum_area_native_pixels":20,"max_tolerance_pixels":0.75}}\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(geo_temporary, geo)
    summary = {"native_shape_yx": [h, w], "native_mpp_xy": report["native_mpp_xy"],
               "windows": count, "window_core_pixels": core,
               "fit_core_pixels_for_new_windows": fit_core,
               "cpu_workers": workers,
               "mask_geojson": str(geo), "mask_raster_dat": str(output / "classified.final.dat"),
               "vector_source_raster": source_name,
               "minimum_area_native_pixels": 20, "maximum_tolerance_pixels": 0.75,
               "exact_geojson_raster_agreement_per_window": True,
               "source_vs_final": dict(totals), "seconds": monotonic()-started}
    atomic_json(output / "VECTOR_COMPLETE.json", summary)
    return summary
