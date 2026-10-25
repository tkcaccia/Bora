"""Global, out-of-core WSI polygonization of an assembled native class mask.

The whole raster is offered to GDAL as one band.  Windowing is used only for
I/O and raster-agreement checks, never for independent contour fitting.  The
route fails closed on excess complexity, disk shortage, invalid coverage, or
failed smoothing guards; it does not launch an encoder or GPU workload.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
import rasterio
from rasterio.features import rasterize, shapes
from rasterio.transform import Affine
from rasterio.windows import Window
import shapely
from shapely.affinity import scale
from shapely.geometry import MultiPolygon, Polygon, box, mapping, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree


FORMAT = "bora-wsi-global-vector-v1"
TILE = 512
MAX_PREFLIGHT_PIXELS = 4 * 1024 * 1024


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    with Path(path).open("w") as stream:
        json.dump(value, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type == "MultiPolygon":
        return list(geometry.geoms)
    raise ValueError("Only polygonal class geometry is supported")


def _count_coordinates(geometry):
    return sum(len(p.exterior.coords) + sum(len(r.coords) for r in p.interiors)
               for p in _parts(geometry))


def _clean_polygon(polygon, minimum_pixels):
    if polygon.area < minimum_pixels:
        return None, 1, 0
    holes = []
    filled = 0
    for ring in polygon.interiors:
        if Polygon(ring).area < minimum_pixels:
            filled += 1
        else:
            holes.append(ring)
    result = Polygon(polygon.exterior, holes)
    if not result.is_valid or result.is_empty:
        raise ValueError("Area cleanup produced invalid class geometry")
    return result, 0, filled


def _require_large_parts(geometry, minimum_pixels):
    for polygon in _parts(geometry):
        if polygon.area + 1e-8 < minimum_pixels:
            return False
        if any(Polygon(ring).area + 1e-8 < minimum_pixels for ring in polygon.interiors):
            return False
    return True


def _validate_coverage(geometries, *, coordinate_limit):
    total = 0
    values = sorted(geometries)
    for value in values:
        geometry = geometries[value]
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError(f"Invalid or empty class geometry for {value}")
        total += _count_coordinates(geometry)
        if total > coordinate_limit:
            raise ValueError("Global vector coordinate complexity bound exceeded")
    for index, value in enumerate(values):
        for other in values[:index]:
            if geometries[value].intersection(geometries[other]).area > 1e-7:
                raise ValueError("Different tissue classes overlap after vector fitting")
    return total


def _exact_component_records(geometries, source_mask_sha256):
    records = []
    for value in sorted(geometries):
        ordered = sorted(_parts(geometries[value]), key=lambda polygon:
            (-polygon.area, polygon.bounds, hashlib.sha256(polygon.wkb).hexdigest()))
        for index, polygon in enumerate(ordered, 1):
            identity = hashlib.sha256(polygon.wkb).hexdigest()[:16]
            records.append(dict(value=value, geometry=polygon,
                component_index=index,
                component_id=f"bora:{source_mask_sha256[:12]}:{value}:{identity}"))
    if len({record["component_id"] for record in records}) != len(records):
        raise ValueError("Global component IDs are not unique")
    return records


def _review_component_records(exact_records, review_geometries):
    """Bind review parts one-to-one to exact global parts by maximal overlap."""
    records = []
    by_class = defaultdict(list)
    for record in exact_records:
        by_class[record["value"]].append(record)
    for value in sorted(review_geometries):
        previous = by_class[value]
        new_parts = _parts(review_geometries[value])
        if len(previous) != len(new_parts):
            raise ValueError("Smoothing changed global component count")
        tree = STRtree([record["geometry"] for record in previous])
        used = set()
        for polygon in new_parts:
            candidates = [(float(polygon.intersection(previous[int(i)]["geometry"]).area), int(i))
                          for i in tree.query(polygon)]
            if not candidates:
                raise ValueError("Smoothed component has no exact counterpart")
            area, index = max(candidates)
            if area <= 0 or index in used:
                raise ValueError("Smoothed components do not map one-to-one to exact regions")
            used.add(index)
            original = previous[index]
            records.append(dict(value=value, geometry=polygon,
                component_index=original["component_index"],
                component_id=original["component_id"]))
    return sorted(records, key=lambda record: (record["value"], record["component_index"]))


def _features(records, class_names, mpp_xy, *, role, identity_status):
    for record in records:
        value, geometry = record["value"], record["geometry"]
        name = class_names[value]
        area = float(geometry.area)
        yield dict(type="Feature", geometry=mapping(geometry),
            properties=dict(value=value, name=name, classification=dict(name=name),
                predicted_tissue_id=value, predicted_tissue=name,
                objectType="annotation", predicted_not_expert_annotation=True,
                tissue_identity_status=identity_status,
                component_id=record["component_id"],
                component_index=record["component_index"],
                boundary_role=role, area_pixels=area,
                area_um2=area * mpp_xy[0] * mpp_xy[1]))


def _write_geojson(path, records, class_names, source, mpp_xy, *,
                   role, identity_status, minimum_area_um2, tolerance_um):
    with Path(path).open("w") as stream:
        header = dict(type="FeatureCollection", coordinate_space="native_image_pixel_edges",
            origin="top_left", native_shape_yx=source["native_shape_yx"],
            native_mpp_xy=mpp_xy, source_frame_id=source["source_frame_id"],
            source_image_sha256=source["source_image_sha256"],
            source_semantic_mask_sha256=source["semantic_mask_sha256"],
            global_model_binding_sha256=source["global_model_binding_sha256"],
            predicted_not_expert_annotation=True, tissue_identity_status=identity_status,
            vector_export=dict(format=FORMAT, global_contours=True,
                independently_fitted_windows=False, minimum_part_and_hole_area_um2=minimum_area_um2,
                geometry_role=role, smoothing_tolerance_um=tolerance_um))
        stream.write(json.dumps(header, separators=(",", ":"), allow_nan=False)[:-1])
        stream.write(',"features":[')
        first = True
        for feature in _features(records, class_names, mpp_xy,
                                  role=role, identity_status=identity_status):
            if not first:
                stream.write(",")
            json.dump(feature, stream, separators=(",", ":"), allow_nan=False)
            first = False
        stream.write("]}\n")
        stream.flush()
        os.fsync(stream.fileno())


class _RasterIndex:
    """Query only global connected components touching a raster-check tile."""

    def __init__(self, geometries):
        self.parts = [(part, int(value)) for value, geometry in geometries.items()
                      for part in _parts(geometry)]
        self.tree = STRtree([part for part, _ in self.parts])

    def tile(self, y, x, height, width):
        footprint = box(x, y, x + width, y + height)
        matches = [self.parts[int(index)] for index in self.tree.query(footprint)]
        if not matches:
            return np.zeros((height, width), np.uint8)
        # Most tissue interiors need no repeated polygon clipping/rasterization.
        for polygon, value in matches:
            if polygon.covers(footprint):
                return np.full((height, width), value, np.uint8)
        candidates = []
        for polygon, value in matches:
            clipped = polygon.intersection(footprint)
            if clipped.is_empty:
                continue
            if clipped.geom_type in ("Polygon", "MultiPolygon"):
                if clipped.area > 0:
                    candidates.append((clipped, value))
            elif clipped.geom_type == "GeometryCollection":
                polygonal = [part for part in clipped.geoms
                             if part.geom_type in ("Polygon", "MultiPolygon") and part.area > 0]
                if polygonal:
                    candidates.append((unary_union(polygonal), value))
        if not candidates:
            return np.zeros((height, width), np.uint8)
        return rasterize(candidates, out_shape=(height, width),
            transform=Affine.translation(x, y), fill=0, dtype=np.uint8)


def _stream_raster_agreement(exact, candidate, shape_yx, *, reference=None,
                             output_path=None):
    height, width = shape_yx
    exact_index = exact if isinstance(exact, _RasterIndex) else _RasterIndex(exact)
    candidate_index = (exact_index if candidate is exact else
                       candidate if isinstance(candidate, _RasterIndex) else _RasterIndex(candidate))
    output = (np.memmap(output_path, mode="w+", dtype=np.uint8, shape=shape_yx)
              if output_path is not None else None)
    confusion = np.zeros((256, 256), np.int64)
    changed_from_reference = 0
    for y in range(0, height, TILE):
        for x in range(0, width, TILE):
            th, tw = min(TILE, height - y), min(TILE, width - x)
            a = exact_index.tile(y, x, th, tw)
            b = candidate_index.tile(y, x, th, tw)
            if output is not None:
                output[y:y+th, x:x+tw] = a
            if reference is not None:
                changed_from_reference += int(np.count_nonzero(
                    a != reference[y:y+th, x:x+tw]))
            confusion += np.bincount(
                a.astype(np.uint16).ravel() * 256 + b.astype(np.uint16).ravel(),
                minlength=256*256).reshape(256, 256)
    if output is not None:
        output.flush()
        del output
    return confusion, changed_from_reference


def _agreement(confusion, class_ids):
    exact_tissue = int(confusion[1:, :].sum())
    review_tissue = int(confusion[:, 1:].sum())
    per_class = {}
    for value in class_ids:
        intersection = int(confusion[value, value])
        union = int(confusion[value, :].sum() + confusion[:, value].sum() - intersection)
        if union:
            per_class[str(value)] = intersection / union
    return dict(changed_pixels=int(confusion.sum() - np.trace(confusion)),
        exact_tissue_pixels=exact_tissue, review_tissue_pixels=review_tissue,
        changed_tissue_area_fraction=(abs(review_tissue - exact_tissue) / exact_tissue
                                      if exact_tissue else 0.0),
        minimum_class_iou=min(per_class.values(), default=1.0),
        class_iou=per_class,
        new_background_in_exact_tissue=int(confusion[1:, 0].sum()),
        new_tissue_in_exact_background=int(confusion[0, 1:].sum()))


def _bounded_support_exchange(agreement, support_delta, exact_support_area, *,
                              maximum_one_way_fraction=.0005,
                              maximum_total_fraction=.001):
    """Reject balanced gain/loss that a net tissue-area check would miss."""
    tissue = agreement["exact_tissue_pixels"]
    lost = agreement["new_background_in_exact_tissue"]
    gained = agreement["new_tissue_in_exact_background"]
    return (tissue > 0 and exact_support_area > 0 and
            lost <= maximum_one_way_fraction * tissue and
            gained <= maximum_one_way_fraction * tissue and
            lost + gained <= maximum_total_fraction * tissue and
            support_delta <= maximum_total_fraction * exact_support_area)


def _simplify_coverage(exact, tolerance_um, mpp_xy):
    if tolerance_um == 0:
        return exact
    values = sorted(exact)
    physically_scaled = [scale(exact[value], xfact=mpp_xy[0], yfact=mpp_xy[1],
                               origin=(0, 0)) for value in values]
    smoothed = shapely.coverage_simplify(physically_scaled, tolerance_um,
                                        simplify_boundary=True)
    return {value: scale(geometry, xfact=1/mpp_xy[0], yfact=1/mpp_xy[1],
                         origin=(0, 0)) for value, geometry in zip(values, smoothed)}


def publish_global_wsi_vectors(assembly_folder, output_folder, class_names, *,
                               identity_status="exploratory_transfer",
                               minimum_area_um2=100.0,
                               smoothing_tolerance_um=3.0,
                               minimum_class_iou=.97,
                               maximum_tissue_area_change=.015,
                               maximum_one_way_support_fraction=.0005,
                               maximum_support_exchange_fraction=.001,
                               maximum_cleanup_change=.02,
                               max_input_regions=250_000,
                               max_coordinates=5_000_000,
                               max_boundary_edges=5_000_000,
                               disk_budget_bytes=64 * 1024**3):
    """Publish exact cleaned and guarded smooth GeoJSON from one global mask.

    The source must be an atomic `assemble_wsi_probability_cores` or
    `materialize_global_group_mask` output.  A
    temporary tiled BigTIFF allows GDAL to polygonize the complete raster as
    one band.  Every returned contour is therefore global, even when the
    raster crosses many I/O tiles.  The final output is a new atomic folder.
    Complexity guards reject pathological fragmentation rather than silently
    returning chopped or incomplete geometries.
    """
    source_folder, folder = Path(assembly_folder), Path(output_folder)
    if folder.exists():
        raise FileExistsError(folder)
    receipt = json.loads((source_folder / "ASSEMBLY_COMPLETE.json").read_text())
    if (receipt.get("format") not in ("bora-wsi-probability-assembly-v1",
                                      "bora-wsi-label-assembly-v1",
                                      "bora-global-group-materialization-v1") or
            not receipt.get("core_partition_complete") or
            receipt.get("independently_fitted_windows") is not False):
        raise ValueError("Global vectorization requires a complete frozen-model assembly")
    h, w = receipt["native_shape_yx"]
    if (type(h) is not int or type(w) is not int or h < 1 or w < 1 or
            h*w > 2**63-1 or w > MAX_PREFLIGHT_PIXELS):
        raise ValueError("Invalid native source-frame mask dimensions")
    mpp_xy = [float(v) for v in receipt["native_mpp_xy"]]
    class_ids = receipt["class_ids"]
    if (not isinstance(class_ids, list) or not 2 <= len(class_ids) <= 32 or
            any(type(value) is not int or not 1 <= value <= 255 for value in class_ids) or
            len(set(class_ids)) != len(class_ids)):
        raise ValueError("Invalid native uint8 class order in source receipt")
    for field in ("semantic_mask_sha256", "source_image_sha256", "global_model_binding_sha256"):
        if not isinstance(receipt.get(field), str) or re.fullmatch(r"[0-9a-f]{64}", receipt[field]) is None:
            raise ValueError("Invalid source-frame checksum or model binding")
    if not isinstance(receipt.get("source_frame_id"), str) or not receipt["source_frame_id"].strip():
        raise ValueError("Missing source frame ID")
    if receipt.get("semantic_mask_path") != "semantic-mask.uint8.dat":
        raise ValueError("Unexpected source-frame semantic mask path")
    names = {int(key): str(value) for key, value in class_names.items()}
    if any(value not in names or not names[value].strip() for value in class_ids):
        raise ValueError("Every probability channel needs a nonempty tissue name")
    if identity_status not in ("exploratory_transfer", "externally_validated"):
        raise ValueError("Declare whether tissue identities are exploratory or validated")
    if (not all(math.isfinite(v) and v > 0 for v in mpp_xy) or
            not math.isfinite(minimum_area_um2) or minimum_area_um2 <= 0 or
            not math.isfinite(smoothing_tolerance_um) or smoothing_tolerance_um < 0 or
            not 0 < minimum_class_iou <= 1 or
            not 0 <= maximum_tissue_area_change <= 1 or
            not 0 <= maximum_one_way_support_fraction <= 1 or
            not 0 <= maximum_support_exchange_fraction <= 1 or
            not 0 <= maximum_cleanup_change <= 1 or
            type(max_input_regions) is not int or max_input_regions < 1 or
            type(max_coordinates) is not int or max_coordinates < 1 or
            type(max_boundary_edges) is not int or max_boundary_edges < 1 or
            type(disk_budget_bytes) is not int or disk_budget_bytes < 3*h*w):
        raise ValueError("Invalid vectorization limits or resource budget")
    mask_path = source_folder / receipt["semantic_mask_path"]
    if mask_path.stat().st_size != h*w or _hash_file(mask_path) != receipt["semantic_mask_sha256"]:
        raise ValueError("Assembled source mask changed or has the wrong byte count")
    folder.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(folder.parent).free < 3*h*w + 1024**3:
        raise ValueError("Insufficient free disk for global vector staging")
    source = np.memmap(mask_path, mode="r", dtype=np.uint8, shape=(h, w))
    valid_value = np.zeros(256, bool)
    valid_value[0] = True
    valid_value[class_ids] = True
    boundary_edges = 2*h + 2*w
    original_tissue = 0
    scan_rows = max(1, min(TILE, MAX_PREFLIGHT_PIXELS // w))
    for y in range(0, h, scan_rows):
        block = source[y:min(h, y+scan_rows)]
        if not np.all(valid_value[block]):
            raise ValueError("Source-frame mask contains an undeclared class value")
        original_tissue += int(np.count_nonzero(block))
        boundary_edges += int(np.count_nonzero(block[:, 1:] != block[:, :-1]))
        if y:
            boundary_edges += int(np.count_nonzero(block[0] != source[y-1]))
        boundary_edges += int(np.count_nonzero(block[1:] != block[:-1]))
        if boundary_edges > max_boundary_edges:
            raise ValueError("Native mask exceeds global boundary-complexity bound")
    staging = Path(tempfile.mkdtemp(prefix=".bora-wsi-vector-", dir=folder.parent))
    try:
        temporary_tif = staging / "global-input.tif"
        with rasterio.open(temporary_tif, "w", driver="GTiff", width=w, height=h,
                count=1, dtype="uint8", tiled=True, blockxsize=TILE,
                blockysize=TILE, compress="deflate", BIGTIFF="IF_SAFER",
                transform=Affine.identity()) as dataset:
            for y in range(0, h, TILE):
                for x in range(0, w, TILE):
                    th, tw = min(TILE, h-y), min(TILE, w-x)
                    dataset.write(source[y:y+th, x:x+tw], 1,
                                  window=Window(x, y, tw, th))
        minimum_pixels = minimum_area_um2 / (mpp_xy[0] * mpp_xy[1])
        parts = defaultdict(list)
        input_regions = input_coordinates = removed_parts = filled_holes = 0
        observed_classes = set()
        with rasterio.open(temporary_tif) as dataset:
            for raw_geometry, raw_value in shapes(rasterio.band(dataset, 1),
                                                  connectivity=4):
                value = int(raw_value)
                if value == 0:
                    continue
                if value not in class_ids:
                    raise ValueError("Source mask contains an undeclared tissue class")
                observed_classes.add(value)
                input_regions += 1
                if input_regions > max_input_regions:
                    raise ValueError("Global vector input-region complexity bound exceeded")
                input_coordinates += (len(raw_geometry["coordinates"][0]) +
                    sum(len(ring) for ring in raw_geometry["coordinates"][1:]))
                if input_coordinates > max_coordinates:
                    raise ValueError("Global vector input-coordinate complexity bound exceeded")
                cleaned, dropped, filled = _clean_polygon(shape(raw_geometry), minimum_pixels)
                removed_parts += dropped
                filled_holes += filled
                if cleaned is not None:
                    parts[value].append(cleaned)
        if not parts:
            raise ValueError("Area cleanup removed all observed tissue")
        if observed_classes != set(parts):
            raise ValueError("Area cleanup would erase an observed tissue class")
        exact = {value: unary_union(polygons) for value, polygons in parts.items()}
        exact_coordinates = _validate_coverage(exact, coordinate_limit=max_coordinates)
        exact_index = _RasterIndex(exact)
        exact_folder, review_folder = staging / "exact", staging / "review"
        exact_folder.mkdir()
        review_folder.mkdir()
        exact_mask_path = exact_folder / "mask.uint8.dat"
        _, cleanup_changed = _stream_raster_agreement(
            exact_index, exact_index, (h, w), reference=source, output_path=exact_mask_path)
        cleanup_fraction = cleanup_changed / max(1, original_tissue)
        if cleanup_fraction > maximum_cleanup_change:
            raise ValueError("Global area cleanup changes too much source tissue")
        tolerances = ([smoothing_tolerance_um / 2**index for index in range(4)] + [0.0]
                      if smoothing_tolerance_um else [0.0])
        exact_records = _exact_component_records(exact, receipt["semantic_mask_sha256"])
        selected = None
        attempts = []
        exact_support = unary_union(list(exact.values()))
        for tolerance in tolerances:
            try:
                candidate = _simplify_coverage(exact, tolerance, mpp_xy)
                coordinates = _validate_coverage(candidate, coordinate_limit=max_coordinates)
                if any(not _require_large_parts(geometry, minimum_pixels)
                       for geometry in candidate.values()):
                    raise ValueError("Smoothing creates a subthreshold part or hole")
                review_records = _review_component_records(exact_records, candidate)
                candidate_index = (exact_index if candidate is exact else _RasterIndex(candidate))
                confusion, _ = _stream_raster_agreement(exact_index, candidate_index, (h, w))
                agreement = _agreement(confusion, exact.keys())
                support_delta = exact_support.symmetric_difference(
                    unary_union(list(candidate.values()))).area
                attempts.append(dict(tolerance_um=tolerance, coordinates=coordinates,
                                     support_symmetric_difference_px2=support_delta,
                                     **agreement))
                if (agreement["minimum_class_iou"] >= minimum_class_iou and
                        agreement["changed_tissue_area_fraction"] <= maximum_tissue_area_change and
                        _bounded_support_exchange(agreement, support_delta,
                            exact_support.area,
                            maximum_one_way_fraction=maximum_one_way_support_fraction,
                            maximum_total_fraction=maximum_support_exchange_fraction)):
                    selected = (candidate, review_records, tolerance, agreement, coordinates)
                    break
            except (ValueError, shapely.GEOSException) as error:
                attempts.append(dict(tolerance_um=tolerance, rejected=str(error)))
        if selected is None:
            raise ValueError("No global smoothing candidate passed geometry/raster guards")
        review, review_records, selected_tolerance, agreement, review_coordinates = selected
        _write_geojson(exact_folder / "mask.geojson", exact_records, names, receipt, mpp_xy,
            role="native_exact_after_area_cleanup", identity_status=identity_status,
            minimum_area_um2=minimum_area_um2, tolerance_um=0.0)
        _write_geojson(review_folder / "mask.geojson", review_records, names, receipt, mpp_xy,
            role="smoothed_review", identity_status=identity_status,
            minimum_area_um2=minimum_area_um2, tolerance_um=selected_tolerance)
        temporary_tif.unlink()
        if _hash_file(mask_path) != receipt["semantic_mask_sha256"]:
            raise ValueError("Source semantic mask changed during global vectorization")
        outputs = {str(path.relative_to(staging)): _hash_file(path) for path in
                   (exact_mask_path, exact_folder / "mask.geojson", review_folder / "mask.geojson")}
        staged_bytes = sum(path.stat().st_size for path in staging.rglob("*") if path.is_file())
        if staged_bytes > disk_budget_bytes:
            raise ValueError("Global vector output exceeds declared disk budget")
        report = dict(format=FORMAT, source_assembly=str(source_folder),
            source_semantic_mask_sha256=receipt["semantic_mask_sha256"],
            exact_mask_sha256=outputs["exact/mask.uint8.dat"],
            native_shape_yx=[h, w], native_mpp_xy=mpp_xy,
            source_frame_id=receipt["source_frame_id"],
            source_image_sha256=receipt["source_image_sha256"],
            global_model_binding_sha256=receipt["global_model_binding_sha256"],
            tissue_identity_status=identity_status, minimum_area_um2=minimum_area_um2,
            maximum_one_way_support_fraction=maximum_one_way_support_fraction,
            maximum_support_exchange_fraction=maximum_support_exchange_fraction,
            minimum_area_strictly_less_removed=True,
            native_boundary_edges=boundary_edges,
            input_regions=input_regions, input_coordinates=input_coordinates,
            removed_small_parts=removed_parts, filled_small_holes=filled_holes,
            exact_coordinates=exact_coordinates, review_coordinates=review_coordinates,
            exact_component_count=len(exact_records), review_component_count=len(review_records),
            component_ids_preserved_between_exact_and_review=True,
            cleanup_changed_pixels=cleanup_changed, cleanup_changed_fraction=cleanup_fraction,
            selected_smoothing_tolerance_um=selected_tolerance,
            smoothing_attempts=attempts, review_raster_agreement=agreement,
            exact_vector_matches_cleaned_mask=True, global_contours=True,
            independently_fitted_vector_windows=False,
            no_macroregion_cuts=True, output_sha256=outputs,
            biological_accuracy_qualified=False)
        _write_json(staging / "GLOBAL_VECTOR_COMPLETE.json", report)
        if folder.exists():
            raise FileExistsError(folder)
        os.rename(staging, folder)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
