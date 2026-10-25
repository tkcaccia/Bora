"""Topology-guarded tissue contours from an existing global Bora vector result.

The exact cleaned mask and exact GeoJSON are immutable inputs. By default both
internal and outer class boundaries are simplified. Small complementary pairs can be
restored atomically from exact geometry; all review candidates are then checked
against the complete native raster before any tissue GeoJSON is published.
This module does not classify tissue or change the exact source mask.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.windows import Window
import shapely
from shapely.affinity import scale
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

from bora.wsi_global_vector import (
    TILE, _RasterIndex, _agreement, _bounded_support_exchange, _count_coordinates,
    _exact_component_records, _parts, _require_large_parts,
    _review_component_records, _stream_raster_agreement,
    _simplify_coverage, _validate_coverage, _write_geojson,
)


FORMAT = "bora-wsi-review-smoothing-v1"
TOLERANCES_UM = (3.0, 1.5, 0.75, 0.375)
OUTER_TOLERANCES_UM = (6.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.75)
MIN_CLASS_IOU = 0.995
MAX_DISPLACEMENT_UM = 5.0
MAX_OUTER_SUPPORT_EXCHANGE_FRACTION = 0.001
MAX_OUTER_SUPPORT_ONE_WAY_FRACTION = 0.0005
MAX_RESTORED_PAIRS = 64
MIN_VERTEX_REDUCTION = 0.50
MAX_STAGING_BYTES = 1 * 1024**3


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _deadline(deadline: float, phase: str) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError(f"Review smoothing deadline reached at {phase}")


def _source_geometry(exact_geojson: Path, vector_receipt: Path,
                     exact_mask: Path) -> tuple[dict, dict, dict, list[dict]]:
    receipt = json.loads(vector_receipt.read_text())
    if (receipt.get("format") != "bora-wsi-global-vector-v1" or
            receipt.get("exact_mask_sha256") != _hash(exact_mask) or
            receipt.get("output_sha256", {}).get("exact/mask.geojson") != _hash(exact_geojson)):
        raise ValueError("Exact cleaned vector source or receipt changed")
    document = json.loads(exact_geojson.read_text())
    shape_yx = document.get("native_shape_yx")
    mpp_xy = document.get("native_mpp_xy")
    if (document.get("coordinate_space") != "native_image_pixel_edges" or
            document.get("source_semantic_mask_sha256") != receipt.get("source_semantic_mask_sha256") or
            shape_yx != receipt.get("native_shape_yx") or
            mpp_xy != receipt.get("native_mpp_xy") or
            not isinstance(shape_yx, list) or len(shape_yx) != 2 or
            not all(type(v) is int and v > 0 for v in shape_yx) or
            exact_mask.stat().st_size != math.prod(shape_yx)):
        raise ValueError("Exact GeoJSON and cleaned mask frame disagree")
    features = document.get("features", [])
    classes = sorted({int(feature["properties"]["value"]) for feature in features})
    if (classes != receipt.get("class_ids", classes) or not 2 <= len(classes) <= 32 or
            len(features) != receipt.get("exact_component_count")):
        raise ValueError("Expected complete multi-class global component set")
    components = {value: [] for value in classes}
    source_records = []
    ids = set()
    for feature in features:
        value = int(feature["properties"]["value"])
        identity = feature["properties"].get("component_id")
        index = feature["properties"].get("component_index")
        polygon = shape(feature["geometry"])
        if (polygon.geom_type != "Polygon" or not polygon.is_valid or polygon.is_empty or
                not isinstance(identity, str) or identity in ids or
                type(index) is not int or index < 1):
            raise ValueError("Invalid or duplicate exact component geometry")
        expected_id = (f"bora:{receipt['source_semantic_mask_sha256'][:12]}:"
                       f"{value}:{hashlib.sha256(polygon.wkb).hexdigest()[:16]}")
        if identity != expected_id:
            raise ValueError("Serialized exact component identity changed")
        ids.add(identity)
        components[value].append(polygon)
        source_records.append(dict(value=value, geometry=polygon,
                                   component_index=index, component_id=identity))
    exact = {value: unary_union(parts) for value, parts in components.items()}
    if _validate_coverage(exact, coordinate_limit=5_000_000) != receipt["exact_coordinates"]:
        raise ValueError("Exact geometry coordinate count changed")
    return exact, document, receipt, source_records


def _internal_coverage_simplify(exact: dict, tolerance_um: float,
                                mpp_xy: list[float]) -> dict:
    values = sorted(exact)
    scaled = [scale(exact[value], xfact=mpp_xy[0], yfact=mpp_xy[1],
                    origin=(0, 0)) for value in values]
    simplified = shapely.coverage_simplify(scaled, tolerance_um,
                                            simplify_boundary=False)
    return {value: scale(geometry, xfact=1 / mpp_xy[0],
                         yfact=1 / mpp_xy[1], origin=(0, 0))
            for value, geometry in zip(values, simplified)}


def _restore_small_complementary_pairs(exact: dict, candidate: dict,
                                        minimum_pixels: float) -> tuple[dict, list[dict]]:
    """Replace each shrunken island and its opposing hole as one partition edit."""
    support = unary_union(list(exact.values()))
    repairs = []
    seen_exact = set()
    original_parts = {value: _parts(exact[value]) for value in exact}
    trees = {value: STRtree(original_parts[value]) for value in exact}
    for value in sorted(candidate):
        for island in _parts(candidate[value]):
            if island.area + 1e-7 >= minimum_pixels:
                continue
            hits = trees[value].query(island)
            if len(hits) == 0:
                raise ValueError("Subthreshold island has no exact counterpart")
            index = max((int(hit) for hit in hits),
                        key=lambda hit: island.intersection(original_parts[value][hit]).area)
            identity = (value, index)
            original = original_parts[value][index]
            overlap = island.intersection(original).area
            if (identity in seen_exact or overlap < .95 * min(island.area, original.area) or
                    island.symmetric_difference(original).area > .2 * original.area or
                    island.boundary.intersection(support.boundary).length > 1e-7 or
                    original.boundary.intersection(support.boundary).length > 1e-7 or
                    original.intersection(candidate[value].difference(island)).area > 1e-7):
                raise ValueError("Cannot safely restore this complementary island and hole")
            seen_exact.add(identity)
            invaded = original.difference(island)
            opponents = [other for other in candidate if other != value and
                         candidate[other].intersection(invaded).area > 1e-7]
            if len(opponents) != 1:
                raise ValueError("Complementary island touches ambiguous opposing classes")
            repairs.append((value, opponents[0], island, original))
    if len(repairs) > MAX_RESTORED_PAIRS:
        raise ValueError("Too many islands would need exact-pair restoration")
    review = dict(candidate)
    report = []
    for value, other, island, original in repairs:
        review[value] = review[value].difference(island).union(original)
        review[other] = review[other].union(island).difference(original)
        report.append({"class_id": value, "candidate_area_pixels": island.area,
                       "opposing_class_id": other,
                       "restored_exact_area_pixels": original.area})
    return review, report


def _internal_border(labels: np.ndarray) -> np.ndarray:
    supported = labels != 0
    border = np.zeros(labels.shape, bool)
    horizontal = (supported[:, :-1] & supported[:, 1:] &
                  (labels[:, :-1] != labels[:, 1:]))
    vertical = (supported[:-1, :] & supported[1:, :] &
                (labels[:-1, :] != labels[1:, :]))
    border[:, :-1] |= horizontal
    border[:, 1:] |= horizontal
    border[:-1, :] |= vertical
    border[1:, :] |= vertical
    return border


def _all_border(labels: np.ndarray) -> np.ndarray:
    """Mark both class interfaces and tissue/background edges."""
    border = np.zeros(labels.shape, bool)
    horizontal = labels[:, :-1] != labels[:, 1:]
    vertical = labels[:-1, :] != labels[1:, :]
    border[:, :-1] |= horizontal
    border[:, 1:] |= horizontal
    border[:-1, :] |= vertical
    border[1:, :] |= vertical
    border[0, :] |= labels[0, :] != 0
    border[-1, :] |= labels[-1, :] != 0
    border[:, 0] |= labels[:, 0] != 0
    border[:, -1] |= labels[:, -1] != 0
    return border


def _changed_distance(exact: dict, review: dict, shape_yx: list[int],
                      *, halo: int = 64, deadline: float = float("inf"),
                      include_outer_boundary: bool = False) -> dict:
    """Bound changed native pixels by distance to the relevant exact edge."""
    height, width = shape_yx
    exact_index, review_index = _RasterIndex(exact), _RasterIndex(review)
    greatest = 0.0
    changed_pixels = 0
    changed_tiles = []
    for y in range(0, height, TILE):
        _deadline(deadline, "changed-pixel displacement")
        for x in range(0, width, TILE):
            th, tw = min(TILE, height - y), min(TILE, width - x)
            before = exact_index.tile(y, x, th, tw)
            after = review_index.tile(y, x, th, tw)
            changed = before != after
            count = int(changed.sum())
            if not count:
                continue
            changed_pixels += count
            changed_tiles.append({"native_yx": [y, x], "changed_pixels": count})
            ey0, ex0 = max(0, y - halo), max(0, x - halo)
            ey1, ex1 = min(height, y + th + halo), min(width, x + tw + halo)
            context = exact_index.tile(ey0, ex0, ey1 - ey0, ex1 - ex0)
            border = (_all_border(context) if include_outer_boundary
                      else _internal_border(context))
            if not border.any():
                raise ValueError("Changed area has no nearby exact internal edge")
            distance = cv2.distanceTransform((~border).astype(np.uint8),
                                              cv2.DIST_L2, 5)
            crop = distance[y-ey0:y-ey0+th, x-ex0:x-ex0+tw]
            greatest = max(greatest, float(crop[changed].max()))
    distance_key = ("maximum_changed_pixel_distance_from_exact_boundary_px"
                    if include_outer_boundary else
                    "maximum_changed_pixel_distance_from_exact_internal_edge_px")
    return {"changed_pixels": changed_pixels, distance_key: greatest,
            "changed_tiles": sorted(changed_tiles,
                                    key=lambda item: -item["changed_pixels"])[:20]}


def _segment_stats(geometries: dict) -> dict:
    axis_count = diagonal_count = 0
    diagonal_length = total_length = 0.0
    for geometry in geometries.values():
        for polygon in _parts(geometry):
            for ring in (polygon.exterior, *polygon.interiors):
                delta = np.diff(np.asarray(ring.coords), axis=0)
                length = np.hypot(delta[:, 0], delta[:, 1])
                diagonal = (np.abs(delta[:, 0]) > 1e-8) & (np.abs(delta[:, 1]) > 1e-8)
                axis_count += int((~diagonal).sum())
                diagonal_count += int(diagonal.sum())
                diagonal_length += float(length[diagonal].sum())
                total_length += float(length.sum())
    return {"axis_aligned_segments": axis_count,
            "diagonal_segments": diagonal_count,
            "diagonal_segment_fraction": diagonal_count / max(1, axis_count + diagonal_count),
            "diagonal_boundary_length_fraction": diagonal_length / max(1, total_length)}


def choose_review_geometry(exact: dict, shape_yx: list[int], mpp_xy: list[float],
                           source_mask_sha256: str, *, deadline: float = float("inf"),
                           exact_records: list[dict] | None = None,
                           minimum_area_um2: float = 100.0,
                           tolerances_um: tuple[float, ...] | None = None,
                           simplify_outer_boundary: bool = True) -> tuple[dict, dict, list[dict]]:
    """Return the simplest passing coverage or fail without a new geometry."""
    if not 2 <= len(exact) <= 32 or len(shape_yx) != 2 or len(mpp_xy) != 2:
        raise ValueError("Expected complete registered tissue classes")
    if not math.isfinite(minimum_area_um2) or minimum_area_um2 <= 0:
        raise ValueError("Require positive physical minimum component area")
    if tolerances_um is None:
        tolerances_um = (OUTER_TOLERANCES_UM if simplify_outer_boundary
                         else TOLERANCES_UM)
    minimum_pixels = minimum_area_um2 / (mpp_xy[0] * mpp_xy[1])
    exact_count = _validate_coverage(exact, coordinate_limit=5_000_000)
    original_counts = {value: len(_parts(exact[value])) for value in exact}
    original_support = unary_union(list(exact.values()))
    if exact_records is None:
        exact_records = _exact_component_records(exact, source_mask_sha256)
    attempts = []
    for tolerance in tolerances_um:
        _deadline(deadline, "next smoothing tolerance")
        attempt = {"tolerance_um": tolerance}
        try:
            if tolerance <= 0 or not math.isfinite(tolerance):
                raise ValueError("Review tolerance must be positive")
            candidate = (_simplify_coverage(exact, tolerance, mpp_xy)
                         if simplify_outer_boundary else
                         _internal_coverage_simplify(exact, tolerance, mpp_xy))
            if {value: len(_parts(candidate[value])) for value in exact} != original_counts:
                raise ValueError("Coverage simplification changed component counts")
            review, repairs = _restore_small_complementary_pairs(exact, candidate,
                                                                   minimum_pixels)
            if (any(not review[value].is_valid or
                    len(_parts(review[value])) != original_counts[value] or
                    not _require_large_parts(review[value], minimum_pixels)
                    for value in exact)):
                raise ValueError("Review has invalid, lost or subthreshold parts/holes")
            values = sorted(review)
            if any(review[left].intersection(review[right]).area > 1e-6
                   for i, left in enumerate(values) for right in values[i + 1:]):
                raise ValueError("Review classes overlap")
            support_delta = original_support.symmetric_difference(
                unary_union(list(review.values()))).area
            if not simplify_outer_boundary and support_delta > 1e-6:
                raise ValueError("Review changed the outer tissue support")
            records = _review_component_records(exact_records, review)
            if len(records) != len(exact_records) or \
                    {record["component_id"] for record in records} != \
                    {record["component_id"] for record in exact_records}:
                raise ValueError("Review lost one-to-one component identities")
            coordinates = _validate_coverage(review, coordinate_limit=5_000_000)
            reduction = 1 - coordinates / exact_count
            if reduction < MIN_VERTEX_REDUCTION:
                raise ValueError("Review does not reduce enough vertices")
            confusion, _ = _stream_raster_agreement(exact, review, shape_yx)
            agreement = _agreement(confusion, sorted(exact))
            if agreement["minimum_class_iou"] < MIN_CLASS_IOU:
                raise ValueError("Review fails full-native raster agreement")
            if simplify_outer_boundary:
                if not _bounded_support_exchange(agreement, support_delta,
                        original_support.area,
                        maximum_one_way_fraction=MAX_OUTER_SUPPORT_ONE_WAY_FRACTION,
                        maximum_total_fraction=MAX_OUTER_SUPPORT_EXCHANGE_FRACTION):
                    raise ValueError("Outer boundary changes too many tissue/background pixels")
            elif (agreement["new_background_in_exact_tissue"] != 0 or
                    agreement["new_tissue_in_exact_background"] != 0 or
                    agreement["changed_tissue_area_fraction"] != 0):
                raise ValueError("Internal smoothing changed tissue support")
            displacement = _changed_distance(exact, review, shape_yx,
                deadline=deadline, include_outer_boundary=simplify_outer_boundary)
            if displacement["changed_pixels"] != agreement["changed_pixels"]:
                raise ValueError("Displacement count disagrees with raster agreement")
            distance_key = ("maximum_changed_pixel_distance_from_exact_boundary_px"
                            if simplify_outer_boundary else
                            "maximum_changed_pixel_distance_from_exact_internal_edge_px")
            max_um = displacement[distance_key] * max(mpp_xy)
            if max_um > MAX_DISPLACEMENT_UM:
                raise ValueError("Review moves a class label too far from its exact edge")
            stats = _segment_stats(review)
            original_stats = _segment_stats(exact)
            if (simplify_outer_boundary and
                    (stats["diagonal_segments"] == 0 or
                     stats["diagonal_segment_fraction"] <=
                         original_stats["diagonal_segment_fraction"] or
                     stats["axis_aligned_segments"] >=
                         original_stats["axis_aligned_segments"] * .5)):
                raise ValueError("Outer review did not replace enough stair steps")
            attempt.update(status="PASSED_DIAGNOSTIC_REVIEW", restored_pairs=repairs,
                original_component_counts=original_counts, review_coordinate_count=coordinates,
                vertex_reduction_fraction=reduction, outer_support_symmetric_difference_px2=support_delta,
                full_native_raster_agreement=agreement, displacement=displacement,
                maximum_changed_pixel_distance_um=max_um,
                segment_stats_before=original_stats,
                segment_stats_after=stats,
                simplify_outer_boundary=simplify_outer_boundary)
            attempts.append(attempt)
            return review, attempt, attempts
        except (ValueError, shapely.GEOSException) as error:
            attempt.update(status="REJECTED", reason=str(error))
            attempts.append(attempt)
    raise ValueError("No nonzero review smoothing passed: " + json.dumps(attempts))


def _outline(labels: np.ndarray) -> np.ndarray:
    edge = _internal_border(labels)
    return cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


def _whole_preview(native_he: Path, exact: dict, review: dict,
                   shape_yx: list[int], output: Path) -> dict:
    height, width = shape_yx
    factor = max(1, (width + 1599) // 1600)
    ph, pw = (height + factor - 1) // factor, (width + factor - 1) // factor
    transform = Affine.scale(width / pw, height / ph)
    def labels(geometries):
        return rasterize([(geometry, value) for value, geometry in geometries.items()],
            out_shape=(ph, pw), transform=transform, fill=0, dtype=np.uint8)
    before, after = labels(exact), labels(review)
    with rasterio.open(native_he) as image:
        he = np.moveaxis(image.read((1, 2, 3), out_shape=(3, ph, pw),
                         resampling=Resampling.bilinear), 0, -1)
    panels = []
    for mask, color in ((before, (255, 224, 0)), (after, (0, 232, 255))):
        panel = he.copy()
        panel[_outline(mask)] = color
        panels.append(panel)
    canvas = Image.new("RGB", (3 * pw, ph + 32), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (name, pixels) in enumerate(zip(
            ("Native H&E", "Exact cleaned Bora boundary", "Smoothed review boundary"),
            (he, *panels))):
        canvas.paste(Image.fromarray(pixels), (column * pw, 32))
        draw.text((column * pw + 8, 8), name, fill="black")
    canvas.save(output, optimize=True)
    return {"path": str(output), "sha256": _hash(output),
            "preview_shape_yx": [ph, pw], "native_resolution_geometry_in_geojson": True}


def _roi_previews(native_he: Path, exact: dict, review: dict,
                  shape_yx: list[int], displacement: dict, output: Path) -> list[dict]:
    height, width = shape_yx
    side = min(1024, height, width)
    sites = []
    for tile in displacement["changed_tiles"]:
        y, x = tile["native_yx"]
        y = min(max(0, y - side // 4), height - side)
        x = min(max(0, x - side // 4), width - side)
        if all(abs(y - sy) >= side or abs(x - sx) >= side for sy, sx in sites):
            sites.append((y, x))
        if len(sites) == 3:
            break
    before_index, after_index = _RasterIndex(exact), _RasterIndex(review)
    entries = []
    with rasterio.open(native_he) as image:
        for index, (y, x) in enumerate(sites, 1):
            rgb = np.moveaxis(image.read((1, 2, 3),
                window=Window(x, y, side, side)), 0, -1)
            before = before_index.tile(y, x, side, side)
            after = after_index.tile(y, x, side, side)
            panel = rgb.copy()
            panel[_outline(before)] = (255, 224, 0)
            panel[_outline(after)] = (0, 232, 255)
            path = output / f"ROI_{index:02d}_native_y{y}_x{x}_exact_yellow_review_cyan.png"
            Image.fromarray(panel).save(path, optimize=True)
            entries.append({"native_yxhw": [y, x, side, side], "path": str(path),
                            "sha256": _hash(path),
                            "changed_pixels": int(np.count_nonzero(before != after))})
    return entries


def publish_review_smoothing(exact_geojson: Path, vector_receipt: Path,
                             exact_mask: Path, native_he: Path, output_folder: Path,
                             *, deadline_seconds: int = 7200,
                             simplify_outer_boundary: bool = True,
                             minimum_free_bytes: int = 1024**3) -> dict:
    """Publish a guarded predicted tissue GeoJSON and previews in a fresh folder."""
    start = time.monotonic()
    if not 60 <= deadline_seconds <= 28200:
        raise ValueError("Review smoothing needs a finite 60..28200-second deadline")
    deadline = start + deadline_seconds
    paths = [Path(p).resolve(strict=True) for p in
             (exact_geojson, vector_receipt, exact_mask, native_he)]
    exact_geojson, vector_receipt, exact_mask, native_he = paths
    output = Path(output_folder)
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError("Use a fresh prediction output with an existing parent")
    if type(minimum_free_bytes) is not int or minimum_free_bytes < 0:
        raise ValueError("Minimum free-space reserve must be a nonnegative integer")
    if shutil.disk_usage(output.parent).free < minimum_free_bytes + MAX_STAGING_BYTES:
        raise ValueError("Review smoothing would breach its free-space reserve")
    package_root = Path(__file__).resolve().parent
    python_tree = {str(path.relative_to(package_root)): _hash(path)
                   for path in sorted(package_root.rglob("*.py"))}
    source_hashes = {str(path): _hash(path) for path in paths}
    source_hashes[str(Path(__file__).resolve())] = _hash(Path(__file__))
    _deadline(deadline, "source hashes")
    exact, document, receipt, exact_records = _source_geometry(
        exact_geojson, vector_receipt, exact_mask)
    if source_hashes[str(native_he)] != document["source_image_sha256"]:
        raise ValueError("Native H&E differs from the exact vector source image")
    shape_yx, mpp_xy = document["native_shape_yx"], document["native_mpp_xy"]
    review, selected, attempts = choose_review_geometry(exact, shape_yx, mpp_xy,
        receipt["source_semantic_mask_sha256"], deadline=deadline,
        exact_records=exact_records,
        minimum_area_um2=float(receipt["minimum_area_um2"]),
        simplify_outer_boundary=simplify_outer_boundary)
    _deadline(deadline, "global geometry/raster guards")
    records = _review_component_records(exact_records, review)
    output.mkdir()
    _atomic_json(output / "RUN_STARTED.json", {"format": FORMAT,
        "status": ("RUNNING_PREDICTED_TISSUE_ANNOTATION" if simplify_outer_boundary
                   else "RUNNING_REVIEW_ONLY"), "source_sha256": source_hashes,
        "staged_bora_python_tree_sha256": python_tree,
        "selected_tolerance_um": selected["tolerance_um"],
        "biological_accuracy_qualified": False, "exact_source_preserved": True})
    source = {"native_shape_yx": shape_yx, "native_mpp_xy": mpp_xy,
        "source_frame_id": document["source_frame_id"],
        "source_image_sha256": document["source_image_sha256"],
        "semantic_mask_sha256": document["source_semantic_mask_sha256"],
        "global_model_binding_sha256": document["global_model_binding_sha256"]}
    class_names = {}
    for value in sorted(exact):
        names = {str(feature["properties"].get("name", ""))
                 for feature in document["features"]
                 if int(feature["properties"]["value"]) == value}
        if len(names) != 1 or not next(iter(names)).strip():
            raise ValueError("Exact vector source has inconsistent class names")
        class_names[value] = next(iter(names))
    geojson = output / ("BORA_TISSUE_ANNOTATIONS.geojson" if simplify_outer_boundary
                        else "SMOOTHED_REVIEW_ONLY.geojson")
    _write_geojson(geojson, records, class_names, source, mpp_xy,
        role=("smoothed_predicted_tissue_annotation" if simplify_outer_boundary
              else "smoothed_review_only"),
        identity_status=document["tissue_identity_status"],
        minimum_area_um2=float(receipt["minimum_area_um2"]),
        tolerance_um=selected["tolerance_um"])
    whole = _whole_preview(native_he, exact, review, shape_yx,
                           output / "FULL_SLIDE_EXACT_VS_SMOOTHED.png")
    rois = _roi_previews(native_he, exact, review, shape_yx,
                         selected["displacement"], output)
    for path in paths:
        if _hash(path) != source_hashes[str(path)]:
            raise ValueError("A source changed during review smoothing")
    if _hash(Path(__file__)) != source_hashes[str(Path(__file__).resolve())]:
        raise ValueError("Review smoothing source changed during execution")
    if {str(path.relative_to(package_root)): _hash(path)
            for path in sorted(package_root.rglob("*.py"))} != python_tree:
        raise ValueError("Staged Bora Python source tree changed during execution")
    _deadline(deadline, "terminal source rehash")
    report = {"format": FORMAT,
        "status": ("PREDICTED_TISSUE_ANNOTATION" if simplify_outer_boundary
                   else "REVIEW_ONLY_NOT_ACCEPTED"),
        "source_sha256": source_hashes, "attempts": attempts,
        "staged_bora_python_tree_sha256": python_tree,
        "selected": selected, "review_geojson_path": str(geojson),
        "review_geojson_sha256": _hash(geojson),
        "full_slide_preview": whole, "native_he_roi_previews": rois,
        "exact_cleaned_mask_and_geojson_unchanged": True,
        "new_tissue_classifier_used": False,
        "simplify_outer_boundary": simplify_outer_boundary,
        "minimum_free_bytes": minimum_free_bytes,
        "probability_seam_status": "not_asserted_by_vector_smoother",
        "biological_accuracy_qualified": False,
        "cellphenotyper_or_viewer_published": False,
        "elapsed_seconds": time.monotonic() - start}
    _atomic_json(output / "SMOOTHING_COMPLETE.json", report)
    return report
