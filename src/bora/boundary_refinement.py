"""Opt-in, bounded H&E-guided placement of uncertain tissue boundaries.

This module consumes an already fitted *global* semantic probability field. It
never fits a tissue identity from H&E or changes high-confidence pixels. Core
tiles are an execution detail: fixed halos and one global boundary band make
the result independent of the chosen processing partition. This is a bounded
crop method, not qualification for arbitrary whole-slide inference.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt, find_objects, label
from shapely.geometry import Polygon, shape
import tifffile

from .compact_geojson import compact
from .spacing import spacing_xy
from .whole_field import publish_exact_mask


FORMAT = "bora-local-he-boundary-v1"
SEMANTIC_CLEANUP_FORMAT = "bora-small-semantic-region-cleanup-v2"
MAX_PIXELS = 20_000_000
MAX_SIDE = 8192


def _digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def remove_tiny_semantic_regions(mask, mpp_xy, *, min_area_um2=100.0,
                                 crop_edge_policy="relabel_single_contact"):
    """Remove enclosed tiny class islands and fill unambiguous tiny holes.

    This is an explicit whole-crop operation on a *categorical* native mask,
    before exact GeoJSON publication. A component is eligible only when its
    physical area is strictly below the cutoff and every different
    four-neighbour pixel has the same nonzero class. A small crop-edge
    component can also be relabelled under the explicit single-contact policy;
    it is otherwise retained, since its continuation outside the crop is
    unknown. No class is erased and no background is introduced.
    All decisions use the original mask, so processing order cannot cascade.
    The input, probabilities, uncertainty, and external background are untouched.
    """
    mask = np.asarray(mask)
    if (mask.ndim != 2 or mask.dtype != np.uint8 or
            min(mask.shape) < 1 or max(mask.shape) > MAX_SIDE or
            mask.size > MAX_PIXELS):
        raise ValueError("Semantic cleanup requires a bounded native uint8 class mask")
    classes = np.unique(mask[mask > 0]).astype(int)
    if len(classes) != 2:
        raise ValueError("Semantic cleanup requires exactly two observed tissue classes")
    try:
        cutoff = float(min_area_um2)
    except (TypeError, ValueError):
        cutoff = float("nan")
    if (isinstance(min_area_um2, (bool, str)) or
            not np.isscalar(min_area_um2) or
            not np.isfinite(cutoff) or not 0 < cutoff <= 100):
        raise ValueError("Require a finite small-region area in (0, 100] µm²")
    if crop_edge_policy not in ("preserve", "relabel_single_contact"):
        raise ValueError("Unknown crop-edge semantic cleanup policy")
    mpp = spacing_xy(mpp_xy)
    pixel_area = float(mpp[0] * mpp[1])
    result = mask.copy()
    four_connected = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    before, removed, changed = {}, {}, {}
    skipped_edge = skipped_ambiguous = skipped_class_erasure = 0
    relabelled_edge = 0
    source_counts = {int(v): int(np.count_nonzero(mask == v)) for v in classes}
    removed_class_pixels = {int(v): 0 for v in classes}
    for value in (0, *classes):
        value = int(value)
        components, count = label(mask == value, structure=four_connected)
        before[str(value)] = int(count)
        removed[str(value)] = 0
        changed[str(value)] = 0
        sizes = np.bincount(components.ravel(), minlength=count + 1)
        for index, bounds in enumerate(find_objects(components), start=1):
            if bounds is None or sizes[index] * pixel_area >= cutoff:
                continue
            ys, xs = bounds
            y0, y1, x0, x1 = ys.start, ys.stop, xs.start, xs.stop
            touches_edge = (y0 == 0 or x0 == 0 or
                            y1 == mask.shape[0] or x1 == mask.shape[1])
            if touches_edge and crop_edge_policy == "preserve":
                skipped_edge += 1
                continue
            component = components[ys, xs] == index
            # Include all four contacts; same-class interior pixels are then
            # ignored. A zero or mixed-class contact is ambiguous for a class
            # island and cannot be silently reassigned.
            neighbours = []
            if y0 > 0:
                neighbours.append(mask[y0 - 1:y1 - 1, xs][component])
            if y1 < mask.shape[0]:
                neighbours.append(mask[y0 + 1:y1 + 1, xs][component])
            if x0 > 0:
                neighbours.append(mask[ys, x0 - 1:x1 - 1][component])
            if x1 < mask.shape[1]:
                neighbours.append(mask[ys, x0 + 1:x1 + 1][component])
            neighbours = np.concatenate(neighbours) if neighbours else np.array([], mask.dtype)
            surrounding = np.unique(neighbours[neighbours != value])
            if len(surrounding) != 1 or surrounding[0] == 0:
                skipped_ambiguous += 1
                continue
            if value and source_counts[value] - removed_class_pixels[value] <= sizes[index]:
                skipped_class_erasure += 1
                continue
            target = int(surrounding[0])
            result[ys, xs][component] = target
            removed[str(value)] += 1
            changed[str(value)] += int(sizes[index])
            if touches_edge:
                relabelled_edge += 1
            if value:
                removed_class_pixels[value] += int(sizes[index])
        del components
    after = {str(int(v)): int(label(result == int(v), structure=four_connected)[1])
             for v in (0, *classes)}
    changed_pixels = int(np.count_nonzero(result != mask))
    if changed_pixels != sum(changed.values()):
        raise AssertionError("Semantic cleanup changed pixels outside selected components")
    if not set(classes).issubset(set(np.unique(result))):
        raise AssertionError("Semantic cleanup erased a represented tissue class")
    if np.any((mask > 0) & (result == 0)):
        raise AssertionError("Semantic cleanup introduced background inside tissue")
    report = dict(method=SEMANTIC_CLEANUP_FORMAT, mpp_xy=mpp.tolist(),
        class_ids=classes.tolist(), min_area_um2=cutoff,
        area_rule="component_pixel_count * mpp_x * mpp_y < min_area_um2",
        connectivity=4, source_role="native_semantic_mask",
        crop_edge_policy=crop_edge_policy,
        relabelled_crop_edge_components=relabelled_edge,
        input_label_sha256=_digest(mask), output_label_sha256=_digest(result),
        component_counts_before=before, component_counts_after=after,
        removed_components_by_source_class=removed,
        changed_pixels_by_source_class=changed, changed_pixels=changed_pixels,
        filled_background_pixels=changed["0"], introduced_background_pixels=0,
        skipped_crop_edge_components=skipped_edge,
        skipped_ambiguous_components=skipped_ambiguous,
        skipped_class_erasure_components=skipped_class_erasure,
        original_mask_preserved=True, probability_field_changed=False,
        uncertainty_field_changed=False, expert_annotation_used=False,
        biological_accuracy_validated=False, exact_vector_requires_new_publication=True)
    return result, report


def _cleanup_receipt_valid(receipt, output_sha, mpp_xy):
    if not isinstance(receipt, dict):
        return False
    area = receipt.get("min_area_um2")
    method = receipt.get("method")
    edge_policy_valid = (method == "bora-small-semantic-region-cleanup-v1" or
        (method == SEMANTIC_CLEANUP_FORMAT and
         receipt.get("crop_edge_policy") in ("preserve", "relabel_single_contact") and
         type(receipt.get("relabelled_crop_edge_components")) is int and
         receipt["relabelled_crop_edge_components"] >= 0))
    return (edge_policy_valid and
        receipt.get("output_label_sha256") == output_sha and
        receipt.get("mpp_xy") == list(mpp_xy) and
        receipt.get("connectivity") == 4 and
        receipt.get("original_mask_preserved") is True and
        receipt.get("introduced_background_pixels") == 0 and
        receipt.get("probability_field_changed") is False and
        receipt.get("uncertainty_field_changed") is False and
        isinstance(receipt.get("input_label_sha256"), str) and
        len(receipt["input_label_sha256"]) == 64 and
        type(area) in (int, float) and np.isfinite(area) and 0 < area <= 100)


def parameters(value):
    """Require an explicit, versioned method rather than hidden defaults."""
    required = {"format", "uncertainty_margin", "maximum_distance_um", "he_blur_um",
                "he_edge_sigma", "smoothness", "iterations"}
    if not isinstance(value, dict) or set(value) != required or value.get("format") != FORMAT:
        raise ValueError("Require the complete versioned boundary-refinement parameters")
    bounds = (("uncertainty_margin", 0., 1.), ("maximum_distance_um", 0., 50.),
              ("he_blur_um", 0., 5.), ("he_edge_sigma", 0., 2.),
              ("smoothness", 0., 10.))
    for key, low, high in bounds:
        val = value[key]
        try:
            number = float(val)
        except (TypeError, ValueError):
            number = float("nan")
        admissible = (0 <= number <= high if key == "he_blur_um"
                      else low < number <= high)
        if isinstance(val, (bool, str)) or not np.isscalar(val) or \
                not np.isfinite(number) or not admissible:
            raise ValueError("Invalid boundary-refinement parameter: " + key)
    if type(value["iterations"]) is not int or not 1 <= value["iterations"] <= 12:
        raise ValueError("Boundary refinement requires 1..12 finite iterations")
    return {key: (value[key] if key in ("format", "iterations") else float(value[key]))
            for key in required}


def preflight(shape_yx, class_count, spec, *, core_size=512, mpp_xy=None,
              memory_budget_bytes=3 * 1024**3):
    """Bound crop, finite halo, and a conservative in-memory working estimate."""
    spec = parameters(spec)
    shape = np.asarray(shape_yx)
    if (shape.shape != (2,) or shape.dtype.kind not in "iu" or np.any(shape < 8)
            or max(shape) > MAX_SIDE or int(np.prod(shape)) > MAX_PIXELS):
        raise ValueError("H&E boundary route admits only a bounded crop, not a full WSI")
    if (type(class_count) is not int or not 2 <= class_count <= 16 or
            type(core_size) is not int or not 64 <= core_size <= 2048 or
            type(memory_budget_bytes) is not int or memory_budget_bytes < 1):
        raise ValueError("Invalid class count, core size or memory budget")
    if mpp_xy is None:
        # Conservative stand-in for direct admission calls; inference passes
        # calibrated spacing and obtains an exact finite patch bound below.
        halo_y, halo_x = 100, 100
    else:
        mpp = spacing_xy(mpp_xy)
        halo_x = int(np.ceil(3 * spec["he_blur_um"] / mpp[0])) + spec["iterations"] + 1
        halo_y = int(np.ceil(3 * spec["he_blur_um"] / mpp[1])) + spec["iterations"] + 1
    if max(halo_y, halo_x) > 512:
        raise ValueError("Physical H&E blur requires an unbounded processing halo")
    # Input P/H&E, global label/band/distance, plus the largest halo patch and
    # its OD, weights, and four iterative probability buffers. The estimate is
    # deliberately conservative; it is not a measured peak or cgroup promise.
    n = int(np.prod(shape))
    estimated = n * (4 * class_count + 3 + 1 + 1 + 8 + 8 + 4 + 2) + \
        min(shape[0], core_size + 2 * halo_y) * \
        min(shape[1], core_size + 2 * halo_x) * (4 * class_count * 5 + 80) + \
        256 * 1024**2
    if estimated > memory_budget_bytes:
        raise ValueError("Boundary-refinement estimated peak exceeds memory budget")
    return dict(format="bora-local-he-boundary-admission-v1", shape_yx=shape.astype(int).tolist(),
        class_count=class_count, core_size=core_size, halo_yx=[halo_y, halo_x],
        estimated_peak_bytes=int(estimated),
        memory_budget_bytes=memory_budget_bytes, estimate_is_measured_peak=False,
        full_wsi_qualified=False, independently_fitted_processing_windows=False)


def _internal_border(labels, support):
    border = np.zeros(labels.shape, bool)
    x = support[:, :-1] & support[:, 1:] & (labels[:, :-1] != labels[:, 1:])
    y = support[:-1, :] & support[1:, :] & (labels[:-1, :] != labels[1:, :])
    border[:, :-1] |= x
    border[:, 1:] |= x
    border[:-1, :] |= y
    border[1:, :] |= y
    return border


def _edge_weights(he, support, mpp, blur_um, edge_sigma):
    od = -np.log((he.astype(np.float32) + 1.) / 256.)
    if blur_um:
        radius_x = int(np.ceil(3 * blur_um / mpp[0]))
        radius_y = int(np.ceil(3 * blur_um / mpp[1]))
        od = cv2.GaussianBlur(od, (2 * radius_x + 1, 2 * radius_y + 1),
            blur_um / mpp[0], sigmaY=blur_um / mpp[1], borderType=cv2.BORDER_REFLECT_101)
    dx = np.sum((od[:, 1:] - od[:, :-1]) ** 2, axis=-1)
    dy = np.sum((od[1:, :] - od[:-1, :]) ** 2, axis=-1)
    wx = np.exp(-dx / (2 * edge_sigma**2)).astype(np.float32)
    wy = np.exp(-dy / (2 * edge_sigma**2)).astype(np.float32)
    wx *= support[:, 1:] & support[:, :-1]
    wy *= support[1:, :] & support[:-1, :]
    return wx, wy


def _refine_patch(p, he, support, movable, mpp, spec):
    original = p.astype(np.float32, copy=True)
    current = original.copy()
    wx, wy = _edge_weights(he, support, mpp, spec["he_blur_um"], spec["he_edge_sigma"])
    strength = spec["smoothness"]
    for _ in range(spec["iterations"]):
        total = original.copy()
        mass = np.ones(original.shape[:2], np.float32)
        total[:, :-1] += (strength * wx)[..., None] * current[:, 1:]
        total[:, 1:] += (strength * wx)[..., None] * current[:, :-1]
        total[:-1, :] += (strength * wy)[..., None] * current[1:, :]
        total[1:, :] += (strength * wy)[..., None] * current[:-1, :]
        mass[:, :-1] += strength * wx
        mass[:, 1:] += strength * wx
        mass[:-1, :] += strength * wy
        mass[1:, :] += strength * wy
        current = total / mass[..., None]
        current[~movable] = original[~movable]
    return current


def refine_probability_boundaries(probability, he, support, mpp_xy, class_ids, spec,
                                  *, core_size=512, memory_budget_bytes=3 * 1024**3,
                                  require_represented_classes=True):
    """Refine only uncertain pixels near a class boundary; return exact labels.

    A single full-field semantic probability map and tissue support are required.
    H&E controls spatial conductance, never class identity. The finite support
    radius is blur radius plus iterations plus one adjacent-pixel edge. Halos
    make every core calculation identical regardless of core_size.
    """
    p, he, support = map(np.asarray, (probability, he, support))
    spec = parameters(spec)
    mpp = spacing_xy(mpp_xy)
    ids = np.asarray(class_ids)
    if type(require_represented_classes) is not bool:
        raise ValueError("Class-preservation scope flag must be boolean")
    if (p.ndim != 3 or p.dtype != np.dtype(np.float32) or p.shape[-1] < 2 or
            he.shape != (*p.shape[:2], 3) or he.dtype != np.uint8 or
            support.shape != p.shape[:2] or support.dtype != np.dtype(bool) or
            ids.shape != (p.shape[-1],) or ids.dtype.kind not in "iu" or
            np.any(ids < 1) or np.any(ids > 255) or len(set(ids.tolist())) != len(ids) or
            not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1) or
            not np.allclose(p.sum(-1), 1, atol=2e-6, rtol=0)):
        raise ValueError("Require aligned finite global probabilities, RGB H&E, support and class IDs")
    admission = preflight(p.shape[:2], p.shape[-1], spec, core_size=core_size,
        mpp_xy=mpp,
        memory_budget_bytes=memory_budget_bytes)
    raw_index = np.argmax(p, axis=-1)
    raw_labels = ids[raw_index].astype(np.uint8)
    raw_labels[~support] = 0
    if p.shape[-1] == 2:
        margin = np.abs(p[..., 0] - p[..., 1])
    else:
        top_two = np.partition(p, p.shape[-1] - 2, axis=-1)[..., -2:]
        margin = top_two[..., 1] - top_two[..., 0]
    border = _internal_border(raw_labels, support)
    if np.any(border):
        distance = distance_transform_edt(~border, sampling=(mpp[1], mpp[0]))
        movable = support & (margin <= spec["uncertainty_margin"]) & \
            (distance <= spec["maximum_distance_um"])
        del distance
    else:
        movable = np.zeros(support.shape, bool)
    del margin, border
    output = raw_labels.copy()
    blur_radius_x = int(np.ceil(3 * spec["he_blur_um"] / mpp[0]))
    blur_radius_y = int(np.ceil(3 * spec["he_blur_um"] / mpp[1]))
    halo_x = blur_radius_x + spec["iterations"] + 1
    halo_y = blur_radius_y + spec["iterations"] + 1
    for y0 in range(0, p.shape[0], core_size):
        y1 = min(y0 + core_size, p.shape[0])
        for x0 in range(0, p.shape[1], core_size):
            x1 = min(x0 + core_size, p.shape[1])
            if not np.any(movable[y0:y1, x0:x1]):
                continue
            a, b = max(0, y0 - halo_y), min(p.shape[0], y1 + halo_y)
            c, d = max(0, x0 - halo_x), min(p.shape[1], x1 + halo_x)
            patch = _refine_patch(p[a:b, c:d], he[a:b, c:d], support[a:b, c:d],
                movable[a:b, c:d], mpp, spec)
            core = patch[y0-a:y1-a, x0-c:x1-c]
            output[y0:y1, x0:x1] = ids[np.argmax(core, axis=-1)].astype(np.uint8)
    output[~support] = 0
    if not np.array_equal(output[~movable], raw_labels[~movable]):
        raise AssertionError("Boundary refinement changed a confident or distant pixel")
    input_classes = set(map(int, np.unique(raw_labels[support])))
    output_classes = set(map(int, np.unique(output[support])))
    represented_classes_preserved = input_classes.issubset(output_classes)
    if require_represented_classes and not represented_classes_preserved:
        raise ValueError("Boundary refinement erased a represented tissue class")
    changed = output != raw_labels
    report = dict(method=FORMAT, **admission, parameters=spec,
        class_ids=ids.astype(int).tolist(), mpp_xy=mpp.tolist(),
        probability_sha256=_digest(p), he_sha256=_digest(he),
        support_sha256=_digest(support), input_label_sha256=_digest(raw_labels),
        output_label_sha256=_digest(output), movable_pixels=int(np.count_nonzero(movable)),
        changed_pixels=int(np.count_nonzero(changed)),
        changed_outside_movable_pixels=int(np.count_nonzero(changed & ~movable)),
        unsupported_output_pixels=int(np.count_nonzero(output[~support])),
        require_represented_classes=bool(require_represented_classes),
        represented_classes_preserved=represented_classes_preserved,
        input_represented_classes=sorted(input_classes),
        output_represented_classes=sorted(output_classes),
        processing_core_size=core_size,
        class_identity_from_he=False, expert_annotation_used=False,
        biological_accuracy_validated=False, partition_invariant_by_finite_halo=True,
        exact_mask_is_authoritative=True)
    return output, report


def publish_boundary_products(mask, folder, mpp_xy, names, *, tolerance_um=3.,
                              boundary_receipt=None, source_mask_path=None,
                              semantic_cleanup_receipt=None,
                              source_frame_id=None, image_sha256=None):
    """Publish exact scientific mask plus separately named compact review.

    The compact GeoJSON and its own TIFF are a matched display pair. The exact
    TIFF/GeoJSON remain authoritative for scientific measurements. A fitted
    review introducing new background inside an exact tissue pixel is rejected.
    """
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError(folder)
    mask = np.asarray(mask)
    if mask.dtype != np.uint8 or mask.ndim != 2:
        raise ValueError("Publish a two-dimensional uint8 native class mask")
    if not np.isfinite(tolerance_um) or tolerance_um <= 0 or tolerance_um > 20:
        raise ValueError("Require a positive bounded physical review tolerance")
    mpp = spacing_xy(mpp_xy)
    if semantic_cleanup_receipt is not None and not _cleanup_receipt_valid(
            semantic_cleanup_receipt, _digest(mask), mpp):
        raise ValueError("Semantic cleanup receipt does not bind the supplied exact mask")
    preceding_sha = (semantic_cleanup_receipt["input_label_sha256"]
                     if semantic_cleanup_receipt is not None else _digest(mask))
    if boundary_receipt is not None and (
            not isinstance(boundary_receipt, dict) or
            boundary_receipt.get("method") != FORMAT or
            boundary_receipt.get("output_label_sha256") != preceding_sha):
        raise ValueError("Boundary receipt does not bind the supplied exact mask")
    source_file_sha256 = None
    if source_mask_path is not None:
        source_path = Path(source_mask_path)
        source_labels = tifffile.imread(source_path)
        if (source_labels.dtype != np.uint8 or source_labels.ndim != 2 or
                not np.array_equal(source_labels, mask)):
            raise ValueError("Source semantic TIFF does not match the exact native mask")
        source_file_sha256 = _file_sha256(source_path)
    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bora-boundary-", dir=folder.parent) as tmp:
        stage = Path(tmp) / "result"
        stage.mkdir()
        exact = publish_exact_mask(mask, stage / "exact", mpp, names,
            source_frame_id=source_frame_id, image_sha256=image_sha256)
        source = json.loads((stage / "exact" / "mask.geojson").read_text())
        # Fitting can create new sub-threshold slivers even when the exact
        # source has no small components. Enforce the requested physical area
        # in the optional display vector too, without changing the exact pair.
        review_minimum_pixels = (max(20, int(np.ceil(
            semantic_cleanup_receipt["min_area_um2"] / float(np.prod(mpp)))))
            if semantic_cleanup_receipt is not None else 20)
        review = compact(source, mpp, mask.shape, tolerance_um=tolerance_um,
            minimum_pixels=review_minimum_pixels)
        review["semantic_boundary"] = dict(format="bora-semantic-boundary-v1",
            role="fitted_review", source_role="native_exact_semantic_boundary",
            source_mask_sha256=exact["source_mask_sha256"],
            source_mask_file_sha256=source_file_sha256,
            source_exact_geojson_sha256=exact["sha256"]["mask.geojson"],
            geometry_simplified=not review["compact_boundary_report"]["original_export_fallback"],
            predicted_not_expert_annotation=True)
        geometries = [shape(feature["geometry"]) for feature in review["features"]]
        if any(not geom.is_valid for geom in geometries):
            raise ValueError("Compact review contains invalid geometry")
        if any(geometries[i].intersection(geometries[j]).area > 1e-7
               for i in range(len(geometries)) for j in range(i)):
            raise ValueError("Compact review contains overlapping classes")
        pairs = [(feature["geometry"], int(feature["properties"]["value"]))
                 for feature in review["features"]]
        display = (rasterize(pairs, out_shape=mask.shape, fill=0, dtype=np.uint8)
                   if pairs else np.zeros_like(mask))
        if np.any((mask > 0) & (display == 0)):
            raise ValueError("Compact review introduced background inside exact tissue")
        view_dir = stage / "review"
        view_dir.mkdir()
        (view_dir / "mask.geojson").write_text(json.dumps(review, allow_nan=False,
            separators=(",", ":")) + "\n")
        tifffile.imwrite(view_dir / "mask.ome.tif", display, ome=True,
            compression="deflate", photometric="minisblack",
            metadata=dict(axes="YX", PhysicalSizeX=float(mpp[0]),
                PhysicalSizeY=float(mpp[1]), PhysicalSizeXUnit="µm", PhysicalSizeYUnit="µm"))
        review_hashes = {name: _file_sha256(view_dir / name)
                         for name in ("mask.geojson", "mask.ome.tif")}
        proof = dict(format="bora-boundary-products-v1", exact=exact,
            boundary_refinement=boundary_receipt,
            semantic_cleanup=semantic_cleanup_receipt,
            source_frame_id=source_frame_id, image_sha256=image_sha256,
            compact_boundary_report=review.get("compact_boundary_report"),
            review_minimum_region_area_pixels=review_minimum_pixels,
            source_role="native_semantic_mask",
            source_mask_sha256=exact["source_mask_sha256"],
            source_mask_file_sha256=source_file_sha256,
            exact_boundary_role="native_exact_semantic_boundary",
            review_boundary_role="fitted_review_boundary",
            review_sha256=review_hashes,
            review_raster_agrees_with_geojson=True,
            review_introduced_background_pixels=0,
            review_changed_pixels=int(np.count_nonzero(display != mask)),
            exact_mask_authoritative=True, full_wsi_qualified=False,
            input_mask_sha256=_digest(mask), review_mask_sha256=_digest(display))
        (stage / "BOUNDARY_PRODUCTS_COMPLETE.json").write_text(json.dumps(proof,
            indent=2, allow_nan=False) + "\n")
        verify_boundary_products(stage, expected_mask_sha256=_digest(mask),
            source_mask_path=source_mask_path)
        if folder.exists():
            raise FileExistsError(folder)
        os.rename(stage, folder)
    return json.loads((folder / "BOUNDARY_PRODUCTS_COMPLETE.json").read_text())


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_boundary_products(folder, *, expected_mask_sha256=None,
                             source_mask_path=None):
    """Fail if a published exact vector is not the native semantic raster.

    The fitted review pair is checked separately. The check never consults
    rendered channel colors, display opacity, or expert annotations. It reads
    bounded native crop artifacts and returns their provenance receipt.
    """
    folder = Path(folder)
    proof = json.loads((folder / "BOUNDARY_PRODUCTS_COMPLETE.json").read_text())
    if (proof.get("format") != "bora-boundary-products-v1" or
            proof.get("source_role") != "native_semantic_mask" or
            proof.get("exact_boundary_role") != "native_exact_semantic_boundary" or
            proof.get("review_boundary_role") != "fitted_review_boundary"):
        raise ValueError("Boundary-products role contract is missing or invalid")
    source_sha = proof.get("source_mask_sha256")
    if (source_sha is None or source_sha != proof["exact"].get("source_mask_sha256") or
            expected_mask_sha256 is not None and source_sha != expected_mask_sha256):
        raise ValueError("Exact boundary source-mask checksum does not match")
    if (proof.get("input_mask_sha256") != source_sha or
            proof.get("exact_mask_authoritative") is not True or
            proof.get("full_wsi_qualified") is not False or
            proof["exact"].get("format") != "bora-exact-whole-field-vector-v1" or
            proof["exact"].get("output_role") != "native_exact_semantic_boundary" or
            proof["exact"].get("exact_input_raster_preserved") is not True or
            proof["exact"].get("exact_geojson_raster_agreement") is not True or
            proof["exact"].get("geometry_simplified") is not False or
            proof["exact"].get("cleanup_applied") is not False):
        raise ValueError("Exact boundary provenance contract is incomplete")
    cleanup = proof.get("semantic_cleanup")
    if cleanup is not None and not _cleanup_receipt_valid(
            cleanup, source_sha, proof["exact"].get("native_mpp_xy")):
        raise ValueError("Semantic cleanup receipt does not bind the exact mask")
    minimum = (max(20, int(np.ceil(cleanup["min_area_um2"] /
        float(np.prod(proof["exact"]["native_mpp_xy"])))))
        if cleanup is not None else 20)
    if proof.get("review_minimum_region_area_pixels", 20) != minimum:
        raise ValueError("Fitted review minimum area is not bound to semantic cleanup")
    receipt = proof.get("boundary_refinement")
    preceding_sha = (cleanup.get("input_label_sha256")
                     if cleanup is not None else source_sha)
    if receipt is not None and (receipt.get("method") != FORMAT or
                                receipt.get("output_label_sha256") != preceding_sha):
        raise ValueError("Boundary refinement receipt does not bind the exact mask")
    source_path = Path(source_mask_path) if source_mask_path is not None else None
    if source_path is not None:
        file_sha = _file_sha256(source_path)
        if (proof.get("source_mask_file_sha256") is not None and
                proof["source_mask_file_sha256"] != file_sha):
            raise ValueError("Source semantic TIFF checksum does not match")
    exact_mask = None
    exact_classes = None
    exact_names = None
    exact_mpp = proof["exact"].get("native_mpp_xy")
    for role, expected_meta, hashes in (
            ("exact", "native_exact", proof["exact"]["sha256"]),
            ("review", "fitted_review", proof["review_sha256"])):
        directory = folder / role
        for name in ("mask.geojson", "mask.ome.tif"):
            if _file_sha256(directory / name) != hashes[name]:
                raise ValueError(role + " boundary artifact checksum does not match")
        data = json.loads((directory / "mask.geojson").read_text())
        meta = data.get("semantic_boundary", {})
        if (meta.get("format") != "bora-semantic-boundary-v1" or
                meta.get("role") != expected_meta or
                meta.get("source_mask_sha256") != source_sha or
                meta.get("predicted_not_expert_annotation") is not True or
                data.get("coordinate_space") != "native_image_pixel_edges" or
                data.get("origin") != "top_left" or
                data.get("native_mpp_xy") != exact_mpp or
                data.get("source_frame_id") != proof.get("source_frame_id") or
                data.get("image_sha256") != proof.get("image_sha256") or
                data.get("source_frame_id") != proof["exact"].get("source_frame_id") or
                data.get("image_sha256") != proof["exact"].get("image_sha256")):
            raise ValueError(role + " boundary lacks native semantic source binding")
        if role == "exact" and meta.get("geometry_simplified") is not False:
            raise ValueError("Exact semantic boundary was fitted or simplified")
        if meta.get("source_role") != ("native_semantic_mask" if role == "exact"
                                      else "native_exact_semantic_boundary"):
            raise ValueError(role + " boundary has the wrong source role")
        if role == "review" and meta.get("source_exact_geojson_sha256") != \
                proof["exact"]["sha256"]["mask.geojson"]:
            raise ValueError("Review boundary is not linked to the exact GeoJSON")
        if role == "review" and meta.get("source_mask_file_sha256") != \
                proof.get("source_mask_file_sha256"):
            raise ValueError("Review boundary is not linked to the source semantic TIFF")
        if role == "review" and (data.get("compact_boundary_report") !=
                                 proof.get("compact_boundary_report") or
                                 meta.get("geometry_simplified") is not
                                 (not data["compact_boundary_report"]["original_export_fallback"])):
            raise ValueError("Fitted review method receipt does not match its vector")
        labels = tifffile.imread(directory / "mask.ome.tif")
        if (labels.dtype != np.uint8 or labels.ndim != 2 or
                list(labels.shape) != proof["exact"]["native_shape_yx"] or
                list(labels.shape) != data.get("native_shape_yx")):
            raise ValueError(role + " boundary has a different native raster grid")
        values = []
        names = {}
        geometries = []
        for feature in data["features"]:
            prop = feature["properties"]
            value = prop.get("value")
            if type(value) is not int or not 1 <= value <= 255:
                raise ValueError(role + " boundary has an invalid class ID")
            name = prop.get("name")
            if (not isinstance(name, str) or not name.strip() or
                    prop.get("predicted_tissue_id") != value or
                    prop.get("predicted_tissue") != name or
                    prop.get("classification", {}).get("name") != name):
                raise ValueError(role + " boundary has inconsistent tissue identity")
            values.append(value)
            names[value] = name
            geom = shape(feature["geometry"])
            if geom.is_empty or not geom.is_valid:
                raise ValueError(role + " boundary has invalid geometry")
            geometries.append(geom)
            if cleanup is not None:
                parts = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
                if any(part.area + 1e-7 < minimum or
                       any(Polygon(ring).area + 1e-7 < minimum for ring in part.interiors)
                       for part in parts):
                    raise ValueError(role + " boundary retains a sub-threshold part or hole")
        if len(values) != len(set(values)):
            raise ValueError(role + " boundary duplicates a class feature")
        if any(geometries[i].intersection(geometries[j]).area > 1e-7
               for i in range(len(geometries)) for j in range(i)):
            raise ValueError(role + " boundary classes overlap")
        pairs = [(feature["geometry"], feature["properties"]["value"])
                 for feature in data["features"]]
        roundtrip = rasterize(pairs, out_shape=labels.shape, fill=0,
                              dtype=np.uint8) if pairs else np.zeros_like(labels)
        if not np.array_equal(labels, roundtrip):
            raise ValueError(role + " GeoJSON does not reproduce its native raster")
        if role == "exact":
            if _digest(labels) != source_sha:
                raise ValueError("Exact GeoJSON is not bound to the source semantic mask")
            exact_mask = labels
            exact_classes = set(np.unique(labels[labels > 0]).astype(int))
            exact_names = names
            if (set(values) != exact_classes or
                    set(proof["exact"].get("class_ids", [])) != exact_classes):
                raise ValueError("Exact boundary classes do not match the semantic mask")
            if source_path is not None:
                source_labels = tifffile.imread(source_path)
                if (source_labels.dtype != np.uint8 or source_labels.ndim != 2 or
                        not np.array_equal(source_labels, exact_mask)):
                    raise ValueError("Source semantic TIFF does not match the exact native mask")
        elif np.any((exact_mask > 0) & (labels == 0)):
            raise ValueError("Fitted review boundary introduces false background")
        else:
            if set(values) != exact_classes or names != exact_names:
                raise ValueError("Fitted review changed exact class identities")
            if (proof.get("review_mask_sha256") != _digest(labels) or
                    proof.get("review_changed_pixels") !=
                    int(np.count_nonzero(labels != exact_mask)) or
                    proof.get("review_introduced_background_pixels") != 0 or
                    proof.get("review_raster_agrees_with_geojson") is not True):
                raise ValueError("Fitted review provenance does not match its raster")
    return proof
