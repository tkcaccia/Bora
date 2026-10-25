"""Explicit bounded-crop admission and exact global vector publication.

This module contains no encoder or launcher. These bounds admit a manageable
crop, not arbitrary whole slides, and estimates are not measured memory proof.
"""
from collections import defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
from rasterio.features import rasterize, shapes
from shapely.geometry import mapping, shape
from shapely.ops import unary_union
import tifffile

from .spacing import spacing_xy

MAX_NATIVE_SIDE = 8192
MAX_NATIVE_PIXELS = 64 * 1024 ** 2
MAX_WORKING_SIDE = 4608
MAX_WORKING_PIXELS = 24 * 1024 ** 2
MAX_BOUNDARY_EDGES = 1_000_000


def _shape(value, name):
    value = np.asarray(value)
    if (value.shape != (2,) or value.dtype.kind not in "iu" or np.any(value < 1)):
        raise ValueError(name + " must contain positive integer height and width")
    return tuple(int(v) for v in value)


def preflight(working_shape_yx, native_shape_yx, class_count, *,
              memory_budget_bytes=10 * 1024 ** 3):
    """Reject oversize whole-field work before allocating RGB probabilities.

    The conservative estimate includes dense probability temporaries, token
    fields and native pooling/vector arrays. Runtime cgroup limits remain
    necessary; passing this calculation does not prove an observed peak.
    """
    working = _shape(working_shape_yx, "Working shape")
    native = _shape(native_shape_yx, "Native shape")
    if (type(class_count) is not int or not 2 <= class_count <= 255 or
            type(memory_budget_bytes) is not int or memory_budget_bytes < 1):
        raise ValueError("Require 2..255 observed classes and a positive integer memory budget")
    if (max(working) > MAX_WORKING_SIDE or np.prod(working) > MAX_WORKING_PIXELS or
            max(native) > MAX_NATIVE_SIDE or np.prod(native) > MAX_NATIVE_PIXELS):
        raise ValueError("Whole-field crop exceeds declared shape bounds; full WSI is not admitted")
    tokens = ((working[0] + 13) // 14) * ((working[1] + 13) // 14)
    estimate = (working[0] * working[1] * (32 * class_count + 96) +
                tokens * 1280 * 12 + native[0] * native[1] * 20 + 1024 ** 3)
    if estimate > memory_budget_bytes:
        raise ValueError("Whole-field conservative memory estimate exceeds declared budget")
    return dict(format="bora-whole-field-admission-v1", working_shape_yx=list(working),
        native_shape_yx=list(native), class_count=class_count,
        estimated_peak_bytes=int(estimate), memory_budget_bytes=memory_budget_bytes,
        estimate_is_measured_peak=False, full_wsi_qualified=False,
        independently_fitted_processing_windows=False)


def _coordinate_count(geometry):
    if geometry.geom_type == "Polygon":
        return len(geometry.exterior.coords) + sum(len(r.coords) for r in geometry.interiors)
    return sum(_coordinate_count(g) for g in geometry.geoms)


def dissolve_same_class(data, *, maximum_coordinates=MAX_BOUNDARY_EDGES * 2):
    """Union only equal-class polygon coverage; never reconcile class conflicts.

    Existing pixel/continuous geometry is retained exactly, including holes.
    Different-class overlap is rejected. This removes redundant export cuts,
    but cannot repair upstream window-dependent tissue predictions.
    """
    groups, properties = defaultdict(list), {}
    lineage = defaultdict(list)
    count = 0
    for index, feature in enumerate(data["features"]):
        prop = feature.get("properties", {})
        value = prop.get("value", prop.get("predicted_tissue_id"))
        if type(value) is not int or not 1 <= value <= 255:
            raise ValueError("Every feature requires an integer class value in 1..255")
        geom = shape(feature["geometry"])
        if geom.geom_type not in ("Polygon", "MultiPolygon") or geom.is_empty or not geom.is_valid:
            raise ValueError("Valid nonempty polygonal input required; geometry is never silently repaired")
        count += _coordinate_count(geom)
        if count > maximum_coordinates:
            raise ValueError("Whole-field vector coordinate bound exceeded")
        names = (prop.get("name"), prop.get("predicted_tissue"), prop.get("classification", {}).get("name"))
        if value in properties:
            previous = properties[value]
            old = (previous.get("name"), previous.get("predicted_tissue"), previous.get("classification", {}).get("name"))
            if old != names:
                raise ValueError("Conflicting tissue names for the same class ID")
        else:
            properties[value] = copy.deepcopy(prop)
        groups[value].append(geom)
        lineage[value].append(index)
    merged = {value: unary_union(parts) for value, parts in groups.items()}
    if any(not geom.is_valid for geom in merged.values()):
        raise ValueError("Invalid same-class union")
    values = sorted(merged)
    for i, value in enumerate(values):
        for other in values[:i]:
            if merged[value].intersection(merged[other]).area > 1e-7:
                raise ValueError("Different-class polygon overlap cannot be dissolved")
    result = copy.deepcopy({key: value for key, value in data.items() if key != "features"})
    result["features"] = []
    mpp = spacing_xy(data["native_mpp_xy"]) if "native_mpp_xy" in data else None
    for value in values:
        prop = properties[value]
        prop.pop("window_core_xywh", None)
        prop.update(value=value, area_pixels=float(merged[value].area),
                    source_feature_indices=lineage[value])
        if mpp is not None:
            prop["area_um2"] = float(merged[value].area * np.prod(mpp))
        result["features"].append(dict(type="Feature", geometry=mapping(merged[value]), properties=prop))
    result["vector_export"] = dict(method="whole-field exact same-class union",
        geometry_simplified=False, input_feature_count=len(data["features"]),
        output_feature_count=len(values), window_divisions_retained=False,
        different_class_disagreement_reconciled=False)
    return result


def _mask_admission(mask):
    mask = np.asarray(mask)
    shape_yx = _shape(mask.shape, "Native mask shape")
    if (mask.dtype != np.uint8 or max(shape_yx) > MAX_NATIVE_SIDE or
            mask.size > MAX_NATIVE_PIXELS):
        raise ValueError("Exact native vector export requires a bounded uint8 crop")
    # Complexity guard before constructing potentially enormous polygon lists.
    edges = 0
    for y in range(0, len(mask), 256):
        part = mask[y:min(len(mask), y + 256)]
        edges += int(np.count_nonzero(part[:, 1:] != part[:, :-1]))
        after = mask[y + 1:min(len(mask), y + 257)]
        edges += int(np.count_nonzero(part[:len(after)] != after))
        if edges > MAX_BOUNDARY_EDGES:
            raise ValueError("Exact crop polygonization exceeds boundary-complexity bound")
    return mask, edges


def publish_exact_mask(mask, folder, mpp_xy, names, *, source_frame_id=None,
                       image_sha256=None):
    """Publish a NEW crop folder, preserving its supplied class mask exactly.

    No cleanup, simplification or minimum component area is applied. If those
    scientific operations are desired, apply them globally and document them
    before calling this function. The final completion record is atomic with
    the directory publication. One MultiPolygon feature per class is allowed.
    """
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError(folder)
    if (source_frame_id is None) != (image_sha256 is None):
        raise ValueError("Source frame and image checksum must be supplied together")
    if source_frame_id is not None and (
            not isinstance(source_frame_id, str) or
            not source_frame_id.strip() or
            not isinstance(image_sha256, str) or
            re.fullmatch(r"[0-9a-f]{64}", image_sha256) is None):
        raise ValueError("Invalid source frame or image SHA-256")
    mask, edges = _mask_admission(mask)
    mpp = spacing_xy(mpp_xy)
    names = {int(k): str(v) for k, v in names.items()}
    observed = np.unique(mask[mask > 0])
    if any(int(value) not in names or not names[int(value)].strip() for value in observed):
        raise ValueError("Every observed positive class needs a tissue name")
    features = []
    for geometry, raw in shapes(mask, mask=mask > 0, connectivity=4):
        value = int(raw)
        name = names[value]
        features.append(dict(type="Feature", geometry=geometry, properties=dict(value=value,
            name=name, classification=dict(name=name), predicted_tissue_id=value,
            predicted_tissue=name, objectType="annotation")))
    mask_sha256 = hashlib.sha256(np.ascontiguousarray(mask).view(np.uint8)).hexdigest()
    data = dissolve_same_class(dict(type="FeatureCollection", features=features,
        coordinate_space="native_image_pixel_edges", origin="top_left",
        native_shape_yx=list(mask.shape), native_mpp_xy=mpp.tolist()))
    if source_frame_id is not None:
        data["source_frame_id"] = source_frame_id
        data["image_sha256"] = image_sha256
    # The displayed colors and opacity are never the segmentation source. This
    # record binds the pixel-edge geometry to the native categorical raster.
    data["semantic_boundary"] = dict(format="bora-semantic-boundary-v1",
        role="native_exact", source_role="native_semantic_mask",
        source_mask_sha256=mask_sha256, geometry_simplified=False,
        rasterization="rasterio.features.rasterize; pixel-edge coordinates",
        predicted_not_expert_annotation=True)
    pairs = [(f["geometry"], f["properties"]["value"]) for f in data["features"]]
    rendered = (rasterize(pairs, out_shape=mask.shape, fill=0, dtype=np.uint8)
                if pairs else np.zeros_like(mask))
    if not np.array_equal(mask, rendered):
        raise ValueError("Exact global vector publication changed the supplied class raster")
    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bora-exact-", dir=folder.parent) as temporary:
        staging = Path(temporary) / "result"
        staging.mkdir()
        (staging / "mask.geojson").write_text(json.dumps(data, allow_nan=False, separators=(",", ":")) + "\n")
        tifffile.imwrite(staging / "mask.ome.tif", mask, ome=True, compression="deflate",
            photometric="minisblack", metadata=dict(axes="YX", PhysicalSizeX=float(mpp[0]),
                PhysicalSizeY=float(mpp[1]), PhysicalSizeXUnit="µm", PhysicalSizeYUnit="µm"))
        hashes = {}
        for name in ("mask.geojson", "mask.ome.tif"):
            digest = hashlib.sha256()
            with (staging / name).open("rb") as stream:
                for block in iter(lambda: stream.read(1024 ** 2), b""):
                    digest.update(block)
            hashes[name] = digest.hexdigest()
        report = dict(format="bora-exact-whole-field-vector-v1", native_shape_yx=list(mask.shape),
            native_mpp_xy=mpp.tolist(), feature_count=len(data["features"]),
            source_frame_id=source_frame_id, image_sha256=image_sha256,
            boundary_edges=edges, class_ids=observed.astype(int).tolist(),
            source_role="native_semantic_mask", source_mask_sha256=mask_sha256,
            output_role="native_exact_semantic_boundary",
            exact_input_raster_preserved=True, exact_geojson_raster_agreement=True,
            geometry_simplified=False, cleanup_applied=False, window_divisions_retained=False,
            full_wsi_qualified=False, sha256=hashes)
        (staging / "EXACT_VECTOR_COMPLETE.json").write_text(json.dumps(report, indent=2) + "\n")
        if folder.exists():
            raise FileExistsError(folder)
        os.rename(staging, folder)
    return report
