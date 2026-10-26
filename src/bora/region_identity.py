"""Frozen Neighbor-context tissue identity for already accepted tissue polygons.

Only properties change. Pooling follows ``groups.describe_groups`` at native
pixel centres, including disconnected pieces of each original anonymous group.
Stripes bound raster memory; tokens are read in blocks from a memory map.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np
from affine import Affine
from rasterio.features import rasterize
from shapely.geometry import shape

from .classifier import NeighborContextClassifier
from .groups import spacing_xy
from .pipeline import digest, json_file


def geometry_digest(features):
    """Order-sensitive canonical hash of geometry alone, excluding properties."""
    return hashlib.sha256(json.dumps([f["geometry"] for f in features],
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _positive_shape(value, name):
    a = np.asarray(value)
    if a.shape != (2,) or a.dtype.kind not in "iu" or np.any(a <= 0):
        raise ValueError(f"{name} must contain two positive integer dimensions")
    return tuple(map(int, a))


def _sha(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(f"{name} requires a lowercase SHA-256")
    return value


def _field(properties, field):
    value = properties
    for part in field.split("."):
        value = value[part]
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 2**31 - 1:
        raise ValueError("Source group IDs must be positive 32-bit integers")
    return value


def _accumulate(regions, row, native_shape, working_shape, token_shape,
                weights, interface_counts, previous):
    h, w = native_shape
    gh, gw = token_shape
    yy = np.minimum((((np.arange(row, row + len(regions)) + .5)
                      * working_shape[0] / h / 14)).astype(np.int64), gh - 1)
    xx = np.minimum((((np.arange(w) + .5) * working_shape[1] / w / 14))
                    .astype(np.int64), gw - 1)
    token_ids = yy[:, None] * gw + xx[None, :]
    for i in range(len(weights)):
        weights[i] += np.bincount(token_ids[regions == i + 1], minlength=gh * gw)
    interfaces = [(regions[:, :-1], regions[:, 1:], 0),
                  (regions[:-1], regions[1:], 1)]
    if previous is not None:
        interfaces.append((previous, regions[0], 1))
    n = len(weights)
    for a, b, axis in interfaces:
        use = (a > 0) & (b > 0) & (a != b)
        codes = (a[use].astype(np.int64) - 1) * n + b[use] - 1
        counts = np.bincount(codes, minlength=n * n).reshape(n, n)
        interface_counts[axis] += counts + counts.T


def describe_region_polygons(tokens, features, native_shape_yx, working_shape_yx,
                             native_mpp_xy, *, group_field="value", stripe_rows=256,
                             maximum_working_bytes=256 * 1024**2):
    """Return source IDs, descriptors, native pixel masses and physical graph.

    Geometry is in native image pixel-edge coordinates. Different source groups
    must not overlap at a native pixel centre. Groups with no covered centre are
    rejected, rather than assigned invented observations. The allocation guard
    covers the descriptor accumulator and principal stripe/block arrays, not the
    model, input GeoJSON, operating-system page cache, or a process memory limit.
    """
    native = _positive_shape(native_shape_yx, "native_shape_yx")
    working = _positive_shape(working_shape_yx, "working_shape_yx")
    mpp = spacing_xy(native_mpp_xy)
    if (isinstance(stripe_rows, bool) or not isinstance(stripe_rows, int) or stripe_rows < 1
            or isinstance(maximum_working_bytes, bool)
            or not isinstance(maximum_working_bytes, int) or maximum_working_bytes < 1):
        raise ValueError("Positive integer stripe_rows and maximum_working_bytes required")
    expected = tuple((np.asarray(working) + 13) // 14)
    if (tokens.shape != (*expected, 1280) or tokens.dtype.kind != "f"
            or tokens.dtype.itemsize > 8):
        raise ValueError("Full 1280-channel tokens must cover the 14-pixel padded working canvas")
    if not features:
        raise ValueError("A nonempty tissue FeatureCollection is required")
    ids = np.asarray(sorted({_field(f["properties"], group_field) for f in features}), np.int64)
    if len(ids) > 255:
        raise ValueError("At most 255 source groups are supported")
    lookup = {int(group): i + 1 for i, group in enumerate(ids)}
    shapes = []
    for feature in features:
        geom = shape(feature["geometry"])
        if geom.geom_type not in ("Polygon", "MultiPolygon") or geom.is_empty or not geom.is_valid:
            raise ValueError("Valid nonempty Polygon or MultiPolygon geometry required")
        x0, y0, x1, y1 = geom.bounds
        if not np.isfinite(geom.bounds).all() or x0 < 0 or y0 < 0 or x1 > native[1] or y1 > native[0]:
            raise ValueError("Tissue geometry lies outside the native image frame")
        shapes.append((feature["geometry"], lookup[_field(feature["properties"], group_field)]))
    shapes.sort(key=lambda item: item[1])
    gh, gw = expected
    block_rows = min(4096, gh * gw)
    fixed_bytes = ((len(ids) + 1) * gh * gw * 8 + len(ids)**2 * 8 * 3
                   + len(ids) * 1280 * 8 * 4 + native[1] * 16
                   + block_rows * (1280 * 12 + len(ids) * 8))
    rows = min(stripe_rows, native[0], (maximum_working_bytes - fixed_bytes) // (native[1] * 40))
    if rows < 1:
        raise ValueError("Descriptor accumulator and one stripe exceed maximum_working_bytes")
    stripe_rows = rows
    weights = np.zeros((len(ids), gh * gw), np.float64)
    interface_counts = np.zeros((2, len(ids), len(ids)), np.int64)
    previous = None
    for row in range(0, native[0], stripe_rows):
        options = dict(out_shape=(min(stripe_rows, native[0] - row), native[1]),
                       transform=Affine.translation(0, row), fill=0, dtype="uint8")
        stripe = rasterize(shapes, **options)
        # A reversed paint order must give the same label at every pixel centre.
        # Same-group overlap is a union; conflicting group overlap is ambiguous.
        if not np.array_equal(stripe, rasterize(shapes[::-1], **options)):
            raise ValueError("Different source groups overlap at native pixel centres")
        _accumulate(stripe, row, native, working, expected, weights, interface_counts, previous)
        previous = stripe[-1].copy()
    mass = weights.sum(1)
    if np.any(mass <= 0):
        raise ValueError("Every original source group must cover native pixel centres")
    flat = tokens.reshape(-1, 1280)
    sums = np.zeros((len(ids), 1280), np.float64)
    for begin in range(0, len(flat), block_rows):
        block = np.asarray(flat[begin:begin + block_rows], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError("Nonfinite Virchow2 spatial token")
        sums += weights[:, begin:begin + len(block)] @ block
    # Preserve the reference group_means float32 rounding before context pooling.
    means = (sums / mass[:, None]).astype(np.float32).astype(np.float64)
    # Integer interface counting makes descriptors independent of stripe height.
    graph = interface_counts[0] * mpp[1] + interface_counts[1] * mpp[0]
    total = graph.sum(1)
    neighbours = means.copy()
    has = total > 0
    neighbours[has] = (graph @ means)[has] / total[has, None]
    return ids, np.concatenate([means, means - neighbours], axis=1), mass, graph


def classify_regions(contours_path, registration_path, tokens_path, model_path,
                     output, *, group_field="value", stripe_rows=256,
                     maximum_working_bytes=256 * 1024**2):
    """Score existing polygons with one frozen head, publishing a new directory.

    The manifest asserts that tokens and polygons share the native/working
    physical field. This function validates dimensions and hashes; it cannot
    independently establish biological suitability or image registration.
    """
    started = perf_counter()
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    paths = dict(contours=Path(contours_path), registration=Path(registration_path),
                 tokens=Path(tokens_path), model=Path(model_path))
    hashes = {key: digest(path) for key, path in paths.items()}
    reg = json.loads(paths["registration"].read_text())
    if (reg.get("format") != "bora-region-registration-v1"
            or reg.get("pixel_aligned") is not True
            or reg.get("registration_method") != "native_same_grid"
            or not isinstance(reg.get("source_frame_id"), str)
            or not reg["source_frame_id"].strip()):
        raise ValueError("A registered native_same_grid bora-region-registration-v1 manifest is required")
    native = _positive_shape(reg["native_shape_yx"], "native_shape_yx")
    working = _positive_shape(reg["working_shape_yx"], "working_shape_yx")
    mpp = spacing_xy(reg["native_mpp_xy"])
    image_sha = _sha(reg["image_sha256"], "image_sha256")
    for key in ("contours", "tokens"):
        if _sha(reg[f"{key}_sha256"], f"{key}_sha256") != hashes[key]:
            raise ValueError(f"Registered {key} hash differs")
    doc = json.loads(paths["contours"].read_text())
    if doc.get("type") != "FeatureCollection":
        raise ValueError("A tissue FeatureCollection is required")
    for key, value in (("native_shape_yx", list(native)), ("native_mpp_xy", mpp.tolist()),
                       ("source_image_sha256", image_sha),
                       ("source_frame_id", reg["source_frame_id"])):
        if key in doc and doc[key] != value:
            raise ValueError(f"Contours {key} differs from registration")
    features = doc["features"]
    before = geometry_digest(features)
    tokens = np.load(paths["tokens"], mmap_mode="r", allow_pickle=False)
    ids, x, mass, graph = describe_region_polygons(tokens, features, native, working, mpp,
        group_field=group_field, stripe_rows=stripe_rows,
        maximum_working_bytes=maximum_working_bytes)
    model = NeighborContextClassifier.load(paths["model"])
    predictions = model.predict(x)
    ranking = np.asarray(model.model_.predict(x, top=len(model.classes_)))
    responses = np.asarray(model.model_.predict_responses(x), np.float64)
    if (responses.shape != (len(ids), len(model.classes_)) or not np.isfinite(responses).all()
            or ranking.shape != responses.shape or not np.array_equal(predictions, ranking[:, 0])):
        raise ValueError("Frozen classifier response/ranking is inconsistent")
    unit = "whole_source_group_including_disconnected_components"
    groups = [dict(source_group_id=int(g), prediction_id=int(predictions[i]),
        prediction_name=model.names_[int(predictions[i])], native_pixels=int(mass[i]),
        physical_area_um2=float(mass[i] * mpp[0] * mpp[1]),
        lda_class_rank=[dict(id=int(c), name=model.names_[int(c)]) for c in ranking[i]],
        pls_class_indicator_responses=[dict(id=int(c), name=model.names_[int(c)],
            response=float(responses[i, j])) for j, c in enumerate(model.classes_)])
        for i, g in enumerate(ids)]
    result = copy.deepcopy(doc)
    by_id = {g["source_group_id"]: g for g in groups}
    for feature in result["features"]:
        props = feature["properties"]
        source = _field(props, group_field)
        group = by_id[source]
        props["source_group_id"] = source
        props["source_group_name"] = props.get("name")
        props.update(value=group["prediction_id"], name=group["prediction_name"],
            classification=dict(props.get("classification") or {}, name=group["prediction_name"]),
            predicted_tissue_id=group["prediction_id"], predicted_tissue=group["prediction_name"],
            tissue_identity_model_sha256=hashes["model"], identity_prediction_unit=unit,
            tissue_identity_status="unvalidated_model_transfer")
    if geometry_digest(result["features"]) != before:
        raise RuntimeError("Tissue polygon geometry changed")
    # Retain old mask/model provenance explicitly as source-group provenance.
    for key, destination in (("source_semantic_mask_sha256", "source_group_mask_sha256"),
                             ("global_model_binding_sha256", "source_group_model_binding_sha256")):
        if key in result:
            result[destination] = result.pop(key)
    result.update(source_frame_id=reg["source_frame_id"], native_shape_yx=list(native),
        source_image_sha256=image_sha, tissue_identity_model_sha256=hashes["model"],
        tissue_identity_status="unvalidated_model_transfer", biological_accuracy_qualified=False,
        identity_prediction_unit=unit,
        tissue_class_order=[dict(id=int(c), name=model.names_[int(c)]) for c in model.classes_])
    for key, path in paths.items():
        if digest(path) != hashes[key]:
            raise RuntimeError(f"Input {key} changed during inference")
    report = dict(format="bora-region-identity-v1", complete=True, source_frame_id=reg["source_frame_id"],
        native_shape_yx=list(native), native_mpp_xy=mpp.tolist(), working_shape_yx=list(working),
        source_image_sha256=image_sha, input_sha256=hashes,
        geometry_sha256=before, geometry_unchanged=True, feature_count=len(features),
        group_count=len(ids), groups=groups, group_field=group_field, identity_prediction_unit=unit,
        descriptor="1280 native-pixel-overlap group means + 1280 physical-boundary-neighbour contrasts",
        responses_are_calibrated_probabilities=False, biological_accuracy_qualified=False,
        registration_asserted_by_caller=True, gpu_used=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bora-identity-", dir=output.parent) as temp:
        folder = Path(temp) / "result"
        folder.mkdir()
        json_file(folder / "tissue-annotations.geojson", result)
        np.savez_compressed(folder / "descriptors.npz", x=x, group_ids=ids, area_pixels=mass,
            physical_boundary_graph=graph, predicted_class=predictions,
            class_ids=model.classes_, class_indicator_responses=responses, lda_ranking=ranking)
        report["output_sha256"] = {name: digest(folder / name)
            for name in ("tissue-annotations.geojson", "descriptors.npz")}
        report["elapsed_seconds"] = perf_counter() - started
        json_file(folder / "report.json", report)
        if output.exists():
            raise FileExistsError(output)
        os.rename(folder, output)
    return report
