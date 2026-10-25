"""Streaming, fail-closed assembly of full-slide semantic probability cores.

This stage never fits a model or creates GeoJSON.  Its input is a row-major
grid of immutable, halo-inclusive ``.npy`` probability shards produced by one
frozen global model.  It writes one disk-backed native uint8 class mask; a
separate global vectorization route is required before publishing boundaries.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np


FORMAT = "bora-wsi-probability-assembly-v1"
MAX_CLASSES = 32  # Includes the currently audited 25-class Neighbor-context head.
MAX_WINDOW_PIXELS = 16 * 1024 * 1024
CHUNK = 128
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _integer_tuple(value, length, *, positive=False):
    if (not isinstance(value, (tuple, list)) or len(value) != length or
            any(type(v) is not int or v < (1 if positive else 0) for v in value)):
        raise ValueError(f"Expected {length} {'positive' if positive else 'nonnegative'} integers")
    return tuple(value)


def _hash_string(value, name):
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _read_bound_npy(path, expected_hash, *, ndim, shape=None, dtype=None):
    path = Path(path)
    _hash_string(expected_hash, "Input checksum")
    if not path.is_file() or path.suffix != ".npy":
        raise ValueError("Probability and support inputs must be existing .npy files")
    before = path.stat()
    if _digest(path) != expected_hash:
        raise ValueError(f"Input checksum mismatch: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != ndim or (shape is not None and array.shape != shape):
        raise ValueError(f"Input array shape mismatch: {path}")
    if dtype is not None and array.dtype != dtype:
        raise ValueError(f"Input array dtype mismatch: {path}")
    return array, (before.st_size, before.st_mtime_ns)


def _check_unchanged(path, identity):
    after = Path(path).stat()
    if (after.st_size, after.st_mtime_ns) != identity:
        raise ValueError(f"Input changed during assembly: {path}")


def _validate_probability(probability, class_count):
    if (probability.ndim != 3 or probability.dtype != np.dtype("float32") or
            probability.shape[-1] != class_count or
            probability.shape[0] * probability.shape[1] > MAX_WINDOW_PIXELS):
        raise ValueError("Probability shard has unsupported shape, size or dtype")
    for y in range(0, probability.shape[0], CHUNK):
        for x in range(0, probability.shape[1], CHUNK):
            block = probability[y:y + CHUNK, x:x + CHUNK]
            if (not np.isfinite(block).all() or np.any(block < 0) or
                    np.any(block > 1) or
                    not np.allclose(block.sum(axis=-1), 1, atol=2e-6, rtol=0)):
                raise ValueError("Probability shard must contain normalized finite scores")


def _compare(a, a_origin, b, b_origin, bounds):
    """Compare a global rectangle in two aligned probability windows."""
    y0, x0, y1, x1 = bounds
    maximum, count = 0.0, 0
    for y in range(y0, y1, CHUNK):
        for x in range(x0, x1, CHUNK):
            yy, xx = min(y1, y + CHUNK), min(x1, x + CHUNK)
            aa = a[y-a_origin[0]:yy-a_origin[0], x-a_origin[1]:xx-a_origin[1]]
            bb = b[y-b_origin[0]:yy-b_origin[0], x-b_origin[1]:xx-b_origin[1]]
            tv = np.abs(aa - bb).sum(axis=-1) / 2
            maximum = max(maximum, float(tv.max()))
            count += tv.size
    return maximum, count


def _intersection(first, second):
    y0, x0 = max(first[0], second[0]), max(first[1], second[1])
    y1, x1 = min(first[2], second[2]), min(first[3], second[3])
    return (y0, x0, y1, x1) if y0 < y1 and x0 < x1 else None


def _compare_neighbors(current, neighbor):
    """Check both halo→core directions; no averaging or seam blending."""
    old = np.load(neighbor["path"], mmap_mode="r", allow_pickle=False)
    maximum, count = 0.0, 0
    for a, b in ((current["extent"], neighbor["core"]),
                 (current["core"], neighbor["extent"])):
        overlap = _intersection(a, b)
        if overlap is not None:
            value, pixels = _compare(current["array"], current["origin"],
                                     old, neighbor["origin"], overlap)
            maximum = max(maximum, value)
            count += pixels
    return maximum, count


def assemble_wsi_probability_cores(windows, shape_yx, class_ids, output_folder, *,
                                   tissue_support_path, tissue_support_sha256,
                                   global_model_binding_sha256, source_frame_id,
                                   source_image_sha256, native_mpp_xy,
                                   max_halo_total_variation=0.0,
                                   disk_budget_bytes=32 * 1024**3):
    """Publish a native uint8 semantic mask from a complete full-slide grid.

    Each window needs ``probability_path``, ``probability_sha256``, global
    ``origin_yx`` and ``core_yxhw``, the same frozen-model binding and
    ``independently_fitted=False``.  Windows must arrive in row-major order;
    all cores in one row have the same height and partition the entire width.
    All interior core sides need at least one halo pixel.  The probability
    shard is checked against both the earlier core and earlier halo for every
    adjacent window.  The support array is a native bool/uint8 .npy on the
    same grid.  Zero is reserved for unsupported tissue; positive class IDs
    describe the channels in their supplied order.

    The new output folder contains ``semantic-mask.uint8.dat`` and a completion
    receipt.  GeoJSON is intentionally not emitted: independently fitting
    vector windows would reintroduce artificial boundaries.
    """
    h, w = _integer_tuple(shape_yx, 2, positive=True)
    if h * w > 2**63 - 1:
        raise ValueError("Global mask dimensions exceed file offset range")
    if (not isinstance(class_ids, (tuple, list)) or not 2 <= len(class_ids) <= MAX_CLASSES or
            any(type(value) is not int or not 1 <= value <= 255 for value in class_ids) or
            len(set(class_ids)) != len(class_ids)):
        raise ValueError("Class IDs must be distinct positive uint8 tissue values")
    if (not isinstance(native_mpp_xy, (tuple, list)) or len(native_mpp_xy) != 2 or
            any(type(value) not in (float, int) or not np.isfinite(value) or
                value <= 0 for value in native_mpp_xy)):
        raise ValueError("Native MPP must contain two positive finite values")
    if not isinstance(source_frame_id, str) or not source_frame_id.strip():
        raise ValueError("A source frame ID is required")
    _hash_string(source_image_sha256, "Source image checksum")
    _hash_string(global_model_binding_sha256, "Frozen-model binding")
    if (type(disk_budget_bytes) is not int or disk_budget_bytes < h * w + 1024**2 or
            not isinstance(max_halo_total_variation, (float, int)) or
            not np.isfinite(max_halo_total_variation) or
            not 0 <= max_halo_total_variation <= 1):
        raise ValueError("Invalid disk budget or halo disagreement bound")
    folder = Path(output_folder)
    if folder.exists():
        raise FileExistsError(folder)
    folder.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(folder.parent).free < h * w + 64 * 1024**2:
        raise ValueError("Insufficient free disk for native semantic mask and staging")
    support, support_identity = _read_bound_npy(tissue_support_path,
        tissue_support_sha256, ndim=2, shape=(h, w))
    if support.dtype not in (np.dtype("bool"), np.dtype("uint8")):
        raise ValueError("Tissue support must be bool or uint8")

    staging = Path(tempfile.mkdtemp(prefix=".bora-wsi-assembly-", dir=folder.parent))
    mask_path = staging / "semantic-mask.uint8.dat"
    next_y = next_x = 0
    row_height = None
    previous_row, current_row = [], []
    window_count = compared_pixels = 0
    maximum = 0.0
    shard_receipts = []
    source_identities = []
    try:
        mask = np.memmap(mask_path, mode="w+", dtype=np.uint8, shape=(h, w))
        for item in windows:
            if not isinstance(item, dict) or item.get("independently_fitted") is not False:
                raise ValueError("Every shard must use one frozen global model")
            if item.get("global_model_binding_sha256") != global_model_binding_sha256:
                raise ValueError("Probability shard has a different frozen-model binding")
            y, x, ch, cw = _integer_tuple(item.get("core_yxhw"), 4)
            if ch == 0 or cw == 0 or y + ch > h or x + cw > w:
                raise ValueError("Core lies outside the slide or has zero area")
            if y != next_y:
                if next_x != w or row_height is None or y != next_y + row_height:
                    raise ValueError("Core grid has a gap, overlap or non-row-major order")
                previous_row, current_row = current_row, []
                next_y, next_x, row_height = y, 0, None
            if x != next_x or (row_height is not None and ch != row_height):
                raise ValueError("Core grid has a gap, overlap or unequal row heights")
            row_height = ch
            next_x = x + cw
            oy, ox = _integer_tuple(item.get("origin_yx"), 2)
            path = Path(item.get("probability_path"))
            probability, identity = _read_bound_npy(path, item.get("probability_sha256"), ndim=3)
            _validate_probability(probability, len(class_ids))
            ey, ex = oy + probability.shape[0], ox + probability.shape[1]
            if (ey > h or ex > w or y < oy or x < ox or y + ch > ey or x + cw > ex or
                    (y > 0 and oy == y) or (x > 0 and ox == x) or
                    (y + ch < h and ey == y + ch) or
                    (x + cw < w and ex == x + cw)):
                raise ValueError("Core/halo does not align with native slide coordinates")
            current = dict(array=probability, path=str(path), origin=(oy, ox),
                extent=(oy, ox, ey, ex), core=(y, x, y + ch, x + cw))
            neighbors = current_row[-1:] + [old for old in previous_row
                if (_intersection(current["extent"], old["core"]) or
                    _intersection(current["core"], old["extent"]))]
            for neighbor in neighbors:
                value, count = _compare_neighbors(current, neighbor)
                maximum = max(maximum, value)
                compared_pixels += count
                if count == 0 or maximum > max_halo_total_variation:
                    raise ValueError("Adjacent probability halos disagree or do not overlap")
            if (x > 0 and not current_row) or (y > 0 and not previous_row):
                raise ValueError("Missing adjacent window for a non-edge core")
            for yy in range(y, y + ch, CHUNK):
                for xx in range(x, x + cw, CHUNK):
                    y1, x1 = min(y + ch, yy + CHUNK), min(x + cw, xx + CHUNK)
                    tissue = support[yy:y1, xx:x1]
                    if support.dtype == np.uint8 and np.any(tissue > 1):
                        raise ValueError("Tissue support must be binary")
                    values = probability[yy-oy:y1-oy, xx-ox:x1-ox]
                    labels = np.asarray(class_ids, np.uint8)[np.argmax(values, axis=-1)]
                    mask[yy:y1, xx:x1] = np.where(tissue, labels, 0)
            _check_unchanged(path, identity)
            shard_receipts.append(dict(core_yxhw=[y, x, ch, cw],
                origin_yx=[oy, ox], probability_shape_yxc=list(probability.shape),
                probability_path=str(path),
                probability_sha256=item["probability_sha256"]))
            source_identities.append((path, identity))
            current_row.append({key: value for key, value in current.items() if key != "array"})
            window_count += 1
        if window_count == 0 or next_x != w or next_y + row_height != h:
            raise ValueError("Core grid leaves uncovered native-slide pixels")
        if window_count > 1 and compared_pixels == 0:
            raise ValueError("Adjacent halo probabilities were not compared")
        _check_unchanged(tissue_support_path, support_identity)
        for source_path, identity in source_identities:
            _check_unchanged(source_path, identity)
        mask.flush()
        del mask
        mask_sha256 = _digest(mask_path)
        receipt = dict(format=FORMAT, native_shape_yx=[h, w],
            native_mpp_xy=list(native_mpp_xy), source_frame_id=source_frame_id,
            source_image_sha256=source_image_sha256, class_ids=list(class_ids),
            tissue_support_path=str(tissue_support_path),
            tissue_support_sha256=tissue_support_sha256,
            global_model_binding_sha256=global_model_binding_sha256,
            independently_fitted_windows=False, window_count=window_count,
            core_partition_complete=True, compared_halo_pixels=compared_pixels,
            maximum_halo_total_variation=maximum,
            allowed_halo_total_variation=float(max_halo_total_variation),
            semantic_mask_sha256=mask_sha256, semantic_mask_dtype="uint8",
            semantic_mask_path="semantic-mask.uint8.dat", shards=shard_receipts,
            geojson_published=False,
            vector_status="blocked: requires globally seamless full-slide vectorization",
            biological_accuracy_qualified=False)
        with (staging / "ASSEMBLY_COMPLETE.json").open("w") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if folder.exists():
            raise FileExistsError(folder)
        os.rename(staging, folder)
        return receipt
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
