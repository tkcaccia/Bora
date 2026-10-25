"""Stream complete, halo-checked uint8 native label cores into one WSI mask.

The categorical input avoids persistent three-channel float probability shards.
Every overlapping shard extent must agree exactly before global vectorization.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .wsi_probability_assembly import (
    CHUNK, MAX_WINDOW_PIXELS, _check_unchanged, _digest, _hash_string,
    _integer_tuple, _intersection, _read_bound_npy,
)


FORMAT = "bora-wsi-label-assembly-v1"


def _compare_exact(current: dict, neighbor: dict) -> int:
    """Check the complete overlap, including halo/halo, in bounded blocks."""
    overlap = _intersection(current["extent"], neighbor["extent"])
    if overlap is None:
        return 0
    old = np.load(neighbor["path"], mmap_mode="r", allow_pickle=False)
    y0, x0, y1, x1 = overlap
    oy, ox = current["origin"]
    py, px = neighbor["origin"]
    for y in range(y0, y1, CHUNK):
        for x in range(x0, x1, CHUNK):
            yy, xx = min(y1, y + CHUNK), min(x1, x + CHUNK)
            if not np.array_equal(
                    current["array"][y-oy:yy-oy, x-ox:xx-ox],
                    old[y-py:yy-py, x-px:xx-px]):
                raise ValueError("Adjacent categorical label halos disagree")
    return (y1 - y0) * (x1 - x0)


def assemble_wsi_label_cores(windows, shape_yx, class_ids, output_folder, *,
                             tissue_support_path, tissue_support_sha256,
                             native_he_path, source_image_sha256,
                             global_model_binding_sha256, source_frame_id,
                             native_mpp_xy, disk_budget_bytes=32 * 1024**3):
    """Publish one complete native mask from row-major, frozen-model label cores.

    Each window is a dict with ``labels_path``, ``labels_sha256``,
    ``origin_yx``, ``core_yxhw``, ``global_model_binding_sha256``,
    ``source_frame_id``, ``source_image_sha256``,
    ``tissue_support_sha256``, and ``independently_fitted=False``. The input
    `.npy` is uint8 and includes a one-pixel-or-larger halo at every interior
    core edge. All overlapping extents, including halo/halo, must match.
    """
    h, w = _integer_tuple(shape_yx, 2, positive=True)
    if h * w > 2**63 - 1:
        raise ValueError("Global mask exceeds file-offset range")
    if (list(class_ids) != [1, 2, 3] or
            not isinstance(native_mpp_xy, (tuple, list)) or
            len(native_mpp_xy) != 2 or
            any(type(value) not in (float, int) or not np.isfinite(value) or
                value <= 0 for value in native_mpp_xy) or
            not isinstance(source_frame_id, str) or not source_frame_id.strip()):
        raise ValueError("Require three ordered groups and calibrated source frame")
    _hash_string(source_image_sha256, "Source image checksum")
    _hash_string(tissue_support_sha256, "Support checksum")
    _hash_string(global_model_binding_sha256, "Frozen-model binding")
    if type(disk_budget_bytes) is not int or disk_budget_bytes < h*w + 1024**2:
        raise ValueError("Invalid label-mask disk budget")
    image_path = Path(native_he_path).resolve(strict=True)
    if not image_path.is_file() or _digest(image_path) != source_image_sha256:
        raise ValueError("Native H&E source bytes differ")
    support_path = Path(tissue_support_path).resolve(strict=True)
    support, support_identity = _read_bound_npy(support_path,
        tissue_support_sha256, ndim=2, shape=(h, w))
    if support.dtype not in (np.dtype("bool"), np.dtype("uint8")):
        raise ValueError("Tissue support must be bool or uint8")
    output = Path(output_folder)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < h*w + 64*1024**2:
        raise ValueError("Insufficient free disk for native label mask")
    staging = Path(tempfile.mkdtemp(prefix=".bora-label-assembly-", dir=output.parent))
    mask_path = staging / "semantic-mask.uint8.dat"
    next_y = next_x = 0
    row_height = None
    previous_row, current_row = [], []
    count = compared = 0
    class_pixels = np.zeros(4, dtype=np.int64)
    source_identities = []
    shard_receipts = []
    try:
        mask = np.memmap(mask_path, mode="w+", dtype=np.uint8, shape=(h, w))
        for item in windows:
            if (not isinstance(item, dict) or
                    item.get("independently_fitted") is not False or
                    item.get("global_model_binding_sha256") != global_model_binding_sha256 or
                    item.get("source_frame_id") != source_frame_id or
                    item.get("source_image_sha256") != source_image_sha256 or
                    item.get("tissue_support_sha256") != tissue_support_sha256):
                raise ValueError("Label shard differs from frozen source binding")
            y, x, ch, cw = _integer_tuple(item.get("core_yxhw"), 4)
            if ch == 0 or cw == 0 or y + ch > h or x + cw > w:
                raise ValueError("Core lies outside slide or has zero area")
            if y != next_y:
                if next_x != w or row_height is None or y != next_y + row_height:
                    raise ValueError("Core grid has a gap, overlap or wrong order")
                previous_row, current_row = current_row, []
                next_y, next_x, row_height = y, 0, None
            if x != next_x or (row_height is not None and ch != row_height):
                raise ValueError("Core grid has a gap, overlap or unequal row heights")
            row_height = ch
            next_x = x + cw
            oy, ox = _integer_tuple(item.get("origin_yx"), 2)
            path = Path(item.get("labels_path")).resolve(strict=True)
            labels, identity = _read_bound_npy(path, item.get("labels_sha256"),
                ndim=2, dtype=np.dtype("uint8"))
            if labels.size > MAX_WINDOW_PIXELS:
                raise ValueError("Categorical shard exceeds bounded window size")
            ey, ex = oy + labels.shape[0], ox + labels.shape[1]
            if (ey > h or ex > w or y < oy or x < ox or y+ch > ey or x+cw > ex or
                    (y > 0 and oy == y) or (x > 0 and ox == x) or
                    (y+ch < h and ey == y+ch) or
                    (x+cw < w and ex == x+cw)):
                raise ValueError("Core/halo does not align with native slide")
            for yy in range(oy, ey, CHUNK):
                for xx in range(ox, ex, CHUNK):
                    block = labels[yy-oy:min(ey,yy+CHUNK)-oy,
                                   xx-ox:min(ex,xx+CHUNK)-ox]
                    tissue = support[yy:min(ey,yy+CHUNK),xx:min(ex,xx+CHUNK)]
                    if (np.any((block != 0) != (tissue != 0)) or
                            np.any((block != 0) & (block != 1) &
                                   (block != 2) & (block != 3)) or
                            (support.dtype == np.uint8 and np.any(tissue > 1))):
                        raise ValueError("Categorical labels disagree with tissue support/classes")
            current = {"array": labels, "path": str(path), "origin": (oy, ox),
                "extent": (oy, ox, ey, ex), "core": (y, x, y+ch, x+cw)}
            neighbors = current_row[-1:] + [old for old in previous_row
                if _intersection(current["extent"], old["extent"])]
            for neighbor in neighbors:
                pixels = _compare_exact(current, neighbor)
                if pixels == 0:
                    raise ValueError("Adjacent categorical windows have no halo overlap")
                compared += pixels
            if (x > 0 and not current_row) or (y > 0 and not previous_row):
                raise ValueError("Missing adjacent window for non-edge core")
            core = labels[y-oy:y+ch-oy, x-ox:x+cw-ox]
            mask[y:y+ch, x:x+cw] = core
            class_pixels += np.bincount(core.ravel(), minlength=4)[:4]
            _check_unchanged(path, identity)
            source_identities.append((path, identity))
            shard_receipts.append({"core_yxhw": [y,x,ch,cw],
                "origin_yx": [oy,ox], "labels_shape_yx": list(labels.shape),
                "labels_path": str(path), "labels_sha256": item["labels_sha256"]})
            current_row.append({key: value for key, value in current.items()
                                if key != "array"})
            count += 1
        if count == 0 or next_x != w or next_y + row_height != h:
            raise ValueError("Core grid leaves uncovered slide pixels")
        if count > 1 and compared == 0:
            raise ValueError("Adjacent categorical halos were not compared")
        if np.any(class_pixels[1:] == 0):
            raise ValueError("Complete mask lost an observed KODAMA group")
        _check_unchanged(support_path, support_identity)
        if _digest(support_path) != tissue_support_sha256:
            raise ValueError("Tissue support changed during assembly")
        for path, identity in source_identities:
            _check_unchanged(path, identity)
        for shard in shard_receipts:
            if _digest(shard["labels_path"]) != shard["labels_sha256"]:
                raise ValueError("Categorical shard changed during assembly")
        if _digest(image_path) != source_image_sha256:
            raise ValueError("Native H&E source changed during assembly")
        mask.flush()
        del mask
        mask_sha256 = _digest(mask_path)
        receipt = {"format": FORMAT, "native_shape_yx": [h,w],
            "native_mpp_xy": list(native_mpp_xy),
            "source_frame_id": source_frame_id,
            "source_image_sha256": source_image_sha256,
            "class_ids": [1,2,3],
            "tissue_support_path": str(support_path),
            "tissue_support_sha256": tissue_support_sha256,
            "global_model_binding_sha256": global_model_binding_sha256,
            "independently_fitted_windows": False,
            "window_count": count, "core_partition_complete": True,
            "compared_halo_pixels": compared,
            "halo_mismatch_pixels": 0,
            "class_pixel_counts": {str(i): int(class_pixels[i]) for i in (1,2,3)},
            "semantic_mask_sha256": mask_sha256,
            "semantic_mask_dtype": "uint8",
            "semantic_mask_path": mask_path.name,
            "shards": shard_receipts,
            "geojson_published": False,
            "biological_accuracy_qualified": False}
        with (staging / "ASSEMBLY_COMPLETE.json").open("x") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if sum(p.stat().st_size for p in staging.iterdir() if p.is_file()) > disk_budget_bytes:
            raise ValueError("Label assembly exceeds declared disk budget")
        if output.exists():
            raise FileExistsError(output)
        os.rename(staging, output)
        return receipt
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
