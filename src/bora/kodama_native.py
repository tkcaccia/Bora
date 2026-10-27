"""Bounded native H&E refinement of one frozen, global three-group score field.

The input field is fitted and scored once across the entire registered token
lattice. Windows are only a memory limit; their positions never fit a model.
The resulting single native mask is the input to global contour export.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage as ndi

from .boundary_refinement import parameters, refine_probability_boundaries
from .wsi import ome_mpp


FORMAT = "bora-kodama-native-v1"
SHA = re.compile(r"[0-9a-f]{64}\Z")


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def _source(manifest: Path, reference, name: str) -> Path:
    if (not isinstance(reference, dict) or set(reference) != {"path", "sha256"}
            or not isinstance(reference["sha256"], str)
            or SHA.fullmatch(reference["sha256"]) is None):
        raise ValueError(f"{name} requires a path and SHA-256")
    path = (manifest.parent / reference["path"]).resolve(strict=True)
    if _digest(path) != reference["sha256"]:
        raise ValueError(f"{name} differs from its frozen hash")
    return path


def project_token_window(tokens: np.ndarray, native_origin_yx: tuple[int, int],
                         native_shape_yx: tuple[int, int], scale_xy: tuple[float, float]
                         ) -> np.ndarray:
    """Interpolate token centres using absolute native pixel coordinates."""
    if (tokens.ndim != 3 or tokens.dtype != np.float32 or
            tokens.shape[-1] != 3 or
            any(not math.isfinite(v) or v <= 0 for v in scale_xy)):
        raise ValueError("Expected finite-scale three-channel token field")
    oy, ox = native_origin_yx
    height, width = native_shape_yx
    if min(oy, ox) < 0 or min(height, width) < 1:
        raise ValueError("Invalid native window")
    yy = ((np.arange(oy, oy + height, dtype=np.float64) + .5) / scale_xy[1]) / 14 - .5
    xx = ((np.arange(ox, ox + width, dtype=np.float64) + .5) / scale_xy[0]) / 14 - .5
    ty0, tx0 = max(0, math.floor(float(yy[0]))), max(0, math.floor(float(xx[0])))
    ty1 = min(tokens.shape[0], math.ceil(float(yy[-1])) + 2)
    tx1 = min(tokens.shape[1], math.ceil(float(xx[-1])) + 2)
    block = np.asarray(tokens[ty0:ty1, tx0:tx1], np.float32)
    if (not block.size or not np.isfinite(block).all() or np.any(block < 0)
            or np.any(block > 1) or
            not np.allclose(block.sum(-1), 1, atol=2e-6, rtol=0)):
        raise ValueError("Frozen token scores are not three-class probabilities")
    grid_y, grid_x = np.meshgrid(yy - ty0, xx - tx0, indexing="ij")
    out = np.empty((height, width, 3), np.float32)
    for channel in range(3):
        out[..., channel] = ndi.map_coordinates(block[..., channel],
            [grid_y, grid_x], order=1, mode="nearest", prefilter=False)
    return out


def _support_window(dataset, y0: int, y1: int, x0: int, x1: int,
                    native_shape_yx: tuple[int, int]) -> np.ndarray:
    height, width = native_shape_yx
    yy = np.minimum(dataset.height - 1,
        np.floor((np.arange(y0, y1) + .5) * dataset.height / height).astype(int))
    xx = np.minimum(dataset.width - 1,
        np.floor((np.arange(x0, x1) + .5) * dataset.width / width).astype(int))
    sy0, sy1, sx0, sx1 = int(yy.min()), int(yy.max())+1, int(xx.min()), int(xx.max())+1
    block = dataset.read(1, window=Window(sx0, sy0, sx1-sx0, sy1-sy0))
    if not np.isin(block, [0, 1, 255, 65535]).all():
        raise ValueError("Tissue-support image must be binary")
    return block[np.ix_(yy-sy0, xx-sx0)] > 0


def refine_global_score_to_native(manifest_path, output_folder) -> dict:
    """Materialize one complete, hash-bound native mask from global scores.

    A fresh output directory is mandatory. An interrupted directory has no
    ``ASSEMBLY_COMPLETE.json`` and cannot be mistaken for a finished mask.
    """
    manifest = Path(manifest_path).resolve(strict=True)
    spec = json.loads(manifest.read_text())
    if (spec.get("format") != FORMAT or
            not isinstance(spec.get("source_frame_id"), str) or
            not spec["source_frame_id"].strip() or
            not isinstance(spec.get("working_shape_yx"), list) or
            len(spec["working_shape_yx"]) != 2 or
            any(type(v) is not int or v < 1 for v in spec["working_shape_yx"])):
        raise ValueError("Require a complete registered native-run manifest")
    score_path = _source(manifest, spec.get("score"), "Global score")
    score_receipt_path = _source(manifest, spec.get("score_receipt"), "Completed score receipt")
    he_path = _source(manifest, spec.get("native_he"), "Native H&E")
    tissue_path = _source(manifest, spec.get("tissue_support"), "Tissue support")
    scores = np.load(score_path, mmap_mode="r", allow_pickle=False)
    working_h, working_w = spec["working_shape_yx"]
    if (scores.dtype != np.float32 or scores.shape !=
            ((working_h+13)//14, (working_w+13)//14, 3)):
        raise ValueError("Score lattice does not match the registered working image")
    score_receipt = json.loads(score_receipt_path.read_text())
    if (score_receipt.get("format") != "bora-kodama-three-group-score-v1" or
            score_receipt.get("status") != "COMPLETE" or
            score_receipt.get("source_frame_id") != spec["source_frame_id"] or
            score_receipt.get("source_image_sha256") != spec["native_he"]["sha256"] or
            score_receipt.get("probability_sha256") != spec["score"]["sha256"] or
            score_receipt.get("shape_yxc") != list(scores.shape) or
            score_receipt.get("class_ids") != [1, 2, 3] or
            not isinstance(score_receipt.get("model_sha256"), str) or
            SHA.fullmatch(score_receipt["model_sha256"]) is None):
        raise ValueError("Score field is not bound to a completed global model")
    mpp = spec.get("native_mpp_xy")
    if (not isinstance(mpp, list) or len(mpp) != 2 or
            any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in mpp)):
        raise ValueError("Native pixel spacing is required")
    working_mpp = spec.get("working_mpp_xy")
    if (not isinstance(working_mpp, list) or len(working_mpp) != 2 or
            any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                for v in working_mpp)):
        raise ValueError("Working pixel spacing is required")
    boundary = parameters(spec.get("boundary_spec"))
    core = spec.get("core_size", 1024)
    halo = spec.get("halo_pixels", 72)
    if (type(core) is not int or not 128 <= core <= 2048 or
            type(halo) is not int or not 1 <= halo <= 512):
        raise ValueError("Native core or halo is outside bounded limits")
    needed = math.ceil(boundary["maximum_distance_um"] / min(mpp)) + \
        math.ceil(3 * boundary["he_blur_um"] / min(mpp)) + boundary["iterations"] + 3
    if halo < needed:
        raise ValueError("Native halo is too small for the declared physical refinement")
    output = Path(output_folder)
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError("Use a fresh native-mask folder")
    with rasterio.open(he_path) as image, rasterio.open(tissue_path) as tissue:
        native_shape = (image.height, image.width)
        if image.count < 3 or tissue.count != 1:
            raise ValueError("Expected RGB H&E and one binary tissue-support band")
    native_h, native_w = native_shape
    ome_shape_xy, ome_mpp_xy = ome_mpp(he_path)
    if (ome_shape_xy != (native_w, native_h) or
            not np.allclose(ome_mpp_xy, mpp, rtol=0, atol=1e-6)):
        raise ValueError("Declared native spacing or shape disagrees with OME metadata")
    if (abs(native_h * float(mpp[1]) - working_h * float(working_mpp[1])) > max(mpp[1], working_mpp[1]) or
            abs(native_w * float(mpp[0]) - working_w * float(working_mpp[0])) > max(mpp[0], working_mpp[0])):
        raise ValueError("Native and working image physical fields disagree")
    if shutil.disk_usage(output.parent).free < native_h * native_w + 2 * 1024**3:
        raise ValueError("Insufficient free disk for native mask and reserve")
    output.mkdir()
    mask_path = output / "semantic-mask.uint8.dat"
    result = np.memmap(mask_path, mode="w+", dtype=np.uint8, shape=native_shape)
    scale_xy = (native_w / working_w, native_h / working_h)
    counts = np.zeros(4, np.int64)
    changed = movable = completed = 0
    with rasterio.open(he_path) as image, rasterio.open(tissue_path) as tissue:
        for y0 in range(0, native_h, core):
            for x0 in range(0, native_w, core):
                y1, x1 = min(native_h, y0+core), min(native_w, x0+core)
                hy0, hx0 = max(0, y0-halo), max(0, x0-halo)
                hy1, hx1 = min(native_h, y1+halo), min(native_w, x1+halo)
                local_tissue = _support_window(tissue, hy0, hy1, hx0, hx1, native_shape)
                if np.any(local_tissue):
                    probability = project_token_window(scores, (hy0, hx0),
                        (hy1-hy0, hx1-hx0), scale_xy)
                    he = np.moveaxis(image.read((1, 2, 3),
                        window=Window(hx0, hy0, hx1-hx0, hy1-hy0)), 0, -1)
                    labels, report = refine_probability_boundaries(
                        probability, he, local_tissue, mpp,
                        np.asarray([1, 2, 3], np.uint8), boundary,
                        core_size=512, memory_budget_bytes=3*1024**3,
                        require_represented_classes=False)
                    native_core = labels[y0-hy0:y1-hy0, x0-hx0:x1-hx0]
                    changed += report["changed_pixels"]
                    movable += report["movable_pixels"]
                else:
                    native_core = np.zeros((y1-y0, x1-x0), np.uint8)
                result[y0:y1, x0:x1] = native_core
                counts += np.bincount(native_core.ravel(), minlength=4)
                completed += 1
            result.flush()
    del result
    if np.any(counts[1:] == 0):
        raise ValueError("Global native mask lost one of the three groups")
    # Recheck all mutable inputs after computation before marking completion.
    for path, key in ((score_path, "score"), (he_path, "native_he"),
                      (tissue_path, "tissue_support"),
                      (score_receipt_path, "score_receipt")):
        if _digest(path) != spec[key]["sha256"]:
            raise ValueError("An input changed during native refinement")
    binding = hashlib.sha256(json.dumps({"manifest_sha256": _digest(manifest),
        "score_sha256": spec["score"]["sha256"],
        "score_receipt_sha256": spec["score_receipt"]["sha256"],
        "model_sha256": score_receipt["model_sha256"],
        "boundary_spec": boundary},
        sort_keys=True).encode()).hexdigest()
    receipt = {"format": "bora-wsi-probability-assembly-v1",
        "core_partition_complete": True, "independently_fitted_windows": False,
        "native_shape_yx": [native_h, native_w], "native_mpp_xy": mpp,
        "class_ids": [1, 2, 3], "class_names": {str(i): f"Group {i}" for i in (1, 2, 3)},
        "source_frame_id": spec["source_frame_id"],
        "source_image_sha256": spec["native_he"]["sha256"],
        "global_model_binding_sha256": binding,
        "semantic_mask_sha256": _digest(mask_path),
        "semantic_mask_path": mask_path.name,
        "biological_accuracy_qualified": False}
    with (output / "ASSEMBLY_COMPLETE.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    summary = {"format": FORMAT, "status": "COMPLETE", "source_manifest_sha256": _digest(manifest),
        "completed_cores": completed, "pixel_counts": counts.tolist(),
        "changed_pixels_in_halo_windows": changed,
        "movable_pixels_in_halo_windows": movable,
        "assembly_sha256": _digest(output / "ASSEMBLY_COMPLETE.json")}
    (output / "NATIVE_COMPLETE.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
