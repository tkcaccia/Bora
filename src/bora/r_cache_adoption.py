"""Adopt verified Virchow2 grids without copying arrays or starting a GPU job.

The manifest is bound to image, model, array and source-receipt hashes. A
full-field historical cache may be paired with a supplied tissue support: the
support controls later Bora admission, while encoder coverage remains explicit.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

import numpy as np

from .r_feature_preflight import TILE, inspect_input, sha256


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _paths(value, name: str, *, distinct: bool = True) -> list[Path]:
    if isinstance(value, (str, Path)):
        value = [value]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a nonempty path vector")
    paths = [Path(item).resolve() for item in value]
    if distinct and len(paths) != len(set(paths)):
        raise ValueError(f"{name} contains repeated paths")
    return paths


def adopt(images, tissue_masks, feature_dir, model_dir, scales,
          precomputed_manifest: str, *, minimum_free_gib: float = 1.0) -> dict:
    """Build a standard preparation manifest from immutable, verified grids."""
    from .r_api import _model_identity  # import here to avoid a module cycle

    if not math.isfinite(minimum_free_gib) or minimum_free_gib < 0:
        raise ValueError("Finite nonnegative free-space reserve is required")

    images = _paths(images, "images")
    masks = ([None] * len(images) if tissue_masks is None else
             _paths(tissue_masks, "tissue_masks", distinct=False))
    if len(masks) != len(images):
        raise ValueError("One tissue mask is required per image when masks are supplied")
    factors = [int(value) for value in ([scales] if isinstance(scales, int) else scales)]
    if not factors or len(factors) != len(set(factors)) or any(f not in (1, 2, 4) for f in factors):
        raise ValueError("Scales must be distinct factors 1, 2, 4")
    manifest_path = Path(precomputed_manifest).resolve()
    spec = json.loads(manifest_path.read_text())
    if spec.get("format") != "bora-precomputed-features-v1":
        raise ValueError("Unrecognized precomputed feature manifest")
    entries = spec.get("images")
    if not isinstance(entries, list) or len(entries) != len(images):
        raise ValueError("Precomputed manifest must have one entry per image")
    model = _model_identity(Path(model_dir).resolve())
    if (spec.get("model_checkpoint_sha256") != model["checkpoint_sha256"] or
            spec.get("model_config_sha256") != model["config_sha256"]):
        raise ValueError("Precomputed model identity differs")

    # Finish all expensive reads and validation before making an output tree.
    checked_cases = []
    for index, (image, mask, entry) in enumerate(zip(images, masks, entries)):
        source = inspect_input(image, mask)
        if (entry.get("image_sha256") != source["image_sha256"] or
                entry.get("image_path") != str(image)):
            raise ValueError(f"Image {index} differs from precomputed source")
        selection = entry.get("encoder_selection")
        if selection not in ("full-field", "supplied-tissue"):
            raise ValueError("Encoder selection must be full-field or supplied-tissue")
        if selection == "supplied-tissue" and (
                mask is None or entry.get("encoder_tissue_sha256") !=
                source["tissue_support_sha256"]):
            raise ValueError("Masked encoder support differs from current tissue support")
        if selection == "full-field" and entry.get("encoder_tissue_sha256") is not None:
            raise ValueError("Full-field cache must not claim a tissue-only encoder mask")
        scale_specs = entry.get("scales")
        if not isinstance(scale_specs, dict) or set(scale_specs) != {str(f) for f in factors}:
            raise ValueError("Precomputed scales differ from requested scales")
        grids = []
        padded_h, padded_w = source["padded_shape_yx"]
        for factor in factors:
            item = scale_specs[str(factor)]
            path = Path(item["path"]).resolve()
            receipt = Path(item["source_receipt_path"]).resolve()
            if (sha256(path) != item.get("sha256") or
                    sha256(receipt) != item.get("source_receipt_sha256")):
                raise ValueError(f"Factor-{factor} array or provenance receipt differs")
            registration = item.get("registration")
            expected_registration = {
                "axis_order": "yxc", "origin_source_px_yx": [0, 0],
                "token_stride_source_px_yx": [14 * factor, 14 * factor],
                "padding": "right-bottom-to-multiple-896"}
            if registration != expected_registration:
                raise ValueError(f"Factor-{factor} row-to-source registration differs")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            wanted = (padded_h // (14 * factor), padded_w // (14 * factor), 1280)
            if array.shape != wanted or array.dtype != np.float16:
                raise ValueError(f"Factor-{factor} grid shape or dtype differs")
            proof = json.loads(receipt.read_text())
            if (proof.get("format") != "bora-precomputed-scale-receipt-v1" or
                    proof.get("status") != "VERIFIED" or
                    proof.get("image_sha256") != source["image_sha256"] or
                    proof.get("model_checkpoint_sha256") != model["checkpoint_sha256"] or
                    proof.get("model_config_sha256") != model["config_sha256"] or
                    proof.get("factor") != factor or
                    proof.get("features_sha256") != item["sha256"] or
                    proof.get("shape_yxc") != list(wanted) or
                    proof.get("registration") != registration or
                    proof.get("encoder_selection") != selection or
                    proof.get("encoder_tissue_sha256") !=
                    entry.get("encoder_tissue_sha256")):
                raise ValueError(f"Factor-{factor} provenance receipt is not source-bound")
            grids.append((factor, path, item["sha256"], list(wanted), receipt,
                          item["source_receipt_sha256"], registration))
        checked_cases.append((source, selection, grids))

    output = Path(feature_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    stats = os.statvfs(output.parent)
    free_bytes = stats.f_bavail * stats.f_frsize
    if free_bytes - (1 << 20) < minimum_free_gib * (1 << 30):
        raise OSError("Cache adoption would breach the free-space reserve")
    output.mkdir(parents=True)
    created = datetime.now(timezone.utc).isoformat()
    plan = {"format": "bora-r-precomputed-adoption-v1", "status": "COMPLETE",
            "images": [str(path) for path in images],
            "tissue_masks": [str(path) if path else None for path in masks],
            "precomputed_manifest": str(manifest_path),
            "precomputed_manifest_sha256": sha256(manifest_path),
            "model": model, "scales": factors, "gpu_workers_launched": 0}
    _write_json(output / "PREPARATION_INPUTS.json", plan)
    _write_json(output / "RESOURCE_ADMISSION.json", {
        "format": "bora-r-precomputed-resource-v1", "new_feature_bytes": 0,
        "precomputed_arrays_reused": True, "gpu_workers_launched": 0,
        "free_bytes_before": free_bytes, "minimum_free_gib": minimum_free_gib})
    cases, scale_results = [], []
    try:
        for index, (source, selection, grids) in enumerate(checked_cases):
            case_dir = output / f"image-{index:03d}"
            case_dir.mkdir()
            counts = source.pop("counts")
            eligible = source.pop("eligible")
            padded_h, padded_w = source["padded_shape_yx"]
            padded = np.zeros((padded_h // TILE, padded_w // TILE), dtype=np.bool_)
            padded[:eligible.shape[0], :eligible.shape[1]] = eligible
            eligible_path = case_dir / "eligible.npy"
            counts_path = case_dir / "tissue-pixel-counts.npy"
            _write_npy(eligible_path, padded)
            _write_npy(counts_path, counts)
            source.update({"eligible_path": str(eligible_path),
                           "eligible_sha256": sha256(eligible_path),
                           "tissue_pixel_counts_path": str(counts_path),
                           "tissue_pixel_counts_sha256": sha256(counts_path),
                           "base_grid_shape_yx": list(padded.shape),
                           "model_checkpoint_sha256": model["checkpoint_sha256"],
                           "model_config_sha256": model["config_sha256"],
                           "scales": factors, "historical_encoder_selection": selection,
                           "new_encoder_job": False})
            _write_json(case_dir / "SOURCE_FROZEN.json", source)
            cases.append({"case_dir": str(case_dir), "source": source,
                          "worker_specs": []})
            for factor, path, digest, shape, old_receipt, old_digest, registration in grids:
                scale_dir = case_dir / f"factor-{factor}"
                scale_dir.mkdir()
                receipt_path = scale_dir / "FEATURES_COMPLETE.json"
                _write_json(receipt_path, {
                    "format": "bora-r-adopted-feature-scale-v1",
                    "status": "COMPLETE", "channels": 1280, "factor": factor,
                    "features_path": str(path), "features_sha256": digest,
                    "shape_yxc": shape, "dtype": "float16",
                    "registration": registration,
                    "encoder_selection": selection,
                    "source_receipt_path": str(old_receipt),
                    "source_receipt_sha256": old_digest})
                scale_results.append({"case_dir": str(case_dir),
                                      "factor": factor,
                                      "receipt_path": str(receipt_path),
                                      "receipt_sha256": sha256(receipt_path),
                                      "lifecycle": {"origin": "adopted verified cache",
                                                    "new_GPU_job": False}})
        result = {"format": "bora-r-preparation-v1", "status": "COMPLETE",
                  "created_utc": created, "completed_utc": datetime.now(timezone.utc).isoformat(),
                  "model": model, "cases": cases, "scales": scale_results,
                  "cache_adoption_not_new_extraction": True,
                  "preparation_inputs_sha256": sha256(output / "PREPARATION_INPUTS.json"),
                  "resource_admission_sha256": sha256(output / "RESOURCE_ADMISSION.json")}
        _write_json(output / "PREPARATION_COMPLETE.json", result)
        return result
    except BaseException as error:
        _write_json(output / "PREPARATION_FAILED.json", {
            "format": "bora-r-precomputed-adoption-failure-v1", "status": "FAILED",
            "time_utc": datetime.now(timezone.utc).isoformat(), "error": repr(error),
            "partial_cases": cases, "partial_scales": scale_results})
        raise
