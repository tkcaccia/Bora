"""Public CPU API for the registered, three-group KODAMA/Virchow2 Bora route.

This module supplies no fitted tissue weights. The caller fits a numeric head
on its own registered training labels, refines native H&E probability windows,
and assembles one complete source-frame mask before publishing contours.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import re
import shutil

import numpy as np

from .boundary_refinement import refine_probability_boundaries
from .contextual_token_head import (
    contextualize_pca, fit_numeric_head, predict_numeric_head,
)
from .wsi_global_vector import publish_global_wsi_vectors
from .wsi_probability_assembly import assemble_wsi_probability_cores
from .wsi_label_assembly import assemble_wsi_label_cores
from .wsi_review_smoothing import publish_review_smoothing


_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _hash(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frozen_ref(manifest_path: Path, value: dict, name: str) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"} or \
            not isinstance(value["sha256"], str) or _SHA.fullmatch(value["sha256"]) is None:
        raise ValueError(f"{name} requires a path and SHA-256")
    path = (manifest_path.parent / value["path"]).resolve(strict=True)
    if not path.is_file() or _hash(path) != value["sha256"]:
        raise ValueError(f"{name} bytes differ from the manifest")
    return path


def fit_three_group_model(fit_manifest, output_folder) -> dict:
    """Fit and freeze a fresh 32-channel, numeric KODAMA head.

    The fit manifest binds a ``samples.npz`` containing ``features`` and
    ``labels`` to its annotation and PCA projection. No model is bundled.
    """
    manifest_path = Path(fit_manifest).resolve(strict=True)
    spec = json.loads(manifest_path.read_text())
    if (spec.get("format") != "bora-kodama-three-group-fit-v1" or
            not isinstance(spec.get("source_frame_id"), str) or
            not spec["source_frame_id"].strip() or
            not isinstance(spec.get("training_annotation_sha256"), str) or
            _SHA.fullmatch(spec["training_annotation_sha256"]) is None or
            not isinstance(spec.get("pca_projection_sha256"), str) or
            _SHA.fullmatch(spec["pca_projection_sha256"]) is None or
            not isinstance(spec.get("source_image_sha256"), str) or
            _SHA.fullmatch(spec["source_image_sha256"]) is None or
            spec.get("class_ids") != [1, 2, 3]):
        raise ValueError("Require a bound three-group fit manifest")
    samples_path = _frozen_ref(manifest_path, spec.get("samples"), "Training samples")
    with np.load(samples_path, allow_pickle=False) as data:
        if set(data.files) != {"features", "labels"}:
            raise ValueError("Training NPZ needs only features and labels")
        features = data["features"]
        labels = data["labels"]
    if (features.ndim != 2 or features.shape[1] != 32 or
            features.dtype != np.float32 or labels.ndim != 1 or
            labels.shape[0] != features.shape[0] or labels.dtype.kind not in "iu" or
            np.any(labels < 1) or np.any(labels > 3)):
        raise ValueError("Three-group head requires 32-channel PCA samples")
    counts = np.bincount(labels.astype(np.int64), minlength=4)[1:4]
    if counts.shape != (3,) or np.any(counts < 2) or counts.sum() != len(labels):
        raise ValueError("Each of three groups needs at least two training observations")
    prior = counts.astype(np.float32) / counts.sum()
    if "training_prior" in spec:
        prior = np.asarray(spec["training_prior"], np.float32)
        if (prior.shape != (3,) or not np.isfinite(prior).all() or
                np.any(prior <= 0) or not np.isclose(prior.sum(), 1, atol=1e-6)):
            raise ValueError("Training prior must be a positive three-group simplex")
    head = fit_numeric_head(features, labels,
        regularization_c=float(spec.get("regularization_c", .1)))
    output = Path(output_folder)
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError("Use a fresh model folder with an existing parent")
    output.mkdir()
    model_file = output / "numeric_head.npz"
    np.savez_compressed(model_file, **head, training_prior=prior)
    receipt = {"format": "bora-kodama-three-group-head-v1",
        "status": "COMPLETE", "model_sha256": _hash(model_file),
        "model_path": str(model_file), "fit_manifest_sha256": _hash(manifest_path),
        "training_samples_sha256": _hash(samples_path),
        "training_annotation_sha256": spec["training_annotation_sha256"],
        "pca_projection_sha256": spec["pca_projection_sha256"],
        "source_frame_id": spec["source_frame_id"],
        "source_image_sha256": spec["source_image_sha256"],
        "class_ids": [1, 2, 3], "class_counts": counts.tolist(),
        "training_prior": prior.tolist(),
        "regularization_c": float(spec.get("regularization_c", .1)),
        "biological_accuracy_qualified": False}
    with (output / "FIT_COMPLETE.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return receipt


def score_three_group_model(score_manifest, model_folder, output_folder, *,
                            sigma_tokens: float, prior_exponent: float,
                            tile_size: int = 256) -> dict:
    """Score a hash-bound global PCA lattice with a frozen numeric head."""
    manifest_path = Path(score_manifest).resolve(strict=True)
    spec = json.loads(manifest_path.read_text())
    if (spec.get("format") != "bora-kodama-three-group-score-v1" or
            spec.get("registration_method") != "same_global_token_grid" or
            not isinstance(spec.get("source_frame_id"), str) or
            not spec["source_frame_id"].strip() or
            not isinstance(spec.get("source_image_sha256"), str) or
            _SHA.fullmatch(spec["source_image_sha256"]) is None):
        raise ValueError("Require a registered global-token scoring manifest")
    field_path = _frozen_ref(manifest_path, spec.get("field"), "PCA field")
    support_path = _frozen_ref(manifest_path, spec.get("support"), "Tissue support")
    model_root = Path(model_folder).resolve(strict=True)
    receipt = json.loads((model_root / "FIT_COMPLETE.json").read_text())
    model_file = model_root / "numeric_head.npz"
    if (receipt.get("format") != "bora-kodama-three-group-head-v1" or
            receipt.get("status") != "COMPLETE" or
            receipt.get("model_sha256") != _hash(model_file) or
            receipt.get("pca_projection_sha256") != spec.get("pca_projection_sha256") or
            receipt.get("source_frame_id") != spec["source_frame_id"] or
            receipt.get("source_image_sha256") != spec["source_image_sha256"] or
            receipt.get("class_ids") != [1, 2, 3]):
        raise ValueError("Frozen model, image, frame or PCA projection differs")
    with np.load(model_file, allow_pickle=False) as data:
        head = {key: data[key] for key in (
            "scaler_mean", "scaler_scale", "logistic_coef",
            "logistic_intercept", "class_ids")}
        prior = data["training_prior"]
    field = np.load(field_path, mmap_mode="r", allow_pickle=False)
    support = np.load(support_path, mmap_mode="r", allow_pickle=False)
    if (field.ndim != 3 or field.shape[2] != 32 or
            field.dtype != np.float32 or field.shape[:2] != support.shape or
            support.dtype != bool or len(field.shape) != 3 or
            type(tile_size) is not int or not 32 <= tile_size <= 2048 or
            not np.isfinite(sigma_tokens) or not 0 <= sigma_tokens <= 8):
        raise ValueError("Require aligned PCA32/support and a bounded scoring tile")
    output = Path(output_folder)
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError("Use a fresh probability folder with an existing parent")
    expected_bytes = field.shape[0] * field.shape[1] * 3 * 4 + 4096
    if shutil.disk_usage(output.parent).free < expected_bytes + 1024**3:
        raise ValueError("Insufficient disk for bounded probability output")
    output.mkdir()
    probability_path = output / "three_group_probability.npy"
    probability = np.lib.format.open_memmap(probability_path, mode="w+",
        dtype=np.float32, shape=(*field.shape[:2], 3))
    # scipy.ndimage uses a Gaussian radius round(4*sigma). Each core includes
    # that exact global-token halo; cropping its center equals whole-field
    # scoring, including at slide edges and encoder-patch seams.
    halo = math.floor(4 * sigma_tokens + .5)
    height, width = field.shape[:2]
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            y1, x1 = min(height, y + tile_size), min(width, x + tile_size)
            ey0, ex0 = max(0, y - halo), max(0, x - halo)
            ey1, ex1 = min(height, y1 + halo), min(width, x1 + halo)
            contextual = contextualize_pca(
                np.asarray(field[ey0:ey1, ex0:ex1], np.float32),
                np.asarray(support[ey0:ey1, ex0:ex1], bool), sigma_tokens)
            core = contextual[y-ey0:y1-ey0, x-ex0:x1-ex0]
            probability[y:y1, x:x1] = predict_numeric_head(core, head,
                training_prior=prior, prior_exponent=prior_exponent)
    probability.flush()
    shape_yxc = list(probability.shape)
    del probability
    if (_hash(field_path) != spec["field"]["sha256"] or
            _hash(support_path) != spec["support"]["sha256"] or
            _hash(model_file) != receipt["model_sha256"]):
        raise ValueError("A frozen scoring source changed during tiled inference")
    result = {"format": "bora-kodama-three-group-score-v1",
        "status": "COMPLETE", "source_frame_id": spec["source_frame_id"],
        "source_image_sha256": spec["source_image_sha256"],
        "registration_method": spec["registration_method"],
        "score_manifest_sha256": _hash(manifest_path),
        "field_sha256": _hash(field_path), "support_sha256": _hash(support_path),
        "model_sha256": receipt["model_sha256"],
        "pca_projection_sha256": receipt["pca_projection_sha256"],
        "probability_path": str(probability_path),
        "probability_sha256": _hash(probability_path),
        "shape_yxc": shape_yxc, "class_ids": [1, 2, 3],
        "sigma_tokens": float(sigma_tokens),
        "prior_exponent": float(prior_exponent),
        "tile_size_yx": [tile_size, tile_size],
        "gaussian_halo_tokens": halo,
        "biological_accuracy_qualified": False}
    with (output / "SCORE_COMPLETE.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return result


def score_three_group_tokens(field: np.ndarray, support: np.ndarray,
                             head: dict, training_prior: np.ndarray, *,
                             sigma_tokens: float, prior_exponent: float) -> np.ndarray:
    """Score one complete registered PCA lattice with a caller-fitted head.

    The returned channels are numeric KODAMA group IDs 1, 2 and 3, not tissue
    identities. Pooling crosses encoder-patch seams but excludes background.
    """
    contextual = contextualize_pca(field, support, sigma_tokens)
    probability = predict_numeric_head(contextual, head,
        training_prior=training_prior, prior_exponent=prior_exponent)
    if probability.shape[-1] != 3:
        raise ValueError("This route requires exactly three KODAMA groups")
    return probability


def refine_three_group_native_window(probability: np.ndarray, he: np.ndarray,
                                     support: np.ndarray, mpp_xy,
                                     boundary_spec: dict, *,
                                     memory_budget_bytes: int = 3 * 1024**3):
    """Apply H&E guidance to a registered native window with an adequate halo.

    The caller must partition the *same frozen global probability field* into
    halo-inclusive windows and verify that overlapping predictions agree before
    assembling their cores. A window here is at most the boundary refiner's
    admitted crop size; this function does not independently fit each window.
    """
    return refine_probability_boundaries(probability, he, support, mpp_xy,
        np.asarray([1, 2, 3], dtype=np.uint8), boundary_spec,
        memory_budget_bytes=memory_budget_bytes,
        require_represented_classes=False)


def publish_three_group_contours(assembly_folder, native_he_tiff, output_folder,
                                 group_names: dict, *,
                                 minimum_area_um2: float = 1000.0,
                                 minimum_free_bytes: int = 1024**3,
                                 deadline_seconds: int = 7200) -> dict:
    """Publish exact mask plus guarded diagonal tissue GeoJSON for a full slide.

    ``assembly_folder`` must contain a complete immutable native-mask receipt
    from ``assemble_wsi_probability_cores`` or an equivalent verified global
    materialization. The caller retains incomplete output for diagnosis if a
    guard fails; only ``COMPLETE.json`` marks an accepted contour publication.
    """
    names = {int(key): str(value) for key, value in group_names.items()}
    if set(names) != {1, 2, 3} or any(not value.strip() for value in names.values()):
        raise ValueError("Supply names for exactly KODAMA groups 1, 2 and 3")
    if type(minimum_free_bytes) is not int or minimum_free_bytes < 0:
        raise ValueError("Minimum free-space reserve must be a nonnegative integer")
    output = Path(output_folder)
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError("Use a fresh contour folder with an existing parent")
    source = json.loads((Path(assembly_folder) / "ASSEMBLY_COMPLETE.json").read_text())
    if (source.get("class_ids") != [1, 2, 3] or
            source.get("core_partition_complete") is not True or
            source.get("independently_fitted_windows") is not False):
        raise ValueError("Require a complete frozen three-group native mask")
    output.mkdir()
    vector_folder = output / "global_vectors"
    vector = publish_global_wsi_vectors(assembly_folder, vector_folder, names,
        identity_status="exploratory_transfer",
        minimum_area_um2=minimum_area_um2)
    exact_document = json.loads((vector_folder / "exact" / "mask.geojson").read_text())
    if {feature["properties"]["value"] for feature in exact_document["features"]} != {1,2,3}:
        raise ValueError("Global cleanup lost one of the three KODAMA groups")
    tissue_folder = output / "tissue_annotations"
    tissue = publish_review_smoothing(
        vector_folder / "exact" / "mask.geojson",
        vector_folder / "GLOBAL_VECTOR_COMPLETE.json",
        vector_folder / "exact" / "mask.uint8.dat",
        native_he_tiff, tissue_folder,
        deadline_seconds=deadline_seconds,
        simplify_outer_boundary=True,
        minimum_free_bytes=minimum_free_bytes)
    if (vector["exact_component_count"] != vector["review_component_count"] or
            tissue["status"] != "PREDICTED_TISSUE_ANNOTATION" or
            tissue["selected"]["full_native_raster_agreement"]["minimum_class_iou"] < .995):
        raise ValueError("Three-group global contour guards did not pass")
    receipt = {"format": "bora-kodama-three-group-contours-v1",
        "status": "COMPLETE", "group_names": {str(k): v for k, v in names.items()},
        "biological_accuracy_qualified": False,
        "minimum_area_um2": float(minimum_area_um2),
        "exact_mask_path": str(vector_folder / "exact" / "mask.uint8.dat"),
        "exact_geojson_path": str(vector_folder / "exact" / "mask.geojson"),
        "predicted_tissue_geojson_path": tissue["review_geojson_path"],
        "predicted_tissue_geojson_sha256": tissue["review_geojson_sha256"],
        "vector_receipt_path": str(vector_folder / "GLOBAL_VECTOR_COMPLETE.json"),
        "smoothing_receipt_path": str(tissue_folder / "SMOOTHING_COMPLETE.json"),
        "component_count": vector["exact_component_count"],
        "selected_outer_tolerance_um": tissue["selected"]["tolerance_um"]}
    with (output / "COMPLETE.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return receipt


__all__ = ["contextualize_pca", "fit_numeric_head", "predict_numeric_head",
           "fit_three_group_model", "score_three_group_model",
           "score_three_group_tokens", "refine_three_group_native_window",
           "assemble_wsi_probability_cores", "assemble_wsi_label_cores",
           "publish_three_group_contours"]
