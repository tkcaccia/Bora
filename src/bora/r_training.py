"""Bounded, source-bound class fitting for the Bora R interface.

This route is intentionally separate from the historical eight-class head.
Annotations are sampled at factor-1 token centres, and every sample keeps its
source image identity. Zero means unscored; it is never silently renamed.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import tempfile

import numpy as np
from sklearn.linear_model import RidgeClassifier

from .r_feature_preflight import sha256


FORMAT = "bora-r-training-v1"
CHANNELS = 1280
TOKEN = 14
SEED = 20261114


def _select_binary_margin_threshold(margin: np.ndarray, targets: np.ndarray,
                                    positive_class: int) -> dict:
    """Maximize class-balanced training accuracy with a deterministic tie break.

    The margin is the class-2 score minus the class-1 score. A positive call
    uses ``margin > threshold``; zero is the historical argmax decision.
    """
    margin = np.asarray(margin, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets).reshape(-1)
    if len(margin) != len(targets) or not len(margin) or not np.isfinite(margin).all():
        raise ValueError("Finite aligned training margins are required")
    positive = targets == positive_class
    if not positive.any() or positive.all():
        raise ValueError("Threshold selection requires both observed classes")
    order = np.argsort(margin, kind="stable")
    values = margin[order]
    truth = positive[order].astype(np.int64)
    ends = np.r_[np.flatnonzero(np.diff(values)), len(values) - 1]
    positives_below = np.cumsum(truth)[ends]
    negatives_below = (ends + 1) - positives_below
    positive_total = int(positive.sum())
    negative_total = len(positive) - positive_total
    accuracy = .5 * ((positive_total - positives_below) / positive_total +
                     negatives_below / negative_total)
    # Include the all-positive decision as a valid candidate.
    thresholds = np.r_[np.nextafter(values[0], -np.inf), values[ends]]
    accuracy = np.r_[.5, accuracy]
    best = np.flatnonzero(np.isclose(accuracy, accuracy.max(), atol=1e-12, rtol=0))
    chosen = min(best, key=lambda i: (abs(thresholds[i]), thresholds[i]))
    threshold = float(thresholds[chosen])
    baseline = .5 * (np.mean(margin[positive] > 0) +
                     np.mean(margin[~positive] <= 0))
    return {"threshold": threshold, "objective": "sampled_training_token_balanced_accuracy",
            "selected_balanced_accuracy": float(accuracy[chosen]),
            "zero_threshold_balanced_accuracy": float(baseline),
            "positive_class": int(positive_class),
            "positive_tokens": positive_total, "negative_tokens": negative_total,
            "decision": "class_2_if_projected_score_margin_gt_threshold",
            "selection_scope": "same fitting images; resubstitution, no held-out calibration"}


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _annotation_grid(path: Path, image_shape: tuple[int, int],
                     mapping: dict | None) -> np.ndarray:
    """Read exact annotation pixels at 14-pixel token centres, not resized IDs."""
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    from rasterio.windows import Window

    height, width = image_shape
    ys = np.arange(7, height, TOKEN)
    xs = np.arange(7, width, TOKEN)
    if not len(ys) or not len(xs):
        raise ValueError("Image is too small for a factor-1 token centre")
    grid = np.zeros((len(ys), len(xs)), np.uint8)
    mapping = None if mapping is None else {str(k): int(v) for k, v in mapping.items()}
    if mapping is not None and (not mapping or any(v < 1 or v > 254 for v in mapping.values())):
        raise ValueError("Class mapping must assign positive uint8 class IDs")
    if path.suffix.lower() in (".json", ".geojson"):
        source = json.loads(path.read_text())
        if source.get("type") != "FeatureCollection":
            raise ValueError("Class GeoJSON must be a FeatureCollection")
        grouped = defaultdict(list)
        for feature in source.get("features", []):
            geometry = feature.get("geometry")
            if geometry is None:
                continue
            props = feature.get("properties") or {}
            source_label = props.get("class_id", props.get("class"))
            if source_label is None:
                raise ValueError("Each class polygon needs class_id or class")
            if mapping is None:
                if not isinstance(source_label, int) or not 1 <= source_label <= 254:
                    raise ValueError("GeoJSON class IDs must be positive integers")
                target = source_label
            else:
                if str(source_label) not in mapping:
                    raise ValueError("GeoJSON class has no supplied mapping")
                target = mapping[str(source_label)]
            grouped[target].append(geometry)
        if not grouped:
            raise ValueError("Class GeoJSON has no mapped polygons")
        for target, geometries in sorted(grouped.items()):
            positive = rasterize(((geometry, 1) for geometry in geometries),
                                 out_shape=grid.shape, transform=Affine.scale(TOKEN),
                                 fill=0, dtype=np.uint8) != 0
            if np.any(positive & (grid != 0) & (grid != target)):
                raise ValueError("Different expert classes overlap at token centres")
            grid[positive] = target
    else:
        with rasterio.open(path) as dataset:
            if (dataset.height, dataset.width) != image_shape or dataset.count != 1:
                raise ValueError("Class raster must be one band at exact image dimensions")
            if not np.issubdtype(np.dtype(dataset.dtypes[0]), np.integer):
                raise ValueError("Class raster must contain integer labels")
            for row, y in enumerate(ys):
                native = dataset.read(1, window=Window(0, int(y), width, 1))[0, xs]
                if mapping is None:
                    if np.any((native < 0) | (native > 254)):
                        raise ValueError("Raster labels must be 0 or 1..254")
                    grid[row] = native
                else:
                    values = set(map(int, np.unique(native)))
                    if values - {0} - {int(k) for k in mapping if k.lstrip("-").isdigit()}:
                        raise ValueError("Raster annotation includes an unmapped class")
                    for original in values - {0}:
                        grid[row, native == original] = mapping[str(original)]
    return grid


def _verified_scales(preparation: dict, case: dict) -> dict[int, np.ndarray]:
    case_dir = case["case_dir"]
    result = {}
    for item in preparation["scales"]:
        if item["case_dir"] != case_dir:
            continue
        receipt_path = Path(item["receipt_path"])
        if sha256(receipt_path) != item["receipt_sha256"]:
            raise ValueError("Feature receipt SHA256 differs")
        receipt = json.loads(receipt_path.read_text())
        factor = int(item["factor"])
        feature_path = Path(receipt["features_path"])
        if (receipt.get("status") != "COMPLETE" or
                receipt.get("channels") != CHANNELS or
                receipt.get("factor") != factor or
                sha256(feature_path) != receipt["features_sha256"]):
            raise ValueError("Feature scale differs from its receipt")
        array = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        if array.dtype != np.float16 or array.ndim != 3 or array.shape[2] != CHANNELS or \
           list(array.shape) != receipt["shape_yxc"]:
            raise ValueError("Feature array geometry or dtype differs")
        result[factor] = array
    if 1 not in result:
        raise ValueError("Training needs factor-1 Virchow2 features")
    return dict(sorted(result.items()))


def _sample_case(grids: dict[int, np.ndarray], labels: np.ndarray,
                 per_class_quota: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    if labels.shape[0] > grids[1].shape[0] or labels.shape[1] > grids[1].shape[1]:
        raise ValueError("Annotation token grid exceeds feature grid")
    columns, targets = [], []
    counts = {}
    for cls in np.unique(labels):
        cls = int(cls)
        if cls == 0:
            continue
        flat = np.flatnonzero(labels.ravel() == cls)
        chosen = rng.choice(flat, size=min(len(flat), per_class_quota), replace=False)
        ys, xs = np.unravel_index(chosen, labels.shape)
        blocks = []
        for factor, grid in grids.items():
            scaled_y = ys // factor
            scaled_x = xs // factor
            if scaled_y.max(initial=0) >= grid.shape[0] or scaled_x.max(initial=0) >= grid.shape[1]:
                raise ValueError("Factor feature grid cannot register token positions")
            blocks.append(np.asarray(grid[scaled_y, scaled_x], np.float32))
        matrix = np.concatenate(blocks, axis=1)
        finite = np.isfinite(matrix).all(axis=1)
        if finite.any():
            columns.append(matrix[finite])
            targets.append(np.full(int(finite.sum()), cls, np.uint8))
        counts[str(cls)] = {"labelled_token_centres": int(len(flat)),
                            "sampled_finite_tokens": int(finite.sum()),
                            "sampled_nonfinite_tokens": int((~finite).sum())}
    if not columns:
        return np.empty((0, CHANNELS * len(grids)), np.float32), np.empty(0, np.uint8), counts
    return np.concatenate(columns), np.concatenate(targets), counts


def train(images, annotations, feature_dir, class_labels, output_dir,
          *, max_tokens_per_class: int = 8000, channels_per_class: int = 32,
          alpha: float = 10.0, optimize_binary_threshold: bool = True) -> dict:
    """Fit a classwise-selected linear head to verified full-channel caches."""
    if isinstance(images, (str, Path)):
        images = [images]
    if isinstance(annotations, (str, Path)):
        annotations = [annotations]
    images = [Path(p).resolve() for p in images]
    annotations = [Path(p).resolve() for p in annotations]
    if not images or len(images) != len(annotations) or len(set(images)) != len(images):
        raise ValueError("Training needs distinct images and one class annotation per image")
    if class_labels is None:
        class_labels = [None] * len(images)
    if len(class_labels) != len(images):
        raise ValueError("Class mappings need one entry per image")
    if not isinstance(max_tokens_per_class, int) or not 1 <= max_tokens_per_class <= 100000 or \
       not isinstance(channels_per_class, int) or not 1 <= channels_per_class <= CHANNELS or \
       not math.isfinite(alpha) or alpha <= 0 or \
       not isinstance(optimize_binary_threshold, bool):
        raise ValueError("Training resource and ridge limits must be finite and positive")
    root = Path(feature_dir).resolve()
    manifest_path = root / "PREPARATION_COMPLETE.json"
    preparation = json.loads(manifest_path.read_text())
    if preparation.get("format") != "bora-r-preparation-v1" or \
       preparation.get("status") != "COMPLETE" or len(preparation["cases"]) != len(images):
        raise ValueError("A complete matching Bora preparation is required")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    input_receipt = {"format": "bora-r-training-inputs-v1",
                     "preparation": str(manifest_path),
                     "preparation_sha256": sha256(manifest_path),
                     "images": [{"image": str(image), "image_sha256": sha256(image),
                                 "annotation": str(annotation),
                                 "annotation_sha256": sha256(annotation),
                                 "class_mapping": mapping}
                                for image, annotation, mapping in
                                zip(images, annotations, class_labels)],
                     "max_tokens_per_class": max_tokens_per_class,
                     "channels_per_class": channels_per_class, "alpha": float(alpha),
                     "seed": SEED,
                     "optimize_binary_threshold": optimize_binary_threshold}
    _atomic_json(output / "TRAINING_INPUTS.json", input_receipt)
    rng = np.random.default_rng(SEED)
    per_case_quota = max(1, math.ceil(max_tokens_per_class / len(images)))
    chunks, labels = [], []
    rows = []
    factors = [1]  # the frozen supervised head uses factor-1, 1,280-channel tokens
    try:
        for index, (image, annotation, mapping) in enumerate(zip(images, annotations, class_labels)):
            case = preparation["cases"][index]
            source = case["source"]
            if Path(source["image"]).resolve() != image or sha256(image) != source["image_sha256"]:
                raise ValueError("Training image differs from prepared source")
            if not annotation.is_file():
                raise FileNotFoundError(annotation)
            grids = _verified_scales(preparation, case)
            target = _annotation_grid(annotation, tuple(source["image_shape_yx"]), mapping)
            features, y, counts = _sample_case({1: grids[1]}, target,
                                                per_case_quota, rng)
            if len(y):
                chunks.append(features)
                labels.append(y)
            rows.append({"index": index, "image": str(image),
                         "image_sha256": source["image_sha256"],
                         "annotation": str(annotation), "annotation_sha256": sha256(annotation),
                         "class_mapping": mapping, "counts": counts})
        if not labels:
            raise ValueError("No finite labelled Virchow2 tokens were sampled")
        matrix = np.concatenate(chunks)
        target = np.concatenate(labels)
        classes, class_counts = np.unique(target, return_counts=True)
        if len(classes) < 2:
            raise ValueError("Training needs at least two observed tissue classes")
        # These are source-group-balanced quotas, not a cross-validation search.
        mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = np.maximum(matrix.std(axis=0, dtype=np.float64).astype(np.float32), 1e-4)
        standardized = (matrix - mean) / std
        picked = set()
        for cls in classes:
            positive = target == cls
            contrast = np.abs(standardized[positive].mean(axis=0) -
                              standardized[~positive].mean(axis=0))
            picked.update(np.argsort(-contrast, kind="stable")[:channels_per_class].tolist())
        selected = np.asarray(sorted(picked), np.int32)
        weights = class_counts[np.searchsorted(classes, target)].astype(np.float64) ** -.5
        weights /= weights.mean()
        model = RidgeClassifier(alpha=alpha, solver="cholesky")
        model.fit(np.ascontiguousarray(standardized[:, selected]), target,
                  sample_weight=weights)
        if not np.array_equal(model.classes_, classes):
            raise ValueError("Ridge class order differs from training targets")
        calibration = None
        if len(classes) == 2:
            decision = model.decision_function(
                np.ascontiguousarray(standardized[:, selected]))
            # _scores writes [-decision, decision] for a binary ridge model.
            margin = 2 * np.asarray(decision, dtype=np.float64)
            calibration = _select_binary_margin_threshold(margin, target,
                                                          int(classes[1]))
            if not optimize_binary_threshold:
                calibration["threshold"] = 0.0
                calibration["selected_balanced_accuracy"] = calibration[
                    "zero_threshold_balanced_accuracy"]
                calibration["selection_scope"] = "disabled; historical zero margin"
        model_path = output / "predictor.npz"
        with tempfile.NamedTemporaryFile(dir=output, prefix=".predictor-", suffix=".npz",
                                         delete=False) as stream:
            temp = Path(stream.name)
            np.savez_compressed(stream, format=FORMAT, model="classwise-ridge",
                                preparation_sha256=sha256(manifest_path),
                                training_inputs_sha256=sha256(output / "TRAINING_INPUTS.json"),
                                factors=np.asarray(factors, np.int16),
                                selected=selected, mean=mean[selected], std=std[selected],
                                coef=np.atleast_2d(model.coef_),
                                intercept=np.atleast_1d(model.intercept_),
                                classes=classes,
                                binary_margin_threshold=np.asarray(
                                    calibration["threshold"] if calibration else 0.0,
                                    dtype=np.float64))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, model_path)
        receipt = {"format": FORMAT, "status": "COMPLETE",
                   "created_utc": datetime.now(timezone.utc).isoformat(),
                   "preparation_sha256": sha256(manifest_path),
                   "model_path": str(model_path), "model_sha256": sha256(model_path),
                   "factor_scales": factors, "all_source_channels_per_scale": CHANNELS,
                   "classes": list(map(int, classes)),
                   "class_token_counts": dict(zip(map(str, classes), map(int, class_counts))),
                   "selected_concatenated_channels": selected.tolist(),
                   "sampled_tokens": int(len(target)), "per_image_class_quota": per_case_quota,
                   "binary_threshold_calibration": calibration,
                   "annotations": rows,
                   "peak_host_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "gpu_used": False}
        _atomic_json(output / "TRAINING_COMPLETE.json", receipt)
        return receipt
    except BaseException as error:
        _atomic_json(output / "TRAINING_FAILED.json",
                     {"format": "bora-r-training-failure-v1", "status": "FAILED",
                      "error": repr(error), "sampled_cases": rows})
        raise
