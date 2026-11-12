"""Stable CPU interface for the frozen eight-class Virchow2 ridge route.

The encoder cache is an input, never fitted or reduced in place.  Historical
campaign scripts remain independent; this module does not import them.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter, binary_dilation, label
from sklearn.linear_model import RidgeClassifier

CHANNELS = 1280
CLASSES = tuple(range(1, 9))
FORMAT = "bora-supervised-ridge-v1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_manifest(path: Path, expected: str) -> dict:
    spec = json.loads(Path(path).read_text())
    if spec.get("format") != expected:
        raise ValueError(f"Expected {expected} manifest")
    return spec


def _input(manifest: Path, spec: dict, name: str) -> Path:
    row = spec[name]
    if not isinstance(row, dict) or not isinstance(row.get("path"), str) or not isinstance(row.get("sha256"), str):
        raise ValueError(f"{name} needs a path and SHA256")
    path = (Path(manifest).parent / row["path"]).resolve()
    if digest(path) != row["sha256"]:
        raise ValueError(f"{name} SHA256 differs")
    return path


def _atomic_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".bora-", delete=False) as stream:
        temp = Path(stream.name)
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _model(path: Path, expected_sha: str | None = None) -> dict:
    if expected_sha is not None and digest(path) != expected_sha:
        raise ValueError("Model SHA256 differs")
    with np.load(path, allow_pickle=False) as data:
        if str(data["format"].item()) != FORMAT:
            raise ValueError("Unsupported supervised model format")
        model = {name: data[name].copy() for name in ("selected", "mean", "std", "coef", "intercept", "classes")}
        model["training_manifest_sha256"] = str(data["training_manifest_sha256"].item())
        model["method"] = str(data["method"].item())
    selected = model["selected"]
    classes = model["classes"]
    binary = model["method"] == "peso-binary-ridge64"
    if model["method"] not in ("eight-class-ridge", "peso-binary-ridge64"):
        raise ValueError("Unsupported frozen supervised method")
    if (selected.ndim != 1 or len(selected) < 1 or len(selected) > CHANNELS or
            len(np.unique(selected)) != len(selected) or selected.min() < 0 or selected.max() >= CHANNELS or
            not np.array_equal(classes, (2, 4) if binary else CLASSES) or
            model["coef"].shape != ((1 if binary else 8), len(selected)) or
            model["intercept"].shape != ((1 if binary else 8),) or
            model["mean"].shape != (len(selected),) or model["std"].shape != (len(selected),) or
            not np.isfinite(model["coef"]).all() or not np.isfinite(model["intercept"]).all() or
            not np.isfinite(model["mean"]).all() or not np.isfinite(model["std"]).all() or
            np.any(model["std"] <= 0)):
        raise ValueError("Invalid eight-class model arrays")
    return model


def fit(manifest: Path, output: Path) -> dict:
    """Fit the fixed class32 ridge head on an already frozen training sample."""
    manifest, output = Path(manifest), Path(output)
    if output.exists() or output.with_suffix(output.suffix + ".json").exists():
        raise FileExistsError("Preserve existing model or receipt; use a new output version")
    spec = _read_manifest(manifest, "bora-supervised-fit-v1")
    if spec.get("split") != "train" or spec.get("source_full_channels") != CHANNELS:
        raise ValueError("Only a frozen 1280-channel training sample is accepted")
    feature = _input(manifest, spec, "features")
    label_path = _input(manifest, spec, "labels")
    x = np.load(feature, mmap_mode="r", allow_pickle=False)
    y = np.load(label_path, allow_pickle=False)
    if (x.ndim != 2 or x.shape[1] != CHANNELS or y.shape != (len(x),) or
            x.dtype not in (np.float16, np.float32) or
            not np.issubdtype(y.dtype, np.integer)):
        raise ValueError("Training arrays must be N×1280 and N")
    if spec.get("old_to_new") is not None:
        mapping = {int(k): int(v) for k, v in spec["old_to_new"].items() if v is not None}
        y = np.asarray([mapping.get(int(v), 0) for v in y], np.uint8)
    if not np.isin(y, (0, *CLASSES)).all():
        raise ValueError("Labels must be 0 (ignored) or 1..8")
    valid = y > 0
    method = spec.get("method", "eight-class-ridge")
    expected_classes = (2, 4) if method == "peso-binary-ridge64" else CLASSES
    if method not in ("eight-class-ridge", "peso-binary-ridge64") or not np.array_equal(np.unique(y[valid]), expected_classes):
        raise ValueError("Training labels differ from the selected frozen head")
    z = np.asarray(x[valid], np.float32)
    if not np.isfinite(z).all():
        raise ValueError("Nonfinite training features")
    target = y[valid]
    mean = z.mean(axis=0)
    std = np.maximum(z.std(axis=0), 1e-4)
    z = (z - mean) / std
    if method == "peso-binary-ridge64":
        contrast = np.abs(z[target == 4].mean(axis=0) - z[target == 2].mean(axis=0))
        selected = np.asarray(np.argsort(-contrast, kind="stable")[:64], np.int32)
        weights = None
    else:
        chosen = set()
        for cls in CLASSES:
            pos = target == cls
            contrast = np.abs(z[pos].mean(axis=0) - z[~pos].mean(axis=0))
            chosen.update(np.argsort(-contrast, kind="stable")[:32].tolist())
        selected = np.asarray(sorted(chosen), np.int32)
        counts = np.bincount(target, minlength=9).astype(np.float64)
        weights = np.maximum(counts[target], 1.0) ** -0.5
        weights /= weights.mean()
    head = RidgeClassifier(alpha=10.0, solver="cholesky")
    head.fit(np.ascontiguousarray(z[:, selected]), target, sample_weight=weights)
    if not np.array_equal(head.classes_, expected_classes):
        raise ValueError("Fitted classes differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".npz", prefix=".bora-", delete=False) as stream:
        temp = Path(stream.name)
    try:
        np.savez_compressed(temp, format=FORMAT, method=method, selected=selected,
                            mean=mean[selected], std=std[selected],
                            coef=np.atleast_2d(head.coef_), intercept=np.atleast_1d(head.intercept_),
                            classes=head.classes_, training_manifest_sha256=digest(manifest))
        _model(temp)
        os.replace(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    receipt = {"status": "COMPLETE", "format": FORMAT, "model": str(output),
               "model_sha256": digest(output), "training_manifest_sha256": digest(manifest),
               "training_tokens": int(valid.sum()), "selected_channels": len(selected),
               "selected_indices": selected.tolist(), "method": method,
               "selection": "binary64-standardized-contrast" if method == "peso-binary-ridge64" else "classwise32-standardized-contrast",
               "classifier": "RidgeClassifier(alpha=10, solver=cholesky)", "gpu_used": False}
    _atomic_json(output.with_suffix(output.suffix + ".json"), receipt)
    return receipt


def _scores(grid: np.ndarray, model: dict, output: Path, *, row_tokens: int = 32768,
            roi_order: bool = False) -> dict:
    """Project bounded token rows with the matching historical numeric order."""
    if grid.ndim != 3 or grid.shape[2] != CHANNELS or grid.dtype not in (np.float16, np.float32):
        raise ValueError("Expected H×W×1280 float16/float32 feature grid")
    height, width = grid.shape[:2]
    selected = model["selected"]
    finite = np.empty((height, width), bool)
    for y in range(height):
        finite[y] = np.isfinite(np.asarray(grid[y])[:, selected]).all(axis=1)
    if not finite.any():
        raise ValueError("No finite selected feature tokens")
    near = None if finite.all() else distance_transform_edt(~finite, return_indices=True)[1]
    weights = model["coef"].T / model["std"][:, None]
    offsets = model["intercept"] - (model["mean"] / model["std"]) @ model["coef"].T
    channels = len(model["classes"])
    temp = output.with_name(output.stem + ".partial.npy")
    if output.exists() or temp.exists():
        raise FileExistsError(output if output.exists() else temp)
    raw = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float32, shape=(channels, height, width))
    step = max(1, min(16, row_tokens // width))
    for y0 in range(0, height, step):
        y1 = min(height, y0 + step)
        if near is None:
            block = np.asarray(grid[y0:y1, :, selected], np.float32)
        else:
            block = np.asarray(grid[near[0, y0:y1], near[1, y0:y1]][:, :, selected], np.float32)
        if roi_order:
            # The ROI validation path normalized float32 tokens before the
            # RidgeClassifier decision function, unlike the WSI affine fold.
            normalized = (block.reshape(-1, len(selected)) - model["mean"]) / model["std"]
            values = np.ascontiguousarray(normalized) @ model["coef"].T + model["intercept"]
        else:
            values = block.reshape(-1, len(selected)).astype(np.float64) @ weights + offsets
        if channels == 2:
            values = np.column_stack((-values.reshape(-1), values.reshape(-1)))
        raw[:, y0:y1] = values.reshape(y1-y0, width, channels).transpose(2, 0, 1)
        raw.flush()
    del raw
    os.replace(temp, output)
    return {"finite_tokens": int(finite.sum()), "imputed_tokens": int((~finite).sum())}


def _roi_mask(scores: np.ndarray, working_wh: tuple[int, int], display_wh: tuple[int, int],
              allowed: list[int], classes: np.ndarray) -> np.ndarray:
    if not allowed or not set(allowed) <= set(map(int, classes)):
        raise ValueError("Provide at least one allowed tissue class")
    token_h, token_w = scores.shape[1:]
    if not (0 < working_wh[0] <= token_w*14 and 0 < working_wh[1] <= token_h*14 and
            min(display_wh) > 0):
        raise ValueError("Invalid registered ROI sizes")
    channels = []
    for score in scores:
        frame = Image.fromarray(np.asarray(score, np.float32), "F").resize(
            (token_w*14, token_h*14), Image.Resampling.BICUBIC)
        channels.append(np.asarray(frame.crop((0, 0, *working_wh)).resize(
            display_wh, Image.Resampling.BICUBIC), np.float32))
    stack = np.stack(channels, axis=2)
    if len(classes) == 2:
        # The PEsO frozen choice is threshold multiplier 0 on the positive
        # ridge decision, which corresponds to display channel 1 > 0.
        if set(allowed) != {2, 4}:
            raise ValueError("The frozen PEsO head requires classes 2 and 4")
        return np.where(stack[:, :, 1] > 0, 4, 2).astype(np.uint8)
    gate = np.isin(classes, allowed)
    return classes[np.argmax(np.where(gate[None, None, :], stack, -np.inf), axis=2)].astype(np.uint8)


def polish_roi(source: np.ndarray, mpp_xy: tuple[float, float]) -> tuple[np.ndarray, dict]:
    """Historical 1.5 µm one-hot contour smoothing, then <400 µm² repair."""
    if source.ndim != 2 or source.dtype != np.uint8 or not np.isin(source, CLASSES).all():
        raise ValueError("Expected a positive eight-class ROI mask")
    if len(mpp_xy) != 2 or not all(math.isfinite(v) and v > 0 for v in mpp_xy):
        raise ValueError("Positive measured X/Y spacing is required")
    sigma_xy = 1.5/mpp_xy[0], 1.5/mpp_xy[1]
    best = np.full(source.shape, -np.inf, np.float32)
    result = np.zeros(source.shape, np.uint8)
    for cls in np.unique(source):
        response = gaussian_filter((source == cls).astype(np.float32),
                                   sigma=(sigma_xy[1], sigma_xy[0]), mode="mirror", truncate=4)
        take = response > best
        result[take], best[take] = cls, response[take]
    smoothed = result.copy()
    threshold = math.ceil(400/(mpp_xy[0]*mpp_xy[1]))
    repairs = 0
    eight = np.ones((3, 3), bool)
    for _ in range(8):
        changed = 0
        for cls in np.unique(result):
            cc, count = label(result == cls, eight)
            sizes = np.bincount(cc.ravel(), minlength=count+1)
            for component in np.flatnonzero((sizes < threshold) & (np.arange(count+1) > 0)):
                island = cc == component
                ring = binary_dilation(island, eight) & ~island
                neighbors = result[ring]
                neighbors = neighbors[neighbors != cls]
                if not len(neighbors):
                    raise ValueError("Positive ROI island has no adjacent class")
                result[island] = np.bincount(neighbors, minlength=25).argmax()
                changed += 1
        repairs += changed
        if not changed:
            break
    else:
        raise ValueError("Area repair did not converge")
    for cls in np.unique(result):
        cc, count = label(result == cls, eight)
        sizes = np.bincount(cc.ravel(), minlength=count+1)[1:]
        if len(sizes) and sizes.min() < threshold:
            raise ValueError("Small class component remains")
    return result, {"sigma_um": 1.5, "minimum_component_um2": 400,
                    "mpp_xy": list(map(float, mpp_xy)), "threshold_pixels": threshold,
                    "area_reassignments": repairs,
                    "smoothed_changed_pixels": int(np.count_nonzero(smoothed != source)),
                    "final_changed_pixels": int(np.count_nonzero(result != source))}


def predict(manifest: Path, output: Path) -> dict:
    manifest, output = Path(manifest), Path(output)
    spec = _read_manifest(manifest, "bora-supervised-predict-v1")
    if output.exists():
        raise FileExistsError(output)
    model_path = _input(manifest, spec, "model")
    feature_path = _input(manifest, spec, "features")
    model = _model(model_path, spec["model"]["sha256"])
    grid = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    output.mkdir(parents=True)
    try:
        raw_path = output / "raw-scores.npy"
        counts = _scores(grid, model, raw_path, roi_order=bool(spec.get("roi")))
        raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
        smooth_path = output / "smoothed-scores.npy"
        temp = output / "smoothed-scores.partial.npy"
        smooth = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float32, shape=raw.shape)
        for cls in range(raw.shape[0]):
            smooth[cls] = gaussian_filter(np.asarray(raw[cls]), sigma=.75, mode="nearest")
            smooth.flush()
        del smooth
        os.replace(temp, smooth_path)
        files = {p.name: digest(p) for p in (raw_path, smooth_path)}
        roi_support = None
        if spec.get("roi"):
            roi = spec["roi"]
            working = tuple(map(int, roi["working_size_wh"]))
            display = tuple(map(int, roi["display_size_wh"]))
            allowed = [int(v) for v in roi["allowed_classes"]]
            mask = _roi_mask(np.load(smooth_path, mmap_mode="r"), working, display, allowed, model["classes"])
            if roi.get("polish", True):
                mpp = (.5*working[0]/display[0], .5*working[1]/display[1]) if "mpp_xy" not in roi else tuple(roi["mpp_xy"])
                mask, contour = polish_roi(mask, mpp)
            else:
                contour = None
            if model["method"] == "peso-binary-ridge64" and "support_mask" not in roi:
                raise ValueError("The corrected PEsO route requires a frozen H&E support mask")
            if "support_mask" in roi:
                support_path = _input(manifest, roi, "support_mask")
                roi_support = {"path": str(support_path), "sha256": digest(support_path)}
                with Image.open(support_path) as image:
                    support = np.asarray(image.convert("L").resize(display, Image.Resampling.NEAREST)) > 0
                if support.shape != mask.shape:
                    raise ValueError("ROI support shape differs")
                mask[~support] = 0
            mask_path = output / "class-mask.png"
            Image.fromarray(mask, "L").save(mask_path, optimize=True)
            files[mask_path.name] = digest(mask_path)
        else:
            contour = None
        full_slide = None
        if spec.get("wsi"):
            if spec.get("roi") or model["method"] != "eight-class-ridge":
                raise ValueError("Full-slide finalization requires the eight-class model without ROI settings")
            from .supervised_wsi import run_wsi
            full_slide = run_wsi(manifest, spec, smooth_path, output / "full-slide")
        receipt = {"status": "COMPLETE", "format": "bora-supervised-prediction-v1",
                   "model_path": str(model_path), "model_sha256": digest(model_path),
                   "features_path": str(feature_path), "features_sha256": digest(feature_path),
                   "manifest_sha256": digest(manifest), "score_shape_class_yx": list(raw.shape),
                   "files": files, "finite_token_count": counts["finite_tokens"],
                   "nearest_imputed_token_count": counts["imputed_tokens"],
                   "roi_contour": contour, "roi_support": roi_support,
                   "full_slide_receipt_sha256": digest(output / "full-slide/FULL_SLIDE_VERIFIED.json") if full_slide else None,
                   "gpu_used": False}
        _atomic_json(output / "PREDICTION_COMPLETE.json", receipt)
        return receipt
    except BaseException as exc:
        (output / "FAILED.json").write_text(json.dumps({"status": "FAILED_PRESERVE_PARTIAL", "error": repr(exc)}) + "\n")
        raise


def _read_mask(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r", allow_pickle=False)
    with Image.open(path) as image:
        return np.asarray(image)


def _metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    if pred.shape != truth.shape or pred.ndim != 2:
        raise ValueError("Prediction and target shapes differ")
    if not np.isin(pred, (0, *CLASSES)).all() or not np.isin(truth, (0, *CLASSES)).all():
        raise ValueError("Mask values must be 0..8")
    valid = truth > 0
    if not valid.any():
        raise ValueError("No scored expert pixels")
    per = {}
    for cls in np.unique(truth[valid]):
        actual, guessed = truth == cls, pred == cls
        tp = int(np.count_nonzero(valid & actual & guessed))
        fp = int(np.count_nonzero(valid & ~actual & guessed))
        fn = int(np.count_nonzero(valid & actual & ~guessed))
        per[str(int(cls))] = {"expert_pixels": tp+fn, "intersection": tp,
                              "false_positive": fp, "false_negative": fn,
                              "iou": tp/(tp+fp+fn), "dice": 2*tp/(2*tp+fp+fn)}
    return {"expert_present_macro_iou": float(np.mean([v["iou"] for v in per.values()])),
            "expert_present_macro_dice": float(np.mean([v["dice"] for v in per.values()])),
            "scored_pixel_accuracy": float(np.mean(pred[valid] == truth[valid])),
            "scored_expert_pixels": int(valid.sum()), "all_pixels": int(truth.size),
            "scored_pixel_fraction": float(valid.mean()), "per_present_class": per}


def evaluate(manifest: Path, output: Path) -> dict:
    manifest, output = Path(manifest), Path(output)
    spec = _read_manifest(manifest, "bora-supervised-evaluate-v1")
    if not spec.get("cases"):
        raise ValueError("At least one case is required")
    from .segmentation_metrics import score as geometry_score
    cases = []
    for row in spec["cases"]:
        pred_path = _input(manifest, row, "prediction")
        target_path = _input(manifest, row, "target")
        pred, truth = _read_mask(pred_path), _read_mask(target_path)
        if pred.shape != truth.shape:
            raise ValueError("Registered prediction and reference frames must match; no scorer resizing")
        spacing = tuple(map(float, row["spacing_yx_um"]))
        if len(spacing) != 2:
            raise ValueError("Measured Y/X spacing is required")
        cases.append({"id": row["id"], "prediction_path": str(pred_path),
                      "target_path": str(target_path), "prediction_sha256": digest(pred_path),
                      "target_sha256": digest(target_path), "spacing_yx_um": list(spacing),
                      "metrics": _metrics(pred, truth),
                      "segmentation": geometry_score(truth, pred, 0, spacing, 5.0,
                                                     aligned_semantic_ids=False)})
    boundary = [r["segmentation"]["geometry"]["boundary_f1"] for r in cases
                if r["segmentation"]["geometry"]["boundary_f1"] is not None]
    result = {"status": "COMPLETE", "format": "bora-supervised-evaluation-v1",
              "manifest_sha256": digest(manifest), "cases": cases, "case_count": len(cases),
              "mean_case_macro_iou": float(np.mean([r["metrics"]["expert_present_macro_iou"] for r in cases])),
              "mean_case_macro_dice": float(np.mean([r["metrics"]["expert_present_macro_dice"] for r in cases])),
              "mean_boundary_f1_5um": float(np.mean(boundary)) if boundary else None,
              "boundary_scored_cases": len(boundary),
              "mean_symmetric_region_covering": float(np.mean([
                  r["segmentation"]["geometry"]["symmetric_region_covering"] for r in cases])),
              "is_independent_test": bool(spec.get("is_independent_test", False))}
    _atomic_json(output, result)
    return result


def verify(path: Path) -> dict:
    path = Path(path)
    if path.is_dir():
        receipt_path = path / "PREDICTION_COMPLETE.json"
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("format") != "bora-supervised-prediction-v1" or receipt.get("status") != "COMPLETE":
            raise ValueError("Invalid prediction receipt")
        for name, expected in receipt["files"].items():
            if digest(path / name) != expected:
                raise ValueError(f"Changed output: {name}")
        if (digest(Path(receipt["model_path"])) != receipt["model_sha256"] or
                digest(Path(receipt["features_path"])) != receipt["features_sha256"]):
            raise ValueError("Changed prediction input")
        if receipt.get("roi_support"):
            if digest(Path(receipt["roi_support"]["path"])) != receipt["roi_support"]["sha256"]:
                raise ValueError("Changed ROI tissue support")
        scores = np.load(path / "smoothed-scores.npy", mmap_mode="r", allow_pickle=False)
        if list(scores.shape) != receipt["score_shape_class_yx"] or scores.dtype != np.float32:
            raise ValueError("Score array shape or type differs")
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite output scores")
        if receipt.get("full_slide_receipt_sha256"):
            from .supervised_wsi import verify_wsi
            if digest(path / "full-slide/FULL_SLIDE_VERIFIED.json") != receipt["full_slide_receipt_sha256"]:
                raise ValueError("Changed full-slide receipt")
            verify_wsi(path / "full-slide")
        return {"status": "VERIFIED", "files": len(receipt["files"]), "receipt_sha256": digest(receipt_path)}
    if path.suffix == ".npz":
        model = _model(path)
        receipt_path = path.with_suffix(path.suffix + ".json")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if (receipt.get("format") != FORMAT or receipt.get("status") != "COMPLETE" or
                    receipt.get("model_sha256") != digest(path) or
                    receipt.get("training_manifest_sha256") != model["training_manifest_sha256"] or
                    receipt.get("selected_indices") != model["selected"].tolist()):
                raise ValueError("Model receipt differs")
        return {"status": "VERIFIED", "format": FORMAT, "selected_channels": len(model["selected"]),
                "model_sha256": digest(path)}
    receipt = json.loads(path.read_text())
    if receipt.get("format") == "bora-supervised-evaluation-v1" and receipt.get("status") == "COMPLETE":
        from .segmentation_metrics import score as geometry_score
        rows = receipt.get("cases", [])
        if len(rows) != receipt.get("case_count") or not rows:
            raise ValueError("Evaluation case count differs")
        for row in rows:
            pred_path, target_path = Path(row["prediction_path"]), Path(row["target_path"])
            if digest(pred_path) != row["prediction_sha256"] or digest(target_path) != row["target_sha256"]:
                raise ValueError("Evaluation input changed")
            pred, truth = _read_mask(pred_path), _read_mask(target_path)
            if _metrics(pred, truth) != row["metrics"]:
                raise ValueError("Evaluation metrics differ")
            if geometry_score(truth, pred, 0, tuple(row["spacing_yx_um"]), 5.0,
                              aligned_semantic_ids=False) != row["segmentation"]:
                raise ValueError("Segmentation geometry metrics differ")
        if not np.isclose(np.mean([r["metrics"]["expert_present_macro_iou"] for r in rows]),
                          receipt["mean_case_macro_iou"], atol=1e-12):
            raise ValueError("Evaluation aggregate differs")
        boundary = [r["segmentation"]["geometry"]["boundary_f1"] for r in rows
                    if r["segmentation"]["geometry"]["boundary_f1"] is not None]
        expected_f1 = float(np.mean(boundary)) if boundary else None
        if (expected_f1 != receipt["mean_boundary_f1_5um"] or
                receipt["boundary_scored_cases"] != len(boundary)):
            raise ValueError("Boundary aggregate differs")
        covering = float(np.mean([r["segmentation"]["geometry"]["symmetric_region_covering"]
                                  for r in rows]))
        if not np.isclose(covering, receipt["mean_symmetric_region_covering"], atol=1e-12):
            raise ValueError("Region-covering aggregate differs")
        return {"status": "VERIFIED", "cases": len(rows), "receipt_sha256": digest(path)}
    raise ValueError("Unsupported verification target")
