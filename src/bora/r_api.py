"""Disk-backed Python bridge called by the Bora R package through reticulate.

GPU work runs in a bounded child process. A child failure latches the whole
preparation bundle; no subsequent scale or image is admitted. The parent does
not import torch and therefore does not retain a CUDA context in the R session.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from .r_feature_preflight import SCALE_PAD, TILE, inspect_input, sha256


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as stream:
        np.save(stream, values, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _owned_pids(pgid: int) -> list[int]:
    found = []
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            values = stat.read_text().split(") ", 1)[1].split()
            if int(values[2]) == pgid:
                found.append(int(stat.parent.name))
        except (OSError, ValueError, IndexError):
            pass
    return found


def _gpu_pids() -> set[int]:
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                             "--format=csv,noheader,nounits"], text=True,
                            capture_output=True, timeout=20, check=True)
    return {int(line.strip()) for line in result.stdout.splitlines()
            if line.strip() and line.strip().isdigit()}


def _cleanup(pgid: int, *, window_seconds: float = 30) -> dict:
    observations = []
    known_pids = {pgid}
    deadline = time.monotonic() + window_seconds
    while True:
        owned = _owned_pids(pgid)
        known_pids.update(owned)
        try:
            active = sorted(_gpu_pids().intersection(known_pids))
            inspection_error = None
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            active, inspection_error = [], repr(error)
        observations.append({"time_utc": _now(), "owned_pids": owned,
                             "owned_gpu_pids": active,
                             "inspection_error": inspection_error})
        if not owned and not active and inspection_error is None:
            return {"status": "VERIFIED", "observations": observations}
        if time.monotonic() >= deadline:
            return {"status": "UNKNOWN_OR_FAILED", "observations": observations}
        time.sleep(3)


def _run_worker(spec_path: Path, max_seconds: int, log_path: Path) -> dict:
    if _gpu_pids():
        raise RuntimeError("GPU compute context exists before Virchow2 admission")
    env = dict(os.environ)
    env.setdefault("OPENBLAS_NUM_THREADS", "2")
    with log_path.open("x") as log:
        child = subprocess.Popen([sys.executable, "-m", "bora.r_feature_worker",
                                  str(spec_path)], stdout=log,
                                 stderr=subprocess.STDOUT,
                                 start_new_session=True, env=env)
        pgid = child.pid
        timed_out = False
        try:
            exit_code = child.wait(timeout=max_seconds + 60)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(pgid, signal.SIGTERM)
            try:
                exit_code = child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                try:
                    exit_code = child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    exit_code = None
        cleanup = _cleanup(pgid)
    result = {"child_pid": child.pid, "child_pgid": pgid,
              "exit_code": exit_code, "timed_out": timed_out,
              "cleanup": cleanup, "log": str(log_path),
              "log_sha256": sha256(log_path)}
    if exit_code != 0 or timed_out or cleanup["status"] != "VERIFIED":
        raise FeatureWorkerFailure(result)
    return result


class FeatureWorkerFailure(RuntimeError):
    def __init__(self, evidence: dict):
        super().__init__("Virchow2 worker failed or cleanup is uncertain")
        self.evidence = evidence


def _model_identity(model_dir: Path) -> dict:
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    checkpoint = model_dir / "model.safetensors"
    config = model_dir / "config.json"
    return {"checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "config_path": str(config), "config_sha256": sha256(config)}


def _case_plan(image: Path, support: Path | None, case_dir: Path,
               model_dir: Path, model: dict, scales: list[int],
               max_seconds: int) -> dict:
    checked = inspect_input(image, support)
    case_dir.mkdir()
    counts = checked.pop("counts")
    eligible = checked.pop("eligible")
    padded_h, padded_w = checked["padded_shape_yx"]
    padded_grid = np.zeros((padded_h // TILE, padded_w // TILE), dtype=np.bool_)
    padded_grid[:eligible.shape[0], :eligible.shape[1]] = eligible
    eligible_path = case_dir / "eligible.npy"
    count_path = case_dir / "tissue-pixel-counts.npy"
    _atomic_npy(eligible_path, padded_grid)
    _atomic_npy(count_path, counts)
    checked.update({"eligible_path": str(eligible_path),
                    "eligible_sha256": sha256(eligible_path),
                    "tissue_pixel_counts_path": str(count_path),
                    "tissue_pixel_counts_sha256": sha256(count_path),
                    "base_grid_shape_yx": list(padded_grid.shape),
                    "model_checkpoint_sha256": model["checkpoint_sha256"],
                    "model_config_sha256": model["config_sha256"],
                    "scales": scales})
    _atomic_json(case_dir / "SOURCE_FROZEN.json", checked)
    specs = []
    for factor in scales:
        worker = {"format": "bora-r-feature-worker-v1",
                  "image": checked["image"],
                  "image_sha256": checked["image_sha256"],
                  "image_shape_yx": checked["image_shape_yx"],
                  "padded_shape_yx": checked["padded_shape_yx"],
                  "eligible_path": str(eligible_path),
                  "eligible_sha256": checked["eligible_sha256"],
                  "base_grid_shape_yx": checked["base_grid_shape_yx"],
                  "model_dir": str(model_dir),
                  "model_checkpoint_sha256": model["checkpoint_sha256"],
                  "model_config_sha256": model["config_sha256"],
                  "factor": factor,
                  "output_dir": str(case_dir / f"factor-{factor}"),
                  "max_seconds": max_seconds}
        spec_path = case_dir / f"factor-{factor}-WORKER_FROZEN.json"
        _atomic_json(spec_path, worker)
        specs.append(spec_path)
    return {"case_dir": str(case_dir), "source": checked,
            "worker_specs": [str(path) for path in specs]}


def prepare(images, tissue_masks, feature_dir, model_dir, scales,
            *, max_seconds: int = 3600, minimum_free_gib: float = 1.0,
            precomputed_manifest: str | None = None) -> dict:
    """Prepare full-1280 feature arrays, one fresh output tree per call."""
    if precomputed_manifest is not None:
        from .r_cache_adoption import adopt
        return adopt(images, tissue_masks, feature_dir, model_dir, scales,
                     precomputed_manifest, minimum_free_gib=minimum_free_gib)
    if isinstance(images, (str, Path)):
        images = [images]
    if isinstance(tissue_masks, (str, Path)):
        tissue_masks = [tissue_masks]
    if isinstance(scales, (int, np.integer)):
        scales = [scales]
    images = [Path(path).resolve() for path in images]
    if not images or len(set(images)) != len(images):
        raise ValueError("Image paths must be distinct and nonempty")
    masks = ([None] * len(images) if tissue_masks is None else
             [Path(path).resolve() for path in tissue_masks])
    if len(masks) != len(images):
        raise ValueError("Tissue support needs one path per image")
    factors = [int(value) for value in scales]
    if not factors or len(set(factors)) != len(factors) or \
       any(factor not in (1, 2, 4) for factor in factors):
        raise ValueError("Supported Virchow2 scales are distinct factors 1, 2, 4")
    if not isinstance(max_seconds, int) or max_seconds < 60 or \
       not math.isfinite(minimum_free_gib) or minimum_free_gib < 0:
        raise ValueError("Finite resource limits are required")
    output = Path(feature_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    model_dir = Path(model_dir).resolve()
    model = _model_identity(model_dir)
    output.mkdir(parents=True)
    plan = {"format": "bora-r-preparation-plan-v1", "status": "RUNNING",
            "created_utc": _now(), "images": [str(p) for p in images],
            "tissue_masks": [str(p) if p else None for p in masks],
            "model": model, "scales": factors,
            "max_seconds_per_scale": max_seconds,
            "minimum_free_gib": minimum_free_gib}
    _atomic_json(output / "PREPARATION_INPUTS.json", plan)
    cases = []
    try:
        for index, (image, support) in enumerate(zip(images, masks)):
            cases.append(_case_plan(image, support, output / f"image-{index:03d}",
                                    model_dir, model, factors, max_seconds))
        # The persistent arrays are float16 H×W×1280 at each scale. A partial
        # array is renamed in place, so there is no second full-size copy.
        expected_bytes = sum(
            int(np.prod((case["source"]["padded_shape_yx"][0] // (14 * factor),
                         case["source"]["padded_shape_yx"][1] // (14 * factor),
                         1280))) * 2
            for case in cases for factor in factors)
        free_bytes = os.statvfs(output).f_bavail * os.statvfs(output).f_frsize
        if free_bytes - expected_bytes < minimum_free_gib * 1024**3:
            raise OSError("Feature arrays would breach the free-space reserve")
        _atomic_json(output / "RESOURCE_ADMISSION.json",
                     {"expected_feature_bytes": expected_bytes,
                      "free_bytes_before": free_bytes,
                      "minimum_free_gib": minimum_free_gib,
                      "time_utc": _now()})
        scale_results = []
        for case in cases:
            for raw in case["worker_specs"]:
                spec_path = Path(raw)
                spec = json.loads(spec_path.read_text())
                log_path = spec_path.with_suffix(".log")
                lifecycle = _run_worker(spec_path, max_seconds, log_path)
                receipt_path = Path(spec["output_dir"]) / "FEATURES_COMPLETE.json"
                receipt = json.loads(receipt_path.read_text())
                if receipt["status"] != "COMPLETE" or receipt["channels"] != 1280 or \
                   sha256(Path(receipt["features_path"])) != receipt["features_sha256"]:
                    raise ValueError("Completed feature scale failed source-only audit")
                scale_results.append({"case_dir": case["case_dir"],
                                      "factor": spec["factor"],
                                      "receipt_path": str(receipt_path),
                                      "receipt_sha256": sha256(receipt_path),
                                      "lifecycle": lifecycle})
        result = {"format": "bora-r-preparation-v1", "status": "COMPLETE",
                  "created_utc": plan["created_utc"],
                  "completed_utc": _now(), "model": model,
                  "cases": cases, "scales": scale_results,
                  "preparation_inputs_sha256": sha256(output / "PREPARATION_INPUTS.json"),
                  "resource_admission_sha256": sha256(output / "RESOURCE_ADMISSION.json")}
        _atomic_json(output / "PREPARATION_COMPLETE.json", result)
        return result
    except BaseException as error:
        failure = {"format": "bora-r-preparation-failure-v1", "status": "FAILED",
                   "time_utc": _now(), "error": repr(error),
                   "partial_cases": cases,
                   "worker_evidence": error.evidence if isinstance(error, FeatureWorkerFailure) else None}
        _atomic_json(output / "PREPARATION_FAILED.json", failure)
        raise


def train(images, annotations, feature_dir, class_labels, output_dir,
          *, max_tokens_per_class: int = 8000,
          channels_per_class: int = 32, alpha: float = 10.0) -> dict:
    """Fit one source-bound classwise ridge head from prepared full channels."""
    from .r_training import train as fit_class_head
    return fit_class_head(images, annotations, feature_dir, class_labels,
                          output_dir, max_tokens_per_class=max_tokens_per_class,
                          channels_per_class=channels_per_class, alpha=alpha)


def predict(preparation, predictor, output_dir, resolutions,
            *, source_mpp_xy=(.5, .5), output_mpp_xy=(4.0, 4.0),
            minimum_area_um2=2000.0, contour_sigma_um=6.0,
            minimum_free_gib=1.0, max_new_gib=None,
            preview_backend="vips-lanczos3", class_names=None) -> dict:
    """Dispatch the R prediction route without changing frozen Python CLIs."""
    if predictor is None:
        raise NotImplementedError("Joint-image KODAMA route is not yet released in the R interface")
    if resolutions is not None:
        raise ValueError("Supervised prediction does not accept cluster resolutions")
    from .r_prediction import predict_supervised
    return predict_supervised(preparation, predictor, output_dir,
        source_mpp_xy=tuple(map(float, source_mpp_xy)),
        output_mpp_xy=tuple(map(float, output_mpp_xy)),
        minimum_area_um2=float(minimum_area_um2),
        contour_sigma_um=float(contour_sigma_um),
        minimum_free_gib=float(minimum_free_gib),
        max_new_gib=None if max_new_gib is None else float(max_new_gib),
        preview_backend=preview_backend, class_names=class_names)


def joint_stage(preparation, output_dir, *, max_joint_tiles: int = 10000,
                minimum_free_gib: float = 1.0) -> dict:
    """Freeze every admitted tile from all prepared images in one joint input."""
    from .r_kodama import stage_joint
    return stage_joint(preparation, output_dir,
                       max_joint_tiles=int(max_joint_tiles),
                       minimum_free_gib=float(minimum_free_gib))


def joint_complete(preparation, stage_dir, cluster_dir, output_dir,
                   *, source_mpp_xy=(.5, .5), output_mpp_xy=(4.0, 4.0),
                   minimum_area_um2=2000.0, contour_sigma_um=6.0,
                   minimum_free_gib=1.0, max_new_gib=None,
                   preview_backend="vips-lanczos3") -> dict:
    """Return joint KODAMA groups to each source H&E image."""
    from .r_kodama import complete_joint
    return complete_joint(preparation, stage_dir, cluster_dir, output_dir,
                          source_mpp_xy=tuple(map(float, source_mpp_xy)),
                          output_mpp_xy=tuple(map(float, output_mpp_xy)),
                          minimum_area_um2=float(minimum_area_um2),
                          contour_sigma_um=float(contour_sigma_um),
                          minimum_free_gib=float(minimum_free_gib),
                          max_new_gib=None if max_new_gib is None else float(max_new_gib),
                          preview_backend=preview_backend)
