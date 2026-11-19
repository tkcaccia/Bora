"""Execution settings for the R bridge; CUDA is never silently emulated."""

from __future__ import annotations

from contextlib import contextmanager
import math

import numpy as np
from threadpoolctl import threadpool_limits


def validate_execution(backend: str, n_cores: int) -> tuple[str, int]:
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend must be 'cpu' or 'cuda'")
    if isinstance(n_cores, bool) or not isinstance(n_cores, int) or n_cores < 1:
        raise ValueError("n_cores must be one positive integer")
    return backend, n_cores


@contextmanager
def cpu_threads(n_cores: int):
    """Limit already-loaded BLAS/OpenMP libraries and OpenCV for this call."""
    validate_execution("cpu", n_cores)
    try:
        import cv2
        old_cv_threads = cv2.getNumThreads()
        cv2.setNumThreads(n_cores)
    except ImportError:
        cv2 = None
        old_cv_threads = None
    try:
        with threadpool_limits(limits=n_cores):
            yield
    finally:
        if cv2 is not None:
            cv2.setNumThreads(old_cv_threads)


def require_cuda():
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA backend requires process-visible PyTorch CUDA support")
    return torch


def ridge_cuda(x: np.ndarray, target: np.ndarray, weights: np.ndarray,
               classes: np.ndarray, alpha: float):
    """Weighted RidgeClassifier normal equations; intercept is unpenalized.

    Only the small selected-channel fit is on CUDA. Sampling and deterministic
    channel selection stay on CPU. The return arrays have sklearn's class order.
    """
    from types import SimpleNamespace

    torch = require_cuda()
    if x.ndim != 2 or len(x) != len(target) or len(x) != len(weights) or \
       len(classes) < 2 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Invalid CUDA ridge inputs")
    dtype = torch.float64
    device = "cuda"
    xx = torch.as_tensor(np.ascontiguousarray(x), dtype=dtype, device=device)
    ww = torch.as_tensor(np.ascontiguousarray(weights), dtype=dtype, device=device)
    if len(classes) == 2:
        yy = np.where(target == classes[1], 1.0, -1.0)[:, None]
    else:
        yy = np.where(target[:, None] == classes[None, :], 1.0, -1.0)
    yy = torch.as_tensor(np.ascontiguousarray(yy), dtype=dtype, device=device)
    total = ww.sum()
    x_mean = (ww[:, None] * xx).sum(dim=0) / total
    y_mean = (ww[:, None] * yy).sum(dim=0) / total
    centered_x = xx - x_mean
    centered_y = yy - y_mean
    lhs = centered_x.T @ (ww[:, None] * centered_x)
    lhs.diagonal().add_(float(alpha))
    rhs = centered_x.T @ (ww[:, None] * centered_y)
    coef = torch.linalg.solve(lhs, rhs).T
    intercept = y_mean - coef @ x_mean
    torch.cuda.synchronize()
    return SimpleNamespace(
        classes_=classes.copy(),
        coef_=coef.cpu().numpy(),
        intercept_=intercept.cpu().numpy(),
        decision_function=lambda data: np.asarray(data) @ coef.cpu().numpy().T +
        intercept.cpu().numpy(),
    )


def scores_cuda(grid: np.ndarray, model: dict, output, *, row_tokens: int = 32768):
    """Project bounded score rows with float64 CUDA affine arithmetic."""
    from pathlib import Path
    import os
    from scipy.ndimage import distance_transform_edt

    torch = require_cuda()
    if grid.ndim != 3 or grid.shape[2] != 1280 or grid.dtype not in (np.float16, np.float32):
        raise ValueError("Expected H×W×1280 float16/float32 feature grid")
    height, width = grid.shape[:2]
    selected = model["selected"]
    finite = np.empty((height, width), bool)
    for y in range(height):
        finite[y] = np.isfinite(np.asarray(grid[y])[:, selected]).all(axis=1)
    if not finite.any():
        raise ValueError("No finite selected feature tokens")
    near = None if finite.all() else distance_transform_edt(
        ~finite, return_indices=True)[1]
    weights = model["coef"].T / model["std"][:, None]
    offsets = model["intercept"] - (model["mean"] / model["std"]) @ model["coef"].T
    w_gpu = torch.as_tensor(np.ascontiguousarray(weights), dtype=torch.float64,
                            device="cuda")
    b_gpu = torch.as_tensor(np.ascontiguousarray(offsets), dtype=torch.float64,
                            device="cuda")
    channels = len(model["classes"])
    output = Path(output)
    temp = output.with_name(output.stem + ".partial.npy")
    if output.exists() or temp.exists():
        raise FileExistsError(output if output.exists() else temp)
    raw = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float32,
                                    shape=(channels, height, width))
    step = max(1, min(16, row_tokens // width))
    for y0 in range(0, height, step):
        y1 = min(height, y0 + step)
        if near is None:
            block = np.take(grid[y0:y1], selected, axis=2).astype(np.float32)
        else:
            block = np.asarray(grid[near[0, y0:y1], near[1, y0:y1]][:, :, selected],
                               np.float32)
        values = (torch.as_tensor(np.ascontiguousarray(block.reshape(-1, len(selected))),
                                  dtype=torch.float64, device="cuda") @ w_gpu +
                  b_gpu).cpu().numpy()
        if channels == 2:
            values = np.column_stack((-values.reshape(-1), values.reshape(-1)))
        raw[:, y0:y1] = values.reshape(y1-y0, width, channels).transpose(2, 0, 1)
        raw.flush()
    del raw
    torch.cuda.synchronize()
    os.replace(temp, output)
    return {"finite_tokens": int(finite.sum()),
            "imputed_tokens": int((~finite).sum())}
