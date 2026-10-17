"""Explicit CPU implementation selection; no silent compiled fallback."""
import ctypes
from pathlib import Path
import numpy as np

_LIB = None


def library():
    global _LIB
    if _LIB is None:
        path = Path(__file__).resolve().parents[2] / "cpp/evidence/libbora_evidence.so"
        if not path.is_file():
            raise RuntimeError(f"Compile the evidence kernels first: {path}")
        _LIB = ctypes.CDLL(str(path))
        ptr = ctypes.c_void_p
        _LIB.bora_linear_logits.argtypes = [
            ptr, ptr, ptr, ptr, ctypes.c_int64, ctypes.c_int,
            ctypes.c_int, ctypes.c_int]
        _LIB.bora_simplex_qp.argtypes = [
            ptr, ptr, ptr, ptr, ptr, ptr, ctypes.c_int64, ctypes.c_int,
            ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int]
        _LIB.bora_semimarkov.argtypes = [
            ptr, ptr, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_double, ctypes.c_int]
        _LIB.bora_adaptive_linear.argtypes = [
            ptr, ptr, ptr, ptr, ptr, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int]
        _LIB.bora_region_counts.argtypes = [
            ptr, ptr, ptr, ptr, ptr, ctypes.c_int64, ctypes.c_int, ctypes.c_int]
        _LIB.bora_white_diversity.argtypes = [
            ptr, ptr, ptr, ptr, ctypes.c_int, ctypes.c_int,
            ctypes.c_int64, ctypes.c_int, ctypes.c_int]
    return _LIB


def logits(x, coefficient, intercept, threads=2):
    x = np.ascontiguousarray(x, dtype=np.float32)
    coefficient = np.ascontiguousarray(coefficient, dtype=np.float64)
    intercept = np.ascontiguousarray(intercept, dtype=np.float64)
    if x.shape[-1] != coefficient.shape[1] or len(intercept) != len(coefficient):
        raise ValueError("Incompatible linear model")
    out = np.empty((*x.shape[:-1], len(coefficient)), dtype=np.float32)
    status = library().bora_linear_logits(
        x.ctypes.data, coefficient.ctypes.data, intercept.ctypes.data,
        out.ctypes.data, int(np.prod(x.shape[:-1])), x.shape[-1],
        len(coefficient), threads)
    if status:
        raise RuntimeError("C++ linear kernel rejected input")
    return out


def simplex_qp(rgb, prior, colors, ridge=.012, tolerance=1e-8, threads=2):
    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    prior = np.ascontiguousarray(prior, dtype=np.float32)
    colors = np.ascontiguousarray(colors, dtype=np.float64)
    if rgb.shape[:-1] != prior.shape[:-1] or colors.shape != (prior.shape[-1], 3):
        raise ValueError("Incompatible palette model")
    shape = rgb.shape[:-1]
    out = np.empty_like(prior)
    residual = np.empty(shape, dtype=np.float64)
    iterations = np.empty(shape, dtype=np.int32)
    status = library().bora_simplex_qp(
        rgb.ctypes.data, prior.ctypes.data, colors.ctypes.data, out.ctypes.data,
        residual.ctypes.data, iterations.ctypes.data, int(np.prod(shape)),
        len(colors), ridge, tolerance, max(100, 8*len(colors)**2), threads)
    if status or not np.isfinite(out).all() or np.max(residual) > tolerance*5:
        raise RuntimeError(f"Simplex solver did not converge: {np.max(residual)}")
    return out, dict(max_kkt_residual=float(residual.max()),
                     mean_iterations=float(iterations.mean()),
                     max_iterations=int(iterations.max()),
                     openmp=bool(library().bora_evidence_openmp()),
                     threads=threads)


def semimarkov(unary, penalty=.6, duration_limit=24, threads=2):
    unary = np.ascontiguousarray(unary, dtype=np.float32)
    if unary.ndim != 3 or not np.isfinite(unary).all():
        raise ValueError("Expected finite rows by columns by class unary costs")
    out = np.empty(unary.shape[:2], np.uint16)
    result = library().bora_semimarkov(
        unary.ctypes.data, out.ctypes.data, *unary.shape,
        duration_limit, penalty, threads)
    if result:
        raise RuntimeError("C++ semi-Markov kernel rejected input")
    return out


def adaptive_linear(rgb, coefficient, intercept, threads=2):
    rgb = np.ascontiguousarray(rgb, np.float32)
    coefficient = np.ascontiguousarray(coefficient, np.float64)
    intercept = np.ascontiguousarray(intercept, np.float64)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or coefficient.shape != (len(intercept), 3):
        raise ValueError("Adaptive linear kernel expects RGB and three model coefficients")
    labels = np.empty(rgb.shape[:2], np.uint16)
    statistics = np.zeros(3, np.int64)
    status = library().bora_adaptive_linear(
        rgb.ctypes.data, coefficient.ctypes.data, intercept.ctypes.data,
        labels.ctypes.data, statistics.ctypes.data, *rgb.shape[:2],
        len(intercept), threads)
    if status:
        raise RuntimeError("Adaptive linear kernel rejected input")
    return labels, dict(visited_blocks=int(statistics[0]),
                        certified_blocks=int(statistics[1]),
                        densely_classified_pixels=int(statistics[2]),
                        total_pixels=int(np.prod(labels.shape)))


def region_counts(truth, prediction, components, component_count, threads=2):
    truth, prediction, components = map(np.asarray, (truth, prediction, components))
    if truth.shape != prediction.shape or truth.shape != components.shape:
        raise ValueError("Native metric geometry mismatch")
    if (truth.min() < 0 or truth.max() > 255 or prediction.min() < 0
            or prediction.max() > 255 or components.min() < 0
            or components.max() >= component_count):
        raise ValueError("Invalid native metric label or component IDs")
    truth = np.ascontiguousarray(truth, np.uint8)
    prediction = np.ascontiguousarray(prediction, np.uint8)
    components = np.ascontiguousarray(components, np.int32)
    confusion = np.zeros((256, 256), np.int64)
    correct = np.zeros(component_count, np.int64)
    status = library().bora_region_counts(
        truth.ctypes.data, prediction.ctypes.data, components.ctypes.data,
        confusion.ctypes.data, correct.ctypes.data, truth.size, component_count, threads)
    if status:
        raise RuntimeError("C++ metric kernel rejected input")
    return confusion, correct


def white_diversity(labels, points, offsets, threads=2):
    labels = np.ascontiguousarray(labels, np.uint16)
    points = np.ascontiguousarray(points, np.int64)
    offsets = np.ascontiguousarray(offsets, np.int32)
    if labels.ndim != 2 or offsets.ndim != 2 or offsets.shape[1] != 2:
        raise ValueError("Invalid categorical window input")
    if points.size and (points.min() < 0 or points.max() >= labels.size):
        raise ValueError("White query point is outside the image")
    result = np.empty(len(points), np.uint8)
    status = library().bora_white_diversity(
        labels.ctypes.data, points.ctypes.data, offsets.ctypes.data,
        result.ctypes.data, *labels.shape, len(points), len(offsets), threads)
    if status:
        raise RuntimeError("White diversity kernel rejected input")
    return result.astype(bool)
