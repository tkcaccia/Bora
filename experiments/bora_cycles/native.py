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
