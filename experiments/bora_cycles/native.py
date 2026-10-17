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
        _LIB.bora_quadratic_simplex.argtypes = [
            ptr, ptr, ptr, ptr, ptr, ptr, ctypes.c_int64, ctypes.c_int,
            ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_int]
        _LIB.bora_pair_palette.argtypes = [
            ptr, ptr, ptr, ptr, ctypes.c_int64, ctypes.c_int,
            ctypes.c_double, ctypes.c_int]
        _LIB.bora_entropy_palette.argtypes = [
            ptr, ptr, ptr, ptr, ptr, ptr, ctypes.c_int64, ctypes.c_int,
            ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int]
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


def gaussian_logits(x, means, precision, intercept, threads=2):
    x = np.ascontiguousarray(x, np.float32)
    means, precision, intercept = [np.ascontiguousarray(a, np.float64)
                                    for a in (means, precision, intercept)]
    k, d = means.shape
    if (x.shape[-1] != d or precision.shape != (k, d, d) or intercept.shape != (k,)
            or not all(np.isfinite(a).all() for a in (x, means, precision, intercept))):
        raise ValueError('Incompatible finite Gaussian model')
    function = library().bora_gaussian_logits
    function.argtypes = [ctypes.c_void_p]*5+[ctypes.c_int64]+[ctypes.c_int]*3
    function.restype = ctypes.c_int
    output = np.empty((*x.shape[:-1], k), np.float32)
    status = function(x.ctypes.data, means.ctypes.data, precision.ctypes.data,
        intercept.ctypes.data, output.ctypes.data, int(np.prod(x.shape[:-1])), d, k, threads)
    if status or not np.isfinite(output).all():
        raise RuntimeError('C++ Gaussian scoring failed')
    return output


def chebyshev_palette(x, prior, colors, ridge=.012, beta=.025,
                      tolerance=1e-7, iterations=600, threads=2):
    x, prior = [np.ascontiguousarray(a, np.float32) for a in (x, prior)]
    colors = np.ascontiguousarray(colors, np.float64)
    if (x.ndim != 2 or x.shape[1] != 3 or prior.ndim != 2 or len(prior) != len(x)
            or colors.shape != (prior.shape[1], 3) or prior.shape[1] < 2
            or ridge <= 0 or beta <= 0 or tolerance <= 0 or iterations < 1
            or not all(np.isfinite(a).all() for a in (x, prior, colors))):
        raise ValueError('Incompatible Chebyshev palette problem')
    function = library().bora_chebyshev_palette
    function.argtypes = [ctypes.c_void_p]*6+[ctypes.c_int64, ctypes.c_int]+[ctypes.c_double]*4+[ctypes.c_int]*2
    function.restype = ctypes.c_int
    output = np.empty_like(prior); gap = np.empty(len(x), np.float64); used = np.empty(len(x), np.int32)
    step = .99/max(float(np.linalg.norm(colors, 2)), 1e-8)
    status = function(x.ctypes.data, prior.ctypes.data, colors.ctypes.data, output.ctypes.data,
        gap.ctypes.data, used.ctypes.data, len(x), prior.shape[1], ridge, beta,
        step, tolerance, iterations, threads)
    if status or not np.isfinite(output).all() or not np.isfinite(gap).all():
        raise RuntimeError('C++ Chebyshev inverse failed')
    return output, dict(max_primal_dual_gap=float(gap.max()),
        unconverged_nodes=int((gap > tolerance).sum()), converged=bool(np.all(gap <= tolerance)),
        max_iterations=int(used.max()), mean_iterations=float(used.mean()),
        ridge=ridge, maximum_channel_weight=beta, threads=threads)


def dominance_qp(x, prior, colors, labels, initial=None, tolerance=1e-7,
                 iterations=400, threads=2):
    x, prior = [np.ascontiguousarray(a, np.float32) for a in (x, prior)]
    colors = np.ascontiguousarray(colors, np.float64)
    labels = np.ascontiguousarray(labels, np.int32)
    initial = np.ascontiguousarray(prior if initial is None else initial, np.float32)
    if (x.ndim != 2 or x.shape[1] != 3 or prior.ndim != 2 or len(x) != len(prior)
            or colors.shape != (prior.shape[1], 3) or labels.shape != (len(x),)
            or initial.shape != prior.shape or np.any(labels < 0)
            or np.any(labels >= prior.shape[1])
            or not all(np.isfinite(a).all() for a in (x, prior, colors, initial))):
        raise ValueError('Incompatible dominance quadratic')
    function = library().bora_dominance_qp
    function.argtypes = [ctypes.c_void_p]*8+[ctypes.c_int64, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int]
    function.restype = ctypes.c_int
    output = np.empty_like(prior)
    residual = np.empty(len(x), np.float64)
    used = np.empty(len(x), np.int32)
    status = function(x.ctypes.data, prior.ctypes.data, colors.ctypes.data,
        initial.ctypes.data, labels.ctypes.data, output.ctypes.data, residual.ctypes.data,
        used.ctypes.data, len(x), prior.shape[1], .012, tolerance, iterations, threads)
    if status or not np.isfinite(output).all() or not np.isfinite(residual).all():
        raise RuntimeError('C++ dominance quadratic failed')
    return output, dict(max_projected_gradient_residual=float(residual.max()),
        unconverged_bags=int((residual > 1e-6).sum()), max_iterations=int(used.max()),
        mean_iterations=float(used.mean()), threads=threads,
        converged=bool(np.all(residual <= 1e-6)))


def local_mixture_logits(x, coefficient, intercept, mixing, threads=2):
    x = np.ascontiguousarray(x, np.float32)
    coefficient = np.ascontiguousarray(coefficient, np.float64)
    intercept = np.ascontiguousarray(intercept, np.float64)
    mixing = np.ascontiguousarray(mixing, np.float32)
    if (mixing.shape != (*x.shape[:-1], len(coefficient)) or coefficient.ndim != 2
            or coefficient.shape[1] != x.shape[-1] or intercept.shape != (len(coefficient),)
            or np.any(mixing < 0) or not np.allclose(mixing.sum(-1), 1, atol=1e-6)
            or not all(np.isfinite(v).all() for v in (x, coefficient, intercept, mixing))):
        raise ValueError("Incompatible spatial mixture")
    function = library().bora_local_mixture_logits
    function.argtypes = [ctypes.c_void_p]*5+[ctypes.c_int64]+[ctypes.c_int]*3
    function.restype = ctypes.c_int
    out = np.empty(x.shape[:-1], np.float32)
    status = function(x.ctypes.data, coefficient.ctypes.data, intercept.ctypes.data,
                      mixing.ctypes.data, out.ctypes.data, out.size, x.shape[-1],
                      len(coefficient), threads)
    if status or not np.isfinite(out).all():
        raise RuntimeError("C++ local mixture rejected input")
    return out


def categorical_winner(probability, labels, threads=2):
    probability = np.ascontiguousarray(probability, dtype=np.float64)
    labels = np.ascontiguousarray(labels, dtype=np.int32)
    if (probability.ndim != 3 or labels.shape != (probability.shape[0],)
            or probability.shape[-1] < 2 or not np.isfinite(probability).all()
            or np.any(probability < 0)
            or not np.allclose(probability.sum(-1), 1, atol=1e-10)):
        raise ValueError("Expected bags of categorical probability vectors")
    function = library().bora_categorical_winner
    function.argtypes = [ctypes.c_void_p]*4+[ctypes.c_int]*4
    function.restype = ctypes.c_int
    logp = np.empty(len(labels), np.float64)
    derivative = np.empty_like(probability)
    status = function(probability.ctypes.data, labels.ctypes.data, logp.ctypes.data,
                      derivative.ctypes.data, *probability.shape, threads)
    if status or not np.isfinite(logp).all() or not np.isfinite(derivative).all():
        raise RuntimeError("C++ categorical count likelihood rejected input")
    return logp, derivative


def hull_depth(x, planes, threads=2):
    x = np.ascontiguousarray(x, dtype=np.float32)
    planes = np.ascontiguousarray(planes, dtype=np.float64)
    if x.ndim != 2 or planes.ndim != 2 or planes.shape[1] != x.shape[1]+1 or not len(planes):
        raise ValueError("Incompatible convex support geometry")
    function = library().bora_hull_depth
    function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                         ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    function.restype = ctypes.c_int
    out = np.empty(len(x), np.float32)
    status = function(x.ctypes.data, planes.ctypes.data, out.ctypes.data,
                      len(x), x.shape[1], len(planes), threads)
    if status:
        raise RuntimeError("C++ convex support kernel rejected input")
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


def quadratic_simplex(hessian, linear, prior, tolerance=1e-8, threads=2):
    prior = np.ascontiguousarray(prior, np.float32)
    linear = np.ascontiguousarray(linear, np.float64)
    hessian = np.ascontiguousarray(hessian, np.float64)
    k, n = prior.shape[-1], int(np.prod(prior.shape[:-1]))
    if linear.shape != prior.shape or hessian.shape not in ((k, k), (*prior.shape[:-1], k, k)):
        raise ValueError("Incompatible general simplex quadratic")
    if not all(np.isfinite(a).all() for a in (prior, linear, hessian)):
        raise ValueError("Nonfinite simplex quadratic")
    out = np.empty_like(prior)
    residual = np.empty(prior.shape[:-1], np.float64)
    iterations = np.empty(prior.shape[:-1], np.int32)
    stride = 0 if hessian.ndim == 2 else k*k
    result = library().bora_quadratic_simplex(
        hessian.ctypes.data, linear.ctypes.data, prior.ctypes.data,
        out.ctypes.data, residual.ctypes.data, iterations.ctypes.data,
        n, k, stride, tolerance, max(100, 8*k*k), threads)
    if result or not np.isfinite(out).all() or residual.max() > 5*tolerance:
        raise RuntimeError(f"General simplex QP failed: {residual.max()}")
    return out, dict(max_kkt_residual=float(residual.max()),
                     mean_iterations=float(iterations.mean()), threads=threads)


def pair_palette(rgb, prior, colors, ridge=.012, threads=2):
    rgb, prior = (np.ascontiguousarray(a, np.float32) for a in (rgb, prior))
    colors = np.ascontiguousarray(colors, np.float64)
    if rgb.shape[:-1] != prior.shape[:-1] or colors.shape != (prior.shape[-1], 3):
        raise ValueError("Incompatible pair palette geometry")
    out = np.empty_like(prior)
    status = library().bora_pair_palette(
        rgb.ctypes.data, prior.ctypes.data, colors.ctypes.data, out.ctypes.data,
        int(np.prod(prior.shape[:-1])), len(colors), ridge, threads)
    if status or not np.isfinite(out).all():
        raise RuntimeError("Pair palette solver failed")
    return out, dict(exact_faces=len(colors)*(len(colors)+1)//2, threads=threads)


def entropy_palette(rgb, prior, colors, ridge=.012, threads=2):
    rgb, prior = (np.ascontiguousarray(a, np.float32) for a in (rgb, prior))
    colors = np.ascontiguousarray(colors, np.float64)
    if rgb.shape[:-1] != prior.shape[:-1] or colors.shape != (prior.shape[-1], 3):
        raise ValueError("Incompatible relative-entropy palette geometry")
    out = np.empty_like(prior)
    residual = np.empty(prior.shape[:-1], np.float64)
    iterations = np.empty(prior.shape[:-1], np.int32)
    status = library().bora_entropy_palette(
        rgb.ctypes.data, prior.ctypes.data, colors.ctypes.data, out.ctypes.data,
        residual.ctypes.data, iterations.ctypes.data, int(np.prod(prior.shape[:-1])),
        len(colors), ridge, 1e-6, 80, threads)
    if status or not np.isfinite(out).all() or residual.max() > 1e-4:
        raise RuntimeError(f"Entropy palette solver failed: {residual.max()}")
    return out, dict(max_stationarity_residual=float(residual.max()),
                     mean_iterations=float(iterations.mean()), threads=threads)
