"""A compact spatial-context head for registered Virchow2 WSI tokens.

This module performs one masked Gaussian pooling of an already projected
feature lattice, then scores it with a numeric three-class logistic head.
Pooling is global across encoder patch seams. Any training labels and class
priors are explicit caller inputs; no tissue identity is inferred from H&E.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def contextualize_pca(field: np.ndarray, support: np.ndarray,
                      sigma_tokens: float) -> np.ndarray:
    """Pool a global PCA feature lattice without bleeding across background."""
    x = np.asarray(field)
    tissue = np.asarray(support)
    if (x.ndim != 3 or x.dtype != np.float32 or x.shape[2] < 2 or
            tissue.shape != x.shape[:2] or tissue.dtype != bool or
            not np.isfinite(x).all() or
            not np.isfinite(sigma_tokens) or not 0 <= sigma_tokens <= 8):
        raise ValueError("Require finite float32 PCA field, aligned support and bounded sigma")
    if sigma_tokens == 0:
        return x.copy()
    weighted = x * tissue[..., None]
    blurred = ndi.gaussian_filter(weighted,
        sigma=(sigma_tokens, sigma_tokens, 0), mode="nearest")
    mass = ndi.gaussian_filter(tissue.astype(np.float32),
        sigma=sigma_tokens, mode="nearest")
    result = blurred / np.maximum(mass[..., None], 1e-6)
    result[~tissue] = x[~tissue]
    return result.astype(np.float32, copy=False)


def fit_numeric_head(samples: np.ndarray, labels: np.ndarray, *,
                     regularization_c: float = .1) -> dict:
    """Fit the established balanced logistic family and return numeric arrays."""
    x = np.asarray(samples)
    y = np.asarray(labels)
    if (x.ndim != 2 or x.dtype != np.float32 or not np.isfinite(x).all() or
            y.ndim != 1 or y.shape[0] != x.shape[0] or y.dtype.kind not in "iu" or
            not np.array_equal(np.unique(y), np.array([1, 2, 3])) or
            not np.isfinite(regularization_c) or
            not 0 < regularization_c <= 10):
        raise ValueError("Require finite aligned labeled PCA samples")
    scaler = StandardScaler().fit(x)
    classifier = LogisticRegression(C=regularization_c,
        class_weight="balanced", max_iter=1000, random_state=17).fit(
            scaler.transform(x), y)
    return {"scaler_mean": scaler.mean_.astype(np.float32),
            "scaler_scale": scaler.scale_.astype(np.float32),
            "logistic_coef": classifier.coef_.astype(np.float32),
            "logistic_intercept": classifier.intercept_.astype(np.float32),
            "class_ids": classifier.classes_.astype(np.int32)}


def predict_numeric_head(field: np.ndarray, head: dict, *,
                         training_prior: np.ndarray | None = None,
                         prior_exponent: float = 0,
                         batch_size: int = 8192) -> np.ndarray:
    """Score all tokens in absolute lattice order; optionally correct priors."""
    x = np.asarray(field)
    if x.ndim != 3 or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("Require finite float32 contextual features")
    if (type(batch_size) is not int or not 1 <= batch_size <= 65536 or
            not np.isfinite(prior_exponent) or not 0 <= prior_exponent <= 1):
        raise ValueError("Invalid numeric scoring parameters")
    if not isinstance(head, dict) or not {
            "scaler_mean", "scaler_scale", "logistic_coef",
            "logistic_intercept", "class_ids"}.issubset(head):
        raise ValueError("Numeric classifier arrays missing")
    mean = np.asarray(head.get("scaler_mean"))
    scale = np.asarray(head.get("scaler_scale"))
    coef = np.asarray(head.get("logistic_coef"))
    intercept = np.asarray(head.get("logistic_intercept"))
    ids = np.asarray(head.get("class_ids"))
    classes = len(ids)
    if (classes < 3 or mean.shape != (x.shape[2],) or
            scale.shape != mean.shape or coef.shape != (classes, x.shape[2]) or
            intercept.shape != (classes,) or not np.array_equal(ids, np.arange(1, classes+1)) or
            not all(np.isfinite(a).all() for a in (mean, scale, coef, intercept)) or
            np.any(scale <= 0)):
        raise ValueError("Numeric classifier arrays or class order invalid")
    if training_prior is None:
        if prior_exponent != 0:
            raise ValueError("Prior exponent requires a training prior")
        prior = np.ones(classes, np.float32)
    else:
        prior = np.asarray(training_prior, np.float32)
        if (prior.shape != (classes,) or not np.isfinite(prior).all() or
                np.any(prior <= 0) or not np.isclose(prior.sum(), 1, atol=1e-6)):
            raise ValueError("Training prior must be a positive simplex")
    result = np.empty((*x.shape[:2], classes), np.float32)
    flat_x = x.reshape(-1, x.shape[2])
    flat_result = result.reshape(-1, classes)
    prior_factor = np.power(prior, prior_exponent).astype(np.float32)
    for start in range(0, len(flat_x), batch_size):
        stop = min(start + batch_size, len(flat_x))
        scores = ((flat_x[start:stop] - mean) / scale) @ coef.T + intercept
        scores -= scores.max(axis=1, keepdims=True)
        probability = np.exp(scores).astype(np.float32)
        probability *= prior_factor
        probability /= probability.sum(axis=1, keepdims=True)
        flat_result[start:stop] = probability
    return result
