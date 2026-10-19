"""Fixed RGB9+19 evidence conversion; no semantic tissue classifier choices.

The logistic and ridge fits below are the unchanged within-image RGB conversion,
not alternatives to the supervised whole-group fastPLS classifier.
"""
from itertools import combinations
import cv2
import numpy as np
from scipy import ndimage as ndi
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


def observations(features, coarse):
    h, w = coarse.shape
    cy = (np.arange(features.shape[0]) + .5) * 14
    cx = (np.arange(features.shape[1]) + .5) * 14
    values, labels = [], []
    for y in range(0, h, 90):
        for x in range(0, w, 90):
            y1, x1 = min(y + 90, h), min(x + 90, w)
            if min(y1 - y, x1 - x) < .8 * 90:
                continue
            ids, counts = np.unique(coarse[y:y1, x:x1], return_counts=True)
            label = int(ids[np.argmax(counts)])
            if label == 0 or counts.max() / counts.sum() < .95:
                continue
            yi = np.flatnonzero((cy >= y + 22) & (cy < y1 - 22))
            xi = np.flatnonzero((cx >= x + 22) & (cx < x1 - 22))
            if len(yi) and len(xi):
                values.append(features[np.ix_(yi, xi)].mean((0, 1), dtype=np.float64))
                labels.append(label)
    if not values:
        raise ValueError("No eligible coarse tile observations")
    return np.asarray(values), np.asarray(labels)


def select_dimensions(values, labels):
    """The frozen 64-channel pair-balanced relevance/redundancy rule."""
    z = StandardScaler().fit_transform(values)
    classes = np.unique(labels)
    means = np.stack([z[labels == g].mean(0) for g in classes])
    variances = np.stack([z[labels == g].var(0) for g in classes])
    fisher = np.array([(means[a] - means[b]) ** 2 / (variances[a] + variances[b] + .01)
                      for a, b in combinations(range(len(classes)), 2)])
    ranks = [np.argsort(-row, kind="stable") for row in fisher]
    pool = np.unique(np.concatenate([r[:128] for r in ranks]))
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.nan_to_num(np.corrcoef(z[:, pool], rowvar=False), nan=0.)
    lookup = {int(feature): i for i, feature in enumerate(pool)}
    chosen = []
    while len(chosen) < 64:
        for row in fisher:
            available = np.array([i for i in pool if int(i) not in chosen])
            if not len(available):
                raise ValueError("Fewer than 64 available conversion dimensions")
            redundancy = (np.max(np.abs(correlation[np.ix_(
                [lookup[int(i)] for i in available], [lookup[i] for i in chosen])]), axis=1)
                if chosen else np.zeros(len(available)))
            chosen.append(int(available[np.argmax(row[available] * (1 - .85 * redundancy))]))
            if len(chosen) == 64:
                break
    return np.asarray(chosen)


def _fit(x, y):
    return LogisticRegression(C=.3, class_weight="balanced", max_iter=2000, random_state=430).fit(x, y)


def _expand(values, shape):
    gh, gw = values.shape[:2]
    return cv2.resize(np.asarray(values, np.float32), (gw * 14, gh * 14),
                      interpolation=cv2.INTER_LINEAR)[:shape[0], :shape[1]]


def _align(values, variant_shape, base_grid, base_shape):
    sh, sw = variant_shape
    y = (np.arange(base_grid[0]) + .5) * 14 * sh / base_shape[0] / 14 - .5
    x = (np.arange(base_grid[1]) + .5) * 14 * sw / base_shape[1] / 14 - .5
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return np.stack([ndi.map_coordinates(values[..., i], [yy, xx], order=1, mode="nearest")
                     for i in range(values.shape[-1])], -1)


def _scale(values, coarse):
    summaries, _ = observations(values, coarse)
    scaler = StandardScaler().fit(summaries)
    scaler.scale_ = np.maximum(scaler.scale_, .05)
    return (scaler.transform(values.reshape(-1, values.shape[-1])).reshape(values.shape).astype(np.float32),
            scaler.transform(summaries))


def _probabilities(values, coarse, classes):
    summaries, labels = observations(values, coarse)
    clf = _fit(summaries, labels)
    if not np.array_equal(clf.classes_, classes):
        raise ValueError("RGB conversion class ordering changed")
    return clf.predict_proba(values.reshape(-1, values.shape[-1])).reshape(
        *values.shape[:2], len(classes)).astype(np.float32)


def convert_rgb(tokens, coarse, palette, variants, *, allow_local_fallback=False):
    """Fit local RGB9+19 once for a new field; replay via RGB9Plus19Recipe.

    The legacy convenience API and explicit homogeneous-WSI fallback remain.
    For repeated use save a fitted recipe and call its transform/predict_proba;
    loading that JSON never refits any scaler or logistic/Ridge model.
    """
    from .rgb_recipe import RGB9Plus19Recipe
    summaries, labels = observations(tokens, coarse)
    classes = np.unique(labels)
    if not np.array_equal(classes, np.unique(coarse[coarse > 0])):
        raise ValueError("Every coarse group needs an eligible tile observation")
    palette = np.asarray(palette, dtype=np.float32)
    if (palette.ndim != 2 or palette.shape[1] != 3 or len(palette) < coarse.max()
            or not np.isfinite(palette).all() or np.any(palette < 0) or np.any(palette > 255)):
        raise ValueError("Finite RGB palette indexed by coarse label minus one required")
    if len(summaries) < 16 or len(classes) < 2:
        if not allow_local_fallback:
            if len(summaries) < 16:
                raise ValueError("At least 16 eligible physical coarse tiles are required")
            raise ValueError("RGB9+19 requires at least two observed coarse groups")
        rgb = np.empty((*coarse.shape, 3), np.uint8)
        rgb[:] = np.rint(np.mean(palette[classes - 1], axis=0)).astype(np.uint8)
        present = coarse > 0
        rgb[present] = np.rint(palette[coarse[present] - 1]).astype(np.uint8)
        return rgb, dict(conversion="coarse-palette fallback",
            reason="fewer than 16 eligible local tiles or only one local group",
            eligible_local_tiles=int(len(summaries)), observed_local_groups=classes.tolist(),
            full_RGB9_plus_19_applied=False, fine_reference_used=False)
    recipe = RGB9Plus19Recipe.fit(tokens, coarse, palette, variants)
    return recipe.transform(tokens, variants)
