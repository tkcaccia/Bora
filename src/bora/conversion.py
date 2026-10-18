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


def convert_rgb(tokens, coarse, palette, variants):
    """Average #9 and #19 class probabilities before 8-bit palette conversion.

    variants contains scale_1 and scale_2: values, original resized shape, and
    selected_dimensions for 64-channel caches (or full 1280-channel arrays).
    Only coarse group observations enter these fits; no fine masks are accepted.
    """
    summaries, labels = observations(tokens, coarse)
    classes = np.unique(labels)
    if len(summaries) < 16:
        raise ValueError("At least 16 eligible physical coarse tiles are required")
    if len(classes) < 2:
        raise ValueError("RGB9+19 requires at least two observed coarse groups")
    if not np.array_equal(classes, np.unique(coarse[coarse > 0])):
        raise ValueError("Every coarse group needs an eligible tile observation")
    palette = np.asarray(palette, dtype=np.float32)
    if (palette.ndim != 2 or palette.shape[1] != 3 or len(palette) < coarse.max()
            or not np.isfinite(palette).all() or np.any(palette < 0) or np.any(palette > 255)):
        raise ValueError("Finite RGB palette indexed by coarse label minus one required")
    selected = select_dimensions(summaries, labels)
    scaler = StandardScaler().fit(summaries[:, selected])
    scaler.scale_ = np.maximum(scaler.scale_, .05)
    tile_z = scaler.transform(summaries[:, selected]).astype(np.float32)
    z = scaler.transform(tokens[..., selected].reshape(-1, 64)).reshape(*tokens.shape[:2], 64).astype(np.float32)
    aligned = {}
    for name in ("scale_1", "scale_2"):
        spec = variants[name]
        values = np.asarray(spec["values"], dtype=np.float32)
        if values.ndim != 3 or not np.isfinite(values).all():
            raise ValueError("Finite spatial scale features required")
        if values.shape[-1] == 1280:
            values = values[..., selected]
        elif values.shape[-1] != 64 or not np.array_equal(spec.get("selected_dimensions"), selected):
            raise ValueError("Scale-cache selection does not match this coarse mask")
        aligned[name] = _align(values, spec["shape_yx"], z.shape[:2], coarse.shape)
    middle, mt = _scale(aligned["scale_1"], coarse)
    wide, wt = _scale(aligned["scale_2"], coarse)
    with np.errstate(invalid="ignore", divide="ignore"):
        correlations = np.stack([np.array([np.corrcoef(tile_z[:, i], other[:, i])[0, 1]
                                          for i in range(64)]) for other in (mt, wt)])
    persistence = np.maximum(np.nan_to_num(correlations.min(0)), 0)
    means = [tile_z[labels == g].mean(0) for g in classes]
    var = [tile_z[labels == g].var(0) for g in classes]
    rankings = []
    for a, b in combinations(range(len(classes)), 2):
        score = (means[a] - means[b]) ** 2 / (var[a] + var[b] + .05)
        score *= .05 + persistence
        rankings.append(np.argsort(-score))
    keep, positions = [], [0] * len(rankings)
    while len(keep) < 32:
        for i, ranking in enumerate(rankings):
            while positions[i] < len(ranking) and int(ranking[positions[i]]) in keep:
                positions[i] += 1
            if positions[i] < len(ranking):
                keep.append(int(ranking[positions[i]])); positions[i] += 1
                if len(keep) == 32:
                    break
    consensus = (z[..., keep] + middle[..., keep] + wide[..., keep]) / 3
    p9 = _expand(_probabilities(consensus, coarse, classes), coarse.shape)
    neighbours = np.empty((*z.shape[:2], 192), np.float32)
    for j, k in enumerate((1, 2, 4)):
        padded = np.pad(z, ((k, k), (k, k), (0, 0)), mode="reflect")
        neighbours[..., j * 64:(j + 1) * 64] = (padded[:-2*k, k:-k] + padded[2*k:, k:-k]
            + padded[k:-k, :-2*k] + padded[k:-k, 2*k:]) / 4
    yy, xx = np.indices(z.shape[:2])
    folds = (yy >= z.shape[0] // 2) * 2 + (xx >= z.shape[1] // 2)
    completed = np.empty_like(z)
    for fold in range(4):
        test = folds == fold
        train = ~ndi.binary_dilation(test, iterations=4)
        if not train.any() or not test.any():
            raise ValueError("Insufficient spatial token extent for the fixed #19 folds")
        model = Ridge(alpha=100).fit(neighbours[train], z[train])
        completed[test] = model.predict(neighbours[test])
    p19 = _expand(_probabilities(completed, coarse, classes), coarse.shape)
    p = np.mean(np.stack([p9, p19]), axis=0, dtype=np.float32)
    rgb = np.rint(np.clip(p, 0, 1) @ palette[classes - 1]).clip(0, 255).astype(np.uint8)
    return rgb, dict(conversion="RGB9+19", components=[9, 19], weights=[.5, .5],
                     selected_dimensions=selected.tolist(), fine_reference_used=False)
