"""Synthetic cached spatial fields, never encoder/model inference."""
import numpy as np


def spatial_case(classes=2, dtype=np.float32):
    rng = np.random.default_rng(123)
    shape = (360, 180 * classes)
    grid = tuple((np.asarray(shape) + 13) // 14)
    coarse = np.broadcast_to(np.arange(shape[1]) // 180 + 1, shape).astype(np.uint8).copy()
    values = rng.normal(0, .4, (*grid, 1280)).astype(dtype)
    for group in range(classes):
        at = ((np.arange(grid[1]) + .5) * 14 // 180) == group
        values[:, at, group * 24:(group + 1) * 24] += 2
    variants = {}
    for name, factor in (("scale_1", 2), ("scale_2", 4)):
        resized = tuple(int(round(v / factor)) for v in shape)
        g = tuple((np.asarray(resized) + 13) // 14)
        f = rng.normal(0, .4, (*g, 1280)).astype(dtype)
        for group in range(classes):
            at = ((np.arange(g[1]) + .5) * 14 * factor // 180) == group
            f[:, at, group * 24:(group + 1) * 24] += 2
        variants[name] = dict(values=f, shape_yx=resized)
    palette = [[210, 35, 75], [35, 170, 70], [40, 90, 220]][:classes]
    return values, coarse, palette, variants


def legacy_rgb9plus19(tokens, coarse, palette, variants):
    """Pre-extraction formula, evaluated with this test runtime's fitted models.

    Frozen from convert_rgb at 370ba9df782b7b1858d0222f13aeeedd2ca8524b,
    src/bora/conversion.py, after input/fallback validation. The unchanged
    observation, selection, alignment and probability helpers are reused;
    none of the fitted-recipe implementation or serialized replay is called.
    Keep the formula here independent of future recipe refactors.
    """
    from itertools import combinations
    from scipy import ndimage as ndi
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from bora.conversion import (observations, select_dimensions, _align,
                                 _expand, _scale, _probabilities)

    summaries, labels = observations(tokens, coarse)
    classes = np.unique(labels)
    selected = select_dimensions(summaries, labels)
    scaler = StandardScaler().fit(summaries[:, selected])
    scaler.scale_ = np.maximum(scaler.scale_, .05)
    tile_z = scaler.transform(summaries[:, selected]).astype(np.float32)
    z = scaler.transform(tokens[..., selected].reshape(-1, 64)).reshape(*tokens.shape[:2], 64).astype(np.float32)
    aligned = {}
    for name in ("scale_1", "scale_2"):
        spec = variants[name]
        values = np.asarray(spec["values"], dtype=np.float32)
        if values.shape[-1] == 1280:
            values = values[..., selected]
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
        model = Ridge(alpha=100).fit(neighbours[train], z[train])
        completed[test] = model.predict(neighbours[test])
    p19 = _expand(_probabilities(completed, coarse, classes), coarse.shape)
    p = np.mean(np.stack([p9, p19]), axis=0, dtype=np.float32)
    colors = np.asarray(palette, np.float32)[classes - 1]
    rgb = np.rint(np.clip(p, 0, 1) @ colors).clip(0, 255).astype(np.uint8)
    return rgb, p, selected


def assert_frozen_rgb_quantization(actual, expected):
    """Allow isolated last-bit threshold crossings, never spatial/color drift."""
    assert actual.dtype == expected.dtype == np.uint8
    assert actual.shape == expected.shape
    delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    assert delta.max(initial=0) <= 1, "Frozen RGB differs by more than one intensity level"
    assert np.count_nonzero(delta) <= actual.size * 1e-5, "Frozen RGB mismatch exceeds 0.001% of channels"
