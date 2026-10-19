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
