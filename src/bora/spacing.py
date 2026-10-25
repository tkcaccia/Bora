"""Physical pixel-spacing validation shared by the opt-in semantic route."""

import numpy as np


def spacing_xy(value):
    spacing = np.asarray(value, dtype=float)
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("mpp_xy must contain two finite positive micrometres-per-pixel values")
    return spacing
