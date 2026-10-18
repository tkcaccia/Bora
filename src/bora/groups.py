"""Whole anonymous groups, pooled from calibrated, unselected Virchow2 tokens."""
import numpy as np
from scipy import sparse


def spacing_xy(value):
    spacing = np.asarray(value, dtype=float)
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("mpp_xy must contain two finite positive micrometres-per-pixel values")
    return spacing


def compact_groups(mask):
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype.kind not in "iu" or np.any(mask < 0):
        raise ValueError("A nonnegative 2D integer group mask is required")
    ids = np.unique(mask[mask > 0])
    if not len(ids):
        raise ValueError("No positive tissue groups survive")
    # Do not allocate a lookup table proportional to arbitrary input IDs.
    groups = np.searchsorted(ids, mask).astype(np.int32) + 1
    groups[mask == 0] = 0
    return groups, ids


def group_means(tokens, regions, working_shape_yx):
    """Exact native-pixel overlap pooling; disconnected pieces stay together.

    Tokens tile the padded working canvas with a 14-pixel stride. Pool all 1280
    channels. Never resize the feature grid to the unpadded image dimensions.
    """
    tokens = np.asarray(tokens)
    regions = np.asarray(regions)
    working = np.asarray(working_shape_yx)
    if (tokens.ndim != 3 or tokens.shape[2] != 1280 or not np.isfinite(tokens).all()
            or working.shape != (2,) or np.any(working < 1)
            or not np.array_equal(working, np.floor(working))):
        raise ValueError("Finite full 1280-channel tokens and positive working shape required")
    if tuple(tokens.shape[:2]) != tuple((working.astype(int) + 13) // 14):
        raise ValueError("Token grid must cover the 14-pixel padded working canvas")
    if regions.ndim != 2 or regions.dtype.kind not in "iu" or np.any(regions < 0):
        raise ValueError("A 2D integer region mask is required")
    ids = np.unique(regions[regions > 0])
    if not np.array_equal(ids, np.arange(1, len(ids) + 1)) or not len(ids):
        raise ValueError("Positive regions must be compact, contiguous group IDs")
    h, w = regions.shape
    gh, gw = tokens.shape[:2]
    yy = np.minimum(((np.arange(h) + .5) * working[0] / h / 14).astype(int), gh - 1)
    xx = np.minimum(((np.arange(w) + .5) * working[1] / w / 14).astype(int), gw - 1)
    token_ids = yy[:, None] * gw + xx[None, :]
    valid = regions > 0
    weights = sparse.coo_matrix((np.ones(int(valid.sum())),
        (regions[valid] - 1, token_ids[valid])), shape=(len(ids), gh * gw)).tocsr()
    mass = np.asarray(weights.sum(1)).ravel()
    flat = np.asarray(tokens, dtype=np.float32).reshape(-1, 1280)
    means = ((weights @ flat) / mass[:, None]).astype(np.float32)
    return means, mass


def boundary_context(regions, means, mpp_xy):
    """Neighbor descriptor: mean and contrast to boundary-weighted neighbours."""
    regions = np.asarray(regions)
    means = np.asarray(means, dtype=np.float64)
    mpp = spacing_xy(mpp_xy)
    if (regions.ndim != 2 or regions.dtype.kind not in "iu" or means.ndim != 2
            or means.shape[1] != 1280 or not np.isfinite(means).all()
            or np.any(regions < 0)
            or not np.array_equal(np.unique(regions[regions > 0]), np.arange(1, len(means) + 1))):
        raise ValueError("Compact regions and their finite 1280-channel means required")
    rows, cols, weights = [], [], []
    # A left/right interface spans one y pixel; an up/down interface spans x.
    for a, b, length in [(regions[:, :-1], regions[:, 1:], mpp[1]),
                          (regions[:-1], regions[1:], mpp[0])]:
        use = (a > 0) & (b > 0) & (a != b)
        i, j = a[use] - 1, b[use] - 1
        rows.extend([i, j]); cols.extend([j, i])
        weights.extend([np.full(len(i), length), np.full(len(i), length)])
    graph = sparse.coo_matrix((np.concatenate(weights),
        (np.concatenate(rows), np.concatenate(cols))), shape=(len(means), len(means))).tocsr()
    total = np.asarray(graph.sum(1)).ravel()
    neighbours = means.copy()
    has = total > 0
    neighbours[has] = (graph @ means)[has] / total[has, None]
    # Isolated groups have zero context contrast, rather than invented neighbours.
    return np.concatenate([means, means - neighbours], axis=1)


def describe_groups(tokens, mask, working_shape_yx, native_mpp_xy):
    regions, ids = compact_groups(mask)
    means, mass = group_means(tokens, regions, working_shape_yx)
    return regions, ids, boundary_context(regions, means, native_mpp_xy), mass
