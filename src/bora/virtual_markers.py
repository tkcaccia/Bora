"""Coarse-label-only channel selection and explicit physical sampling plans.

These helpers do not run a model or accept a reference segmentation mask.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def sampling_plan(shape_yx, native_mpp_xy, target_mpp):
    """Preserve field of view, accounting for integer output dimensions."""
    h, w = map(int, shape_yx)
    mx, my = map(float, native_mpp_xy)
    target = float(target_mpp)
    if min(h, w, mx, my, target) <= 0 or not np.isfinite([mx, my, target]).all():
        raise ValueError("Image dimensions and micrometres per pixel must be positive")
    nh, nw = max(1, round(h * my / target)), max(1, round(w * mx / target))
    return {"native_shape_yx": [h, w], "inference_shape_yx": [nh, nw],
            "native_mpp_xy": [mx, my], "requested_mpp": target,
            "effective_inference_mpp_xy": [w * mx / nw, h * my / nh],
            "field_of_view_um_xy": [w * mx, h * my],
            "pixel_centres": "(x + 0.5) * mpp_x, (y + 0.5) * mpp_y",
            "optical_resolution_improved_by_resampling": False}


def select_markers(maps_hwc, coarse, tile_size, names, *, top_k=6,
                   min_tiles=3, excluded=("DAPI", "TRITC", "Cy5")):
    """Rank channels by largest absolute Hedges g between coarse tile groups.

    One central-half mean per original tile is the observation. Groups must
    have at least min_tiles; a 0.01 activation-SD floor limits unstable ratios.
    Spatial correlation is not modelled: these are descriptive effects, with
    no significance claim. Fine reference labels must never be passed here.
    """
    values = np.asarray(maps_hwc)
    labels = np.asarray(coarse)
    if (values.ndim != 3 or values.shape[:2] != labels.shape or
            values.shape[-1] != len(names) or not np.isfinite(values).all()):
        raise ValueError("Finite HWC marker maps must align to coarse labels")
    if tile_size < 2 or top_k < 1 or min_tiles < 2:
        raise ValueError("Invalid tile/selection settings")
    records, observations = [], {}
    h, w = labels.shape
    margin = max(1, tile_size // 4)
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            y1, x1 = min(y + tile_size, h), min(x + tile_size, w)
            if min(y1-y, x1-x) < tile_size * 0.8:
                continue
            region = labels[y:y1, x:x1]
            ids, counts = np.unique(region, return_counts=True)
            label = int(ids[np.argmax(counts)])
            if label == 0 or counts.max() / region.size < 0.95:
                continue
            sample = values[y+margin:y1-margin, x+margin:x1-margin]
            means = sample.mean(axis=(0, 1), dtype=np.float64)
            observations.setdefault(label, []).append(means)
            records.append({"x": x, "y": y, "label": label,
                            "central_mean": means.tolist()})
    groups = {key: np.asarray(vals) for key, vals in observations.items()}
    comparisons = []
    scores = np.zeros(len(names))
    for a, b in combinations(sorted(groups), 2):
        va, vb = groups[a], groups[b]
        na, nb = len(va), len(vb)
        if min(na, nb) < min_tiles:
            continue
        df = na + nb - 2
        sd = np.maximum(np.sqrt(((na-1)*va.var(0, ddof=1) +
                                 (nb-1)*vb.var(0, ddof=1)) / df), 0.01)
        delta = va.mean(0) - vb.mean(0)
        g = (1 - 3 / (4*df - 1)) * delta / sd
        scores = np.maximum(scores, np.abs(g))
        for j, name in enumerate(names):
            comparisons.append({"channel": name, "channel_index": j,
                "group_a": a, "group_b": b, "n_a": na, "n_b": nb,
                "mean_a": float(va[:, j].mean()), "mean_b": float(vb[:, j].mean()),
                "delta_a_minus_b": float(delta[j]), "hedges_g": float(g[j])})
    eligible = [j for j, name in enumerate(names) if name not in excluded and scores[j] > 0]
    ranked = sorted(eligible, key=lambda j: (-scores[j], j))
    if not ranked:
        raise ValueError("No eligible channel contrasts: need two groups with enough tiles")
    selected = ranked[:top_k]
    return {"selected_indices": selected, "selected_channels": [names[j] for j in selected],
            "ranking": [{"channel": names[j], "index": j, "max_abs_hedges_g": float(scores[j])}
                        for j in ranked],
            "group_tile_counts": {str(k): len(v) for k, v in groups.items()},
            "excluded_channels": list(excluded), "comparisons": comparisons,
            "tiles": records, "top_k": top_k, "minimum_tiles_per_group": min_tiles,
            "reference_mask_used_for_selection": False,
            "inference": "descriptive only; tiles are spatially correlated"}


def standardize_markers(values, support):
    """Robust unit-IQR scaling; 0.05 activation floor avoids noise expansion."""
    values = np.asarray(values, np.float32)
    support = np.asarray(support, bool)
    if values.ndim != 3 or values.shape[:2] != support.shape or not support.any():
        raise ValueError("Marker scaling needs an aligned, nonempty coarse support")
    if not np.isfinite(values).all():
        raise ValueError("Marker activations must be finite")
    sample = values[support]
    median = np.median(sample, axis=0)
    q25, q75 = np.percentile(sample, [25, 75], axis=0)
    scale = np.maximum(q75-q25, 0.05)
    result = np.clip((values-median)/scale, -10, 10).astype(np.float32)
    return result, {"median": median.tolist(), "scale": scale.tolist(),
                    "scale_rule": "max(IQR over coarse foreground, 0.05)", "clip": [-10, 10]}
