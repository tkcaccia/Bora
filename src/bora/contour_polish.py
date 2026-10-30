"""Slide-wide, physical-scale smoothing of a categorical Bora tissue mask.

This produces a new raster candidate. Run the whole-slide minimum-area repair
afterwards, then vectorize and audit the final raster and native GeoJSON.
The reviewed tissue support is authoritative and is never expanded.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


def smooth_block(labels: np.ndarray, support: np.ndarray,
                 sigma_xy_pixels: tuple[float, float]) -> np.ndarray:
    """Compete Gaussian-smoothed class indicators inside immutable support."""
    labels = np.asarray(labels, np.uint8)
    support = np.asarray(support, bool)
    if labels.ndim != 2 or labels.shape != support.shape:
        raise ValueError('Labels and reviewed support must share one 2D frame')
    if labels.max() > 32 or any(not math.isfinite(v) or v <= 0 for v in sigma_xy_pixels):
        raise ValueError('Use 1..32 classes and positive finite physical smoothing widths')
    if np.any((labels > 0) & ~support):
        raise ValueError('The class mask extends outside reviewed tissue support')
    active = support & (labels > 0)
    groups = [int(value) for value in np.unique(labels) if value > 0]
    if not groups:
        return np.zeros(labels.shape, np.uint8)
    best = np.full(labels.shape, -1.0, np.float32)
    result = np.zeros(labels.shape, np.uint8)
    for group in groups:
        response = cv2.GaussianBlur((labels == group).astype(np.float32), (0, 0),
                                    sigmaX=sigma_xy_pixels[0], sigmaY=sigma_xy_pixels[1],
                                    borderType=cv2.BORDER_REFLECT_101)
        winner = response > best
        result[winner] = group
        best[winner] = response[winner]
    if np.any(active & (best <= 0)):
        raise ValueError('Smoothing has no class evidence for assigned tissue pixels')
    # The previous global cleanup may leave zero-class holes within the
    # reviewed tissue area. Those are fixed support boundaries, not a new
    # class to be blurred into neighboring tissue.
    result[~active] = 0
    return result


def smooth_global(labels: np.ndarray, support: np.ndarray, mpp_xy: tuple[float, float],
                  output: Path, *, sigma_um: float = 6.0, block_size: int = 1024) -> dict:
    """Stream a global candidate using overlap wide enough for exact block seams."""
    labels = np.asarray(labels)
    support = np.asarray(support)
    if (labels.ndim != 2 or labels.shape != support.shape or labels.dtype != np.uint8 or
            support.dtype not in (np.dtype('uint8'), np.dtype('bool'))):
        raise ValueError('Expected matching uint8 categorical mask and reviewed support')
    if (len(mpp_xy) != 2 or not all(math.isfinite(float(v)) and v > 0 for v in mpp_xy) or
            not math.isfinite(sigma_um) or sigma_um <= 0 or block_size < 128):
        raise ValueError('Measured X/Y pixel spacing and positive smoothing width required')
    if output.exists():
        raise FileExistsError(output)
    sigma = (sigma_um / float(mpp_xy[0]), sigma_um / float(mpp_xy[1]))
    halo = max(16, math.ceil(4 * max(sigma)))
    height, width = labels.shape
    result = np.lib.format.open_memmap(output, mode='w+', dtype=np.uint8,
                                       shape=(height, width))
    changed = supported = 0
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            y1, x1 = min(height, y + block_size), min(width, x + block_size)
            ya, xa = max(0, y - halo), max(0, x - halo)
            yb, xb = min(height, y1 + halo), min(width, x1 + halo)
            source = np.asarray(labels[ya:yb, xa:xb], np.uint8)
            inside = np.asarray(support[ya:yb, xa:xb] > 0)
            candidate = smooth_block(source, inside, sigma)
            core = candidate[y-ya:y1-ya, x-xa:x1-xa]
            old = source[y-ya:y1-ya, x-xa:x1-xa]
            core_support = inside[y-ya:y1-ya, x-xa:x1-xa] & (old > 0)
            if not np.array_equal(core > 0, old > 0):
                raise AssertionError('Contour smoothing changed final class support')
            changed += int(np.count_nonzero((core != old) & core_support))
            supported += int(core_support.sum())
            result[y:y1, x:x1] = core
        result.flush()
    checks = []
    for x, y in ((block_size, block_size), (2 * block_size, block_size),
                 (block_size, 2 * block_size)):
        if x + 64 >= width or y + 64 >= height:
            continue
        xa, ya = max(0, x - 64 - halo), max(0, y - 64 - halo)
        xb, yb = min(width, x + 64 + halo), min(height, y + 64 + halo)
        predicted = smooth_block(np.asarray(labels[ya:yb, xa:xb], np.uint8),
                                 np.asarray(support[ya:yb, xa:xb] > 0), sigma)
        core = predicted[y-64-ya:y+64-ya, x-64-xa:x+64-xa]
        mismatch = int(np.count_nonzero(core != result[y-64:y+64, x-64:x+64]))
        checks.append({'center_xy': [x, y], 'different_pixels': mismatch})
        if mismatch:
            raise AssertionError('Contour smoothing block seam differs')
    return {'sigma_um': sigma_um, 'sigma_xy_pixels': list(sigma),
            'halo_pixels': halo, 'block_size': block_size,
            'changed_supported_pixels': changed, 'supported_pixels': supported,
            'changed_supported_fraction': changed / max(1, supported),
            'exact_block_seam_checks': checks,
            'support_policy': 'reviewed tissue GeoJSON is an upper bound; exact existing positive/zero support is immutable'}
