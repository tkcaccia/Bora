"""Geometry and optional semantic scores for registered tissue label rasters.

The geometry endpoints deliberately ignore the numerical tissue names.  They
score internal partition interfaces and connected-region overlap on annotated
pixels only.  A full WSI must be split into prespecified annotated windows;
the CLI refuses unbounded arrays and never resizes or harmonizes masks.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, label


# Covers every frozen Benchmark 1 ROI (largest: 20,267,382 pixels) while
# rejecting unbounded whole-slide inputs. Process one ROI at a time.
MAX_PIXELS = 24_000_000
CONNECTIVITY = np.ones((3, 3), dtype=np.uint8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_labels(path: Path) -> np.ndarray:
    if path.suffix.lower() == '.npy':
        array = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as image:
            array = np.asarray(image)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f'Expected 2D integer label raster: {path}')
    if array.size > MAX_PIXELS:
        raise ValueError('Array exceeds bounded ROI/window scorer; select registered annotated windows')
    return array


def internal_interfaces(classes: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Mark both sides of each four-neighbor label transition inside valid pixels."""
    edge = np.zeros(classes.shape, dtype=bool)
    horizontal = valid[:, :-1] & valid[:, 1:] & (classes[:, :-1] != classes[:, 1:])
    vertical = valid[:-1, :] & valid[1:, :] & (classes[:-1, :] != classes[1:, :])
    edge[:, :-1] |= horizontal
    edge[:, 1:] |= horizontal
    edge[:-1, :] |= vertical
    edge[1:, :] |= vertical
    return edge


def boundary_score(reference: np.ndarray, prediction: np.ndarray, valid: np.ndarray,
                   spacing_yx_um: tuple[float, float], tolerance_um: float) -> dict:
    ref = internal_interfaces(reference, valid)
    pred = internal_interfaces(prediction, valid)
    nr, npred = int(ref.sum()), int(pred.sum())
    result = {'reference_interface_pixels': nr, 'predicted_interface_pixels': npred,
              'tolerance_um': tolerance_um}
    if nr == 0 and npred == 0:
        return {**result, 'boundary_f1': None, 'boundary_precision': None,
                'boundary_recall': None, 'symmetric_hd95_um': None,
                'reason': 'Neither partition contains an internal annotated interface'}
    if nr == 0 or npred == 0:
        return {**result, 'boundary_f1': 0.0, 'boundary_precision': 0.0,
                'boundary_recall': 0.0, 'symmetric_hd95_um': None,
                'reason': 'One partition has no internal annotated interface'}
    distance_to_ref = distance_transform_edt(~ref, sampling=spacing_yx_um)[pred]
    distance_to_pred = distance_transform_edt(~pred, sampling=spacing_yx_um)[ref]
    precision = float(np.mean(distance_to_ref <= tolerance_um))
    recall = float(np.mean(distance_to_pred <= tolerance_um))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {**result, 'boundary_f1': f1, 'boundary_precision': precision,
            'boundary_recall': recall,
            'symmetric_hd95_um': float(max(np.percentile(distance_to_ref, 95),
                                           np.percentile(distance_to_pred, 95))),
            'reason': None}


def components(classes: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eight-connected regions for every distinct label, without assigning meaning to it."""
    result = np.zeros(classes.shape, dtype=np.int32)
    offset = 0
    for value in np.unique(classes[valid]):
        parts, count = label(valid & (classes == value), structure=CONNECTIVITY)
        if count:
            present = parts != 0
            result[present] = parts[present] + offset
            offset += int(count)
    return result, np.bincount(result[valid], minlength=offset + 1)


def region_covering(reference: np.ndarray, prediction: np.ndarray, valid: np.ndarray) -> dict:
    ref, ref_sizes = components(reference, valid)
    pred, pred_sizes = components(prediction, valid)
    if len(ref_sizes) <= 1 or len(pred_sizes) <= 1:
        raise ValueError('No annotated connected regions')
    best_ref = np.zeros(len(ref_sizes), dtype=np.float64)
    best_pred = np.zeros(len(pred_sizes), dtype=np.float64)
    # Count co-occurrences in bounded strips to avoid a whole-image pair matrix.
    pairs: dict[tuple[int, int], int] = {}
    for top in range(0, ref.shape[0], 128):
        rv = ref[top:top + 128][valid[top:top + 128]].astype(np.uint64)
        pv = pred[top:top + 128][valid[top:top + 128]].astype(np.uint64)
        keys, counts = np.unique((rv << np.uint64(32)) | pv, return_counts=True)
        for key, count in zip(keys, counts):
            pair = (int(key >> np.uint64(32)), int(key & np.uint64(0xffffffff)))
            pairs[pair] = pairs.get(pair, 0) + int(count)
    for (i, j), intersection in pairs.items():
        union = int(ref_sizes[i]) + int(pred_sizes[j]) - intersection
        overlap = intersection / union
        best_ref[i] = max(best_ref[i], overlap)
        best_pred[j] = max(best_pred[j], overlap)
    ref_cover = float(np.dot(ref_sizes[1:], best_ref[1:]) / ref_sizes[1:].sum())
    pred_cover = float(np.dot(pred_sizes[1:], best_pred[1:]) / pred_sizes[1:].sum())
    return {'reference_regions': len(ref_sizes) - 1, 'predicted_regions': len(pred_sizes) - 1,
            'reference_weighted_covering': ref_cover,
            'prediction_weighted_covering': pred_cover,
            'symmetric_region_covering': (ref_cover + pred_cover) / 2}


def semantic_score(reference: np.ndarray, prediction: np.ndarray, valid: np.ndarray) -> dict:
    rows = []
    for class_id in np.unique(reference[valid]):
        actual = valid & (reference == class_id)
        guessed = valid & (prediction == class_id)
        tp = int((actual & guessed).sum())
        n_true, n_pred = int(actual.sum()), int(guessed.sum())
        union = n_true + n_pred - tp
        rows.append({'class_id': int(class_id), 'reference_pixels': n_true,
                     'predicted_pixels': n_pred, 'iou': tp / union,
                     'dice': 2 * tp / (n_true + n_pred)})
    return {'expert_present_macro_iou': float(np.mean([r['iou'] for r in rows])),
            'expert_present_macro_dice': float(np.mean([r['dice'] for r in rows])),
            'pixel_accuracy': float(np.mean(reference[valid] == prediction[valid])),
            'predicted_only_ids': [int(v) for v in np.setdiff1d(np.unique(prediction[valid]),
                                                                 np.unique(reference[valid]))],
            'per_reference_class': rows}


def score(reference: np.ndarray, prediction: np.ndarray, ignore_id: int,
          spacing_yx_um: tuple[float, float], tolerance_um: float,
          aligned_semantic_ids: bool = False) -> dict:
    if reference.ndim != 2 or prediction.shape != reference.shape:
        raise ValueError('Reference and prediction must have identical 2D shape; no resizing is permitted')
    if not np.issubdtype(reference.dtype, np.integer) or not np.issubdtype(prediction.dtype, np.integer):
        raise TypeError('Integer labels required')
    if reference.size > MAX_PIXELS:
        raise ValueError('Score prespecified annotated windows, not a full WSI in RAM')
    if min(*spacing_yx_um, tolerance_um) <= 0:
        raise ValueError('Measured spacing and fixed tolerance must be positive')
    valid = reference != ignore_id
    if not np.any(valid):
        raise ValueError('No annotated reference pixels')
    result = {'shape_yx': list(reference.shape), 'ignore_id': ignore_id,
              'annotated_pixels': int(valid.sum()), 'annotated_fraction': float(valid.mean()),
              'spacing_yx_um': list(spacing_yx_um),
              'geometry': {**boundary_score(reference, prediction, valid, spacing_yx_um, tolerance_um),
                           **region_covering(reference, prediction, valid)}}
    if aligned_semantic_ids:
        result['semantic'] = semantic_score(reference, prediction, valid)
    return result
