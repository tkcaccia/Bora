#!/usr/bin/env python3
"""CPU-only, bounded-row projection of the training-selected ridge head."""
import hashlib
import json
import os
from pathlib import Path
import sys
from time import monotonic

os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

import joblib
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
HERE = Path(__file__).resolve().parent
OUT = DATA / 'benchmark2-supervised-ridge-2026-11-12-v3/wsi-scores'
CAMPAIGN = ROOT / 'migration/benchmark2-unified-wsi-panels-2026-11-11-v17/CAMPAIGN_VERIFIED.json'
MODEL = ROOT / 'migration/benchmark1-eight-class-head-search-2026-11-12-v1/FINAL_MODEL_FROZEN.json'
ONTOLOGY = ROOT / 'migration/benchmark-label-reduction-2026-11-11-v1/ONTOLOGY_FROZEN.json'
PRESENCE = ROOT / 'migration/benchmark2-supervised-ridge-2026-11-12-v2/CLASS_PRESENCE_FROZEN.json'
PRESENCE_AUDIT = ROOT / 'migration/benchmark2-supervised-ridge-2026-11-12-v2/PRESENCE_AUDIT_VERIFIED.json'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def source_for_slug(slug):
    base = DATA / 'feature-cache/Lion-Data-Bora/analysis'
    candidates = list(base.glob(f'full1280-public-10x-*/{slug}/full-channel-arrays/virchow2-full1280-factor-1.npy'))
    candidates += list(DATA.glob(f'benchmark2-unified-wsi-panels-2026-11-11-v11/assembled-features/{slug}/features-full1280.npy'))
    if len(candidates) != 1:
        raise ValueError(f'Expected one full-channel cache for {slug}; found {len(candidates)}')
    path = candidates[0]
    if path.name == 'features-full1280.npy':
        receipt = path.with_name('ASSEMBLED_FEATURES_VERIFIED.json')
        meta = json.loads(receipt.read_text())
        expected = meta['feature_sha256']
    else:
        receipt = path.parent.parent / 'CACHE_COMPLETE.json'
        meta = json.loads(receipt.read_text())
        expected = meta['features']['1']['sha256']
    assert sha(path) == expected, slug
    return path, receipt, expected


def project(slug, model, model_receipt):
    if 'pm016' in slug.lower():
        raise ValueError('PM016 is excluded')
    target = OUT / slug
    if target.exists():
        receipt = target / 'RIDGE_SCORE_GRID_VERIFIED.json'
        if receipt.exists():
            data = json.loads(receipt.read_text())
            if (data['status'] == 'VERIFIED_EIGHT_CLASS_RIDGE_CPU_SCORE_GRID'
                    and sha(target / 'smoothed-scores.npy') == data['smoothed_scores_sha256']
                    and sha(target / 'raw-scores.npy') == data['raw_scores_sha256']):
                print('REUSED_VERIFIED', slug, flush=True)
                return
        raise FileExistsError('Partial or failed slide preserved: ' + str(target))
    feature_path, source_receipt, feature_hash = source_for_slug(slug)
    grid = np.load(feature_path, mmap_mode='r', allow_pickle=False)
    if grid.ndim != 3 or grid.shape[-1] != 1280 or grid.dtype != np.float16:
        raise ValueError('Unexpected full-channel grid shape or dtype: ' + slug)
    selected = model['selected']
    if not np.array_equal(model['classifier'].classes_, np.arange(1, 9)):
        raise ValueError('Eight-class model identities differ')
    weights = model['classifier'].coef_.T / model['std'][:, None]
    offsets = model['classifier'].intercept_ - (model['mean'] / model['std']) @ model['classifier'].coef_.T
    target.mkdir(parents=True)
    state = target / 'STATE.json'
    state.write_text(json.dumps({'status': 'RUNNING', 'rows_complete': 0}) + '\n')
    started = monotonic()
    try:
        height, width = grid.shape[:2]
        finite = np.empty((height, width), bool)
        for y in range(height):
            finite[y] = np.isfinite(np.asarray(grid[y])[:, selected]).all(axis=1)
        if not finite.any():
            raise ValueError('No finite Virchow2 tokens')
        nearest = None if finite.all() else distance_transform_edt(~finite, return_indices=True)[1]
        temp = target / 'raw-scores.partial.npy'
        raw = np.lib.format.open_memmap(temp, mode='w+', dtype=np.float32, shape=(8, height, width))
        row_step = max(1, min(16, 32768 // width))
        for y0 in range(0, height, row_step):
            y1 = min(height, y0 + row_step)
            if nearest is None:
                block = np.asarray(grid[y0:y1, :, selected], np.float32)
            else:
                block = np.asarray(grid[nearest[0, y0:y1], nearest[1, y0:y1]][:, :, selected], np.float32)
            scores = block.reshape(-1, len(selected)).astype(np.float64) @ weights + offsets
            raw[:, y0:y1] = scores.reshape(y1 - y0, width, 8).transpose(2, 0, 1)
            raw.flush()
            state.write_text(json.dumps({'status': 'RUNNING', 'rows_complete': y1,
                                         'rows_total': height}) + '\n')
        del raw
        os.replace(temp, target / 'raw-scores.npy')
        source = np.load(target / 'raw-scores.npy', mmap_mode='r')
        smooth_temp = target / 'smoothed-scores.partial.npy'
        smooth = np.lib.format.open_memmap(smooth_temp, mode='w+', dtype=np.float32, shape=(8, height, width))
        for cls in range(8):
            smooth[cls] = gaussian_filter(np.asarray(source[cls]), sigma=.75, mode='nearest')
            smooth.flush()
        del smooth
        os.replace(smooth_temp, target / 'smoothed-scores.npy')
        result = {'status': 'VERIFIED_EIGHT_CLASS_RIDGE_CPU_SCORE_GRID', 'slug': slug,
                  'source_feature_path': str(feature_path), 'source_feature_sha256': feature_hash,
                  'source_receipt_sha256': sha(source_receipt),
                  'ontology_sha256': sha(ONTOLOGY), 'model_receipt_sha256': sha(MODEL),
                  'presence_receipt_sha256': sha(PRESENCE),
                  'presence_audit_sha256': sha(PRESENCE_AUDIT),
                  'model_sha256': model_receipt['model_sha256'],
                  'score_shape_class_yx': [8, height, width], 'dtype': 'float32',
                  'finite_token_count': int(finite.sum()),
                  'nearest_imputed_token_count': int((~finite).sum()),
                  'raw_scores_sha256': sha(target / 'raw-scores.npy'),
                  'smoothed_scores_sha256': sha(target / 'smoothed-scores.npy'),
                  'elapsed_seconds': monotonic() - started,
                  'source_code_sha256': sha(Path(__file__)),
                  'gpu_used': False, 'whole_slide_accuracy': None, 'pm016_included': False}
        receipt = target / 'RIDGE_SCORE_GRID_VERIFIED.json'
        receipt.write_text(json.dumps(result, indent=2) + '\n')
        state.write_text(json.dumps({'status': 'COMPLETE', 'receipt_sha256': sha(receipt)}) + '\n')
        print('PROJECTED', slug, height, width, round(result['elapsed_seconds'], 1), flush=True)
    except BaseException as exc:
        state.write_text(json.dumps({'status': 'FAILED_PRESERVE_PARTIAL', 'error': repr(exc)}) + '\n')
        raise


def main():
    campaign = json.loads(CAMPAIGN.read_text())
    assert campaign['status'] == 'THIRTEEN_PUBLIC_WSI_PAGES_SEVEN_PANELS_TEN_VIEWS_VERIFIED'
    assert campaign['pm016_included'] is False
    model_receipt = json.loads(MODEL.read_text())
    assert model_receipt['ontology_sha256'] == sha(ONTOLOGY)
    assert model_receipt['training_images'] == 397 and model_receipt['validation_images_opened'] == 0
    assert model_receipt['selected'] == 'class32_ridge'
    presence = json.loads(PRESENCE.read_text())
    audit = json.loads(PRESENCE_AUDIT.read_text())
    assert presence['status'] == 'TRAINING_ONLY_RIDGE_CLASS_PRESENCE_RULE_FROZEN'
    assert presence['development_masks_opened'] == presence['whole_slide_expert_masks_opened'] == 0
    assert audit['frozen_sha256'] == sha(PRESENCE)
    assert sha(model_receipt['model_path']) == model_receipt['model_sha256']
    model = joblib.load(model_receipt['model_path'])
    slugs = [r['slug'] for r in campaign['pages']]
    if len(sys.argv) == 2:
        if sys.argv[1] not in slugs:
            raise ValueError('Not a public campaign slug')
        slugs = [sys.argv[1]]
    elif len(sys.argv) != 1:
        raise ValueError('One optional public slug')
    controller_state = HERE / 'WSI_PROJECTION_STATE.json'
    controller_state.write_text(json.dumps({'status': 'RUNNING', 'total': len(slugs), 'complete': 0}) + '\n')
    try:
        for i, slug in enumerate(slugs, 1):
            project(slug, model, model_receipt)
            controller_state.write_text(json.dumps({'status': 'RUNNING', 'total': len(slugs),
                                                     'complete': i, 'last': slug}) + '\n')
        controller_state.write_text(json.dumps({'status': 'COMPLETE', 'total': len(slugs),
                                                 'complete': len(slugs)}) + '\n')
    except BaseException as exc:
        controller_state.write_text(json.dumps({'status': 'FAILED_LATCHED', 'total': len(slugs),
                                                 'error': repr(exc)}) + '\n')
        raise


if __name__ == '__main__':
    main()
