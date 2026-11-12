#!/usr/bin/env python3
"""Independently check all public WSI score grids against the frozen classifier."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
HERE = Path(__file__).resolve().parent
MODEL = ROOT / 'migration/benchmark1-eight-class-head-search-2026-11-12-v1/FINAL_MODEL_FROZEN.json'
CAMPAIGN = ROOT / 'migration/benchmark2-unified-wsi-panels-2026-11-11-v17/CAMPAIGN_VERIFIED.json'
PRESENCE = ROOT / 'migration/benchmark2-supervised-ridge-2026-11-12-v2/CLASS_PRESENCE_FROZEN.json'
PRESENCE_AUDIT = ROOT / 'migration/benchmark2-supervised-ridge-2026-11-12-v2/PRESENCE_AUDIT_VERIFIED.json'
OUT = DATA / 'benchmark2-supervised-ridge-2026-11-12-v3/wsi-scores'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    target = HERE / 'WSI_PROJECTION_AUDIT.json'
    if target.exists():
        raise FileExistsError(target)
    frozen = json.loads(MODEL.read_text())
    assert sha(frozen['model_path']) == frozen['model_sha256']
    model = joblib.load(frozen['model_path'])
    slugs = [r['slug'] for r in json.loads(CAMPAIGN.read_text())['pages']]
    assert len(slugs) == 13 and all('pm016' not in slug.lower() for slug in slugs)
    rows = []
    rng = np.random.default_rng(20261111)
    for slug in slugs:
        folder = OUT / slug
        receipt_path = folder / 'RIDGE_SCORE_GRID_VERIFIED.json'
        r = json.loads(receipt_path.read_text())
        assert r['status'] == 'VERIFIED_EIGHT_CLASS_RIDGE_CPU_SCORE_GRID' and r['slug'] == slug
        assert r['model_receipt_sha256'] == sha(MODEL)
        assert r['presence_receipt_sha256'] == sha(PRESENCE)
        assert r['presence_audit_sha256'] == sha(PRESENCE_AUDIT)
        for name, key in [('raw-scores.npy', 'raw_scores_sha256'),
                          ('smoothed-scores.npy', 'smoothed_scores_sha256')]:
            assert sha(folder / name) == r[key], slug + ' ' + name
        raw = np.load(folder / 'raw-scores.npy', mmap_mode='r')
        smooth = np.load(folder / 'smoothed-scores.npy', mmap_mode='r')
        features = np.load(r['source_feature_path'], mmap_mode='r')
        assert raw.shape == smooth.shape == tuple(r['score_shape_class_yx'])
        assert features.shape[:2] == raw.shape[1:] and features.shape[2] == 1280
        checked = 0
        maximum_difference = 0.0
        for _ in range(200):
            if checked == 32:
                break
            y = int(rng.integers(features.shape[0]))
            x = int(rng.integers(features.shape[1]))
            vector = np.asarray(features[y, x, model['selected']], np.float32)
            if not np.isfinite(vector).all():
                continue
            expected = model['classifier'].decision_function(
                ((vector - model['mean']) / model['std'])[None, :])[0]
            observed = np.asarray(raw[:, y, x])
            delta = float(np.max(np.abs(expected - observed)))
            assert np.allclose(expected, observed, atol=2e-4, rtol=2e-5), (slug, delta)
            assert np.isfinite(smooth[:, y, x]).all()
            maximum_difference = max(maximum_difference, delta)
            checked += 1
        if checked < 16:
            raise ValueError('Too few independently checked finite tokens: ' + slug)
        rows.append({'slug': slug, 'raw_sha256': r['raw_scores_sha256'],
                     'smoothed_sha256': r['smoothed_scores_sha256'],
                     'receipt_sha256': sha(receipt_path), 'sampled_finite_tokens': checked,
                     'max_abs_logit_difference': maximum_difference})
        print('AUDITED', len(rows), '/13', slug, flush=True)
    result = {'status': 'THIRTEEN_PUBLIC_EIGHT_CLASS_RIDGE_SCORE_GRIDS_INDEPENDENTLY_VERIFIED',
              'model_receipt_sha256': sha(MODEL), 'campaign_sha256': sha(CAMPAIGN),
              'slugs': rows, 'pm016_included': False, 'whole_slide_expert_accuracy': None,
              'source_sha256': sha(Path(__file__))}
    target.write_text(json.dumps(result, indent=2) + '\n')
    print('COMPLETE', len(rows))


if __name__ == '__main__':
    main()
