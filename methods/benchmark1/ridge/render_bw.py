#!/usr/bin/env python3
"""Render eight training-selected ridge response maps for 87 development ROIs."""
import hashlib
import json
import os
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'

import joblib
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.special import softmax

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
HERE = Path(__file__).resolve().parent
OUT = DATA / 'benchmark1-eight-class-head-search-2026-11-12-v1/development-bw'
INPUT = ROOT / 'migration/benchmark1-supervised-virchow2-response-2026-11-10-v1/INPUTS_FROZEN.json'
MODEL = HERE / 'FINAL_MODEL_FROZEN.json'
ONTOLOGY = ROOT / 'migration/benchmark-label-reduction-2026-11-11-v1/ONTOLOGY_FROZEN.json'
GEOM = DATA / 'benchmark1-supervised-virchow2-response-2026-11-10-v1'
SUPPORT = Path('/tmp/bora-analysis-bw106-v2/site/benchmark1-comparators')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    output = HERE / 'DEVELOPMENT_BW_COMPLETE.json'
    if output.exists() or OUT.exists():
        raise FileExistsError('No in-place response rerun')
    fitted = json.loads(MODEL.read_text())
    assert fitted['selected'] == 'class32_ridge' and fitted['validation_images_opened'] == 0
    assert fitted['ontology_sha256'] == sha(ONTOLOGY)
    assert sha(fitted['model_path']) == fitted['model_sha256']
    model = joblib.load(fitted['model_path'])
    selected = model['selected']
    rows = {r['case']: r for r in json.loads(INPUT.read_text())['rows'] if r['split'] == 'validation'}
    cases = json.loads((SUPPORT / 'cases.json').read_text())['cases']
    assert len(rows) == len(cases) == 87
    OUT.mkdir(parents=True)
    state = HERE / 'BW_STATE.json'
    state.write_text(json.dumps({'status': 'RUNNING', 'complete': 0}) + '\n')
    results = []
    try:
        for case in cases:
            cid, source = case['id'], rows[case['case']]
            grid_path = Path(source['feature_path'])
            if sha(grid_path) != source['feature_sha256']:
                raise ValueError('Feature SHA differs: ' + cid)
            grid = np.load(grid_path, mmap_mode='r')
            finite = np.isfinite(np.asarray(grid[:, :, selected])).all(axis=2)
            if not finite.any():
                raise ValueError('No finite tokens: ' + cid)
            if finite.all():
                nearest = np.arange(grid.shape[0] * grid.shape[1])
            else:
                _, near = distance_transform_edt(~finite, return_indices=True)
                nearest = np.ravel_multi_index(near, grid.shape[:2]).ravel()
            flat = grid.reshape(-1, 1280)
            x = np.asarray(flat[nearest[:, None], selected[None, :]], np.float32)
            x = (x - model['mean']) / model['std']
            logits = model['classifier'].decision_function(np.ascontiguousarray(x))
            assert logits.shape == (grid.shape[0] * grid.shape[1], 8)
            logits = logits.reshape(*grid.shape[:2], 8)
            for k in range(8):
                logits[:, :, k] = gaussian_filter(logits[:, :, k], sigma=.75, mode='nearest')
            response = softmax(logits, axis=2)
            geom = json.loads((GEOM / cid / 'RESPONSE_COMPLETE.json').read_text())
            work_wh = tuple(geom['working_size_wh'])
            display_wh = tuple(geom['combined_mask']['display_size_wh'])
            support_path = SUPPORT / case['he_background_support']['image']
            if sha(support_path) != case['he_background_support']['sha256']:
                raise ValueError('Support SHA differs: ' + cid)
            with Image.open(support_path) as im:
                support = np.asarray(im.getchannel('A').resize(display_wh, Image.Resampling.NEAREST)) > 0
            folder = OUT / cid
            folder.mkdir()
            assets = []
            for k in range(8):
                full = Image.fromarray(response[:, :, k].astype(np.float32), 'F').resize(
                    (grid.shape[1] * 14, grid.shape[0] * 14), Image.Resampling.BICUBIC)
                displayed = np.asarray(full.crop((0, 0, *work_wh)).resize(display_wh, Image.Resampling.BICUBIC), np.float32)
                bw = np.rint(np.clip(displayed, 0, 1) * 255).astype(np.uint8)
                bw[~support] = 0
                path = folder / f'class-{k+1:02}.webp'
                Image.fromarray(bw, 'L').save(path, 'WEBP', quality=90, method=6)
                assets.append({'class_id': k + 1, 'sha256': sha(path), 'bytes': path.stat().st_size})
            results.append({'id': cid, 'case': case['case'], 'feature_sha256': source['feature_sha256'],
                            'support_sha256': case['he_background_support']['sha256'],
                            'display_size_wh': list(display_wh), 'assets': assets})
            state.write_text(json.dumps({'status': 'RUNNING', 'complete': len(results), 'last': cid}) + '\n')
            print('BW', len(results), '/87', cid, flush=True)
        receipt = {'status': 'EIGHT_CLASS_VIRCHOW2_BW_RESPONSE_MAPS_COMPLETE',
                   'case_count': 87, 'map_count': 696, 'model_sha256': fitted['model_sha256'],
                   'ontology_sha256': sha(ONTOLOGY), 'source_code_sha256': sha(Path(__file__)),
                   'response_definition': 'softmax of eight class logits after 0.75-token Gaussian smoothing; display resampling and shared tissue support; not calibrated probabilities',
                   'cases': results, 'pm016_included': False, 'beetle_included': False}
        output.write_text(json.dumps(receipt, indent=2) + '\n')
        state.write_text(json.dumps({'status': 'COMPLETE', 'complete': 87,
                                     'receipt_sha256': sha(output)}) + '\n')
    except BaseException as exc:
        state.write_text(json.dumps({'status': 'FAILED_PRESERVE_PARTIAL',
                                     'complete': len(results), 'error': repr(exc)}) + '\n')
        raise


if __name__ == '__main__':
    main()
