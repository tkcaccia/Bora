#!/usr/bin/env python3
"""Fit the audited training-selected eight-class head on frozen training tokens."""
import hashlib
import json
import os
from pathlib import Path
from time import monotonic

os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

import joblib
import numpy as np

from search import fit_head, select_columns

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
HERE = Path(__file__).resolve().parent
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data/benchmark1-eight-class-head-search-2026-11-12-v1')
SAMPLE = ROOT / 'migration/benchmark1-supervised-strategy-search-2026-11-10-v1/SAMPLE_COMPLETE.json'
ONTOLOGY = ROOT / 'migration/benchmark-label-reduction-2026-11-11-v1/ONTOLOGY_FROZEN.json'
FROZEN = ROOT / 'migration/benchmark-label-reduction-2026-11-11-v1/SUPERVISED_MODEL_FROZEN.json'
SEARCH = HERE / 'TRAINING_SEARCH_COMPLETE.json'
AUDIT = HERE / 'SEARCH_AUDIT_VERIFIED.json'
PROTOCOL = HERE / 'PROTOCOL_FROZEN.json'
RECEIPT = HERE / 'FINAL_MODEL_FROZEN.json'
STATE = HERE / 'FIT_STATE.json'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def write_state(**kwargs):
    temp = STATE.with_suffix('.json.tmp')
    temp.write_text(json.dumps(kwargs, indent=2) + '\n')
    os.replace(temp, STATE)


def main():
    if RECEIPT.exists():
        raise FileExistsError(RECEIPT)
    sample, ontology, frozen, search, audit, protocol = [json.loads(p.read_text()) for p in
        (SAMPLE, ONTOLOGY, FROZEN, SEARCH, AUDIT, PROTOCOL)]
    assert audit['training_receipt_sha256'] == sha(SEARCH)
    assert audit['status'] == 'TRAINING_SEARCH_INDEPENDENTLY_VERIFIED'
    assert search['selected'] == audit['selected'] == 'class32_ridge'
    assert sample['validation_cases_opened'] == search['validation_images_opened'] == 0
    assert sha(sample['feature_path']) == sample['feature_sha256']
    assert sha(sample['index_path']) == sample['index_sha256']
    cfg = next(c for c in protocol['candidates'] if c['id'] == search['selected'])
    idx = np.load(sample['index_path'])
    lut = np.zeros(256, np.uint8)
    for old_id, new_id in ontology['old_to_new'].items():
        if new_id is not None:
            lut[int(old_id)] = new_id
    y = lut[idx['label']]
    valid = y > 0
    x = np.asarray(np.load(sample['feature_path'], mmap_mode='r')[valid], np.float32)
    write_state(status='RUNNING', valid_training_tokens=int(valid.sum()))
    try:
        tic = monotonic()
        mean = x.mean(axis=0)
        std = np.maximum(x.std(axis=0), 1e-4)
        z = (x - mean) / std
        del x
        selected = select_columns(z, y[valid], cfg['selector'])
        head = fit_head(z, y[valid], selected, cfg)
        elapsed = monotonic() - tic
        assert list(head.classes_) == list(range(1, 9))
        assert np.isfinite(head.decision_function(np.ascontiguousarray(z[:16, selected]))).all()
        model = {'selected': selected, 'mean': mean[selected], 'std': std[selected],
                 'classifier': head, 'channels': int(len(selected)),
                 'method': cfg['id'], 'source_full_channels': 1280}
        DATA.mkdir(parents=True, exist_ok=True)
        model_path = DATA / 'supervised-eight-class-class32-ridge.joblib'
        if model_path.exists():
            raise FileExistsError(model_path)
        joblib.dump(model, model_path, compress=3)
        receipt = {'status': 'TRAINING_SELECTED_EIGHT_CLASS_MODEL_FROZEN',
                   'training_images': 397, 'validation_images_opened': 0,
                   'training_tokens_total': len(y), 'training_tokens_used': int(valid.sum()),
                   'selected': cfg['id'], 'selected_channels': len(selected),
                   'original_channel_indices': selected.tolist(),
                   'fit_seconds': elapsed, 'model_path': str(model_path),
                   'model_sha256': sha(model_path), 'sample_sha256': sha(SAMPLE),
                   'ontology_sha256': sha(ONTOLOGY), 'search_receipt_sha256': sha(SEARCH),
                   'search_audit_sha256': sha(AUDIT), 'protocol_sha256': sha(PROTOCOL),
                   'source_sha256': sha(Path(__file__)),
                   'eligible_classes_by_dataset': frozen['eligible_classes_by_dataset']}
        RECEIPT.write_text(json.dumps(receipt, indent=2) + '\n')
        write_state(status='COMPLETE', receipt_sha256=sha(RECEIPT))
        print('MODEL_FROZEN', len(selected), elapsed, flush=True)
    except BaseException as exc:
        write_state(status='FAILED', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
