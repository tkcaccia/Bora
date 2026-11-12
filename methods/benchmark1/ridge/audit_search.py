#!/usr/bin/env python3
"""Independent receipt and split audit without opening development images."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
HERE = Path(__file__).resolve().parent
SAMPLE = ROOT / 'migration/benchmark1-supervised-strategy-search-2026-11-10-v1/SAMPLE_COMPLETE.json'
SPLITS = ROOT / 'migration/benchmark1-supervised-strategy-search-2026-11-10-v1/SEARCH_COMPLETE.json'
PROTOCOL = HERE / 'PROTOCOL_FROZEN.json'
RESULT = HERE / 'TRAINING_SEARCH_COMPLETE.json'
OUT = HERE / 'SEARCH_AUDIT_VERIFIED.json'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    sample, splits, proto, result = [json.loads(p.read_text()) for p in (SAMPLE, SPLITS, PROTOCOL, RESULT)]
    assert result['status'] == 'TRAINING_ONLY_HEAD_SEARCH_COMPLETE'
    assert result['validation_images_opened'] == 0
    for field, path in [('sample_sha256', SAMPLE), ('splits_sha256', SPLITS), ('protocol_sha256', PROTOCOL)]:
        assert result[field] == digest(path)
    assert result['feature_sha256'] == sample['feature_sha256'] == digest(sample['feature_path'])
    assert result['index_sha256'] == sample['index_sha256'] == digest(sample['index_path'])
    idx = np.load(sample['index_path'])
    cases = idx['case_index']
    wanted = {c['id'] for c in proto['candidates']}
    assert len(result['fold_rows']) == 2 * len(wanted)
    for number, split in enumerate(splits['folds'], 1):
        tr, te = set(split['training_case_indices']), set(split['heldout_case_indices'])
        assert not tr & te and len(tr | te) == 397
        assert not {sample['source_groups'][n] for n in tr} & {sample['source_groups'][n] for n in te}
        rows = [r for r in result['fold_rows'] if r['fold'] == number]
        assert {r['candidate'] for r in rows} == wanted
        for row in rows:
            assert row['training_cases'] == len(tr) and row['heldout_cases'] == len(te)
            assert row['training_tokens'] <= np.isin(cases, list(tr)).sum()
            assert row['heldout_tokens'] <= np.isin(cases, list(te)).sum()
            assert row['metrics']['cases'] <= len(te)
            assert 0 <= row['metrics']['four_multiclass_dataset_macro_iou'] <= 1
            assert len(row['metrics']['classwise']) == 8
            for label in range(1, 9):
                metric = row['metrics']['classwise'][str(label)]
                for field in ('iou', 'dice'):
                    assert metric[field] is None or 0 <= metric[field] <= 1
    aggregates = []
    for candidate in sorted(wanted):
        rows = [r for r in result['fold_rows'] if r['candidate'] == candidate]
        primary = np.mean([r['metrics']['four_multiclass_dataset_macro_iou'] for r in rows])
        fit = np.mean([r['fit_seconds'] for r in rows])
        inference = np.mean([r['prediction_seconds_per_10000_tokens'] for r in rows])
        channels = np.mean([r['selected_channels'] for r in rows])
        recorded = next(r for r in result['aggregate'] if r['candidate'] == candidate)
        for field, value in [('four_multiclass_dataset_macro_iou', primary), ('mean_fit_seconds', fit),
                             ('mean_prediction_seconds_per_10000_tokens', inference), ('mean_selected_channels', channels)]:
            assert np.isclose(value, recorded[field], rtol=0, atol=1e-10)
        aggregates.append((candidate, primary, fit + inference, channels))
    best = max(r[1] for r in aggregates)
    contenders = [r for r in aggregates if best - r[1] <= 0.005]
    winner = min(contenders, key=lambda r: (r[2], r[3], r[0]))
    assert winner[0] == result['selected']
    out = {'status': 'TRAINING_SEARCH_INDEPENDENTLY_VERIFIED', 'training_cases': 397,
           'training_folds': 2, 'candidates': len(wanted), 'development_images_opened': 0,
           'selected': winner[0], 'primary_cv_iou': winner[1],
           'best_raw_primary_cv_iou': best, 'training_receipt_sha256': digest(RESULT),
           'source_sha256': digest(Path(__file__))}
    OUT.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
