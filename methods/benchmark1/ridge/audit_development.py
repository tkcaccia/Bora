#!/usr/bin/env python3
"""Independent image/hash/score audit of the one-time development evaluation."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
HERE = Path(__file__).resolve().parent
RAW = HERE / 'DEVELOPMENT_EVALUATION_COMPLETE.json'
POLISHED = HERE / 'DEVELOPMENT_CONTOUR_COMPLETE.json'
OLD = ROOT / 'migration/benchmark-label-reduction-2026-11-11-v3/SUPERVISED_CONTOUR_COMPLETE.json'
FROZEN = ROOT / 'migration/benchmark1-supervised-virchow2-response-2026-11-10-v1/INPUTS_FROZEN.json'
ONTOLOGY = ROOT / 'migration/benchmark-label-reduction-2026-11-11-v1/ONTOLOGY_FROZEN.json'
HARM = ROOT / 'benchmark/methods/benchmark1/tissue_label_harmonization.json'
BASE = Path('/mnt/sata_ssd/ChatGPT/Bora-Data/benchmark1-supervised-mask-overlay-score-2026-11-10-v1/SCORES_AND_FIGURES.json')
OUT = HERE / 'DEVELOPMENT_AUDIT_VERIFIED.json'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def iou(pred, truth):
    valid = truth > 0
    scores = []
    per = {}
    for label in np.unique(truth[valid]):
        tp = np.count_nonzero(valid & (pred == label) & (truth == label))
        union = np.count_nonzero(valid & ((pred == label) | (truth == label)))
        value = float(tp / union)
        per[str(int(label))] = value
        scores.append(value)
    return float(np.mean(scores)), per


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    raw, polished, old, frozen, ontology, harmony, baseline = [json.loads(p.read_text()) for p in
        (RAW, POLISHED, OLD, FROZEN, ONTOLOGY, HARM, BASE)]
    assert raw['status'] == 'EIGHT_CLASS_87_CASE_DEVELOPMENT_EVALUATION_COMPLETE'
    assert polished['status'] == 'EIGHT_CLASS_87_CASE_SUPERVISED_CONTOURS_POLISHED'
    assert raw['validation_cases_used_for_selection'] == 0 and not raw['is_independent_test']
    assert polished['source_validation_sha256'] == sha(RAW)
    assert len(raw['cases']) == len(polished['cases']) == len(old['cases']) == len(baseline['cases']) == 87
    roles = frozen['classes']
    role_to_new = {role: ontology['old_to_new'][old_id] for old_id, role in roles.items()}
    harmonized = harmony['mapping']
    before_rows = {r['id']: r for r in raw['cases']}
    old_rows = {r['id']: r for r in old['cases']}
    references = {r['id']: r for r in baseline['cases']}
    rows = []
    for row in polished['cases']:
        cid = row['id']
        prior, reference = before_rows[cid], references[cid]
        assert row['case'] == prior['case'] == reference['case']
        assert row['dataset'] == prior['dataset'] == reference['dataset']
        assert sha(prior['prediction_path']) == prior['prediction_sha256'] == row['input_sha256']
        assert sha(row['output_path']) == row['output_sha256']
        assert sha(reference['source_mask_path']) == reference['source_mask_sha256']
        shape = tuple(reference['scoring_shape_yx'])
        with Image.open(row['output_path']) as img:
            pred = np.asarray(img.resize(shape[::-1], Image.Resampling.NEAREST), np.uint8)
        assert pred.shape == shape and np.isin(pred, np.arange(1, 9)).all()
        with Image.open(reference['source_mask_path']) as img:
            raw_expert = np.asarray(img.resize(shape[::-1], Image.Resampling.NEAREST))
        truth = np.zeros(shape, np.uint8)
        for code in np.unique(raw_expert):
            spec = harmonized[row['dataset']].get(str(int(code)))
            if spec and spec['use'] == 'score':
                label = role_to_new.get(spec.get('canonical_role'))
                if label is not None:
                    truth[raw_expert == code] = label
        value, per = iou(pred, truth)
        assert abs(value - row['after']['expert_present_macro_iou']) < 1e-10
        rows.append({'id': cid, 'dataset': row['dataset'], 'macro_iou': value,
                     'old_macro_iou': old_rows[cid]['after']['expert_present_macro_iou'],
                     'classwise_iou': per})
    mean = float(np.mean([r['macro_iou'] for r in rows]))
    by = {d: float(np.mean([r['macro_iou'] for r in rows if r['dataset'] == d]))
          for d in ('bcss', 'gcss', 'ignite', 'peso', 'tiger')}
    assert abs(mean - polished['mean_macro_iou_after']) < 1e-10
    assert abs(np.mean([by[d] for d in ('bcss', 'gcss', 'ignite', 'tiger')]) -
               polished['four_multiclass_dataset_mean_macro_iou_after']) < 1e-10
    result = {'status': 'DEVELOPMENT_87_INDEPENDENTLY_VERIFIED',
              'is_independent_test': False, 'cases': 87, 'prediction_hashes_checked': 174,
              'expert_hashes_checked': 87, 'mean_case_macro_iou': mean,
              'four_multiclass_dataset_mean_case_macro_iou': float(np.mean([by[d] for d in ('bcss', 'gcss', 'ignite', 'tiger')])),
              'per_dataset': by,
              'previous_mean_case_macro_iou': old['mean_macro_iou_after'],
              'previous_four_multiclass_dataset_mean_case_macro_iou': old['four_multiclass_dataset_mean_macro_iou_after'],
              'raw_receipt_sha256': sha(RAW), 'contour_receipt_sha256': sha(POLISHED),
              'previous_receipt_sha256': sha(OLD), 'source_sha256': sha(Path(__file__))}
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
