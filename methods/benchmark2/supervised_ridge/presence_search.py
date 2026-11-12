#!/usr/bin/env python3
"""Training-only class-presence search for the selected ridge WSI head."""
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

import numpy as np

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
HERE = Path(__file__).resolve().parent
HEAD = ROOT / 'migration/benchmark1-eight-class-head-search-2026-11-12-v1'
sys.path.insert(0, str(HEAD))
from search import select_columns, fit_head

SAMPLE = ROOT / 'migration/benchmark1-supervised-strategy-search-2026-11-10-v1/SAMPLE_COMPLETE.json'
SPLITS = ROOT / 'migration/benchmark1-supervised-strategy-search-2026-11-10-v1/SEARCH_COMPLETE.json'
ONTOLOGY = ROOT / 'migration/benchmark-label-reduction-2026-11-11-v1/ONTOLOGY_FROZEN.json'
MODEL = HEAD / 'FINAL_MODEL_FROZEN.json'
PROTOCOL = HEAD / 'PROTOCOL_FROZEN.json'
THRESHOLDS = (0.0, .005, .01, .02, .03, .05)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(4 << 20):
            h.update(block)
    return h.hexdigest()


def atomic(path, data):
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(tmp, path)


def gated(scores, floor):
    winners = np.argmax(scores, axis=1)
    fractions = np.bincount(winners, minlength=8) / len(winners)
    kept = fractions >= floor
    if not kept.any():
        kept[int(np.argmax(fractions))] = True
    return np.argmax(np.where(kept[None, :], scores, -np.inf), axis=1) + 1


def case_iou(pred, truth):
    return float(np.mean([
        np.count_nonzero((pred == cls) & (truth == cls)) /
        np.count_nonzero((pred == cls) | (truth == cls))
        for cls in np.unique(truth)]))


def main():
    receipt = HERE / 'TRAINING_PRESENCE_SEARCH.json'
    frozen_path = HERE / 'CLASS_PRESENCE_FROZEN.json'
    if receipt.exists() or frozen_path.exists():
        raise FileExistsError('No in-place training search rerun')
    sample = json.loads(SAMPLE.read_text())
    splits = json.loads(SPLITS.read_text())
    ontology = json.loads(ONTOLOGY.read_text())
    model = json.loads(MODEL.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    assert sample['training_cases'] == 397 and sample['validation_cases_opened'] == 0
    assert splits['validation_cases_opened'] == model['validation_images_opened'] == 0
    assert model['selected'] == 'class32_ridge'
    assert sha(sample['feature_path']) == sample['feature_sha256']
    assert sha(sample['index_path']) == sample['index_sha256']
    assert sha(MODEL) and sha(PROTOCOL) and sha(ONTOLOGY)
    cfg, = [c for c in protocol['candidates'] if c['id'] == model['selected']]
    x = np.load(sample['feature_path'], mmap_mode='r')
    idx = np.load(sample['index_path'])
    old_y, cases, datasets = (idx[k] for k in ('label', 'case_index', 'dataset_id'))
    lut = np.zeros(256, np.uint8)
    for old, new in ontology['old_to_new'].items():
        if new is not None:
            lut[int(old)] = int(new)
    y = lut[old_y]
    valid = y > 0
    rows = []
    state = HERE / 'STATE.json'
    atomic(state, {'status': 'RUNNING', 'folds_complete': 0})
    try:
        for fold_no, fold in enumerate(splits['folds'], 1):
            train_cases, held_cases = set(fold['training_case_indices']), set(fold['heldout_case_indices'])
            assert not train_cases & held_cases
            assert not ({sample['source_groups'][c] for c in train_cases} &
                        {sample['source_groups'][c] for c in held_cases})
            train = valid & np.isin(cases, list(train_cases))
            held = valid & np.isin(cases, list(held_cases))
            xtr = np.asarray(x[train], np.float32)
            mean = xtr.mean(axis=0)
            std = np.maximum(xtr.std(axis=0), 1e-4)
            ztr = (xtr - mean) / std
            columns = select_columns(ztr, y[train], cfg['selector'])
            classifier = fit_head(ztr, y[train], columns, cfg)
            assert np.array_equal(classifier.classes_, np.arange(1, 9))
            xte = np.asarray(x[held][:, columns], np.float32)
            scores = classifier.decision_function((xte - mean[columns]) / std[columns])
            held_cases_arr, held_datasets, held_truth = cases[held], datasets[held], y[held]
            for did, name in enumerate(sample['dataset_names']):
                subset = held_datasets == did
                if subset.any():
                    allowed = np.isin(np.arange(1, 9), model['eligible_classes_by_dataset'][name])
                    assert allowed.any()
                    scores[np.ix_(subset, ~allowed)] = -np.inf
            for case in np.unique(held_cases_arr):
                subset = held_cases_arr == case
                s, truth = scores[subset], held_truth[subset]
                dataset = sample['dataset_names'][int(held_datasets[subset][0])]
                variants = {}
                for floor in THRESHOLDS:
                    pred = gated(s, floor)
                    variants[str(floor)] = {
                        'macro_iou': case_iou(pred, truth),
                        'pixel_accuracy': float(np.mean(pred == truth)),
                        'predicted_classes': int(len(np.unique(pred))),
                        'false_positive_classes': int(len(set(np.unique(pred)) - set(np.unique(truth)))),
                    }
                rows.append({'fold': fold_no, 'case_index': int(case), 'dataset': dataset,
                             'tokens': int(subset.sum()), 'expert_classes': int(len(np.unique(truth))),
                             'variants': variants})
            atomic(state, {'status': 'RUNNING', 'folds_complete': fold_no,
                           'heldout_cases_complete': len(rows)})
            print('TRAINING_FOLD', fold_no, 'CASES', len(rows), flush=True)
        summary = []
        for floor in THRESHOLDS:
            variant = [r['variants'][str(floor)] for r in rows]
            summary.append({'minimum_fraction': floor,
                            'mean_case_macro_iou': float(np.mean([v['macro_iou'] for v in variant])),
                            'mean_case_accuracy': float(np.mean([v['pixel_accuracy'] for v in variant])),
                            'mean_predicted_classes': float(np.mean([v['predicted_classes'] for v in variant])),
                            'mean_false_positive_classes': float(np.mean([v['false_positive_classes'] for v in variant]))})
        baseline = summary[0]['mean_case_macro_iou']
        viable = [r for r in summary if r['mean_case_macro_iou'] >= baseline - .005]
        selected = max(viable, key=lambda r: r['minimum_fraction'])
        report = {'status': 'TRAINING_ONLY_RIDGE_CLASS_PRESENCE_SEARCH',
                  'training_images': 397, 'heldout_cases': len(rows),
                  'development_masks_opened': 0, 'whole_slide_expert_masks_opened': 0,
                  'selection_rule': 'largest tested floor within 0.005 mean case macro IoU of no floor',
                  'thresholds': THRESHOLDS, 'summary': summary, 'cases': rows,
                  'sample_sha256': sha(SAMPLE), 'splits_sha256': sha(SPLITS),
                  'model_receipt_sha256': sha(MODEL), 'ontology_sha256': sha(ONTOLOGY),
                  'protocol_sha256': sha(PROTOCOL), 'source_sha256': sha(__file__)}
        atomic(receipt, report)
        frozen = {'status': 'TRAINING_ONLY_RIDGE_CLASS_PRESENCE_RULE_FROZEN',
                  'minimum_predicted_tissue_fraction': selected['minimum_fraction'],
                  'baseline_mean_case_macro_iou': baseline,
                  'selected_mean_case_macro_iou': selected['mean_case_macro_iou'],
                  'baseline_mean_false_positive_classes': summary[0]['mean_false_positive_classes'],
                  'selected_mean_false_positive_classes': selected['mean_false_positive_classes'],
                  'training_heldout_cases': len(rows), 'development_masks_opened': 0,
                  'whole_slide_expert_masks_opened': 0,
                  'training_search_sha256': sha(receipt),
                  'selection_rule': report['selection_rule']}
        atomic(frozen_path, frozen)
        atomic(state, {'status': 'COMPLETE', 'folds_complete': len(splits['folds']),
                       'receipt_sha256': sha(receipt), 'frozen_sha256': sha(frozen_path)})
        print('SELECTED', selected, flush=True)
    except BaseException as exc:
        atomic(state, {'status': 'FAILED_PRESERVE_EVIDENCE', 'folds_complete': fold_no - 1,
                       'heldout_cases_complete': len(rows), 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
