#!/usr/bin/env python3
"""Frozen training-only CPU search for eight-class Virchow2 token heads."""
import hashlib
import json
import os
from pathlib import Path
from time import monotonic

os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
HERE = Path(__file__).resolve().parent
SAMPLE = ROOT / 'migration/benchmark1-supervised-strategy-search-2026-11-10-v1/SAMPLE_COMPLETE.json'
SPLITS = ROOT / 'migration/benchmark1-supervised-strategy-search-2026-11-10-v1/SEARCH_COMPLETE.json'
ONTOLOGY = ROOT / 'migration/benchmark-label-reduction-2026-11-11-v1/ONTOLOGY_FROZEN.json'
FROZEN = ROOT / 'migration/benchmark-label-reduction-2026-11-11-v1/SUPERVISED_MODEL_FROZEN.json'
PROTOCOL = HERE / 'PROTOCOL_FROZEN.json'
STATE = HERE / 'SEARCH_STATE.json'
RECEIPT = HERE / 'TRAINING_SEARCH_COMPLETE.json'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(tmp, path)


def select_columns(z, y, selector):
    if selector == 'global ANOVA top 256':
        scores, _ = f_classif(z, y)
        scores = np.nan_to_num(scores, nan=-np.inf)
        return np.argsort(-scores, kind='stable')[:256]
    n = 32 if 'top 32 ' in selector else 64
    chosen = []
    for label in sorted(np.unique(y)):
        pos = y == label
        # z is globally standardized; the one-vs-rest contrast is expressed
        # in training-fold standard-deviation units, without target leakage.
        contrast = np.abs(z[pos].mean(axis=0) - z[~pos].mean(axis=0))
        chosen.extend(np.argsort(-contrast, kind='stable')[:n].tolist())
    return np.asarray(sorted(set(chosen)), dtype=np.int32)


def fit_head(z, y, columns, cfg):
    counts = np.bincount(y, minlength=9).astype(np.float64)
    weights = np.maximum(counts[y], 1.0) ** -0.5
    weights /= weights.mean()
    if cfg['classifier'] == 'multinomial logistic':
        head = LogisticRegression(C=cfg['C'], solver='lbfgs', max_iter=150, tol=1e-3)
    else:
        head = RidgeClassifier(alpha=cfg['alpha'], solver='cholesky')
    head.fit(np.ascontiguousarray(z[:, columns]), y, sample_weight=weights)
    return head


def score(pred, truth, cases, datasets, dataset_names):
    rows = []
    for case in np.unique(cases):
        mask = cases == case
        t, p = truth[mask], pred[mask]
        present = np.unique(t)
        iou, dice = [], []
        for label in present:
            tp = np.count_nonzero((t == label) & (p == label))
            union = np.count_nonzero((t == label) | (p == label))
            iou.append(tp / union)
            dice.append(2 * tp / (np.count_nonzero(t == label) + np.count_nonzero(p == label)))
        rows.append({'case_index': int(case), 'dataset': dataset_names[int(datasets[mask][0])],
                     'macro_iou': float(np.mean(iou)), 'macro_dice': float(np.mean(dice)),
                     'accuracy': float(np.mean(t == p))})
    by = {name: float(np.mean([r['macro_iou'] for r in rows if r['dataset'] == name]))
          for name in dataset_names if any(r['dataset'] == name for r in rows)}
    per_class = {}
    for label in range(1, 9):
        t = truth == label
        p = pred == label
        support = int(t.sum())
        intersection = int((t & p).sum())
        union = int((t | p).sum())
        per_class[str(label)] = {'support': support,
                                 'predicted': int(p.sum()),
                                 'iou': float(intersection / union) if union else None,
                                 'dice': float(2 * intersection / (support + int(p.sum())))
                                 if support + int(p.sum()) else None}
    return {'case_macro_iou': float(np.mean([r['macro_iou'] for r in rows])),
            'case_macro_dice': float(np.mean([r['macro_dice'] for r in rows])),
            'four_multiclass_dataset_macro_iou': float(np.mean([v for k, v in by.items() if k != 'peso'])),
            'dataset_mean': by, 'classwise': per_class, 'cases': len(rows)}


def main():
    if RECEIPT.exists():
        raise FileExistsError(RECEIPT)
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol['status'] == 'TRAINING_ONLY_PROTOCOL_FROZEN'
    sample, splits, ontology, frozen = [json.loads(p.read_text()) for p in (SAMPLE, SPLITS, ONTOLOGY, FROZEN)]
    assert sample['training_cases'] == 397 and sample['validation_cases_opened'] == 0
    assert splits['validation_cases_opened'] == 0
    assert sha(Path(sample['feature_path'])) == sample['feature_sha256']
    assert sha(Path(sample['index_path'])) == sample['index_sha256']
    assert len(sample['case_ids']) == len(sample['source_groups']) == 397
    idx = np.load(sample['index_path'])
    old_y, cases, datasets = [idx[k] for k in ('label', 'case_index', 'dataset_id')]
    lut = np.zeros(256, np.uint8)
    for old_id, new_id in ontology['old_to_new'].items():
        if new_id is not None:
            lut[int(old_id)] = new_id
    y = lut[old_y]
    valid = y > 0
    x = np.load(sample['feature_path'], mmap_mode='r')
    assert x.shape == (139736, 1280) and len(y) == x.shape[0]
    eligible = frozen['eligible_classes_by_dataset']
    rows = []
    atomic_json(STATE, {'status': 'RUNNING', 'completed': 0, 'protocol_sha256': sha(PROTOCOL)})
    try:
        for fold_no, split in enumerate(splits['folds'], 1):
            tr_cases = set(split['training_case_indices'])
            te_cases = set(split['heldout_case_indices'])
            assert not tr_cases & te_cases
            tr_groups = {sample['source_groups'][c] for c in tr_cases}
            te_groups = {sample['source_groups'][c] for c in te_cases}
            assert not tr_groups & te_groups
            tr = valid & np.isin(cases, list(tr_cases))
            te = valid & np.isin(cases, list(te_cases))
            xtr = np.asarray(x[tr], dtype=np.float32)
            xte = np.asarray(x[te], dtype=np.float32)
            mean = xtr.mean(axis=0)
            std = np.maximum(xtr.std(axis=0), 1e-4)
            ztr = (xtr - mean) / std
            del xtr
            for cfg in protocol['candidates']:
                tic = monotonic()
                columns = select_columns(ztr, y[tr], cfg['selector'])
                head = fit_head(ztr, y[tr], columns, cfg)
                fit_seconds = monotonic() - tic
                tic = monotonic()
                zte = (xte[:, columns] - mean[columns]) / std[columns]
                decision = head.decision_function(np.ascontiguousarray(zte))
                assert decision.ndim == 2 and decision.shape[1] == len(head.classes_)
                for did, name in enumerate(sample['dataset_names']):
                    subset = datasets[te] == did
                    if subset.any():
                        allowed = np.isin(head.classes_, eligible[name])
                        assert allowed.any()
                        decision[np.ix_(subset, ~allowed)] = -np.inf
                pred = head.classes_[np.argmax(decision, axis=1)]
                prediction_seconds = monotonic() - tic
                metrics = score(pred, y[te], cases[te], datasets[te], sample['dataset_names'])
                rows.append({'fold': fold_no, 'candidate': cfg['id'], 'training_cases': len(tr_cases),
                             'heldout_cases': len(te_cases), 'training_tokens': int(tr.sum()),
                             'heldout_tokens': int(te.sum()), 'selected_channels': int(len(columns)),
                             'fit_seconds': fit_seconds, 'prediction_seconds': prediction_seconds,
                             'prediction_seconds_per_10000_tokens': prediction_seconds / te.sum() * 10000,
                             'metrics': metrics})
                atomic_json(STATE, {'status': 'RUNNING', 'completed': len(rows),
                                    'last': [fold_no, cfg['id']], 'protocol_sha256': sha(PROTOCOL)})
                print('TRAINING_FOLD', fold_no, cfg['id'],
                      metrics['four_multiclass_dataset_macro_iou'], flush=True)
            del ztr, xte
        aggregate = []
        for cfg in protocol['candidates']:
            part = [r for r in rows if r['candidate'] == cfg['id']]
            aggregate.append({'candidate': cfg['id'],
                              'four_multiclass_dataset_macro_iou': float(np.mean([r['metrics']['four_multiclass_dataset_macro_iou'] for r in part])),
                              'mean_fit_seconds': float(np.mean([r['fit_seconds'] for r in part])),
                              'mean_prediction_seconds_per_10000_tokens': float(np.mean([r['prediction_seconds_per_10000_tokens'] for r in part])),
                              'mean_selected_channels': float(np.mean([r['selected_channels'] for r in part]))})
        best_score = max(a['four_multiclass_dataset_macro_iou'] for a in aggregate)
        contenders = [a for a in aggregate if best_score - a['four_multiclass_dataset_macro_iou'] <= 0.005]
        contenders.sort(key=lambda a: (a['mean_fit_seconds'] + a['mean_prediction_seconds_per_10000_tokens'],
                                       a['mean_selected_channels'], a['candidate']))
        selected = contenders[0]['candidate']
        receipt = {'status': 'TRAINING_ONLY_HEAD_SEARCH_COMPLETE', 'training_images': 397,
                   'validation_images_opened': 0, 'sample_sha256': sha(SAMPLE),
                   'feature_sha256': sample['feature_sha256'], 'index_sha256': sample['index_sha256'],
                   'splits_sha256': sha(SPLITS), 'ontology_sha256': sha(ONTOLOGY),
                   'protocol_sha256': sha(PROTOCOL), 'source_sha256': sha(Path(__file__)),
                   'selected': selected, 'aggregate': aggregate, 'fold_rows': rows}
        atomic_json(RECEIPT, receipt)
        atomic_json(STATE, {'status': 'COMPLETE', 'completed': len(rows),
                            'receipt_sha256': sha(RECEIPT)})
    except BaseException as exc:
        atomic_json(STATE, {'status': 'FAILED', 'completed': len(rows), 'error': repr(exc),
                            'protocol_sha256': sha(PROTOCOL)})
        raise


if __name__ == '__main__':
    main()
