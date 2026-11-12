#!/usr/bin/env python3
"""One-time development evaluation of training-selected eight-class head."""
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
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
HERE = Path(__file__).resolve().parent
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data/benchmark1-eight-class-head-search-2026-11-12-v1')
OUT = DATA / 'development-predictions'
FROZEN = ROOT / 'migration/benchmark1-supervised-virchow2-response-2026-11-10-v1/INPUTS_FROZEN.json'
HARM = ROOT / 'benchmark/methods/benchmark1/tissue_label_harmonization.json'
GEOMETRY = Path('/mnt/sata_ssd/ChatGPT/Bora-Data/benchmark1-supervised-virchow2-response-2026-11-10-v1')
BASE = Path('/mnt/sata_ssd/ChatGPT/Bora-Data/benchmark1-supervised-mask-overlay-score-2026-11-10-v1/SCORES_AND_FIGURES.json')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def score(pred, truth):
    valid = truth > 0
    by_class = {}
    for cid in np.unique(truth[valid]):
        actual, guessed = truth == cid, pred == cid
        tp = int(np.count_nonzero(valid & actual & guessed))
        fp = int(np.count_nonzero(valid & ~actual & guessed))
        fn = int(np.count_nonzero(valid & actual & ~guessed))
        by_class[str(int(cid))] = {'expert_pixels': tp + fn, 'intersection': tp,
                                   'false_positive': fp, 'false_negative': fn,
                                   'iou': tp / (tp + fp + fn),
                                   'dice': 2 * tp / (2 * tp + fp + fn)}
    return {'expert_present_macro_iou': float(np.mean([v['iou'] for v in by_class.values()])),
            'expert_present_macro_dice': float(np.mean([v['dice'] for v in by_class.values()])),
            'scored_pixel_accuracy': float(np.mean(pred[valid] == truth[valid])),
            'scored_expert_pixels': int(valid.sum()), 'all_pixels': int(truth.size),
            'scored_pixel_fraction': float(valid.mean()), 'per_present_class': by_class}


def main():
    receipt = HERE / 'DEVELOPMENT_EVALUATION_COMPLETE.json'
    if receipt.exists() or OUT.exists():
        raise FileExistsError('No in-place validation rerun')
    ontology_path = ROOT / 'migration/benchmark-label-reduction-2026-11-11-v1/ONTOLOGY_FROZEN.json'
    model_receipt_path = HERE / 'FINAL_MODEL_FROZEN.json'
    ontology = json.loads(ontology_path.read_text())
    fitted = json.loads(model_receipt_path.read_text())
    assert fitted['validation_images_opened'] == ontology['validation_masks_opened'] == 0
    assert fitted['selected'] == 'class32_ridge'
    search_audit = HERE / 'SEARCH_AUDIT_VERIFIED.json'
    assert sha(search_audit) == fitted['search_audit_sha256']
    assert sha(ontology_path) == fitted['ontology_sha256']
    assert sha(fitted['model_path']) == fitted['model_sha256']
    model = joblib.load(fitted['model_path'])
    assert np.array_equal(model['classifier'].classes_, np.arange(1, 9))
    selected = model['selected']
    freeze = json.loads(FROZEN.read_text())
    validation = {r['case']: r for r in freeze['rows'] if r['split'] == 'validation'}
    baseline = json.loads(BASE.read_text())
    assert len(validation) == len(baseline['cases']) == 87
    harmony = json.loads(HARM.read_text())['mapping']
    old_names = freeze['classes']
    role_to_new = {role: ontology['old_to_new'][old_id] for old_id, role in old_names.items()}
    OUT.mkdir(parents=True)
    state = HERE / 'DEVELOPMENT_EVALUATION_STATE.json'
    state.write_text(json.dumps({'status': 'RUNNING', 'complete': 0}) + '\n')
    rows = []
    try:
        for old in baseline['cases']:
            case, case_id = old['case'], old['id']
            source = validation[case]
            geometry = json.loads((GEOMETRY / case_id / 'RESPONSE_COMPLETE.json').read_text())
            working_wh = tuple(geometry['working_size_wh'])
            display_wh = tuple(geometry['combined_mask']['display_size_wh'])
            feature_path = Path(source['feature_path'])
            if sha(feature_path) != source['feature_sha256']:
                raise ValueError('Feature SHA mismatch: ' + case_id)
            grid = np.load(feature_path, mmap_mode='r')
            assert tuple(grid.shape) == tuple(source['feature_shape'])
            finite = np.isfinite(np.asarray(grid[:, :, selected])).all(axis=2)
            if not finite.any():
                raise ValueError('No finite feature: ' + case_id)
            if finite.all():
                nearest = np.arange(grid.shape[0] * grid.shape[1])
            else:
                _, near = distance_transform_edt(~finite, return_indices=True)
                nearest = np.ravel_multi_index(near, grid.shape[:2]).ravel()
            flat = grid.reshape(-1, 1280)
            tic = monotonic()
            x = np.asarray(flat[nearest[:, None], selected[None, :]], np.float32)
            x = (x - model['mean']) / model['std']
            logits = model['classifier'].decision_function(np.ascontiguousarray(x))
            logits = logits.reshape(*grid.shape[:2], 8)
            display = np.empty((*display_wh[::-1], 8), np.float32)
            for cid in range(8):
                token = gaussian_filter(logits[:, :, cid], sigma=.75, mode='nearest')
                full = Image.fromarray(token.astype(np.float32), 'F').resize(
                    (grid.shape[1] * 14, grid.shape[0] * 14), Image.Resampling.BICUBIC)
                display[:, :, cid] = np.asarray(full.crop((0, 0, *working_wh)).resize(
                    display_wh, Image.Resampling.BICUBIC), np.float32)
            allowed = np.isin(np.arange(1, 9), fitted['eligible_classes_by_dataset'][old['dataset']])
            assert allowed.any()
            pred_display = np.argmax(np.where(allowed[None, None, :], display, -np.inf), axis=2).astype(np.uint8) + 1
            inference_seconds = monotonic() - tic
            pred_path = OUT / f'{case_id}-class-mask.png'
            Image.fromarray(pred_display, 'L').save(pred_path, optimize=True)
            frame_wh = tuple(old['scoring_shape_yx'][::-1])
            pred = np.asarray(Image.fromarray(pred_display, 'L').resize(frame_wh, Image.Resampling.NEAREST))
            mask_path = Path(old['source_mask_path'])
            if sha(mask_path) != old['source_mask_sha256']:
                raise ValueError('Expert SHA mismatch: ' + case_id)
            with Image.open(mask_path) as image:
                raw = np.asarray(image.resize(frame_wh, Image.Resampling.NEAREST))
            truth = np.zeros(raw.shape, np.uint8)
            for code in np.unique(raw):
                spec = harmony[old['dataset']].get(str(int(code)))
                if spec is None and code not in (0, 255):
                    raise ValueError('Unmapped expert code: ' + case_id)
                if spec and spec['use'] == 'score':
                    new_id = role_to_new.get(spec.get('canonical_role'))
                    if new_id is not None:
                        truth[raw == code] = new_id
            metrics = score(pred, truth)
            rows.append({'id': case_id, 'case': case, 'dataset': old['dataset'],
                         'prediction_path': str(pred_path), 'prediction_sha256': sha(pred_path),
                         'feature_sha256': source['feature_sha256'],
                         'expert_sha256': old['source_mask_sha256'],
                         'inference_seconds': inference_seconds, 'metrics': metrics})
            state.write_text(json.dumps({'status': 'RUNNING', 'complete': len(rows),
                                         'last': case_id}) + '\n')
            print('VALIDATED', len(rows), '/87', case_id,
                  round(metrics['expert_present_macro_iou'], 4), flush=True)
        receipt_data = {'status': 'EIGHT_CLASS_87_CASE_DEVELOPMENT_EVALUATION_COMPLETE',
                        'ontology_sha256': sha(ontology_path),
                        'model_receipt_sha256': sha(model_receipt_path),
                        'input_freeze_sha256': sha(FROZEN),
                        'baseline_case_list_sha256': sha(BASE),
                        'training_images': 397, 'validation_cases': len(rows),
                        'validation_cases_used_for_selection': 0,
                        'is_independent_test': False,
                        'mean_case_macro_iou': float(np.mean([r['metrics']['expert_present_macro_iou'] for r in rows])),
                        'mean_case_macro_dice': float(np.mean([r['metrics']['expert_present_macro_dice'] for r in rows])),
                        'mean_scored_pixel_fraction': float(np.mean([r['metrics']['scored_pixel_fraction'] for r in rows])),
                        'four_multiclass_dataset_mean_case_macro_iou': float(np.mean([
                            np.mean([r['metrics']['expert_present_macro_iou'] for r in rows if r['dataset'] == dataset])
                            for dataset in sorted({r['dataset'] for r in rows} - {'peso'})])),
                        'cases': rows, 'source_sha256': sha(Path(__file__))}
        receipt.write_text(json.dumps(receipt_data, indent=2) + '\n')
        state.write_text(json.dumps({'status': 'COMPLETE', 'complete': 87,
                                     'receipt_sha256': sha(receipt)}) + '\n')
        print('VALIDATION_COMPLETE', receipt_data['mean_case_macro_iou'], flush=True)
    except BaseException as exc:
        state.write_text(json.dumps({'status': 'FAILED', 'complete': len(rows), 'error': repr(exc)}) + '\n')
        raise


if __name__ == '__main__':
    main()
