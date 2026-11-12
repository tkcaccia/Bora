#!/usr/bin/env python3
"""Polish the training-selected head with the frozen Bora ROI contour rule."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
HERE = Path(__file__).resolve().parent
OUT = DATA / 'benchmark1-eight-class-head-search-2026-11-12-v1/development-polished'
BASE = HERE / 'DEVELOPMENT_EVALUATION_COMPLETE.json'
ONTOLOGY = ROOT / 'migration/benchmark-label-reduction-2026-11-11-v1/ONTOLOGY_FROZEN.json'
GEOM = DATA / 'benchmark1-supervised-virchow2-response-2026-11-10-v1'
HARM = ROOT / 'benchmark/methods/benchmark1/tissue_label_harmonization.json'
FROZEN = ROOT / 'migration/benchmark1-supervised-virchow2-response-2026-11-10-v1/INPUTS_FROZEN.json'
SCORES = DATA / 'benchmark1-supervised-mask-overlay-score-2026-11-10-v1/SCORES_AND_FIGURES.json'
CONTOUR_SOURCE = ROOT / 'migration/benchmark1-supervised-ontology-contour-2026-11-11-v1/run.py'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    receipt = HERE / 'DEVELOPMENT_CONTOUR_COMPLETE.json'
    if receipt.exists() or OUT.exists():
        raise FileExistsError('No in-place contour rerun')
    spec = importlib.util.spec_from_file_location('frozen_bora_contour', CONTOUR_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = json.loads(BASE.read_text())
    ontology = json.loads(ONTOLOGY.read_text())
    assert base['ontology_sha256'] == sha(ONTOLOGY)
    old_roles = json.loads(FROZEN.read_text())['classes']
    role_to_new = {role: ontology['old_to_new'][cid] for cid, role in old_roles.items()}
    harmony = json.loads(HARM.read_text())['mapping']
    references = {r['id']: r for r in json.loads(SCORES.read_text())['cases']}
    assert len(base['cases']) == len(references) == 87
    OUT.mkdir(parents=True)
    state = HERE / 'DEVELOPMENT_CONTOUR_STATE.json'
    records = []
    try:
        for row in base['cases']:
            cid = row['id']
            assert sha(row['prediction_path']) == row['prediction_sha256']
            prior = np.asarray(Image.open(row['prediction_path']), np.uint8)
            geometry = json.loads((GEOM / cid / 'RESPONSE_COMPLETE.json').read_text())
            working_wh = geometry['working_size_wh']
            display_wh = geometry['combined_mask']['display_size_wh']
            assert prior.shape == tuple(display_wh[::-1])
            mpp = tuple(.5 * working_wh[i] / display_wh[i] for i in (0, 1))
            result, report = module.polish(prior, mpp)
            assert np.isin(result, range(1, 9)).all()
            path = OUT / f'{cid}-class-mask.png'
            Image.fromarray(result, 'L').save(path, optimize=True)
            reference = references[cid]
            assert sha(reference['source_mask_path']) == reference['source_mask_sha256']
            shape = tuple(reference['scoring_shape_yx'])
            with Image.open(reference['source_mask_path']) as im:
                raw = np.asarray(im.resize(shape[::-1], Image.Resampling.NEAREST))
            truth = np.zeros(shape, np.uint8)
            for code in np.unique(raw):
                entry = harmony[row['dataset']].get(str(int(code)))
                if entry and entry['use'] == 'score':
                    new_id = role_to_new.get(entry.get('canonical_role'))
                    if new_id is not None:
                        truth[raw == code] = new_id
            before = np.asarray(Image.fromarray(prior).resize(shape[::-1], Image.Resampling.NEAREST))
            after = np.asarray(Image.fromarray(result).resize(shape[::-1], Image.Resampling.NEAREST))
            before_score = module.score(before, truth)
            after_score = module.score(after, truth)
            assert abs(before_score['expert_present_macro_iou'] - row['metrics']['expert_present_macro_iou']) < 1e-9
            assert before_score['scored_expert_pixels'] == after_score['scored_expert_pixels']
            records.append({'id': cid, 'case': row['case'], 'dataset': row['dataset'],
                            'input_sha256': row['prediction_sha256'], 'output_path': str(path),
                            'output_sha256': sha(path), 'contour': report,
                            'before_macro_iou': before_score['expert_present_macro_iou'],
                            'after': after_score})
            state.write_text(json.dumps({'status': 'RUNNING', 'complete': len(records), 'last': cid}) + '\n')
            print('POLISHED', len(records), '/87', cid, flush=True)
        result_receipt = {'status': 'EIGHT_CLASS_87_CASE_SUPERVISED_CONTOURS_POLISHED',
                          'ontology_sha256': sha(ONTOLOGY), 'source_validation_sha256': sha(BASE),
                          'contour_rule_source_sha256': sha(CONTOUR_SOURCE),
                          'source_code_sha256': sha(Path(__file__)),
                          'mean_macro_iou_before': float(np.mean([r['before_macro_iou'] for r in records])),
                          'mean_macro_iou_after': float(np.mean([r['after']['expert_present_macro_iou'] for r in records])),
                          'four_multiclass_dataset_mean_macro_iou_after': float(np.mean([
                              np.mean([r['after']['expert_present_macro_iou'] for r in records if r['dataset'] == dataset])
                              for dataset in ('bcss', 'gcss', 'ignite', 'tiger')])),
                          'cases': records}
        receipt.write_text(json.dumps(result_receipt, indent=2) + '\n')
        state.write_text(json.dumps({'status': 'COMPLETE', 'complete': 87,
                                     'receipt_sha256': sha(receipt)}) + '\n')
        print('COMPLETE', result_receipt['mean_macro_iou_after'], flush=True)
    except BaseException as exc:
        state.write_text(json.dumps({'status': 'FAILED_PRESERVE_PARTIAL', 'complete': len(records),
                                     'error': repr(exc)}) + '\n')
        raise


if __name__ == '__main__':
    main()
