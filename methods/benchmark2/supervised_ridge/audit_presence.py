#!/usr/bin/env python3
"""Independently recompute the frozen ridge class-presence decision."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEARCH = HERE / 'TRAINING_PRESENCE_SEARCH.json'
FROZEN = HERE / 'CLASS_PRESENCE_FROZEN.json'
SPLITS = Path('/mnt/sata_ssd/ChatGPT/Bora/migration/benchmark1-supervised-strategy-search-2026-11-10-v1/SEARCH_COMPLETE.json')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(4 << 20):
            h.update(chunk)
    return h.hexdigest()


def main():
    search, frozen, splits = [json.loads(p.read_text()) for p in (SEARCH, FROZEN, SPLITS)]
    assert search['status'] == 'TRAINING_ONLY_RIDGE_CLASS_PRESENCE_SEARCH'
    assert frozen['status'] == 'TRAINING_ONLY_RIDGE_CLASS_PRESENCE_RULE_FROZEN'
    assert search['development_masks_opened'] == search['whole_slide_expert_masks_opened'] == 0
    assert frozen['development_masks_opened'] == frozen['whole_slide_expert_masks_opened'] == 0
    assert sha(SEARCH) == frozen['training_search_sha256']
    assert sha(SPLITS) == search['splits_sha256']
    assert len(search['cases']) == frozen['training_heldout_cases'] == 149
    assert len({(r['fold'], r['case_index']) for r in search['cases']}) == 149
    for row in search['cases']:
        assert row['case_index'] in splits['folds'][row['fold'] - 1]['heldout_case_indices']
        assert row['tokens'] > 0
    summary = []
    for floor in search['thresholds']:
        variants = [r['variants'][str(floor)] for r in search['cases']]
        result = {'minimum_fraction': floor,
                  'mean_case_macro_iou': float(np.mean([v['macro_iou'] for v in variants])),
                  'mean_case_accuracy': float(np.mean([v['pixel_accuracy'] for v in variants])),
                  'mean_predicted_classes': float(np.mean([v['predicted_classes'] for v in variants])),
                  'mean_false_positive_classes': float(np.mean([v['false_positive_classes'] for v in variants]))}
        expected, = [r for r in search['summary'] if r['minimum_fraction'] == floor]
        assert all(np.isclose(expected[k], value) for k, value in result.items())
        summary.append(result)
    baseline = summary[0]
    selected = max((r for r in summary if r['mean_case_macro_iou'] >= baseline['mean_case_macro_iou'] - .005),
                   key=lambda r: r['minimum_fraction'])
    assert selected['minimum_fraction'] == frozen['minimum_predicted_tissue_fraction']
    assert selected['mean_case_macro_iou'] == frozen['selected_mean_case_macro_iou']
    assert baseline['mean_case_macro_iou'] == frozen['baseline_mean_case_macro_iou']
    receipt = {'status': 'TRAINING_ONLY_RIDGE_PRESENCE_INDEPENDENTLY_VERIFIED',
               'search_sha256': sha(SEARCH), 'frozen_sha256': sha(FROZEN),
               'heldout_source_disjoint_cases': 149,
               'selected_minimum_fraction': selected['minimum_fraction'],
               'baseline_mean_case_macro_iou': baseline['mean_case_macro_iou'],
               'selected_mean_case_macro_iou': selected['mean_case_macro_iou'],
               'baseline_mean_false_positive_classes': baseline['mean_false_positive_classes'],
               'selected_mean_false_positive_classes': selected['mean_false_positive_classes'],
               'development_or_whole_slide_expert_masks_opened': 0}
    (HERE / 'PRESENCE_AUDIT_VERIFIED.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
