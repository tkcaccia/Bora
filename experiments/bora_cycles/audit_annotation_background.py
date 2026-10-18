"""Reproduce the posthoc TIGER71 annotation/background disagreement check.

The rectangle was selected after visual review. It is not independent glass
ground truth, a new cohort metric, or permission to rewrite frozen scores.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile

from .run import digest, save


CASE = 'tiger_71'
BOX = (16, 768, 160, 160)


def inspect_rectangle(source, root, data):
    source, root, data = map(Path, (source, root, data))
    protocol = json.loads((root/'PROTOCOL.json').read_text())
    frozen = json.loads((root/'ALL_PREDICTIONS_FROZEN.json').read_text())
    if (protocol['cycle'] != 11 or protocol['pilot']
            or frozen['fine_reference_read']
            or frozen['protocol_sha256'] != digest(root/'PROTOCOL.json')
            or digest(source/'PLAN.json') != protocol['input_plan_sha256']):
        raise ValueError('Matching full cycle11 prediction freeze required')
    src, dst = source/'cases'/CASE, root/'cases'/CASE
    if digest(dst/'PREDICTIONS.json') != frozen['manifests'][CASE]:
        raise ValueError('Prediction manifest changed')
    predictions = json.loads((dst/'PREDICTIONS.json').read_text())
    if digest(src/'INPUTS_FROZEN.json') != predictions['input_manifest_sha256']:
        raise ValueError('Input manifest changed')
    inputs = json.loads((src/'INPUTS_FROZEN.json').read_text())
    plan = json.loads((source/'PLAN.json').read_text())
    sample = next(s for s in plan['samples'] if s['case'] == CASE)
    if (digest(src/'reference.native.tif') != sample['reference_native_sha256']
            or digest(src/'original.native.png')
            != inputs['files_sha256']['original.native.png']):
        raise ValueError('Native inputs changed')
    raw_rgb, raw_mask = (data/sample['source'][k] for k in ('image', 'mask'))
    if (digest(raw_rgb) != sample['source']['image_sha256']
            or digest(raw_mask) != sample['source']['mask_sha256']):
        raise ValueError('Raw source files changed')
    with Image.open(src/'original.native.png') as image:
        he = np.asarray(image.convert('RGB'))
    with Image.open(raw_rgb) as image:
        original = np.asarray(image.convert('RGB'))
    with Image.open(raw_mask) as image:
        original_mask = np.asarray(image)
    truth = tifffile.imread(src/'reference.native.tif')
    # This case is a full unscaled ROI with identity raw/compact labels.
    if (sample['native_crop_xywh'] != [0, 0, he.shape[1], he.shape[0]]
            or not np.array_equal(original, he)
            or not np.array_equal(original_mask, truth)):
        raise ValueError('The specific source/native identity no longer holds')
    x, y, width, height = BOX
    rgb = he[y:y+height, x:x+width]
    target = truth[y:y+height, x:x+width]
    if rgb.shape != (height, width, 3):
        raise ValueError('Rectangle outside native ROI')
    fractions, identities = {}, {}
    for method in protocol['methods']:
        path = dst/method/'native.tif'
        identities[method] = digest(path)
        if identities[method] != predictions['files_sha256'][method+'/native.tif']:
            raise ValueError('Frozen native mask changed: '+method)
        labels = tifffile.imread(path)[y:y+height, x:x+width]
        values, counts = np.unique(labels, return_counts=True)
        fractions[method] = {str(int(v)): float(n/(width*height))
                             for v, n in zip(values, counts)}
    return dict(cycle=11, case=CASE, posthoc_selected_after_actual_overview=True,
        region_description='Visually empty exterior rectangle at the left of a cut '
            'tissue boundary; selected after overview, not independent glass annotation.',
        native_box_xywh=list(BOX), native_mpp_xy=sample['native_mpp_xy'],
        raw_rgb_matches_native_pixelwise=True,
        raw_mask_matches_compacted_reference_pixelwise=True,
        reference_class_fractions={str(int(v)): float((target == v).mean())
                                   for v in np.unique(target)},
        class_names=sample['class_names'],
        rgb_quantiles=np.quantile(rgb.reshape(-1, 3), [0, .01, .5, .99, 1], axis=0).tolist(),
        rgb_quantile_probabilities=[0, .01, .5, .99, 1],
        prediction_class_fractions=fractions,
        protocol_sha256=digest(root/'PROTOCOL.json'),
        results_sha256=digest(root/'RESULTS.json'),
        prediction_freeze_sha256=digest(root/'ALL_PREDICTIONS_FROZEN.json'),
        raw_image_sha256=digest(raw_rgb), raw_mask_sha256=digest(raw_mask),
        native_image_sha256=digest(src/'original.native.png'),
        native_reference_sha256=digest(src/'reference.native.tif'),
        native_mask_sha256=identities, predictions_modified=False,
        scores_modified=False, cohort_metric=False,
        interpretation='Positive source annotation may include visible exterior glass. '
            'The existing frozen metrics remain unchanged; filling this rectangle '
            'receives positive-class credit. This is an annotation/background '
            'disagreement diagnostic, not a correction or independently measured '
            'glass error rate.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'root', 'data', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--verify-existing', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve the existing audit receipt')
    report = inspect_rectangle(args.source, args.root, args.data)
    if args.verify_existing:
        if json.loads(args.verify_existing.read_text()) != report:
            raise ValueError('Reproduction differs from the existing observation')
    report['audit_source_sha256'] = digest(Path(__file__))
    report['verified_existing_report_sha256'] = (
        digest(args.verify_existing) if args.verify_existing else None)
    save(args.output, report)
    print(json.dumps(dict(case=CASE, output=str(args.output),
        verified_existing=bool(args.verify_existing), predictions_modified=False)))


if __name__ == '__main__':
    main()
