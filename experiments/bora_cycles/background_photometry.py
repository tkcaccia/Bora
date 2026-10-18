"""Cycle11 posthoc checks of three visibly blank native-image rectangles.

These rectangles were chosen after actual overview inspection. They are not
independent glass annotations or a cohort-level background-quality metric.
The script reads frozen masks and references without changing any result.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import tifffile

from . import diagnostics
from .run import digest


RECTANGLES = {
    'camelyon16-subset_06': (.1, 0, .9, .2),
    'camelyon16-subset_07': (.1, 0, .9, .2),
    'beetle_01': (.9, 0, 1, .1),
}
METHODS = ('J01', 'B23', 'P10', 'P11', 'P12')


def inspect_background(source, root):
    source, root = Path(source), Path(root)
    protocol = json.loads((root/'PROTOCOL.json').read_text())
    frozen = json.loads((root/'ALL_PREDICTIONS_FROZEN.json').read_text())
    if (protocol['cycle'] != 11 or protocol['pilot']
            or frozen['fine_reference_read']
            or frozen['protocol_sha256'] != digest(root/'PROTOCOL.json')
            or digest(source/'PLAN.json') != protocol['input_plan_sha256']
            or set(frozen['manifests']) != set(protocol['cases'])):
        raise ValueError('Matching whole-run cycle11 prediction freeze required')
    plan = json.loads((source/'PLAN.json').read_text())
    samples = {x['case']: x for x in plan['samples']}
    records = []
    for case, rectangle in RECTANGLES.items():
        src, result = source/'cases'/case, root/'cases'/case
        if digest(result/'PREDICTIONS.json') != frozen['manifests'][case]:
            raise ValueError('Prediction manifest changed: '+case)
        predictions = json.loads((result/'PREDICTIONS.json').read_text())
        if digest(src/'INPUTS_FROZEN.json') != predictions['input_manifest_sha256']:
            raise ValueError('Input manifest changed: '+case)
        inputs = json.loads((src/'INPUTS_FROZEN.json').read_text())
        source_names = ('original.native.png', 'reference.native.tif', 'tissue.native.tif')
        identities = {n: digest(src/n) for n in source_names}
        # Fine references are intentionally absent from prediction-input files.
        # Their independently frozen identity belongs to the source PLAN.
        expected = dict(inputs['files_sha256'])
        expected['reference.native.tif'] = samples[case]['reference_native_sha256']
        if any(expected[n] != h for n, h in identities.items()):
            raise ValueError('Input changed: '+case)
        with Image.open(src/'original.native.png') as image:
            width, height = image.size
            box = tuple(round(v*n) for v, n in zip(rectangle, (width, height, width, height)))
            rgb = np.asarray(image.crop(box).convert('RGB'))
        x0, y0, x1, y1 = box
        white, _, policy = diagnostics.white_regions(rgb)
        truth = tifffile.imread(src/'reference.native.tif')[y0:y1, x0:x1]
        common = tifffile.imread(src/'tissue.native.tif')[y0:y1, x0:x1] > 0
        fractions, hashes = {}, {}
        for method in METHODS:
            path = result/method/'native.tif'
            hashes[method] = digest(path)
            if hashes[method] != predictions['files_sha256'][method+'/native.tif']:
                raise ValueError('Native mask changed: '+case+'/'+method)
            mask = tifffile.imread(path)[y0:y1, x0:x1]
            fractions[method] = float((mask > 0).mean())
        spacing = samples[case]['native_mpp_xy']
        records.append(dict(case=case, native_box_xyxy=box,
            native_mpp_xy=spacing,
            box_physical_size_um_xy=[(x1-x0)*spacing[0], (y1-y0)*spacing[1]],
            rgb_quantile_probabilities=[.01, .5, .99],
            rgb_quantiles=np.quantile(rgb.reshape(-1, 3), [.01, .5, .99], axis=0).tolist(),
            photometric_white_fraction=float(white.mean()),
            reference0_fraction=float((truth == 0).mean()),
            common_support_fraction=float(common.mean()),
            prediction_foreground_fractions=fractions, white_policy=policy,
            input_sha256=identities, native_mask_sha256=hashes))
    return dict(cycle=11, posthoc_selected_cases=True, cases=records,
        protocol_sha256=digest(root/'PROTOCOL.json'),
        prediction_freeze_sha256=digest(root/'ALL_PREDICTIONS_FROZEN.json'),
        source_sha256=digest(Path(__file__)),
        white_policy_module_sha256=digest(Path(diagnostics.__file__)),
        fine_references_read=True, quality_scores_read=False,
        predictions_modified=False, scores_modified=False,
        interpretation='Selected visible-background rectangles, not independent glass truth '
            'or a new cohort quality metric. Reference0 establishes unscored annotation '
            'status, not glass identity. Foreground fractions describe the existing masks.')


def main():
    cv2.setNumThreads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve the existing posthoc observation')
    report = inspect_background(args.source, args.root)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(posthoc_cases=len(report['cases']), masks_modified=False)))


if __name__ == '__main__':
    main()
