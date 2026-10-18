"""Post-prediction input limits; never changes masks or primary scores.

The upper bound assumes perfect classification wherever the true class is
available and support permits it. It ignores unavoidable false positives from
unrepresented classes, so it is deliberately optimistic, not an achievable
oracle prediction or a correction to measured accuracy.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tifffile

from .run import digest, save
from experiments.bora_extension.task_adapter import geometry, report_units


def summarize(truth, support, coarse, class_names, mpp_xy, *, sample=None,white=None,ocelot_white_negative=False):
    truth, support, coarse = np.asarray(truth), np.asarray(support, bool), np.asarray(coarse)
    if truth.shape != support.shape:
        raise ValueError('Native reference/support geometry mismatch')
    spacing = np.asarray(mpp_xy, float)
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('Positive physical spacing required')
    available = set(np.unique(coarse[coarse > 0]).tolist())
    from experiments.bora_extension.task_adapter import binary_negative
    negative=binary_negative(sample,ocelot_white_negative=ocelot_white_negative) if sample is not None else None
    if negative is not None and (white is None or np.asarray(white).shape!=truth.shape):
        raise ValueError('The binary support bound requires the same native RGB-white mask as scoring')
    classes = {}
    for label in np.unique(truth[truth > 0]):
        take = truth == label
        n, inside = int(take.sum()), int(np.count_nonzero(take & support))
        allowed = int(label) in available
        valid_negative = (sample is not None and sample['dataset'] == 'rings-tumor'
            and str(sample.get('raw_to_compact', {}).get('0')) == str(int(label)))
        reachable=inside if allowed else 0
        if negative is not None and negative[0]==int(label):
            reachable=int(np.count_nonzero(take & (support|np.asarray(white,bool)))) if allowed else int(np.count_nonzero(take&np.asarray(white,bool)))
        classes[str(int(label))] = dict(name=class_names.get(str(int(label)), str(int(label))),
            truth_pixels=n, supported_truth_pixels=inside, deleted_positive_pixels=n-inside,
            class_available_in_coarse=allowed,
            optimistic_iou_upper_bound=1. if valid_negative else reachable/n)
    if not classes:
        raise ValueError('No positive reference pixels')
    deleted = sum(v['deleted_positive_pixels'] for v in classes.values())
    return dict(positive_reference_pixels=int(np.count_nonzero(truth)),
        deleted_positive_pixels=deleted, deleted_positive_area_um2=float(deleted*spacing.prod()),
        unannotated_supported_pixels=int(np.count_nonzero((truth == 0) & support)),
        reference_classes_absent_from_coarse=[int(k) for k,v in classes.items() if not v['class_available_in_coarse']],
        optimistic_macro_iou_upper_bound=float(np.mean([v['optimistic_iou_upper_bound'] for v in classes.values()])),
        per_class=classes)


def audit_inputs(source, root):
    source, root = Path(source), Path(root)
    protocol = json.loads((root/'PROTOCOL.json').read_text())
    frozen = json.loads((root/'ALL_PREDICTIONS_FROZEN.json').read_text())
    if frozen['protocol_sha256'] != digest(root/'PROTOCOL.json') or frozen['fine_reference_read']:
        raise ValueError('The prediction freeze is invalid')
    if set(frozen['manifests']) != set(protocol['cases']):
        raise ValueError('The complete prediction scope must be frozen before this audit')
    plan = json.loads((source/'PLAN.json').read_text())
    if digest(source/'PLAN.json') != protocol['input_plan_sha256']:
        raise ValueError('Input plan differs from frozen prediction plan')
    samples = {s['case']:s for s in plan['samples']}
    cases = []
    for case in protocol['cases']:
        src = source/'cases'/case
        sample = samples[case]
        inputs = json.loads((src/'INPUTS_FROZEN.json').read_text())
        expected = dict(inputs['files_sha256'])
        expected['reference.native.tif'] = sample['reference_native_sha256']
        names = ['reference.native.tif', 'coarse.tif', 'tissue.native.tif']
        hashes = {name:digest(src/name) for name in names}
        if any(hashes[name] != expected[name] for name in names):
            raise ValueError('Audit input differs from frozen input: '+case)
        truth = tifffile.imread(src/'reference.native.tif')
        support = tifffile.imread(src/'tissue.native.tif') > 0
        coarse = tifffile.imread(src/'coarse.tif')
        coordinates = geometry(sample, native=True)
        from PIL import Image
        from .diagnostics import white_regions
        white=white_regions(np.array(Image.open(src/'original.native.png').convert('RGB')))[0]
        report = summarize(truth,support,coarse,sample['class_names'],coordinates['algorithm_spacing_xy'],
            sample=sample if protocol.get('benchmark_scope') else None,white=white,
            ocelot_white_negative=protocol.get('benchmark_scope',{}).get('version')=='calibrated-tissues-v1')
        cases.append(dict(case=case,dataset=sample['dataset'],source_region_type=sample['source_region_type'],
            native_mpp_xy=sample['native_mpp_xy'],input_sha256=hashes,
            coordinate_units=coordinates['units'], **report_units(report, coordinates['units'])))
    datasets = {}
    for dataset in sorted({c['dataset'] for c in cases}):
        rows = [c for c in cases if c['dataset'] == dataset]
        datasets[dataset] = dict(cases=len(rows),
            positive_reference_pixels=sum(c['positive_reference_pixels'] for c in rows),
            deleted_positive_pixels=sum(c['deleted_positive_pixels'] for c in rows),
            unannotated_supported_pixels=sum(c['unannotated_supported_pixels'] for c in rows),
            cases_with_reference_class_absent_from_coarse=sum(bool(c['reference_classes_absent_from_coarse']) for c in rows),
            mean_optimistic_macro_iou_upper_bound=float(np.mean([c['optimistic_macro_iou_upper_bound'] for c in rows])))
    return dict(cases=cases,datasets=datasets,input_plan_sha256=digest(source/'PLAN.json'),
        prediction_freeze_sha256=digest(root/'ALL_PREDICTIONS_FROZEN.json'),
        mean_optimistic_macro_iou_upper_bound=float(np.mean([c['optimistic_macro_iou_upper_bound'] for c in cases])),
        reporting_only=True,predictions_modified=False,scores_modified=False,
        interpretation='Common-support bounds do not bind own-native-support interventions. Positive reference pixels outside support and classes absent from coarse remain errors, except declared negative targets can be represented by removed background0 under their explicit scoring policy. '
        'Reference zero is unscored. The bound assumes perfect supported classification and ignores extra '
        'false positives forced by missing class IDs; it can be unattainable. It is not a normalized quality score.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh audit path; do not overwrite evidence')
    report = audit_inputs(args.source,args.root)
    save(args.output,report)
    print(json.dumps(dict(cases=len(report['cases']),datasets=report['datasets'],
        mean_optimistic_macro_iou_upper_bound=report['mean_optimistic_macro_iou_upper_bound'])))
