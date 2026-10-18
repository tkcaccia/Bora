"""Export planned review figures after a full frozen, scored and audited run."""
import argparse
import json
from pathlib import Path

from experiments.bora_cycles.review_zoom import export_zoom
from experiments.bora_cycles.run import digest, save


def main():
    import cv2
    cv2.setNumThreads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'root', 'plan'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    protocol = json.loads((root/'PROTOCOL.json').read_text())
    posthoc = json.loads((root/'POSTHOC_COMPLETE.json').read_text())
    plan = json.loads(args.plan.read_text())
    from .benchmark_scope import validate_scope
    validate_scope(protocol)
    if (protocol['pilot'] or protocol['cycle'] != plan['cycle']
            or (plan.get('input_plan_sha256') is not None and plan['input_plan_sha256']!=protocol['input_plan_sha256'])
            or digest(args.source/'PLAN.json') != protocol['input_plan_sha256']
            or posthoc['primary_protocol_sha256'] != digest(root/'PROTOCOL.json')
            or posthoc['primary_results_sha256'] != digest(root/'RESULTS.json')):
        raise ValueError('Full audited evidence for the planned cycle is required')
    outputs = []
    for crop in plan['crops']:
        path = export_zoom(args.source, root, crop['case'],
                           crop['native_crop_xywh'], protocol['methods'])
        outputs.append(dict(case=crop['case'], folder=str(path.relative_to(root)),
                            provenance_sha256=digest(path/'PROVENANCE.json'),
                            labels_sha256=digest(path/'labels.png')))
    save(root/'NATIVE_CROPS_EXPORTED.json', dict(
        cycle=protocol['cycle'], plan_sha256=digest(args.plan), crops=outputs,
        exporter_source_sha256=digest(Path(__file__)),
        predictions_modified=False, scores_modified=False,
        actual_reviews_created=False, selection_created=False))
    print(json.dumps(dict(exported_crops=len(outputs), actual_reviews_created=False)))


if __name__ == '__main__':
    main()
