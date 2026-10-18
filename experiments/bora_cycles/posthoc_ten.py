"""Complete frozen-score reporting without creating a visual review."""
import argparse
import json
from pathlib import Path
from .audit_cycle import audit
from .benchmark_scope import validate_scope
from .input_limits import audit_inputs
from .paired_eight import analyze
from .publish_diagnostics import publish
from .run import digest, save
from .small_region_atlas import export
from .numerical_nine import summarize


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'root', 'previous', 'executed-code'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    protocol = json.loads((root/'PROTOCOL.json').read_text())
    validate_scope(protocol)
    if protocol['cycle'] != 10 or protocol['pilot'] or (root/'SELECTION.json').exists():
        raise ValueError('Unselected full cycle-ten results required')
    stage = json.loads((root/'STAGE.json').read_text())
    if stage['state'] != 'awaiting_visual_review' or stage['cases'] != 349:
        raise ValueError('Wait for full prediction freeze, scoring and rendering')
    reports = [root/name for name in ('INPUT_LIMITS.json', 'SMALL_REGION_ATLAS.json',
                                     'FULL_AUDIT.json', 'PAIRED_ANALYSIS.json')]
    if any(p.exists() for p in reports) or (root/'POSTHOC_COMPLETE.json').exists():
        raise ValueError('Existing reporting output requires explicit inspection')
    save(reports[0], audit_inputs(args.source, root))
    export(args.source, root, reports[1])
    publish(args.source, root)
    checked = audit(args.source, root, args.previous, args.executed_code)
    save(reports[2], checked)
    if (checked['failed_methods'] or checked['native_masks'] != 7678
            or checked['unchanged_survivor_comparisons'] != 1745
            or checked['constant_evidence_contract_comparisons'] != 660):
        raise ValueError('Full mask/parity/constant-evidence audit incomplete')
    save(reports[3], analyze(root, args.previous, reports[2]))
    numerical = root/'NUMERICAL_STATUS.json'
    if numerical.exists():
        raise ValueError('Existing numerical report requires explicit inspection')
    save(numerical, summarize(root))
    reports.append(numerical)
    reports.append(root/'diagnostics-v2/MANIFEST.json')
    save(root/'POSTHOC_COMPLETE.json', dict(cycle=10, cases=349, native_masks=7678,
        report_sha256={str(p.relative_to(root)):digest(p) for p in reports},
        primary_protocol_sha256=digest(root/'PROTOCOL.json'),
        primary_results_sha256=digest(root/'RESULTS.json'),
        actual_reviews_required=349, automated_audits_are_not_visual_review=True,
        visual_reviews_created=False, survivors_selected=False, completed_cycles=9,
        predictions_modified=False, scores_modified=False, complete_goal=False))
    print(json.dumps(dict(posthoc='complete', cases=349, completed_cycles=9,
                         awaiting_actual_visual_review=True)), flush=True)


if __name__ == '__main__':
    main()
