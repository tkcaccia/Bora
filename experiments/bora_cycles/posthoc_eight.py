"""Sequential reporting after full inference; never completes a cycle review."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--executed-code', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    protocol = json.loads((root/'PROTOCOL.json').read_text())
    validate_scope(protocol)
    if protocol['cycle'] != 8 or protocol['pilot'] or (root/'SELECTION.json').exists():
        raise ValueError('Unselected, full cycle-eight inference required')
    if not (root/'RESULTS.json').exists() or not (root/'ALL_PREDICTIONS_FROZEN.json').exists():
        raise ValueError('Wait for full predictions and scoring to finish')
    stage = json.loads((root/'STAGE.json').read_text())
    if stage['state'] != 'awaiting_visual_review' or stage['cases'] != 349:
        raise ValueError('The main full run has not completed rendering')
    if (root/'POSTHOC_COMPLETE.json').exists():
        raise ValueError('Reporting already completed; do not regenerate evidence')
    # Each reader validates its inputs. A final independent audit checks every
    # frozen mask and exact display raster after the publication-only writes.
    limits = root/'INPUT_LIMITS.json'
    atlas = root/'SMALL_REGION_ATLAS.json'
    integrity = root/'FULL_AUDIT.json'
    paired = root/'PAIRED_ANALYSIS.json'
    if any(p.exists() for p in (limits,atlas,integrity,paired)):
        raise ValueError('Partial posthoc outputs exist; inspect before resuming explicitly')
    save(limits, audit_inputs(args.source, root))
    print(json.dumps(dict(posthoc='input_limits_complete')),flush=True)
    export(args.source, root, atlas)
    print(json.dumps(dict(posthoc='small_region_atlas_complete')),flush=True)
    publish(args.source, root)
    print(json.dumps(dict(posthoc='rare_colour_publication_complete')),flush=True)
    checked = audit(args.source,root,args.previous,args.executed_code)
    save(integrity, checked)
    if (checked['failed_methods'] or checked['native_masks'] != 7678
            or checked['unchanged_survivor_comparisons'] != 745
            or checked['constant_evidence_contract_comparisons'] != 660):
        raise ValueError('Full mask/parity/constant-evidence audit is incomplete')
    save(paired, analyze(root,args.previous,integrity))
    reports = [limits,atlas,integrity,paired,root/'diagnostics-v2/MANIFEST.json']
    code = Path(__file__).resolve().parent
    save(root/'POSTHOC_COMPLETE.json', dict(cycle=8,cases=349,native_masks=7678,
        report_sha256={str(p.relative_to(root)):digest(p) for p in reports},
        report_source_sha256={p.name:digest(p) for p in code.glob('*.py')},
        primary_protocol_sha256=digest(root/'PROTOCOL.json'),
        primary_results_sha256=digest(root/'RESULTS.json'),
        actual_reviews_required=349,automated_audits_are_not_visual_review=True,
        visual_reviews_created=False,survivors_selected=False,completed_cycles=7,
        predictions_modified=False,scores_modified=False,complete_goal=False))
    print(json.dumps(dict(posthoc='complete',cases=349,completed_cycles=7,
        awaiting_actual_visual_review=True)),flush=True)


if __name__ == '__main__':
    main()
