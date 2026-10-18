"""Paired, cohort-explicit analysis of frozen expanded-cycle scores only."""
import json
from pathlib import Path

import numpy as np

from .benchmark_scope import validate_scope
from .eight_manifest import CHILDREN
from .run import digest
from .runtime_policy import active_policy, adjusted_score


SCORE_CORE = ('macro_iou', 'macro_dice', 'pixel_accuracy', 'annotated_unassigned_pixels',
              'valid_pixels', 'per_class', 'regions', 'missed_regions')


def delta_summary(rows, child, parent, policy):
    failed = [r['case'] for r in rows if any(r['methods'].get(k, {}).get('status') != 'ok'
                                          for k in (child, parent))]
    if failed:
        return dict(cases=len(rows), complete=False, failed_cases=failed)
    if not rows:
        return dict(cases=0, complete=True, mean_iou_delta=None, mean_utility_delta=None,
                    wins=0, losses=0, ties=0, median_iou_delta=None)
    difference, utilities = [], []
    for row in rows:
        c, p = row['methods'][child], row['methods'][parent]
        difference.append(c['score']['macro_iou']-p['score']['macro_iou'])
        utilities.append(adjusted_score(c['score']['macro_iou'], c['median_seconds'], policy)
                         -adjusted_score(p['score']['macro_iou'], p['median_seconds'], policy))
    delta = np.asarray(difference)
    return dict(cases=len(rows), complete=True, mean_iou_delta=float(delta.mean()),
        median_iou_delta=float(np.median(delta)), mean_utility_delta=float(np.mean(utilities)),
        wins=int((delta > 1e-12).sum()), losses=int((delta < -1e-12).sum()),
        ties=int((np.abs(delta) <= 1e-12).sum()))


def parent_score_parity(rows, previous_rows, historical, parents):
    """New task adapters must not alter any historical accuracy definition."""
    current = {row['case']:row for row in rows}
    previous = {row['case']:row for row in previous_rows}
    if not set(historical) <= set(current) or set(historical) != set(previous):
        raise ValueError('Historical score scope is incomplete or changed')
    checked = 0
    for case in historical:
        for key in parents:
            a, b = current[case]['methods'][key], previous[case]['methods'][key]
            if a['status'] != 'ok' or b['status'] != 'ok':
                raise ValueError('An unchanged parent has a failed historical score')
            for field in SCORE_CORE:
                if a['score'][field] != b['score'][field]:
                    raise ValueError('Historical parent score changed: '+case+'/'+key+'/'+field)
            checked += 1
    return checked


def analyze(root, previous, audit_path):
    root, previous, audit_path = Path(root), Path(previous), Path(audit_path)
    result = json.loads((root/'RESULTS.json').read_text())
    protocol = result['protocol']
    validate_scope(protocol)
    cycle = protocol['cycle']
    if cycle not in (8, 9, 10) or protocol['pilot']:
        raise ValueError('Full expanded cycle-eight/nine/ten results required')
    if cycle == 10:
        from .ten_manifest import CHILDREN as children_specs
    elif cycle == 9:
        from .nine_manifest import CHILDREN as children_specs
    else:
        children_specs = CHILDREN
    audit = json.loads(audit_path.read_text())
    if (audit['results_sha256'] != digest(root/'RESULTS.json')
            or audit['protocol_sha256'] != digest(root/'PROTOCOL.json')
            or audit['cases'] != 349 or audit['methods'] != 22):
        raise ValueError('Full scored evidence does not match the completed integrity audit')
    old_selection = json.loads((previous/'SELECTION.json').read_text())
    if (digest(previous/'SELECTION.json') != protocol['lineage']['selection_sha256']
            or digest(previous/'RESULTS.json') != old_selection['results_sha256']):
        raise ValueError('Historical signed parent results changed')
    old_result = json.loads((previous/'RESULTS.json').read_text())
    historical = (protocol['benchmark_scope']['historical_cases'] if cycle == 8
                  else protocol['cases'])
    parents = protocol['lineage']['unchanged_survivors']
    checked = parent_score_parity(result['cases'], old_result['cases'], historical, parents)
    policy = active_policy(protocol)
    if policy is None:
        raise ValueError('Frozen excess-one-second policy required')
    rows = result['cases']
    cohorts = dict(all349=rows,
        historical149=[r for r in rows if r['historical_subset']],
        additional200=[r for r in rows if not r['historical_subset']],
        informative319=[r for r in rows if not r['single_eligible_class']],
        constant30=[r for r in rows if r['single_eligible_class']])
    if [len(cohorts[k]) for k in ('all349','historical149','additional200','informative319','constant30')] != [349,149,200,319,30]:
        raise ValueError('Documented expansion/degeneracy cohorts changed')
    for name, active_set in (('affine_palette',False),('ambiguous_palette',True)):
        cohorts[name] = [r for r in rows if not r['single_eligible_class'] and
                        r['methods']['C17'].get('details',{}).get('active_set_required') is active_set]
    if len(cohorts['affine_palette'])+len(cohorts['ambiguous_palette']) != 319:
        raise ValueError('Nonconstant parent palette branch is undocumented')
    children = {}
    for item in children_specs:
        child, parent = item['id'], item['parent']
        per_case = []
        for row in rows:
            c, p = row['methods'][child], row['methods'][parent]
            per_case.append(dict(case=row['case'], dataset=row['dataset'],
                coordinate_units=row['coordinate_units'], single_eligible_class=row['single_eligible_class'],
                iou_delta=(c['score']['macro_iou']-p['score']['macro_iou'])
                if c['status'] == p['status'] == 'ok' else None))
        children[child] = dict(parent=parent, title=item['title'],
            cohorts={name:delta_summary(cases,child,parent,policy) for name,cases in cohorts.items()},
            datasets={d:delta_summary([r for r in rows if r['dataset']==d],child,parent,policy)
                      for d in sorted({r['dataset'] for r in rows})}, per_case=per_case)
    return dict(cycle=cycle, cases=349, historical_parent_score_comparisons=checked,
        historical_score_fields_exact=list(SCORE_CORE),
        historical_accuracy_definition_unchanged=True, children=children,
        palette_branch_cases={k:len(cohorts[k]) for k in ('affine_palette','ambiguous_palette')},
        palette_child_interpretation=('C17 uses a direct pixel affine inverse when identifiable. '
            'N13 instead decodes categorical codewords densely; N14/N15 use occupied24-bin '
            'colour averages and a coarse prior under their stated alternative forward models. '
            'These are complete implementations, not isolated loss ablations.' if cycle == 10 else
            'C17 uses a direct pixel affine inverse when identifiable. '
            'Palette children instead use24-bin occupied-colour nodes and their stated decision objective '
            'on those cases. These are comparisons of complete implementations, not isolated '
            'likelihood ablations; affine and ambiguous cohorts are reported separately.'),
        protocol_sha256=digest(root/'PROTOCOL.json'), results_sha256=digest(root/'RESULTS.json'),
        previous_results_sha256=digest(previous/'RESULTS.json'), audit_sha256=digest(audit_path),
        analysis_source_sha256=digest(Path(__file__)), runtime_policy=policy,
        interpretation='Paired development differences, not independent validation or significance evidence. '
            'No cross-cycle speedup claim; utility differences use only current-cycle per-image times. '
            'Constant evidence uses a common fallback; it is not a refinement gain.',
        predictions_modified=False, scores_modified=False, visual_reviews_created=False,
        survivors_selected=False)
