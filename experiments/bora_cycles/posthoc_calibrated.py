"""Audit and compare calibrated frozen outputs; never manufacture reviews."""
import argparse
import json
from pathlib import Path

from .audit_cycle import audit
from .benchmark_scope import requires_parent_parity,validate_scope
from .input_limits import audit_inputs
from .numerical_nine import summarize
from .paired_eight import SCORE_CORE,delta_summary
from .publish_diagnostics import publish
from .run import digest,save
from .runtime_policy import active_policy
from .small_region_atlas import export


def paired(result,previous,audited):
    protocol=result['protocol'];validate_scope(protocol,pilot=protocol['pilot'])
    cycle=protocol['cycle']
    if cycle<11 or protocol['benchmark_scope']['version']!='calibrated-tissues-v1':
        raise ValueError('A prospective calibrated tissue cycle is required')
    if audited['cases']!=len(protocol['cases']) or audited['methods']!=len(protocol['methods']):
        raise ValueError('Integrity audit scope differs from paired comparison')
    old={r['case']:r for r in previous['cases']};same=changed=0;changes=[]
    rows=result['cases'];parents=protocol['lineage']['unchanged_survivors']
    for row in rows:
        case=row['case']
        if not requires_parent_parity(protocol,case):continue
        for method in parents:
            a,b=row['methods'][method],old[case]['methods'][method]
            if a['status']!=b['status'] or a['status']!='ok':raise ValueError('Matched parent failed')
            different=[k for k in SCORE_CORE if a['score'][k]!=b['score'][k]]
            if cycle==11 and row['dataset']=='ocelot':
                changed+=1
                changes.append(dict(case=case,method=method,changed_fields=different,
                    previous_iou=b['score']['macro_iou'],current_iou=a['score']['macro_iou']))
            else:
                if different:raise ValueError('Unchanged-ontology parent score changed: '+case+'/'+method)
                same+=1
    policy=active_policy(protocol)
    if policy is None:raise ValueError('Runtime penalty must be preregistered')
    groups=dict(all_calibrated=rows,previous_cohort=[r for r in rows if r['historical_subset']],
        new_tissue_datasets=[r for r in rows if not r['historical_subset']])
    children={}
    specs=protocol['lineage']['children']
    if len(specs)!=15 or {r['id'] for r in specs}!=set(protocol['roster'])-set(parents):
        raise ValueError('Frozen child lineage differs from this twenty-candidate roster')
    for spec in specs:
        child,parent=spec['id'],spec['parent']
        children[child]=dict(parent=parent,title=spec['title'],
            cohorts={k:delta_summary(v,child,parent,policy) for k,v in groups.items()},
            datasets={d:delta_summary([r for r in rows if r['dataset']==d],child,parent,policy)
                for d in sorted({r['dataset'] for r in rows})})
    return dict(cycle=cycle,pilot=protocol['pilot'],cases=len(rows),children=children,
        parent_native_parity_comparisons=audited['unchanged_survivor_comparisons'],
        unchanged_ontology_parent_score_comparisons=same,
        prospective_ocelot_rescored_comparisons=changed,ocelot_score_changes=changes,
        interpretation='Current-cohort paired development differences, not independent validation or statistical significance. '
            +('Cycle11 prospectively rescored OCELOT white-only broad negatives. ' if cycle==11 else 'Every retained-parent accuracy core must exactly match the preceding calibrated cycle. ')
            +'No historical mean change or cross-cycle timing difference is an algorithm gain.',
        predictions_modified=False,scores_modified=False,visual_reviews_created=False,survivors_selected=False)


def main():
    import cv2
    cv2.setNumThreads(2)
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','root','previous','executed-code'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args();root=args.root
    protocol=json.loads((root/'PROTOCOL.json').read_text());validate_scope(protocol,pilot=args.pilot)
    if protocol['pilot']!=args.pilot or protocol['cycle']<11 or (root/'SELECTION.json').exists():
        raise ValueError('Unselected calibrated cycle with matching explicit pilot status required')
    state=json.loads((root/'STAGE.json').read_text())
    if state['state']!=('pilot_complete' if args.pilot else 'awaiting_visual_review') or state['cases']!=len(protocol['cases']):
        raise ValueError('Wait for the whole prediction, score and render process')
    names=['INPUT_LIMITS.json','SMALL_REGION_ATLAS.json','FULL_AUDIT.json','PAIRED_ANALYSIS.json','NUMERICAL_STATUS.json']
    if any((root/n).exists() for n in names+['POSTHOC_COMPLETE.json']):raise ValueError('Use fresh reporting evidence')
    checked=audit(args.source,root,args.previous,args.executed_code)
    expected_parity=5*sum(requires_parent_parity(protocol,c) for c in protocol['cases'])
    if (checked['failed_methods'] or checked['native_masks']!=23*len(protocol['cases'])
            or checked['unchanged_survivor_comparisons']!=expected_parity):
        raise ValueError('Complete23-method mask and matched parent audit required')
    save(root/names[2],checked)
    save(root/names[0],audit_inputs(args.source,root))
    export(args.source,root,root/names[1])
    publish(args.source,root)
    result=json.loads((root/'RESULTS.json').read_text());old=json.loads((args.previous/'RESULTS.json').read_text())
    selection=json.loads((args.previous/'SELECTION.json').read_text())
    if digest(args.previous/'RESULTS.json')!=selection['results_sha256']:raise ValueError('Signed parent scores changed')
    save(root/names[3],paired(result,old,checked));save(root/names[4],summarize(root))
    names.append('diagnostics-v2/MANIFEST.json')
    save(root/'POSTHOC_COMPLETE.json',dict(cycle=protocol['cycle'],pilot=args.pilot,cases=len(protocol['cases']),
        native_masks=checked['native_masks'],report_sha256={n:digest(root/n) for n in names},
        reporting_source_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
        primary_protocol_sha256=digest(root/'PROTOCOL.json'),primary_results_sha256=digest(root/'RESULTS.json'),
        actual_reviews_required=len(protocol['cases']),automated_audits_are_not_visual_review=True,
        visual_reviews_created=False,survivors_selected=False,completed_cycles=protocol['cycle']-1,complete_goal=False))
    print(json.dumps(dict(posthoc='complete',pilot=args.pilot,cases=len(protocol['cases']),awaiting_actual_visual_review=True)),flush=True)


if __name__=='__main__':main()
