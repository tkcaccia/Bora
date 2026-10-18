"""Compare frozen pilot/full evidence without re-running or re-scoring it.

This is scoped repeatability on the actual overlap, never proof of an unseen
full-cohort rerun. Timings and repeat counts are deliberately not compared.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tifffile

from .paired_eight import SCORE_CORE
from .run import digest,save,verify


REPORT_ONLY={'experiments/bora_cycles/'+name for name in (
    'posthoc_calibrated.py','selected_viewer.py','selected_assets.py','publish_diagnostics.py',
    'numerical_nine.py','export_planned_crops.py','audit_repeatability.py','finish_calibrated.py')}


def load(path):return json.loads(Path(path).read_text())


def frozen_run(root):
    protocol=load(root/'PROTOCOL.json');frozen=load(root/'ALL_PREDICTIONS_FROZEN.json')
    result=load(root/'RESULTS.json')
    if (result['protocol']!=protocol or frozen['protocol_sha256']!=digest(root/'PROTOCOL.json')
            or frozen['fine_reference_read'] or set(frozen['manifests'])!=set(protocol['cases'])
            or {r['case'] for r in result['cases']}!=set(protocol['cases'])
            or len(result['cases'])!=len(protocol['cases'])):
        raise ValueError('Changed or incomplete prediction/score freeze')
    for name,expected in protocol['source_hashes'].items():
        if digest(root/'runtime_sources'/name)!=expected:raise ValueError('Archived execution source changed')
    return protocol,frozen,{r['case']:r for r in result['cases']}


def compare(full,pilot):
    full,pilot=Path(full),Path(pilot)
    p,f,rows=frozen_run(full);q,g,old=frozen_run(pilot)
    if (p['pilot'] or not q['pilot'] or p['cycle']!=q['cycle']
            or p['input_plan_sha256']!=q['input_plan_sha256'] or p['methods']!=q['methods']
            or not set(q['cases'])<=set(p['cases']) or not q['cases']
            or p['primary_conversion']!=q['primary_conversion']
            or p.get('benchmark_scope')!=q.get('benchmark_scope')
            or p.get('study_protocol')!=q.get('study_protocol')):
        raise ValueError('Pilot/full study contract or overlap differs')
    # Changed output paths and report code cannot justify changing prediction code.
    source_names=set(p['source_hashes'])|set(q['source_hashes'])
    changed=sorted(n for n in source_names if p['source_hashes'].get(n)!=q['source_hashes'].get(n))
    if set(changed)-REPORT_ONLY:raise ValueError('Prediction source differs between pilot and full')
    counts=dict(native_masks=0,native_supports=0,accuracy_core_comparisons=0,rarity_arrays=0)
    records=[]
    for case in q['cases']:
        a,b=rows[case],old[case]
        if a['sample']!=b['sample']:raise ValueError('Pilot/full case geometry, ontology or sample identity changed')
        x,y=full/'cases'/case,pilot/'cases'/case
        if digest(x/'PREDICTIONS.json')!=f['manifests'][case] or digest(y/'PREDICTIONS.json')!=g['manifests'][case]:
            raise ValueError('Prediction manifest changed')
        first,second=load(x/'PREDICTIONS.json'),load(y/'PREDICTIONS.json')
        verify(x,first['files_sha256']);verify(y,second['files_sha256'])
        for key in ('input_manifest_sha256','rgb_sha256'):
            if first[key]!=second[key]:raise ValueError('Pilot/full frozen inputs differ')
        for method in p['methods']:
            aa,bb=a['methods'][method],b['methods'][method]
            if aa['status']!='ok' or bb['status']!='ok':raise ValueError('Repeatability requires successful overlapping methods')
            for field in SCORE_CORE:
                if aa['score'][field]!=bb['score'][field]:raise ValueError('Pilot/full score differs: '+case+'/'+method+'/'+field)
            counts['accuracy_core_comparisons']+=1
            if not np.array_equal(tifffile.imread(x/method/'native.tif'),tifffile.imread(y/method/'native.tif')):
                raise ValueError('Pilot/full native labels differ: '+case+'/'+method)
            counts['native_masks']+=1
            support=method+'/support.native.tif'
            if (support in first['files_sha256'])!=(support in second['files_sha256']):
                raise ValueError('Pilot/full support contract differs')
            if support in first['files_sha256']:
                if not np.array_equal(tifffile.imread(x/support),tifffile.imread(y/support)):
                    raise ValueError('Pilot/full native support differs: '+case+'/'+method)
                counts['native_supports']+=1
        for diagnostic in ('NOVELTY.npz','RARITY.npz'):
            if (diagnostic in first['files_sha256'])!=(diagnostic in second['files_sha256']):
                raise ValueError('Diagnostic freeze differs')
            if diagnostic not in first['files_sha256']:continue
            with np.load(x/diagnostic) as aa,np.load(y/diagnostic) as bb:
                if aa.files!=bb.files or any(not np.array_equal(aa[k],bb[k]) for k in aa.files):
                    raise ValueError('Pilot/full annotation-free proposal differs: '+case+'/'+diagnostic)
                counts['rarity_arrays']+=len(aa.files) if diagnostic=='RARITY.npz' else 0
        records.append(dict(case=case,methods=len(p['methods']),same_frozen_inputs=True,
            all_native_labels_equal=True,all_score_core_fields_equal=True))
    return dict(cycle=p['cycle'],full_cases=len(p['cases']),pilot_cases=len(q['cases']),
        **counts,score_core_fields=list(SCORE_CORE),reporting_only_source_changes=changed,records=records,
        full_protocol_sha256=digest(full/'PROTOCOL.json'),pilot_protocol_sha256=digest(pilot/'PROTOCOL.json'),
        full_results_sha256=digest(full/'RESULTS.json'),pilot_results_sha256=digest(pilot/'RESULTS.json'),
        source_sha256=digest(Path(__file__)),timings_compared=False,predictions_modified=False,
        scores_modified=False,visual_reviews_created=False,selection_created=False,
        interpretation='Exact frozen pilot/full overlap only. Does not prove an independent full-cohort rerun, anatomical validity, convergence, or controlled timing.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('full','pilot','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve existing audit evidence')
    result=compare(args.full,args.pilot);save(args.output,result)
    print(json.dumps({k:result[k] for k in ('full_cases','pilot_cases','native_masks','native_supports','accuracy_core_comparisons')}))


if __name__=='__main__':main()
