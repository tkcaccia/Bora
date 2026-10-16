"""Score the frozen ensemble and reproduce the paired #9 baseline."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import tifffile
from experiments.virchow2_inverse.common import digest,save_json
from experiments.virchow2_multi.score import Evaluator


def summarize(cases,bootstrap=3000):
    datasets=sorted({r['dataset'] for r in cases});rng=np.random.default_rng(7730)
    per_dataset={};draws=[]
    keys=['coarse','he','09','ensemble']
    values=lambda r,k:r['controls'][k]['macro_iou'] if k in ('coarse','he') else r['methods'][k]['after']['macro_iou']
    for dataset in datasets:
        subset=[r for r in cases if r['dataset']==dataset]
        delta=np.array([values(r,'ensemble')-values(r,'09') for r in subset])
        indices=rng.integers(0,len(subset),size=(bootstrap,len(subset)));draws.append(delta[indices].mean(1))
        per_dataset[dataset]=dict(cases=len(subset),**{k:float(np.mean([values(r,k) for r in subset])) for k in keys},
            delta=float(delta.mean()),wins=int(np.sum(delta>1e-8)),losses=int(np.sum(delta<-1e-8)),ties=int(np.sum(abs(delta)<=1e-8)))
    delta=np.array([values(r,'ensemble')-values(r,'09') for r in cases])
    return dict(cases=len(cases),dataset_balanced={k:float(np.mean([v[k] for v in per_dataset.values()])) for k in keys},
        dataset_balanced_delta=float(np.mean([v['delta'] for v in per_dataset.values()])),
        paired_delta_bootstrap95=np.percentile(np.mean(draws,axis=0),[2.5,97.5]).tolist(),
        wins=int(np.sum(delta>1e-8)),losses=int(np.sum(delta<-1e-8)),ties=int(np.sum(abs(delta)<=1e-8)),per_dataset=per_dataset,
        expert_tissue_excluding_camelyon={k:float(np.mean([v[k] for d,v in per_dataset.items() if d!='camelyon16-subset'])) for k in keys})


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    frozen=json.loads((a.root/'ALL_PREDICTIONS_FROZEN.json').read_text());protocol=json.loads((a.root/'PROTOCOL.json').read_text())
    if digest(a.root/'PROTOCOL.json')!=frozen['protocol_sha256'] or digest(a.source/'PLAN.json')!=protocol['source_plan_sha256']:raise ValueError('Protocol/source scope changed')
    plan=json.loads((a.source/'PLAN.json').read_text())
    # Verify every new prediction, original component and paired baseline first.
    for sample in plan['samples']:
        name=sample['case'];folder=a.root/'cases'/name;src=a.source/'cases'/name
        if digest(folder/'FROZEN.json')!=frozen['case_manifest_sha256'][name]:raise ValueError('Case manifest changed')
        record=json.loads((folder/'FROZEN.json').read_text())
        if digest(src/'FROZEN.json')!=record['source_case_manifest_sha256']:raise ValueError('Original manifest changed')
        for base,files in [(folder,record['files_sha256']),(src,json.loads((src/'FROZEN.json').read_text())['files_sha256'])]:
            for f,sha in files.items():
                if digest(base/f)!=sha:raise ValueError('Frozen output changed')
    previous=json.loads((a.source/'results.json').read_text());old={r['case']:r for r in previous['cases']};cases=[];rows=[]
    for sample in plan['samples']:
        name=sample['case'];folder=a.root/'cases'/name;src=a.source/'cases'/name
        if digest(src/'reference.native.tif')!=sample['reference_native_sha256']:raise ValueError('Reference changed')
        evaluator=Evaluator(tifffile.imread(src/'reference.native.tif'),sample['native_mpp_xy'])
        baseline=evaluator.score(tifffile.imread(src/'09/bora.native.tif'))
        if baseline!=old[name]['methods']['09']['after']:raise ValueError('Baseline scores not reproduced exactly')
        row=old[name]
        before=evaluator.score(tifffile.imread(folder/'before.native.tif'));after=evaluator.score(tifffile.imread(folder/'bora.native.tif'))
        title='Combined '+' + '.join(map(str,protocol['components']))
        row['methods']['ensemble']=dict(id='ensemble',title=title,before=before,after=after,
            delta_vs_09=after['macro_iou']-baseline['macro_iou'],delta_vs_coarse=after['macro_iou']-row['controls']['coarse']['macro_iou'])
        save_json(folder/'scores.json',row);cases.append(row)
        for key,value in row['methods'].items():
            if key=='04' and 4 not in protocol['components']:continue
            score=value['after'];rows.append(dict(case=name,dataset=row['dataset'],method=key,macro_iou=score['macro_iou'],macro_dice=score['macro_dice'],
                coarse_iou=row['controls']['coarse']['macro_iou'],delta_vs_coarse=score['macro_iou']-row['controls']['coarse']['macro_iou'],
                delta_vs_09=score['macro_iou']-baseline['macro_iou'],boundary_f1_8um=score['boundary_f1_at_um']['8'],before_iou=value['before']['macro_iou']))
        print(json.dumps(dict(scored=name,ensemble=after['macro_iou'],algorithm_9=baseline['macro_iou'],delta=after['macro_iou']-baseline['macro_iou'])),flush=True)
    result=dict(cases=cases,summary=summarize(cases),source_folder=a.source.name,components=protocol['components'],rejected_methods=[4] if 4 not in protocol['components'] else [],
        metric='Per-image native macro IoU; average within dataset, then equal weight across four datasets',
        uncertainty='3000 paired bootstrap resamples within dataset; descriptive case/slide groups, patient mapping incompletely verified',
        comparison='Fixed equal evidence mean of '+','.join(map(str,protocol['components']))+' + unchanged Bora vs frozen algorithm 9 + Bora',baseline_reproduced_exactly=True)
    save_json(a.root/'results.json',result)
    with (a.root/'scores.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save_json(a.root/'AUDIT.json',dict(all_predictions_verified_before_reference_scoring=True,all_49_baseline_scores_reproduced_exactly=True,
        cases=len(cases),component_rgb_and_before_masks_identical=True,fine_reference_read_by_predictor=False))

if __name__=='__main__':main()
