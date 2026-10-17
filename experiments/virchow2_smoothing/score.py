"""Score identical frozen inputs; bootstrap cases, not correlated conversions."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import tifffile
from threadpoolctl import threadpool_limits
from experiments.virchow2_inverse.common import digest,save_json
from experiments.virchow2_multi.score import Evaluator
from .run import PRIMARY,METHODS,VARIANTS,verify


def summary(cases,method=None):
    datasets=sorted({r['dataset'] for r in cases});rng=np.random.default_rng(7730)
    indices={d:rng.integers(0,len([r for r in cases if r['dataset']==d]),size=(3000,len([r for r in cases if r['dataset']==d]))) for d in datasets}
    rows=[];keys=[method] if method else PRIMARY
    def value(r,v,metric):
        vals=[r['methods'][m][v]['score']['boundary_f1_at_um']['8'] if metric=='boundary' else r['methods'][m][v]['score'][metric] for m in keys]
        vals=[x for x in vals if x is not None]
        return float(np.mean(vals)) if vals else None
    for v,(title,_) in VARIANTS.items():
        per={};draws=[];deltas=[]
        for d in datasets:
            sub=[r for r in cases if r['dataset']==d]
            a=np.array([value(r,v,'macro_iou') for r in sub]);b=np.array([value(r,'median9','macro_iou') for r in sub]);delta=a-b
            bf=[value(r,v,'boundary') for r in sub];bf=[x for x in bf if x is not None]
            per[d]=dict(cases=len(sub),macro_iou=float(a.mean()),delta=float(delta.mean()),boundary_f1_8um=float(np.mean(bf)) if bf else None)
            draws.append(delta[indices[d]].mean(axis=1));deltas.extend(delta)
        delta=np.array(deltas);boundary=[r['boundary_f1_8um'] for r in per.values() if r['boundary_f1_8um'] is not None]
        rows.append(dict(variant=v,title=title,dataset_balanced_macro_iou=float(np.mean([r['macro_iou'] for r in per.values()])),
            dataset_balanced_boundary_f1_8um=float(np.mean(boundary)) if boundary else None,
            delta_vs_current=float(np.mean([r['delta'] for r in per.values()])),paired_delta_bootstrap95=np.percentile(np.mean(draws,axis=0),[2.5,97.5]).tolist(),
            wins=int((delta>1e-8).sum()),losses=int((delta<-1e-8).sum()),ties=int((abs(delta)<=1e-8).sum()),per_dataset=per))
    rows.sort(key=lambda r:-r['dataset_balanced_macro_iou'])
    for i,r in enumerate(rows):r['rank']=i+1
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    frozen=json.loads((a.root/'ALL_PREDICTIONS_FROZEN.json').read_text());protocol=json.loads((a.root/'PROTOCOL.json').read_text())
    if digest(a.root/'PROTOCOL.json')!=frozen['protocol_sha256'] or digest(a.source/'ALL_PREDICTIONS_FROZEN.json')!=protocol['source_frozen_sha256']:raise ValueError('Frozen protocol changed')
    originals=a.source.parent/protocol['original_folder'];old={r['case']:r for r in json.loads((a.source/'results.json').read_text())['cases']}
    # No fine-reference reads until every prediction in every case is verified.
    for case,sha in frozen['case_manifest_sha256'].items():
        folder=a.root/'cases'/case
        if digest(folder/'FROZEN.json')!=sha:raise ValueError('Manifest changed')
        verify(folder,json.loads((folder/'FROZEN.json').read_text()))
    rows=[];csvrows=[]
    with threadpool_limits(limits=2):
        for case in protocol['cases']:
            folder=a.root/'cases'/case;record=json.loads((folder/'FROZEN.json').read_text());sample=record['sample']
            if digest(originals/'cases'/case/'reference.native.tif')!=sample['reference_native_sha256']:raise ValueError('Reference changed')
            if (folder/'scores.json').exists():
                row=json.loads((folder/'scores.json').read_text())
            else:
                evaluator=Evaluator(tifffile.imread(originals/'cases'/case/'reference.native.tif'),sample['native_mpp_xy']);methods={}
                for m in METHODS:
                    methods[m]={}
                    for v in VARIANTS:
                        mask=tifffile.imread(folder/m/v/'bora.native.tif');score=evaluator.score(mask)
                        baseline=old[case]['controls'][m] if m in ('he','features') else old[case]['methods'][m]['after']
                        if v=='median9' and abs(score['macro_iou']-baseline['macro_iou'])>1e-14:raise ValueError('Baseline score not reproduced')
                        methods[m][v]=dict(score=score,delta_vs_current=score['macro_iou']-baseline['macro_iou'],**record['diagnostics'][m][v])
                    print(json.dumps(dict(scored=case,method=m)),flush=True)
                row=dict(case=case,dataset=sample['dataset'],sample=sample,coverage=old[case]['coverage'],methods=methods)
                save_json(folder/'scores.json',row)
            rows.append(row)
            for m in METHODS:
                for v in VARIANTS:
                    r=row['methods'][m][v];s=r['score'];csvrows.append(dict(case=case,dataset=sample['dataset'],method=m,variant=v,
                        macro_iou=s['macro_iou'],macro_dice=s['macro_dice'],boundary_f1_4um=s['boundary_f1_at_um']['4'],boundary_f1_8um=s['boundary_f1_at_um']['8'],
                        boundary_f1_16um=s['boundary_f1_at_um']['16'],delta_vs_current=r['delta_vs_current'],changed_native_pixels=r['changed_native_pixels_vs_current']))
    results=dict(cases=rows,ranking=summary(rows),per_method={m:summary(rows,m) for m in METHODS},protocol=protocol,
        uncertainty='3000 paired case resamples within dataset; the seven conversions of each case stay together; descriptive on reused cases')
    save_json(a.root/'results.json',results)
    with (a.root/'scores.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    save_json(a.root/'SCORING_AUDIT.json',dict(all_predictions_frozen_before_scoring=True,original_quality_support_unchanged=True,
        legacy_baseline_reproduced=True,cases=len(rows),scores=len(csvrows)))


if __name__=='__main__':main()
