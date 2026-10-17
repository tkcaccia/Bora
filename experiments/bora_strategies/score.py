"""Paired native scoring after every strategy prediction is frozen."""
import argparse,csv,json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import cv2
import numpy as np
import tifffile
from threadpoolctl import threadpool_limits
from experiments.virchow2_inverse.common import digest,save_json
from experiments.virchow2_smoothing.run import verify
from experiments.virchow2_multi.score import Evaluator
from .registry import STRATEGIES


def summarize(cases,conversions):
    datasets=sorted({r['dataset'] for r in cases});rng=np.random.default_rng(7730)
    indices={d:rng.integers(0,len([r for r in cases if r['dataset']==d]),size=(3000,len([r for r in cases if r['dataset']==d]))) for d in datasets}
    def value(row,key,metric):
        a=[row['conversions'][m]['strategies'][key]['score']['boundary_f1_at_um']['8'] if metric=='boundary' else row['conversions'][m]['strategies'][key]['score'][metric] for m in conversions]
        a=[x for x in a if x is not None];return float(np.mean(a)) if a else None
    ranking=[]
    for key,(title,_) in STRATEGIES.items():
        per={};draws=[];deltas=[]
        for d in datasets:
            subset=[r for r in cases if r['dataset']==d];a=np.array([value(r,key,'macro_iou') for r in subset]);b=np.array([value(r,'B01','macro_iou') for r in subset]);delta=a-b
            bf=[value(r,key,'boundary') for r in subset];bf=[v for v in bf if v is not None]
            per[d]=dict(cases=len(subset),macro_iou=float(a.mean()),delta_vs_current=float(delta.mean()),boundary_f1_8um=float(np.mean(bf)) if bf else None)
            draws.append(delta[indices[d]].mean(1));deltas.extend(delta)
        deltas=np.array(deltas);boundaries=[r['boundary_f1_8um'] for r in per.values() if r['boundary_f1_8um'] is not None]
        ranking.append(dict(strategy=key,title=title,dataset_balanced_macro_iou=float(np.mean([r['macro_iou'] for r in per.values()])),
            delta_vs_current=float(np.mean([r['delta_vs_current'] for r in per.values()])),paired_delta_bootstrap95=np.percentile(np.mean(draws,axis=0),[2.5,97.5]).tolist(),
            boundary_f1_8um=float(np.mean(boundaries)) if boundaries else None,wins=int((deltas>1e-8).sum()),losses=int((deltas<-1e-8).sum()),ties=int((abs(deltas)<=1e-8).sum()),per_dataset=per,
            expert_tissue_dataset_balanced_iou=float(np.mean([r['macro_iou'] for d,r in per.items() if d!='camelyon16-subset']))))
    ranking.sort(key=lambda r:-r['dataset_balanced_macro_iou'])
    for i,r in enumerate(ranking):r['rank']=i+1
    return ranking


def score_case(case,root,source,protocol,prior):
    originals=source.parent/protocol['original_folder']
    previous={case:prior}
    csvrows=[]
    with threadpool_limits(limits=2):
        out=root/'cases'/case;sample=json.loads((out/'SAMPLE.json').read_text());path=originals/'cases'/case/'reference.native.tif'
        if digest(path)!=sample['reference_native_sha256']:raise ValueError('Original reference changed')
        truth=tifffile.imread(path);evaluator=Evaluator(truth,sample['native_mpp_xy'])
        tissue=tifffile.imread(source/'cases'/case/'tissue.native.tif')>0
        coarse=tifffile.imread(source/'cases'/case/'coarse.completed.tif');coarse=cv2.resize(coarse,tuple(sample['native_shape_yx'][::-1]),interpolation=cv2.INTER_NEAREST)
        conversions={}
        for conversion in protocol['conversions']:
            folder=out/conversion;f=json.loads((folder/'FROZEN.json').read_text());scorepath=folder/'scores.json'
            if scorepath.exists():
                record=json.loads(scorepath.read_text())
                if record['prediction_manifest_sha256']!=digest(folder/'FROZEN.json'):raise ValueError('Score cache mismatch')
            else:
                methods={};baseline=previous[case]['methods'][conversion]['after'];current=tifffile.imread(folder/'B01/bora.native.tif')
                for key in STRATEGIES:
                    pred=tifffile.imread(folder/key/'bora.native.tif');s=evaluator.score(pred)
                    if key=='B01' and (abs(s['macro_iou']-baseline['macro_iou'])>1e-14 or abs(s['macro_dice']-baseline['macro_dice'])>1e-14):raise ValueError('Current score mismatch')
                    methods[key]=dict(score=s,delta_vs_current=s['macro_iou']-baseline['macro_iou'],seconds=f['records'][key]['seconds'],
                        changed_tissue_percent_vs_current=float(100*np.mean(pred[tissue]!=current[tissue])),changed_tissue_percent_vs_tiles=float(100*np.mean(pred[tissue]!=coarse[tissue])),
                        missing_predicted_classes=sorted(set(map(int,evaluator.classes))-set(map(int,np.unique(pred[evaluator.valid])))))
                record=dict(strategies=methods,prediction_manifest_sha256=digest(folder/'FROZEN.json'),analysis_mpp_xy=f['analysis_mpp_xy'],analysis_shape_yx=f['analysis_shape_yx'])
                save_json(scorepath,record)
            conversions[conversion]=record
            for key,row in record['strategies'].items():
                s=row['score'];csvrows.append(dict(case=case,dataset=sample['dataset'],conversion=conversion,strategy=key,
                    macro_iou=s['macro_iou'],macro_dice=s['macro_dice'],boundary_f1_4um=s['boundary_f1_at_um']['4'],boundary_f1_8um=s['boundary_f1_at_um']['8'],boundary_f1_16um=s['boundary_f1_at_um']['16'],
                    delta_vs_current=row['delta_vs_current'],changed_tissue_percent=row['changed_tissue_percent_vs_current'],seconds=row['seconds'],missing_predicted_classes=','.join(map(str,row['missing_predicted_classes']))))
            print(json.dumps(dict(scored=case,conversion=conversion)),flush=True)
        row=dict(case=case,dataset=sample['dataset'],sample=sample,coverage=previous[case]['coverage'],conversions=conversions)
        save_json(out/'scores.json',row)
    return row,csvrows


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,choices=(1,2,3),default=3);p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    protocol=json.loads((a.root/'PROTOCOL.json').read_text());frozen=json.loads((a.root/'ALL_PREDICTIONS_FROZEN.json').read_text())
    if digest(a.root/'PROTOCOL.json')!=frozen['protocol_sha256'] or digest(a.source/'ALL_PREDICTIONS_FROZEN.json')!=protocol['source_frozen_sha256']:raise ValueError('Protocol changed')
    for pair,sha in frozen['conversion_manifest_sha256'].items():
        folder=a.root/'cases'/pair
        if digest(folder/'FROZEN.json')!=sha:raise ValueError('Prediction manifest changed')
        verify(folder,json.loads((folder/'FROZEN.json').read_text()))
    originals=a.source.parent/protocol['original_folder'];previous={r['case']:r for r in json.loads((a.source/'results.json').read_text())['cases']}
    cases=[];csvrows=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=get_context('spawn')) as pool:
        futures=[pool.submit(score_case,case,a.root,a.source,protocol,previous[case]) for case in protocol['cases']]
        for future in futures:
            row,records=future.result();cases.append(row);csvrows.extend(records)
    results=dict(cases=cases,protocol=protocol,ranking=summarize(cases,protocol['conversions']),
        per_conversion={m:summarize(cases,[m]) for m in protocol['conversions']},
        metric='native macro IoU; average seven conversions per case, cases within dataset, equal four dataset weights',
        uncertainty='3000 paired case resamples within dataset; conversions stay together; exploratory reused cases')
    save_json(a.root/'results.json',results)
    with (a.root/'scores.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    save_json(a.root/'SCORING_AUDIT.json',dict(all_predictions_verified_before_fine_reference_read=True,native_scores=len(csvrows),
        original_quality_support_unchanged=True,current_baseline_reproduced=True,unannotated_tissue_unscored=True))


if __name__=='__main__':main()
