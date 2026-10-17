"""Native scores against unchanged annotation support; paired tile-mask controls."""
import argparse,csv,json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import numpy as np
import tifffile
from threadpoolctl import threadpool_limits
from experiments.virchow2_inverse.common import digest,save_json
from experiments.virchow2_multi.score import Evaluator
from .prepare import verify
from .registry import STRATEGIES


def summarize(cases,conversions):
    datasets=sorted({r['dataset'] for r in cases});rng=np.random.default_rng(7730)
    indices={d:rng.integers(0,sum(r['dataset']==d for r in cases),size=(3000,sum(r['dataset']==d for r in cases))) for d in datasets}
    def value(row,key,metric):
        if key=='tiles':return row['tiles'][metric]
        values=[row['conversions'][m]['strategies'][key]['score']['boundary_f1_at_um']['8'] if metric=='boundary' else row['conversions'][m]['strategies'][key]['score'][metric] for m in conversions]
        values=[v for v in values if v is not None];return float(np.mean(values)) if values else None
    ranking=[]
    for key,(title,_) in STRATEGIES.items():
        per={};draws=[];deltas=[]
        for d in datasets:
            rows=[r for r in cases if r['dataset']==d];a=np.array([value(r,key,'macro_iou') for r in rows]);b=np.array([value(r,'tiles','macro_iou') for r in rows]);delta=a-b
            bf=[value(r,key,'boundary') for r in rows];bf=[v for v in bf if v is not None]
            per[d]=dict(cases=len(rows),macro_iou=float(a.mean()),tiles_iou=float(b.mean()),delta_vs_tiles=float(delta.mean()),boundary_f1_8um=float(np.mean(bf)) if bf else None)
            draws.append(delta[indices[d]].mean(1));deltas.extend(delta)
        deltas=np.array(deltas);bf=[r['boundary_f1_8um'] for r in per.values() if r['boundary_f1_8um'] is not None]
        ranking.append(dict(strategy=key,title=title,dataset_balanced_macro_iou=float(np.mean([r['macro_iou'] for r in per.values()])),
            delta_vs_tiles=float(np.mean([r['delta_vs_tiles'] for r in per.values()])),paired_delta_bootstrap95=np.percentile(np.mean(draws,axis=0),[2.5,97.5]).tolist(),
            boundary_f1_8um=float(np.mean(bf)) if bf else None,wins=int((deltas>1e-8).sum()),losses=int((deltas<-1e-8).sum()),ties=int((abs(deltas)<=1e-8).sum()),per_dataset=per))
    ranking.sort(key=lambda r:-r['dataset_balanced_macro_iou'])
    for i,row in enumerate(ranking):row['rank']=i+1
    return ranking


def score_case(case,root,source,protocol,strategies=None):
    strategies=STRATEGIES if strategies is None else strategies
    cvrows=[];out=root/'cases'/case;src=source/'cases'/case;sample=json.loads((out/'SAMPLE.json').read_text())
    if digest(src/'reference.native.tif')!=sample['reference_native_sha256']:raise ValueError('Reference changed')
    truth=tifffile.imread(src/'reference.native.tif');evaluator=Evaluator(truth,sample['native_mpp_xy']);tissue=tifffile.imread(src/'tissue.native.tif')>0
    tiles=tifffile.imread(out/'tiles.native.tif');baseline=evaluator.score(tiles);conversions={}
    with threadpool_limits(limits=2):
        for m in protocol['conversions']:
            folder=out/m;frozen=json.loads((folder/'FROZEN.json').read_text());methods={}
            for key in strategies:
                pred=tifffile.imread(folder/key/'bora.native.tif');score=evaluator.score(pred)
                methods[key]=dict(score=score,delta_vs_tiles=score['macro_iou']-baseline['macro_iou'],seconds=frozen['records'][key]['seconds'],changed_tissue_percent_vs_tiles=float(100*np.mean(pred[tissue]!=tiles[tissue])),
                    missing_predicted_classes=sorted(set(map(int,evaluator.classes))-set(map(int,np.unique(pred[evaluator.valid])))))
                cvrows.append(dict(case=case,dataset=sample['dataset'],cohort=sample['benchmark_cohort'],conversion=m,strategy=key,macro_iou=score['macro_iou'],macro_dice=score['macro_dice'],
                    tile_macro_iou=baseline['macro_iou'],delta_vs_tiles=methods[key]['delta_vs_tiles'],boundary_f1_8um=score['boundary_f1_at_um']['8'],seconds=methods[key]['seconds']))
            conversions[m]=dict(strategies=methods,prediction_manifest_sha256=digest(folder/'FROZEN.json'),analysis_mpp_xy=frozen['analysis_mpp_xy'])
            save_json(folder/'scores.json',conversions[m])
    row=dict(case=case,dataset=sample['dataset'],cohort=sample['benchmark_cohort'],sample=sample,tiles=baseline,coverage=dict(annotated_pixels=int(evaluator.valid.sum()),detected_tissue_pixels=int(tissue.sum()),unannotated_tissue_pixels=int((tissue&~evaluator.valid).sum())),conversions=conversions)
    save_json(out/'scores.json',row);print(json.dumps(dict(scored=case)),flush=True);return row,cvrows


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    protocol=json.loads((a.root/'PROTOCOL.json').read_text());frozen=json.loads((a.root/'ALL_PREDICTIONS_FROZEN.json').read_text())
    if digest(a.root/'PROTOCOL.json')!=frozen['protocol_sha256'] or digest(a.source/'ALL_INPUTS_FROZEN.json')!=protocol['inputs_sha256']:raise ValueError('Protocol changed')
    for pair,sha in frozen['conversion_manifest_sha256'].items():
        folder=a.root/'cases'/pair
        if digest(folder/'FROZEN.json')!=sha:raise ValueError('Prediction manifest changed')
        record=json.loads((folder/'FROZEN.json').read_text());verify(folder,record)
        if digest(folder.parent/'tiles.native.tif')!=record['tiles_native_sha256']:raise ValueError('Tile baseline changed')
    cases=[];csvrows=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=get_context('spawn')) as pool:
        for row,records in pool.map(score_case,protocol['cases'],[a.root]*len(protocol['cases']),[a.source]*len(protocol['cases']),[protocol]*len(protocol['cases'])):cases.append(row);csvrows.extend(records)
    fresh=[r for r in cases if r['cohort']=='new_extension'];old=[r for r in cases if r['cohort']=='previously_inspected']
    results=dict(cases=cases,protocol=protocol,ranking=summarize(cases,protocol['conversions']),new_image_ranking=summarize(fresh,protocol['conversions']),previous_image_ranking=summarize(old,protocol['conversions']),
        per_conversion={m:summarize(cases,[m]) for m in protocol['conversions']},new_per_conversion={m:summarize(fresh,[m]) for m in protocol['conversions']},
        metric='mean native macro IoU over five conversions per case, then cases within dataset, then equal datasets',uncertainty='3000 paired case resamples within dataset; conversions kept together; coarse-reference-derived exploratory supervision')
    save_json(a.root/'results.json',results)
    with (a.root/'scores.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    save_json(a.root/'SCORING_AUDIT.json',dict(all_predictions_verified_before_fine_reference_read=True,native_scores=len(csvrows),original_quality_support_unchanged=True,unannotated_tissue_unscored=True))

if __name__=='__main__':main()
