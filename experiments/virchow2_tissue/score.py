"""Annotation-only scores, never intersected with RGB tissue estimates."""
import argparse,csv,json
from pathlib import Path
import cv2
import numpy as np
import tifffile
from experiments.virchow2_inverse.common import digest,save_json
from experiments.virchow2_multi.score import Evaluator
from experiments.virchow2_ensemble.score import summarize
from .run import ACTIVE,TITLES


def overlap(evaluator,pred):
    truth=evaluator.truth;valid=evaluator.valid;rows={}
    for g in evaluator.classes:
        a=(pred==g)&valid;b=truth==g;i=int((a&b).sum());u=int((a|b).sum());n=int(a.sum()+b.sum())
        rows[str(int(g))]=dict(iou=i/u,dice=2*i/n,truth_pixels=int(b.sum()),prediction_pixels=int(a.sum()))
    return dict(macro_iou=float(np.mean([r['iou'] for r in rows.values()])),macro_dice=float(np.mean([r['dice'] for r in rows.values()])),
        worst_class_iou=min(r['iou'] for r in rows.values()),pixel_accuracy=float(np.mean(pred[valid]==truth[valid])),
        per_class=rows,valid_pixels=int(valid.sum()),ignore_code=0,boundary_f1_at_um={str(t):None for t in (4,8,16)})


def ranking(cases):
    datasets=sorted({r['dataset'] for r in cases});rows=[]
    for key in [*[f'{n:02d}' for n in ACTIVE],'ensemble']:
        per_dataset={}
        for d in datasets:
            sub=[r for r in cases if r['dataset']==d]
            per_dataset[d]=dict(cases=len(sub),macro_iou=float(np.mean([r['methods'][key]['after']['macro_iou'] for r in sub])))
        rows.append(dict(method=key,title=TITLES[key],dataset_balanced_macro_iou=float(np.mean([r['macro_iou'] for r in per_dataset.values()])),
            expert_tissue_dataset_balanced_iou=float(np.mean([r['macro_iou'] for d,r in per_dataset.items() if d!='camelyon16-subset'])),per_dataset=per_dataset))
    rows.sort(key=lambda r:-r['dataset_balanced_macro_iou'])
    for i,row in enumerate(rows):row['rank']=i+1
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    frozen=json.loads((a.root/'ALL_PREDICTIONS_FROZEN.json').read_text());protocol=json.loads((a.root/'PROTOCOL.json').read_text())
    if digest(a.root/'PROTOCOL.json')!=frozen['protocol_sha256'] or digest(a.source/'PLAN.json')!=protocol['source_plan_sha256']:raise ValueError('Protocol changed')
    plan=json.loads((a.source/'PLAN.json').read_text())
    for sample in plan['samples']:
        folder=a.root/'cases'/sample['case']
        if digest(folder/'FROZEN.json')!=frozen['case_manifest_sha256'][sample['case']]:raise ValueError('Frozen manifest changed')
        for name,sha in json.loads((folder/'FROZEN.json').read_text())['files_sha256'].items():
            if digest(folder/name)!=sha:raise ValueError('Prediction changed')
    old={r['case']:r for r in json.loads((a.source/'results.json').read_text())['cases']};cases=[];csv_rows=[]
    previous_ensemble=a.source.with_name('virchow2-ensemble-9-19-2026-10-16')
    old_ensemble={r['case']:r for r in json.loads((previous_ensemble/'results.json').read_text())['cases']}
    for sample in plan['samples']:
        name=sample['case'];folder=a.root/'cases'/name;src=a.source/'cases'/name
        if digest(src/'reference.native.tif')!=sample['reference_native_sha256']:raise ValueError('Fine reference changed')
        truth=tifffile.imread(src/'reference.native.tif');evaluator=Evaluator(truth,sample['native_mpp_xy']);tissue=tifffile.imread(folder/'tissue.native.tif')>0
        original=tifffile.imread(src/'coarse.tif');completed=tifffile.imread(folder/'coarse.completed.tif')
        native_size=tuple(sample['native_shape_yx'][::-1]);native_full=cv2.resize(completed,native_size,interpolation=cv2.INTER_NEAREST);native_coarse=native_full.copy();native_coarse[~tissue]=0
        tifffile.imwrite(folder/'coarse.full.native.tif',native_full,compression='deflate')
        tifffile.imwrite(folder/'coarse.native.tif',native_coarse,compression='deflate')
        tifffile.imwrite(folder/'evaluation.native.tif',evaluator.valid.astype(np.uint8),compression='deflate')
        controls=dict(coarse=overlap(evaluator,native_full),coarse_tissue=overlap(evaluator,native_coarse),original_coarse=overlap(evaluator,cv2.resize(original,native_size,interpolation=cv2.INTER_NEAREST)))
        for key in ('he','features'):controls[key]=overlap(evaluator,tifffile.imread(folder/key/'bora.native.tif'))
        methods={}
        for key in [*[f'{n:02d}' for n in ACTIVE],'ensemble']:
            before=overlap(evaluator,tifffile.imread(folder/key/'before.native.tif'));after=evaluator.score(tifffile.imread(folder/key/'bora.native.tif'))
            prior=(old_ensemble[name] if key=='ensemble' else old[name])['methods'][key]['after']['macro_iou']
            methods[key]=dict(id=key,title=TITLES[key],before=before,after=after,previous_iou=prior,delta_vs_previous=after['macro_iou']-prior,
                delta_vs_coarse=after['macro_iou']-controls['coarse']['macro_iou'])
            csv_rows.append(dict(case=name,dataset=sample['dataset'],method=key,macro_iou=after['macro_iou'],macro_dice=after['macro_dice'],
                boundary_f1_8um=after['boundary_f1_at_um']['8'],completed_tile_iou=controls['coarse']['macro_iou'],original_tile_iou=controls['original_coarse']['macro_iou'],
                delta_vs_completed_tiles=methods[key]['delta_vs_coarse'],previous_protocol_iou=prior,delta_vs_previous_protocol=after['macro_iou']-prior,
                annotated_pixels=int(evaluator.valid.sum()),detected_tissue_pixels=int(tissue.sum()),unannotated_tissue_pixels=int((tissue&~evaluator.valid).sum())))
        missing=sorted(set(np.unique(truth))-{0}-set(np.unique(original)))
        row=dict(case=name,dataset=sample['dataset'],group=sample['group'],sample=sample,controls=controls,methods=methods,missing_coarse_classes=list(map(int,missing)),
            coverage=dict(native_pixels=int(truth.size),annotated_pixels=int(evaluator.valid.sum()),detected_tissue_pixels=int(tissue.sum()),
                unannotated_tissue_pixels=int((tissue&~evaluator.valid).sum()),annotated_pixels_outside_detected_tissue=int((evaluator.valid&~tissue).sum()),
                prediction_coverage_of_detected_tissue=1.,prediction_pixels_outside_tissue=0,quality_region='original reference > 0, unchanged by tissue detection'))
        save_json(folder/'scores.json',row);cases.append(row);print(json.dumps(dict(scored=name,unannotated_tissue_pixels=row['coverage']['unannotated_tissue_pixels'])),flush=True)
    results=dict(cases=cases,ranking=ranking(cases),summary=summarize(cases),source_folder=a.source.name,components=[9,19],rejected_methods=[4],
        metric='Native macro IoU on original annotated pixels only; equal dataset weighting',quality_support='reference > 0, not predicted tissue',
        all_tissue_predicted=True,background_removed_after_bora=True,predictions_frozen_before_scoring=True)
    save_json(a.root/'results.json',results)
    with (a.root/'scores.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)
    save_json(a.root/'SCORING_AUDIT.json',dict(cases=len(cases),annotation_valid_pixels_unchanged=True,all_predictions_verified_before_fine_reference_read=True,
        fine_reference_ignored_outside_original_annotations=True,tissue_detector_false_negatives_remain_penalized=True))

if __name__=='__main__':main()
