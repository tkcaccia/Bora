"""Freeze development predictions, then score them in a separate function."""
import argparse,json,time,gc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import cv2
import numpy as np
from PIL import Image
import tifffile
from threadpoolctl import threadpool_limits
from .core import predict,TITLES,resize
from experiments.virchow2_inverse.common import digest,save_json
from experiments.bora_strategies.context import Context
from experiments.bora_strategies.methods import run
from experiments.bora_strategies.registry import STRATEGIES
from experiments.bora_strategies.prepare import verify
from experiments.virchow2_multi.score import Evaluator


def freeze(args):
    source,root,sample,keys,repeats=args;src=source/'cases'/sample['case'];out=root/'cases'/sample['case'];out.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(2);verify(src,json.loads((src/'INPUTS_FROZEN.json').read_text()))
    rgb=np.array(Image.open(src/'ensemble/representation.png').convert('RGB'));he=np.array(Image.open(src/'original.png').convert('RGB'))
    coarse=tifffile.imread(src/'coarse.tif');completed=tifffile.imread(src/'coarse.completed.tif');support=tifffile.imread(src/'tissue.working.tif')>0;native_support=tifffile.imread(src/'tissue.native.tif')>0
    baseline=resize(completed,sample['native_shape_yx'],True);baseline[~native_support]=0;tifffile.imwrite(out/'tiles.tif',baseline,compression='deflate')
    records={}
    with threadpool_limits(limits=2):
        for key in keys:
            times=[]
            for repeat in range(repeats):
                start=time.perf_counter()
                if key in STRATEGIES:
                    c=Context(rgb,he,coarse,completed,support,sample['working_mpp_xy'])
                    if key=='B19':
                        for k in ('B02','B03','B08'):run(k,c)
                    pred=c.finish(run(key,c));detail=dict(analysis_shape_yx=list(c.shape),analysis_mpp_xy=c.mpp_xy.tolist(),revision=3 if key=='B19' else 1)
                    del c
                else:pred,detail=predict(key,rgb,coarse,support,sample['working_mpp_xy'],palette=sample['palette'])
                native=resize(pred,sample['native_shape_yx'],True);native[~native_support]=0;times.append(time.perf_counter()-start)
                if repeat==0:
                    if native.shape!=tuple(sample['native_shape_yx']) or not np.all(native[native_support]>0):raise ValueError('Coverage failure')
                    tifffile.imwrite(out/(key+'.tif'),native,compression='deflate');sha=digest(out/(key+'.tif'))
                elif not np.array_equal(native,tifffile.imread(out/(key+'.tif'))):raise ValueError('Nondeterministic prediction')
            records[key]=dict(seconds=times,median_seconds=float(np.median(times)),mask_sha256=sha,details=detail)
            gc.collect()
    save_json(out/'FROZEN.json',dict(records=records,input_sha256=digest(src/'INPUTS_FROZEN.json'),rgb_sha256=digest(src/'ensemble/representation.png'),fine_reference_read=False));print(json.dumps(dict(frozen=sample['case'],methods=len(records))),flush=True)
    return sample['case']


def score(source,root,samples,keys):
    rows=[]
    for s in samples:
        src=source/'cases'/s['case'];out=root/'cases'/s['case'];record=json.loads((out/'FROZEN.json').read_text())
        truth=tifffile.imread(src/'reference.native.tif');assert digest(src/'reference.native.tif')==s['reference_native_sha256'];ev=Evaluator(truth,s['native_mpp_xy'])
        row=dict(case=s['case'],tiles=ev.score(tifffile.imread(out/'tiles.tif')),methods={})
        for k in keys:
            assert digest(out/(k+'.tif'))==record['records'][k]['mask_sha256']
            row['methods'][k]=dict(score=ev.score(tifffile.imread(out/(k+'.tif'))),**record['records'][k])
        rows.append(row)
    ranking=[]
    for k in keys:
        ranking.append(dict(method=k,title=TITLES.get(k,STRATEGIES.get(k,('',''))[0]),mean_iou=float(np.mean([r['methods'][k]['score']['macro_iou'] for r in rows])),mean_dice=float(np.mean([r['methods'][k]['score']['macro_dice'] for r in rows])),mean_boundary_f1_8um=float(np.mean([r['methods'][k]['score']['boundary_f1_at_um']['8'] or 0 for r in rows])),mean_seconds=float(np.mean([r['methods'][k]['median_seconds'] for r in rows]))))
    ranking.sort(key=lambda r:-r['mean_iou']);result=dict(cases=rows,ranking=ranking,tile_mean_iou=float(np.mean([r['tiles']['macro_iou'] for r in rows])),primary_metric='arithmetic mean native per-image macro IoU, fixed ensemble conversion',source=str(source))
    save_json(root/'RESULTS.json',result);print(json.dumps(dict(ranking=ranking,tile_mean_iou=result['tile_mean_iou']),indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--keys',nargs='+',default=list(STRATEGIES)+list(TITLES));p.add_argument('--repeats',type=int,default=2);a=p.parse_args()
    plan=json.loads(Path(__file__).with_name('PLAN.json').read_text());all_samples=json.loads((a.source/'PLAN.json').read_text())['samples'];samples=[s for s in all_samples if s['case'] in plan['development_cases']];assert len(samples)==9
    a.root.mkdir(parents=True,exist_ok=True);protocol=dict(plan=plan,keys=a.keys,repeats=a.repeats,source_sha256=digest(a.source/'ALL_INPUTS_FROZEN.json'),sources_sha256={f.name:digest(f) for f in Path(__file__).parent.glob('*.py')})
    save_json(a.root/'PROTOCOL.json',protocol)
    with ProcessPoolExecutor(max_workers=3,mp_context=get_context('spawn')) as pool:list(pool.map(freeze,[(a.source,a.root,s,a.keys,a.repeats) for s in samples]))
    save_json(a.root/'ALL_FROZEN.json',dict(manifests={s['case']:digest(a.root/'cases'/s['case']/'FROZEN.json') for s in samples},fine_reference_read=False))
    score(a.source,a.root,samples,a.keys)

if __name__=='__main__':main()
