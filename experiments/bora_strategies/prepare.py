"""Prepare retained RGB conversions on full tissue, without running any Bora solver."""
import argparse,gc,json,time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import tifffile
import torch
from threadpoolctl import threadpool_limits
from experiments.virchow2_inverse.common import digest,save_json,MODEL_SHA256,MODEL_REVISION
from experiments.virchow2_inverse.gpu_ideas import load_model
from experiments.virchow2_multi.methods import Context,align_variant,run_method
from experiments.virchow2_ensemble.run import mean_evidence
from experiments.virchow2_tissue.support import tissue_support,complete_tiles,encoder_coverage,POLICY
from .encoder import encode
from .registry import CONVERSIONS


def verify(folder,record):
    for name,sha in record['files_sha256'].items():
        if digest(folder/name)!=sha:raise ValueError('Frozen file changed: '+str(folder/name))


def prepare_case(root,sample,model,previous):
    folder=root/'cases'/sample['case'];frozen=folder/'INPUTS_FROZEN.json'
    if frozen.exists():verify(folder,json.loads(frozen.read_text()));return
    started=time.monotonic();image=np.array(Image.open(folder/'original.png').convert('RGB'));native=np.array(Image.open(folder/'original.native.png').convert('RGB'))
    coarse=tifffile.imread(folder/'coarse.tif')
    if digest(folder/'original.png')!=sample['original_sha256'] or digest(folder/'coarse.tif')!=sample['coarse_sha256']:raise ValueError('Input changed')
    features=folder/'features';features.mkdir(exist_ok=True)
    cache=previous/'cases'/sample['case']/'features' if sample['benchmark_cohort']=='previously_inspected' else features
    base=cache/'base.npy';ready=cache/'READY.json'
    if not ready.exists():
        f,meta=encode(model,image);np.save(base,f);save_json(ready,dict(**meta,sha256=digest(base),image_sha256=sample['original_sha256']))
    meta=json.loads(ready.read_text())
    if digest(base)!=meta['sha256'] or meta['image_sha256']!=sample['original_sha256']:raise ValueError('Feature cache changed')
    native_support,policy=tissue_support(native,sample['native_mpp_xy'])
    support=cv2.resize(native_support.astype(np.float32),tuple(image.shape[:2][::-1]),interpolation=cv2.INTER_AREA)>.2
    f=np.load(base,mmap_mode='r');c=Context(image,coarse,f,sample['palette'],analysis_support=support)
    save_json(folder/'selection.json',c.selection);save_json(folder/'tissue_policy.json',policy)
    save_json(folder/'encoder_coverage.json',dict(**encoder_coverage(meta,f.shape[:2],image.shape[:2]),checkpoint_sha256=MODEL_SHA256,revision=MODEL_REVISION))
    tifffile.imwrite(folder/'tissue.native.tif',native_support.astype(np.uint8),compression='deflate');tifffile.imwrite(folder/'tissue.working.tif',support.astype(np.uint8),compression='deflate')
    completed,completion=complete_tiles(coarse,c.native_p,c.classes,support)
    tifffile.imwrite(folder/'coarse.completed.tif',completed,compression='deflate');save_json(folder/'completion.json',completion)
    variants={};feature_manifest={'base':dict(path=str(base),sha256=meta['sha256'])}
    for name,factor in [('scale_1',2),('scale_2',4)]:
        p=cache/(name+'.npy');mp=cache/(name+'.json');scaled=np.array(Image.fromarray(image).resize((round(image.shape[1]/factor),round(image.shape[0]/factor)),Image.Resampling.LANCZOS))
        if not mp.exists():
            values,metadata=encode(model,scaled,c.selected);np.save(p,values)
            save_json(mp,dict(**metadata,shape_yx=list(scaled.shape[:2]),sha256=digest(p),mpp_xy=[sample['fov_um_xy'][0]/scaled.shape[1],sample['fov_um_xy'][1]/scaled.shape[0]]))
        metadata=json.loads(mp.read_text())
        if digest(p)!=metadata['sha256'] or metadata['selected_dimensions']!=c.selected.tolist():raise ValueError('Variant feature cache changed')
        variants[name]=align_variant(np.load(p).astype(np.float32),metadata['shape_yx'],c)
        feature_manifest[name]=dict(path=str(p),sha256=metadata['sha256'],mpp_xy=metadata['mpp_xy'])
    evidence={};files={}
    for n in (9,10,15,19):
        rgb,_,details,_,prob=run_method(n,c,variants,return_probabilities=True);out=folder/f'{n:02d}';out.mkdir(exist_ok=True)
        Image.fromarray(rgb).save(out/'representation.png');save_json(out/'details.json',details)
        if n in (9,19):evidence[n]=prob
    combined=mean_evidence([evidence[9],evidence[19]],[c.classes,c.classes]);out=folder/'ensemble';out.mkdir(exist_ok=True)
    Image.fromarray(c.probability_rgb(combined)).save(out/'representation.png');save_json(out/'details.json',dict(components=[9,19],weights=[.5,.5],fine_reference_used=False))
    for name in ['original.png','original.native.png','coarse.tif','coarse.completed.tif','selection.json','completion.json','tissue.native.tif','tissue.working.tif','tissue_policy.json','encoder_coverage.json']:
        files[name]=digest(folder/name)
    for key in CONVERSIONS:
        for name in ('representation.png','details.json'):files[key+'/'+name]=digest(folder/key/name)
    save_json(frozen,dict(files_sha256=files,feature_cache=feature_manifest,conversions=list(CONVERSIONS),fine_reference_read=False,seconds=time.monotonic()-started))
    print(json.dumps(dict(prepared=sample['case'],seconds=round(time.monotonic()-started,2))),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.35);cv2.setNumThreads(2)
    plan=json.loads((a.root/'PLAN.json').read_text());model=load_model(a.model_dir)
    with threadpool_limits(limits=2):
        for sample in plan['samples']:prepare_case(a.root,sample,model,a.previous);gc.collect()
    save_json(a.root/'ALL_INPUTS_FROZEN.json',dict(plan_sha256=digest(a.root/'PLAN.json'),cases=len(plan['samples']),conversions=list(CONVERSIONS),
        manifests={s['case']:digest(a.root/'cases'/s['case']/'INPUTS_FROZEN.json') for s in plan['samples']},tissue_policy=POLICY,fine_reference_read=False))

if __name__=='__main__':main()
