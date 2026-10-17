"""Predict every tissue pixel; freeze before fine-reference scoring."""
import argparse,gc,json,time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import tifffile
from threadpoolctl import threadpool_limits
from experiments.virchow2_inverse.common import digest,save_json
from experiments.virchow2_inverse.protocol import bora_rgb
from experiments.virchow2_multi.methods import Context,align_variant,run_method,METHODS
from experiments.virchow2_ensemble.run import mean_evidence
from .support import POLICY,tissue_support,complete_tiles,encoder_coverage

ACTIVE=(9,10,15,19)
TITLES={**{f'{n:02d}':METHODS[n] for n in ACTIVE},'ensemble':'Equal evidence 9 + 19','he':'Original H&E','features':'Plain selected-feature RGB'}


def freeze_case(source,root,sample):
    started=time.monotonic();src=source/'cases'/sample['case'];out=root/'cases'/sample['case'];out.mkdir(parents=True,exist_ok=True)
    if (out/'FROZEN.json').exists():
        for name,sha in json.loads((out/'FROZEN.json').read_text())['files_sha256'].items():
            if digest(out/name)!=sha:raise ValueError('Frozen output changed')
        return
    for name,key in [('original.png','original_sha256'),('coarse.tif','coarse_sha256')]:
        if digest(src/name)!=sample[key]:raise ValueError('Original input changed')
    image=np.array(Image.open(src/'original.png').convert('RGB'));native_rgb=np.array(Image.open(src/'original.native.png').convert('RGB'))
    if list(native_rgb.shape[:2])!=sample['native_shape_yx']:raise ValueError('Native geometry changed')
    original_coarse=tifffile.imread(src/'coarse.tif')
    native_support,policy=tissue_support(native_rgb,sample['native_mpp_xy'])
    work_support=cv2.resize(native_support.astype(np.float32),(image.shape[1],image.shape[0]),interpolation=cv2.INTER_AREA)>.2
    if not work_support.any():raise ValueError('No tissue detected')
    tifffile.imwrite(out/'tissue.native.tif',native_support.astype(np.uint8),compression='deflate')
    tifffile.imwrite(out/'tissue.working.tif',work_support.astype(np.uint8),compression='deflate')
    Image.fromarray(native_support.astype(np.uint8)*255).save(out/'tissue.png')
    save_json(out/'tissue_policy.json',policy)
    ready=json.loads((src/'features/READY.json').read_text());feature_path=src/'features/base.npy'
    if digest(feature_path)!=ready['sha256']:raise ValueError('Feature cache changed')
    f=np.load(feature_path,mmap_mode='r');coverage=encoder_coverage(ready,f.shape[:2],image.shape[:2])
    save_json(out/'encoder_coverage.json',dict(**coverage,feature_sha256=ready['sha256'],cached_full_image_features_reused=True))
    c=Context(image,original_coarse,f,sample['palette'],analysis_support=work_support)
    save_json(out/'selection.json',c.selection)
    completed,completion=complete_tiles(original_coarse,c.native_p,c.classes,work_support)
    tifffile.imwrite(out/'coarse.completed.tif',completed,compression='deflate');save_json(out/'completion.json',completion)
    variants={}
    for name in ('scale_1','scale_2'):
        meta=json.loads((src/'features'/f'{name}.json').read_text());path=src/'features'/f'{name}.npy'
        if digest(path)!=meta['sha256']:raise ValueError('Variant cache changed')
        values=np.load(path).astype(np.float32)
        variants[name]=values if name.startswith('stain') else align_variant(values,meta['shape_yx'],c)
    files={};records=[]
    def store(key,rgb,p,details):
        folder=out/key;folder.mkdir(exist_ok=True);Image.fromarray(rgb).save(folder/'representation.png')
        before=c.classes[p.argmax(-1)].astype(np.uint8) if p is not None else None
        raw=bora_rgb(rgb,completed).astype(np.uint8)
        if np.any(raw==0):raise ValueError('Bora left an unassigned full-field pixel')
        # The requested background removal is a separate final operation.
        tifffile.imwrite(folder/'bora.pre_background.working.tif',raw,compression='deflate')
        for name,mask in [('bora',raw)]+([('before',before)] if before is not None else []):
            native=cv2.resize(mask,tuple(sample['native_shape_yx'][::-1]),interpolation=cv2.INTER_NEAREST);native[~native_support]=0
            working=mask.copy();working[~work_support]=0
            if np.any(native[native_support]==0) or np.any(native[~native_support]):raise ValueError('Incomplete tissue or retained background')
            tifffile.imwrite(folder/f'{name}.working.tif',working,compression='deflate');tifffile.imwrite(folder/f'{name}.native.tif',native,compression='deflate')
        save_json(folder/'details.json',{**details,'background_removal':'after Bora using independent native RGB support','all_detected_tissue_assigned':True,'fine_reference_used':False})
        for path in folder.iterdir():files[str(path.relative_to(out))]=digest(path)
        records.append(dict(method=key,title=TITLES[key]));print(json.dumps(dict(case=sample['case'],method=key,complete=True)),flush=True)
    store('he',image,None,dict(control='same completed tile input; H&E supplied to Bora'))
    store('features',c.probability_rgb(c.native_p),c.native_p,dict(control='plain coarse-trained selected-feature probabilities'))
    evidence={}
    for number in ACTIVE:
        rgb,_,details,extras,p=run_method(number,c,variants,return_probabilities=True)
        store(f'{number:02d}',rgb,p,details)
        if number in (9,19):evidence[number]=p
        del extras
    combined=mean_evidence([evidence[9],evidence[19]],[c.classes,c.classes])
    store('ensemble',c.probability_rgb(combined),combined,dict(components=[9,19],weights=[.5,.5]))
    for name in ('tissue.native.tif','tissue.working.tif','tissue.png','tissue_policy.json','encoder_coverage.json','selection.json','coarse.completed.tif','completion.json'):files[name]=digest(out/name)
    save_json(out/'FROZEN.json',dict(files_sha256=files,methods=records,source_original_sha256=sample['original_sha256'],
        source_native_original_sha256=digest(src/'original.native.png'),source_coarse_sha256=sample['coarse_sha256'],fine_reference_read=False,
        classes=c.classes.tolist(),seconds=time.monotonic()-started,timestamp_unix=time.time()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=True);plan=json.loads((a.source/'PLAN.json').read_text())
    protocol=dict(source_root=str(a.source),source_plan_sha256=digest(a.source/'PLAN.json'),cases=[r['case'] for r in plan['samples']],
        methods=list(TITLES),rejected_methods=[4],tissue_policy=POLICY,quality_support='original native reference > 0, never intersected with predicted tissue',
        coarse_completion='same plain-feature probability extension for every method; original labelled tiles train classifiers',
        background='removed only after Bora; final native mask positive on all detected tissue and zero elsewhere',fine_references_read_by_predictor=False)
    if (a.root/'PROTOCOL.json').exists() and json.loads((a.root/'PROTOCOL.json').read_text())!=protocol:raise ValueError('Protocol changed')
    save_json(a.root/'PROTOCOL.json',protocol)
    with threadpool_limits(limits=2):
        for sample in plan['samples']:freeze_case(a.source,a.root,sample);gc.collect()
    save_json(a.root/'ALL_PREDICTIONS_FROZEN.json',dict(cases=len(plan['samples']),protocol_sha256=digest(a.root/'PROTOCOL.json'),
        case_manifest_sha256={r['case']:digest(a.root/'cases'/r['case']/'FROZEN.json') for r in plan['samples']},fine_references_read_by_predictor=False,timestamp_unix=time.time()))

if __name__=='__main__':main()
