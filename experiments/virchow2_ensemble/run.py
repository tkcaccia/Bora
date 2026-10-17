"""Combine coarse-trained evidence; freeze all predictions before scoring.

No fine-reference pixels or previous scores are read by this module.
"""
import argparse,gc,json,time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import tifffile
from threadpoolctl import threadpool_limits
from experiments.virchow2_inverse.common import digest,save_json
from experiments.virchow2_inverse.protocol import bora_rgb
from experiments.virchow2_multi.methods import Context,align_variant,run_method

COMPONENTS=(9,19)


def mean_evidence(probabilities,classes):
    """Keep actual class columns; average before palette/8-bit conversion."""
    if len(probabilities)<2 or len(classes)!=len(probabilities):raise ValueError("At least two matching components required")
    target=np.asarray(classes[0])
    if len(np.unique(target))!=len(target):raise ValueError("Duplicate class IDs")
    aligned=[]
    shape=probabilities[0].shape
    for p,ids in zip(probabilities,classes):
        ids=np.asarray(ids)
        if p.shape!=shape or p.shape[-1]!=len(ids) or set(ids)!=set(target):raise ValueError("Unmatched component classes/geometry")
        if not np.isfinite(p).all() or np.min(p)<-1e-6 or not np.allclose(p.sum(-1),1,atol=2e-5):raise ValueError("Invalid probability mass")
        aligned.append(p[..., [int(np.flatnonzero(ids==g)[0]) for g in target]])
    return np.mean(np.stack(aligned),axis=0,dtype=np.float32)


def freeze_case(source,root,sample,components):
    name=sample['case'];src=source/'cases'/name;out=root/'cases'/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'FROZEN.json').exists():
        record=json.loads((out/'FROZEN.json').read_text())
        for f,sha in record['files_sha256'].items():
            if digest(out/f)!=sha:raise ValueError('Frozen output changed')
        return
    started=time.monotonic()
    for f,key in [('original.png','original_sha256'),('coarse.tif','coarse_sha256')]:
        if digest(src/f)!=sample[key]:raise ValueError('Input changed')
    ready=json.loads((src/'features/READY.json').read_text())
    if digest(src/'features/base.npy')!=ready['sha256']:raise ValueError('Feature cache changed')
    image=np.array(Image.open(src/'original.png').convert('RGB'));coarse=tifffile.imread(src/'coarse.tif')
    c=Context(image,coarse,np.load(src/'features/base.npy',mmap_mode='r'),sample['palette'])
    variants={}
    for key in ('scale_1','scale_2'):
        meta=json.loads((src/'features'/f'{key}.json').read_text());path=src/'features'/f'{key}.npy'
        if digest(path)!=meta['sha256']:raise ValueError('Scale cache changed')
        variants[key]=align_variant(np.load(path).astype(np.float32),meta['shape_yx'],c)
    evidence=[];parity=[]
    for number in components:
        rgb,raw,details,extras,p=run_method(number,c,variants,return_probabilities=True)
        previous=src/f'{number:02d}'
        original_rgb=np.array(Image.open(previous/'representation.png').convert('RGB'))
        before=tifffile.imread(previous/'before.working.tif')
        if not np.array_equal(rgb,original_rgb) or not np.array_equal(raw,before):raise ValueError(f'Component #{number} parity failure in {name}')
        evidence.append(p)
        parity.append(dict(method=number,rgb_identical=True,before_mask_identical=True,representation_sha256=digest(previous/'representation.png')))
        del extras
    p=mean_evidence(evidence,[c.classes]*len(components));rgb=c.probability_rgb(p)
    before=c.classes[p.argmax(-1)].astype(np.uint8);before[coarse==0]=0
    pred=bora_rgb(rgb,coarse).astype(np.uint8)
    if not set(np.unique(pred)).issubset(set(np.unique(coarse))) or np.any(pred[coarse==0]):raise ValueError('Bora label/support mismatch')
    Image.fromarray(rgb).save(out/'representation.png')
    native_size=tuple(sample['native_shape_yx'][::-1])
    for key,mask in [('before',before),('bora',pred)]:
        tifffile.imwrite(out/f'{key}.working.tif',mask,compression='deflate')
        tifffile.imwrite(out/f'{key}.native.tif',cv2.resize(mask,native_size,interpolation=cv2.INTER_NEAREST),compression='deflate')
    save_json(out/'details.json',dict(component_methods=list(components),weights=[1/len(components)]*len(components),combination='arithmetic mean of aligned class probabilities before RGB quantization',
        classes=c.classes.tolist(),fine_reference_used=False,component_parity=parity,probability_mass_max_error=float(np.max(np.abs(p.sum(-1)-1))),
        selected_dimensions=c.selected.tolist(),working_mpp_xy=sample['working_mpp_xy'],bora_preset='same BoundaryConfig defaults and downsample=2 as baseline #9'))
    files=['representation.png','before.working.tif','before.native.tif','bora.working.tif','bora.native.tif','details.json']
    save_json(out/'FROZEN.json',dict(case=name,files_sha256={f:digest(out/f) for f in files},source_case_manifest_sha256=digest(src/'FROZEN.json'),
        seconds=time.monotonic()-started,fine_reference_read=False,timestamp_unix=time.time()))
    print(json.dumps(dict(case=name,complete=True,seconds=time.monotonic()-started,component_parity=True)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--components',type=int,nargs='+',choices=[4,9,19],default=list(COMPONENTS));a=p.parse_args()
    if len(a.components)<2 or len(set(a.components))!=len(a.components):raise ValueError('Specify at least two distinct components')
    a.root.mkdir(parents=True,exist_ok=True);plan=json.loads((a.source/'PLAN.json').read_text())
    protocol=dict(source_root=str(a.source),source_plan_sha256=digest(a.source/'PLAN.json'),cases=[r['case'] for r in plan['samples']],
        components=a.components,weights=[1/len(a.components)]*len(a.components),weight_selection='fixed before new inference/scoring; no reference-driven weight search',
        source='cached Virchow2 features, same crops/coarse masks/classes/palette/physical sampling/Bora settings',fine_reference_read_by_predictor=False)
    if (a.root/'PROTOCOL.json').exists() and json.loads((a.root/'PROTOCOL.json').read_text())!=protocol:raise ValueError('Protocol changed')
    save_json(a.root/'PROTOCOL.json',protocol)
    with threadpool_limits(limits=2):
        for sample in plan['samples']:freeze_case(a.source,a.root,sample,a.components);gc.collect()
    save_json(a.root/'ALL_PREDICTIONS_FROZEN.json',dict(cases=len(plan['samples']),protocol_sha256=digest(a.root/'PROTOCOL.json'),
        case_manifest_sha256={r['case']:digest(a.root/'cases'/r['case']/'FROZEN.json') for r in plan['samples']},fine_references_read_by_predictor=False,timestamp_unix=time.time()))

if __name__=='__main__':main()
