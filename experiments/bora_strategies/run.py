"""Freeze every retained prediction before native fine-reference scoring."""
import argparse,gc,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import tifffile
from threadpoolctl import threadpool_limits
from experiments.virchow2_inverse.common import digest,save_json
from .prepare import verify
from .context import Context,resize
from .methods import run
from .registry import STRATEGIES,PARAMETERS,CONVERSIONS,REJECTED_STRATEGIES,REJECTED_CONVERSIONS,CONSENSUS


def freeze_case(source,root,sample):
    cv2.setNumThreads(2);case=sample['case'];src=source/'cases'/case;out=root/'cases'/case;out.mkdir(parents=True,exist_ok=True)
    verify(src,json.loads((src/'INPUTS_FROZEN.json').read_text()));manifests={}
    known=tifffile.imread(src/'coarse.tif');completed=tifffile.imread(src/'coarse.completed.tif')
    support=tifffile.imread(src/'tissue.native.tif')>0;work_support=tifffile.imread(src/'tissue.working.tif')>0
    he=np.array(Image.open(src/'original.png').convert('RGB'));save_json(out/'SAMPLE.json',sample)
    tiles=resize(completed,sample['native_shape_yx'],True);tiles[~support]=0
    tifffile.imwrite(out/'tiles.native.tif',tiles,compression='deflate')
    with threadpool_limits(limits=2):
        for conversion in CONVERSIONS:
            folder=out/conversion;folder.mkdir(exist_ok=True)
            if (folder/'FROZEN.json').exists():verify(folder,json.loads((folder/'FROZEN.json').read_text()));manifests[case+'/'+conversion]=digest(folder/'FROZEN.json');continue
            rgb=np.array(Image.open(src/conversion/'representation.png').convert('RGB'));c=Context(rgb,he,known,completed,work_support,sample['working_mpp_xy']);files={};records={}
            for key in STRATEGIES:
                started=time.monotonic();analysis=c.finish(run(key,c));native=resize(analysis,sample['native_shape_yx'],True);native[~support]=0
                if np.any(native[support]==0) or np.any(native[~support]) or not set(np.unique(native))<=(set(c.ids)|{0}):raise ValueError('Coverage or labels changed')
                target=folder/key;target.mkdir(exist_ok=True)
                for name,array in [('mask.analysis.tif',analysis),('bora.native.tif',native)]:
                    path=target/name;tifffile.imwrite(path,array,compression='deflate');files[str(path.relative_to(folder))]=digest(path)
                records[key]=dict(seconds=time.monotonic()-started,diagnostics=c.diagnostics.get(key,{}),fine_reference_used=False)
                save_json(target/'details.json',records[key]);files[key+'/details.json']=digest(target/'details.json')
            save_json(folder/'FROZEN.json',dict(files_sha256=files,records=records,tiles_native_sha256=digest(out/'tiles.native.tif'),input_manifest_sha256=digest(src/'INPUTS_FROZEN.json'),rgb_sha256=digest(src/conversion/'representation.png'),
                source_coarse_sha256=sample['coarse_sha256'],analysis_shape_yx=list(c.shape),analysis_mpp_xy=c.mpp_xy.tolist(),native_shape_yx=sample['native_shape_yx'],native_mpp_xy=sample['native_mpp_xy'],fine_reference_read=False))
            manifests[case+'/'+conversion]=digest(folder/'FROZEN.json');del c;gc.collect()
            print(json.dumps(dict(predicted=case,conversion=conversion,strategies=len(STRATEGIES))),flush=True)
    return manifests


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,choices=(1,2,3),default=2);a=p.parse_args()
    plan=json.loads((a.source/'PLAN.json').read_text());inputs=json.loads((a.source/'ALL_INPUTS_FROZEN.json').read_text())
    if inputs['plan_sha256']!=digest(a.source/'PLAN.json'):raise ValueError('Plan changed')
    for sample in plan['samples']:
        if digest(a.source/'cases'/sample['case']/'INPUTS_FROZEN.json')!=inputs['manifests'][sample['case']]:raise ValueError('Inputs changed')
    code={p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py')) if p.name not in ('score.py','publish.py','audit.py')}
    protocol=dict(source_folder=a.source.name,source_plan_sha256=digest(a.source/'PLAN.json'),inputs_sha256=digest(a.source/'ALL_INPUTS_FROZEN.json'),cases=[s['case'] for s in plan['samples']],
        conversions=list(CONVERSIONS),strategies={k:dict(title=t,hypothesis=d) for k,(t,d) in STRATEGIES.items()},parameters=PARAMETERS,
        rejected_strategies=list(REJECTED_STRATEGIES),rejected_conversions=list(REJECTED_CONVERSIONS),consensus_components=list(CONSENSUS),consensus_revision=3,
        predictor_sources_sha256=code,compute=dict(workers=a.workers,threads_per_worker=2),baseline='same completed input tile mask with final independent RGB background clipping',
        quality='original native reference > 0, never intersected with predicted tissue; unannotated tissue unscored; removed annotated tissue penalized',
        supervision='per-image original reference-derived coarse labels; no fine-boundary access during prediction',fine_reference_read_by_predictor=False)
    a.root.mkdir(parents=True,exist_ok=True)
    if (a.root/'PROTOCOL.json').exists() and json.loads((a.root/'PROTOCOL.json').read_text())!=protocol:raise ValueError('Protocol changed')
    save_json(a.root/'PROTOCOL.json',protocol);manifests={}
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=get_context('spawn')) as pool:
        futures=[pool.submit(freeze_case,a.source,a.root,s) for s in plan['samples']]
        for future in as_completed(futures):manifests.update(future.result())
    save_json(a.root/'ALL_PREDICTIONS_FROZEN.json',dict(protocol_sha256=digest(a.root/'PROTOCOL.json'),conversion_manifest_sha256=manifests,cases=len(plan['samples']),
        conversions=len(CONVERSIONS),strategies=len(STRATEGIES),native_masks=len(manifests)*len(STRATEGIES),fine_reference_read=False))

if __name__=='__main__':main()
