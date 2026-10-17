"""Predict the entire frozen scope before any native fine-reference scoring."""
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
from experiments.virchow2_smoothing.run import verify,PRIMARY
from .context import Context,resize
from .registry import STRATEGIES,PARAMETERS
from .methods import run


def freeze_case(source,root,sample,conversions):
    original=source.parent/'virchow2-multicase-2026-10-16'
    source_frozen=json.loads((source/'ALL_PREDICTIONS_FROZEN.json').read_text())
    cv2.setNumThreads(2)
    manifests={}
    with threadpool_limits(limits=2):
        case=sample['case'];src=source/'cases'/case;orig=original/'cases'/case;out=root/'cases'/case;out.mkdir(parents=True,exist_ok=True)
        if digest(src/'FROZEN.json')!=source_frozen['case_manifest_sha256'][case]:raise ValueError('Source manifest changed')
        verify(src,json.loads((src/'FROZEN.json').read_text()))
        if digest(orig/'coarse.tif')!=sample['coarse_sha256'] or digest(orig/'original.png')!=sample['original_sha256']:raise ValueError('Original coarse/image changed')
        known=tifffile.imread(orig/'coarse.tif');completed=tifffile.imread(src/'coarse.completed.tif')
        support=tifffile.imread(src/'tissue.native.tif')>0;work_support=tifffile.imread(src/'tissue.working.tif')>0
        he=np.array(Image.open(orig/'original.png').convert('RGB'));save_json(out/'SAMPLE.json',sample)
        for conversion in conversions:
            folder=out/conversion;folder.mkdir(exist_ok=True)
            if (folder/'FROZEN.json').exists():verify(folder,json.loads((folder/'FROZEN.json').read_text()));manifests[case+'/'+conversion]=digest(folder/'FROZEN.json');continue
            rgb=np.array(Image.open(src/conversion/'representation.png').convert('RGB'))
            context=Context(rgb,he,known,completed,work_support,sample['working_mpp_xy'])
            baseline=tifffile.imread(src/conversion/'bora.native.tif');baseline_raw=tifffile.imread(src/conversion/'bora.pre_background.working.tif')
            context.outputs['B01']=np.searchsorted(context.ids,resize(baseline_raw,context.shape,True)).astype(np.int32)
            files={};records={}
            for strategy in STRATEGIES:
                start=time.monotonic();pred=run(strategy,context);analysis=context.finish(pred)
                native=baseline.copy() if strategy=='B01' else resize(analysis,sample['native_shape_yx'],True)
                native[~support]=0
                if np.any(native[support]==0) or np.any(native[~support]) or not set(np.unique(native))<=(set(context.ids)|{0}):raise ValueError('Coverage or labels changed')
                target=folder/strategy;target.mkdir(exist_ok=True)
                for name,array in [('mask.analysis.tif',analysis),('bora.native.tif',native)]:
                    path=target/name;tifffile.imwrite(path,array,compression='deflate');files[str(path.relative_to(folder))]=digest(path)
                records[strategy]=dict(seconds=time.monotonic()-start,changed_native_pixels_vs_current=int(np.count_nonzero(native!=baseline)),
                    diagnostics=context.diagnostics.get(strategy,{}),fine_reference_used=False)
                save_json(target/'details.json',records[strategy]);files[str((target/'details.json').relative_to(folder))]=digest(target/'details.json')
                print(json.dumps(dict(predicted=case,conversion=conversion,strategy=strategy,seconds=round(records[strategy]['seconds'],3))),flush=True)
            save_json(folder/'FROZEN.json',dict(files_sha256=files,records=records,source_case_manifest_sha256=digest(src/'FROZEN.json'),
                rgb_sha256=digest(src/conversion/'representation.png'),source_coarse_sha256=sample['coarse_sha256'],analysis_shape_yx=list(context.shape),analysis_mpp_xy=context.mpp_xy.tolist(),
                native_shape_yx=sample['native_shape_yx'],native_mpp_xy=sample['native_mpp_xy'],anchor_pixels=int((context.anchors>=0).sum()),
                training_pixels=int((context.core>=0).sum()),fine_reference_read=False))
            manifests[case+'/'+conversion]=digest(folder/'FROZEN.json');del context;gc.collect()
    return manifests


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--workers',type=int,choices=(1,2,3),default=3);p.add_argument('--cases',nargs='+');p.add_argument('--conversions',nargs='+');p.add_argument('--pilot',action='store_true');a=p.parse_args()
    original=a.source.parent/'virchow2-multicase-2026-10-16';plan=json.loads((original/'PLAN.json').read_text())
    samples=[s for s in plan['samples'] if a.cases is None or s['case'] in a.cases];conversions=a.conversions or list(PRIMARY)
    if not a.pilot and (len(samples)!=49 or conversions!=list(PRIMARY)):raise ValueError('Production run must cover all 49 cases and seven retained conversions')
    if not samples or not set(conversions)<=set(PRIMARY):raise ValueError('Invalid scope')
    source_frozen=json.loads((a.source/'ALL_PREDICTIONS_FROZEN.json').read_text())
    code={path.name:digest(path) for path in sorted(Path(__file__).parent.glob('*.py')) if path.name not in ('score.py','publish.py','audit.py')}
    protocol=dict(pilot=a.pilot,source_folder=a.source.name,original_folder=original.name,source_frozen_sha256=digest(a.source/'ALL_PREDICTIONS_FROZEN.json'),
        source_plan_sha256=digest(original/'PLAN.json'),cases=[s['case'] for s in samples],conversions=conversions,rejected_conversion=[4],
        strategies={key:dict(title=title,hypothesis=description) for key,(title,description) in STRATEGIES.items()},parameters=PARAMETERS,
        predictor_sources_sha256=code,compute=dict(workers=a.workers,threads_per_worker=2,priority='nice 12'),baseline='B01 native mask copied pixel-identically from the previous full-tissue result',
        training='original nonzero coarse-label interiors only; never completed pseudo labels as training targets',
        analysis='whole converted RGB field, including all unannotated tissue; sparse coarse tile-center anchors for most solvers',
        geometry='new solvers use approximately 1 micrometre/pixel per-axis geometry, projected nearest-neighbor to original native coordinates',
        final_support='unchanged independent native RGB tissue support; zero background after segmentation, every detected tissue pixel assigned',
        quality='original native reference > 0, unchanged; unannotated tissue excluded, removed annotated tissue penalized',
        primary_summary='equal datasets and equal seven conversions within each case; paired case resamples keep conversions together',
        fine_reference_read_by_predictor=False)
    a.root.mkdir(parents=True,exist_ok=True)
    if (a.root/'PROTOCOL.json').exists() and json.loads((a.root/'PROTOCOL.json').read_text())!=protocol:raise ValueError('Frozen protocol/code changed')
    save_json(a.root/'PROTOCOL.json',protocol);cv2.setNumThreads(2)
    manifests={}
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=get_context('spawn')) as pool:
        futures=[pool.submit(freeze_case,a.source,a.root,sample,conversions) for sample in samples]
        for future in as_completed(futures):manifests.update(future.result())
    save_json(a.root/'ALL_PREDICTIONS_FROZEN.json',dict(protocol_sha256=digest(a.root/'PROTOCOL.json'),conversion_manifest_sha256=manifests,
        cases=len(samples),conversions=len(conversions),strategies=len(STRATEGIES),native_masks=len(manifests)*len(STRATEGIES),fine_reference_read=False))


if __name__=='__main__':main()
