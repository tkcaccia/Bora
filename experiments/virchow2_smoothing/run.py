"""Freeze all regularization predictions before reading fine references."""
import argparse, json, time
from dataclasses import asdict, replace
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import tifffile
from threadpoolctl import threadpool_limits
from bora.boundary import BoundaryConfig
from experiments.virchow2_inverse.common import digest, save_json
from experiments.virchow2_inverse.protocol import bora_rgb

PRIMARY = ('09', '10', '15', '19', 'ensemble')
METHODS = (*PRIMARY, 'he', 'features')
VARIANTS = {
    'median9': ('Current median 9 × 9', BoundaryConfig()),
    'none': ('No label smoothing', replace(BoundaryConfig(), internal_regularizer='none', internal_median_size=1)),
    'median3': ('Small median 3 × 3', replace(BoundaryConfig(), internal_median_size=3)),
    'majority3': ('Categorical vote 3 × 3', replace(BoundaryConfig(), internal_regularizer='majority', internal_median_size=3)),
    'edge': ('Edge-aware competition', replace(BoundaryConfig(), internal_regularizer='edge_potts', internal_median_size=1)),
}


def verify(folder, record):
    for name, sha in record['files_sha256'].items():
        if digest(folder / name) != sha:
            raise ValueError(f'Changed frozen file: {folder / name}')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--source', type=Path, required=True); p.add_argument('--root', type=Path, required=True); a = p.parse_args()
    plan = json.loads((a.source.parent / 'virchow2-multicase-2026-10-16/PLAN.json').read_text())
    old = json.loads((a.source / 'ALL_PREDICTIONS_FROZEN.json').read_text())
    protocol = dict(source_folder=a.source.name, original_folder='virchow2-multicase-2026-10-16',
        source_frozen_sha256=digest(a.source / 'ALL_PREDICTIONS_FROZEN.json'),
        source_plan_sha256=digest(a.source.parent / 'virchow2-multicase-2026-10-16/PLAN.json'),
        cases=[s['case'] for s in plan['samples']], primary_methods=list(PRIMARY), controls=['he', 'features'], rejected_methods=[4],
        variants={k:dict(title=t, config=asdict(c)) for k,(t,c) in VARIANTS.items()},
        unchanged='same converted RGBs, completed coarse mask, ds2, unary, seed connectivity, native projection and RGB support',
        smoothing_scope='only categorical label postfilter changes; Gaussian LAB preblur sigma 1.4 and local-model averaging remain fixed',
        primary_metric='native macro IoU; equal datasets, equal seven primary representations within each case',
        secondary_metric='class-specific boundary F1 at 4, 8, 16 micrometres; ignore interfaces excluded',
        quality_support='original reference > 0; never intersected with RGB tissue support', fine_reference_read_by_predictor=False)
    a.root.mkdir(parents=True, exist_ok=True)
    if (a.root/'PROTOCOL.json').exists() and json.loads((a.root/'PROTOCOL.json').read_text()) != protocol:
        raise ValueError('Protocol changed')
    save_json(a.root/'PROTOCOL.json', protocol)
    cv2.setNumThreads(2)
    with threadpool_limits(limits=2):
        for sample in plan['samples']:
            start=time.monotonic(); name=sample['case']; src=a.source/'cases'/name; out=a.root/'cases'/name; out.mkdir(parents=True,exist_ok=True)
            if (out/'FROZEN.json').exists(): verify(out,json.loads((out/'FROZEN.json').read_text())); continue
            old_record=json.loads((src/'FROZEN.json').read_text())
            if digest(src/'FROZEN.json')!=old['case_manifest_sha256'][name]: raise ValueError('Source manifest changed')
            verify(src,old_record)
            coarse=tifffile.imread(src/'coarse.completed.tif'); support=tifffile.imread(src/'tissue.native.tif')>0
            files={}; inputs={}; diagnostics={}
            for method in METHODS:
                rgb=np.array(Image.open(src/method/'representation.png').convert('RGB')); inputs[method]=digest(src/method/'representation.png')
                baseline=tifffile.imread(src/method/'bora.native.tif'); diagnostics[method]={}
                for variant,(_,config) in VARIANTS.items():
                    folder=out/method/variant;folder.mkdir(parents=True,exist_ok=True)
                    raw=bora_rgb(rgb,coarse,config).astype(np.uint8)
                    native=cv2.resize(raw,tuple(sample['native_shape_yx'][::-1]),interpolation=cv2.INTER_NEAREST);native[~support]=0
                    if np.any(raw==0) or np.any(native[support]==0) or np.any(native[~support]): raise ValueError('Tissue/background coverage changed')
                    if not set(np.unique(raw)).issubset(set(np.unique(coarse))): raise ValueError('Invented class')
                    if variant=='median9' and (not np.array_equal(native,baseline) or not np.array_equal(raw,tifffile.imread(src/method/'bora.pre_background.working.tif'))):
                        raise ValueError(f'Legacy baseline not reproduced: {name}/{method}')
                    for filename,values in [('bora.pre_background.working.tif',raw),('bora.native.tif',native)]:
                        path=folder/filename;tifffile.imwrite(path,values,compression='deflate');files[str(path.relative_to(out))]=digest(path)
                    diagnostics[method][variant]=dict(changed_native_pixels_vs_current=int(np.count_nonzero(native!=baseline)), baseline_reproduced=variant=='median9')
                print(json.dumps(dict(predicted=name,method=method,seconds=round(time.monotonic()-start,1))),flush=True)
            save_json(out/'FROZEN.json',dict(files_sha256=files,input_rgb_sha256=inputs,source_manifest_sha256=digest(src/'FROZEN.json'),
                diagnostics=diagnostics, fine_reference_read=False, sample=sample, seconds=time.monotonic()-start))
    save_json(a.root/'ALL_PREDICTIONS_FROZEN.json',dict(protocol_sha256=digest(a.root/'PROTOCOL.json'),
        case_manifest_sha256={s['case']:digest(a.root/'cases'/s['case']/'FROZEN.json') for s in plan['samples']},
        cases=len(plan['samples']),native_masks=len(plan['samples'])*len(METHODS)*len(VARIANTS), fine_reference_read=False))


if __name__=='__main__': main()
