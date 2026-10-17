"""Check complete tissue coverage, zero background and unchanged quality region."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import tifffile
from experiments.virchow2_inverse.common import digest,save_json
from experiments.virchow2_multi.publish import rgba
from .support import tissue_support
from .run import TITLES


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    results=json.loads((a.root/'results.json').read_text());frozen=json.loads((a.root/'ALL_PREDICTIONS_FROZEN.json').read_text());visuals={};masks=0;parity=0
    for row in results['cases']:
        out=a.root/'cases'/row['case'];src=a.source/'cases'/row['case'];s=row['sample']
        if digest(out/'FROZEN.json')!=frozen['case_manifest_sha256'][row['case']]:raise ValueError('Manifest changed')
        record=json.loads((out/'FROZEN.json').read_text())
        for name,sha in record['files_sha256'].items():
            if digest(out/name)!=sha:raise ValueError('Frozen prediction changed')
        rgb=np.array(Image.open(src/'original.native.png').convert('RGB'))
        if digest(src/'original.native.png')!=record['source_native_original_sha256']:raise ValueError('Original changed')
        tissue=tifffile.imread(out/'tissue.native.tif')>0;recomputed,_=tissue_support(rgb,s['native_mpp_xy'])
        if not np.array_equal(tissue,recomputed):raise ValueError('Support policy mismatch')
        truth=tifffile.imread(src/'reference.native.tif');valid=truth>0;stored=tifffile.imread(out/'evaluation.native.tif')>0
        if not np.array_equal(valid,stored) or int(valid.sum())!=row['coverage']['annotated_pixels']:raise ValueError('Quality region changed')
        if digest(src/'reference.native.tif')!=s['reference_native_sha256']:raise ValueError('Reference changed')
        original=tifffile.imread(src/'coarse.tif');completed=tifffile.imread(out/'coarse.completed.tif')
        if digest(src/'coarse.tif')!=record['source_coarse_sha256']:raise ValueError('Original training input changed')
        if json.loads((out/'selection.json').read_text())!=json.loads((src/'selection.json').read_text()):raise ValueError('Feature selection changed beyond original coarse supervision')
        if not np.array_equal(completed[original>0],original[original>0]) or np.any(completed==0):raise ValueError('Coarse extension mismatch')
        for key in TITLES:
            folder=out/key;raw=tifffile.imread(folder/'bora.pre_background.working.tif');native=tifffile.imread(folder/'bora.native.tif')
            projected=cv2.resize(raw,tuple(s['native_shape_yx'][::-1]),interpolation=cv2.INTER_NEAREST);projected[~tissue]=0
            if np.any(raw==0) or np.any(native[tissue]==0) or np.any(native[~tissue]) or not np.array_equal(native,projected):raise ValueError('Tissue/background/native projection failed')
            if not np.array_equal(np.array(Image.open(folder/'mask.png')),rgba(native,s['palette'])):raise ValueError('Colour mask mismatch')
            if key not in ('15','ensemble'):
                previous=np.array(Image.open(src/key/'representation.png'))
                if not np.array_equal(np.array(Image.open(folder/'representation.png')),previous):raise ValueError('Fixed conversion unexpectedly changed')
                parity+=1
            if key in row['methods'] and row['methods'][key]['after']['valid_pixels']!=int(valid.sum()):raise ValueError('Scoring excluded valid reference')
            masks+=1
        for path in out.rglob('*'):
            if path.suffix.lower() in ('.png','.jpg'):
                with Image.open(path) as im:im.verify()
                visuals[str(path.relative_to(a.root))]=digest(path)
    for name in ('gallery.html','detail.html','all_outputs.html','all_inputs.jpg','all_masks.jpg','scores.csv','REPORT.md','COMPLETE.json'):
        if not (a.root/name).is_file():raise FileNotFoundError(name)
    audit=dict(all_checks_passed=True,cases=len(results['cases']),final_native_masks_verified=masks,zero_unassigned_detected_tissue_pixels=True,
        zero_final_labels_outside_tissue=True,annotation_only_quality_region_unchanged=True,rgb_support_reproduced=True,
        full_image_encoder_coverage_verified=all(json.loads((a.root/'cases'/r['case']/'encoder_coverage.json').read_text())['all_original_tokens_covered'] for r in results['cases']),
        unchanged_conversion_rgb_parities=parity,visual_images_verified=len(visuals),rejected_methods=[4])
    save_json(a.root/'AUDIT.json',audit);save_json(a.root/'VISUAL_MANIFEST.json',visuals);print(json.dumps(audit),flush=True)

if __name__=='__main__':main()
