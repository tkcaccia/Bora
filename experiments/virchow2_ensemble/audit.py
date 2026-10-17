"""Verify native masks, colours, geometry, matched inputs and complete visuals."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import tifffile
from experiments.virchow2_inverse.common import digest,save_json
from experiments.virchow2_multi.publish import rgba


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    results=json.loads((a.root/'results.json').read_text());protocol=json.loads((a.root/'PROTOCOL.json').read_text())
    if [r['case'] for r in results['cases']]!=protocol['cases'] or len(results['cases'])!=49:raise ValueError('Cohort mismatch')
    frozen=json.loads((a.root/'ALL_PREDICTIONS_FROZEN.json').read_text())
    visual_files=[]
    for row in results['cases']:
        folder=a.root/'cases'/row['case'];src=a.source/'cases'/row['case'];sample=row['sample']
        if digest(folder/'FROZEN.json')!=frozen['case_manifest_sha256'][row['case']]:raise ValueError('Manifest changed')
        record=json.loads((folder/'FROZEN.json').read_text())
        for f,sha in record['files_sha256'].items():
            if digest(folder/f)!=sha:raise ValueError('Prediction changed')
        coarse=tifffile.imread(src/'coarse.tif');pred=tifffile.imread(folder/'bora.working.tif');native=tifffile.imread(folder/'bora.native.tif')
        if pred.shape!=coarse.shape or np.any(pred[coarse==0]) or not set(np.unique(pred))<=set(np.unique(coarse)):raise ValueError('Invalid working mask')
        projected=cv2.resize(pred,tuple(sample['native_shape_yx'][::-1]),interpolation=cv2.INTER_NEAREST)
        if not np.array_equal(native,projected):raise ValueError('Native projection mismatch')
        if not np.array_equal(np.array(Image.open(folder/'mask.png')),rgba(native,sample['palette'])):raise ValueError('Colour mask mismatch')
        details=json.loads((folder/'details.json').read_text())
        if details['component_methods']!=protocol['components'] or [r['method'] for r in details['component_parity']]!=protocol['components']:raise ValueError('Unexpected/rejected component used')
        if not all(r['rgb_identical'] and r['before_mask_identical'] for r in details['component_parity']):raise ValueError('Component mismatch')
        for file in ['original.jpg','reference.jpg','coarse.jpg','09.jpg','thumb.jpg','overlay.jpg','comparison.jpg','four_panel.jpg','mask.png','representation.png']:
            with Image.open(folder/file) as im:im.verify()
            visual_files.append(folder/file)
        for file in ['original.native.png','expert.mask.png','coarse.mask.png','reference.native.tif','coarse.tif']:
            if not (src/file).is_file():raise FileNotFoundError(src/file)
        for key in ('04','05','08','09','10','15','19','he'):
            for file in ('mask.png','bora.native.tif','representation.png'):
                if not (src/key/file).is_file():raise FileNotFoundError(src/key/file)
    for file in ('gallery.html','all_outputs.html','scores.csv','REPORT.md','COMPLETE.json'):
        if not (a.root/file).is_file():raise FileNotFoundError(file)
    audit=json.loads((a.root/'AUDIT.json').read_text());audit.update(selected_components_verified=protocol['components'],native_masks_and_projections_verified=49,colour_masks_verified=49,
        visual_images_verified=len(visual_files),all_original_reference_tiled_and_prior_method_assets_present=True,all_checks_passed=True)
    save_json(a.root/'AUDIT.json',audit)
    save_json(a.root/'VISUAL_MANIFEST.json',{str(f.relative_to(a.root)):digest(f) for f in visual_files})
    print(json.dumps(audit),flush=True)

if __name__=='__main__':main()
