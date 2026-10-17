"""Audit native geometry, unchanged annotation support, scope and saved visuals."""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
import tifffile
from experiments.virchow2_inverse.common import digest,save_json
from .prepare import verify
from .registry import STRATEGIES,CONVERSIONS,REJECTED_STRATEGIES,CONSENSUS


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--previous-results',type=Path,required=True);a=p.parse_args()
    results=json.loads((a.root/'results.json').read_text());protocol=results['protocol'];frozen=json.loads((a.root/'ALL_PREDICTIONS_FROZEN.json').read_text())
    assert digest(a.root/'PROTOCOL.json')==frozen['protocol_sha256'];assert digest(a.source/'ALL_INPUTS_FROZEN.json')==protocol['inputs_sha256']
    assert set(STRATEGIES).isdisjoint(REJECTED_STRATEGIES) and set(CONSENSUS)<=set(STRATEGIES)
    assert protocol['conversions']==list(CONVERSIONS) and set(protocol['strategies'])==set(STRATEGIES)
    for name,sha in protocol['predictor_sources_sha256'].items():assert digest(Path(__file__).with_name(name))==sha
    groups=set();oldgroups={(r['dataset'],r['sample']['group']) for r in results['cases'] if r['cohort']=='previously_inspected'};masks=parity=rgbparity=0
    for row in results['cases']:
        case=row['case'];out=a.root/'cases'/case;src=a.source/'cases'/case;group=(row['dataset'],row['sample']['group']);assert group not in groups;groups.add(group)
        if row['cohort']=='new_extension':assert group not in oldgroups
        verify(src,json.loads((src/'INPUTS_FROZEN.json').read_text()));truth=tifffile.imread(src/'reference.native.tif');valid=truth>0;support=tifffile.imread(src/'tissue.native.tif')>0
        assert digest(src/'reference.native.tif')==row['sample']['reference_native_sha256'];assert int(valid.sum())==row['coverage']['annotated_pixels']
        assert json.loads((src/'encoder_coverage.json').read_text())['all_original_tokens_covered'];allowed=set(map(int,np.unique(tifffile.imread(src/'coarse.tif'))))|{0}
        tiles=tifffile.imread(out/'tiles.native.tif')
        import cv2
        expected=cv2.resize(tifffile.imread(src/'coarse.completed.tif'),tuple(truth.shape[::-1]),interpolation=cv2.INTER_NEAREST);expected[~support]=0
        assert np.array_equal(tiles,expected)
        for m in CONVERSIONS:
            folder=out/m;record=json.loads((folder/'FROZEN.json').read_text());assert digest(folder/'FROZEN.json')==frozen['conversion_manifest_sha256'][case+'/'+m];verify(folder,record);assert record['tiles_native_sha256']==digest(out/'tiles.native.tif')
            assert record['input_manifest_sha256']==digest(src/'INPUTS_FROZEN.json') and record['rgb_sha256']==digest(src/m/'representation.png')
            if row['cohort']=='previously_inspected':
                previous_rgb=a.previous_results.parent/'virchow2-full-tissue-2026-10-16'/'cases'/case/m/'representation.png'
                assert digest(previous_rgb)==digest(src/m/'representation.png');rgbparity+=1
            for key in STRATEGIES:
                target=folder/key;mask=tifffile.imread(target/'bora.native.tif');assert mask.shape==truth.shape and np.all(mask[support]>0) and np.all(mask[~support]==0)
                assert set(map(int,np.unique(mask)))<=allowed
                visual=Image.open(target/'mask.png');assert visual.mode=='P' and visual.info['transparency']==0 and np.array_equal(np.array(visual),mask)
                colours=np.array(visual.getpalette(),np.uint8).reshape(-1,3)
                for label in np.unique(mask[mask>0]):assert np.array_equal(colours[label],row['sample']['palette'][int(label)-1])
                assert np.array_equal(np.array(Image.open(target/'difference.mask.png'))>0,mask!=tiles)
                score=row['conversions'][m]['strategies'][key]['score'];assert score['valid_pixels']==int(valid.sum()) and set(score['per_class'])==set(map(str,np.unique(truth[valid])))
                if row['cohort']=='previously_inspected' and key!='B19':assert np.array_equal(mask,tifffile.imread(a.previous_results/'cases'/case/m/key/'bora.native.tif'));parity+=1
                masks+=1
        print(json.dumps(dict(audited=case)),flush=True)
    visual=json.loads((a.root/'VISUAL_MANIFEST.json').read_text())['files_sha256']
    for name,sha in visual.items():assert digest(a.root/name)==sha
    audit=dict(cases=len(results['cases']),cohorts=dict(Counter(r['cohort'] for r in results['cases'])),conversions=len(CONVERSIONS),strategies=len(STRATEGIES),native_masks=masks,
        unchanged_retained_native_masks=parity,unchanged_retained_rgb_inputs=rgbparity,consensus_revision=3,new_groups_disjoint_from_previous=True,
        all_detected_tissue_assigned=True,no_final_background_labels=True,original_quality_support_preserved=True,all_reference_classes_scored=True,
        exact_difference_maps=True,all_frozen_hashes_verified=True,predictor_source_hashes_verified=True,verified_visuals=len(visual),all_checks_passed=True)
    save_json(a.root/'AUDIT.json',audit);save_json(a.root/'COMPLETE.json',audit);print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
