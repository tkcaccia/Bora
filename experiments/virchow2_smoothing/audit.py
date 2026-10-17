"""Check native coverage, fixed inputs, and every score/visual artifact."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import tifffile
from experiments.virchow2_inverse.common import digest,save_json
from .run import METHODS,VARIANTS,verify


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    results=json.loads((a.root/'results.json').read_text());frozen=json.loads((a.root/'ALL_PREDICTIONS_FROZEN.json').read_text())
    protocol=results['protocol'];originals=a.source.parent/protocol['original_folder']
    assert digest(a.root/'PROTOCOL.json')==frozen['protocol_sha256']
    assert digest(a.source/'ALL_PREDICTIONS_FROZEN.json')==protocol['source_frozen_sha256']
    masks=0;baseline=0;visuals=0
    for row in results['cases']:
        name=row['case'];out=a.root/'cases'/name;src=a.source/'cases'/name;record=json.loads((out/'FROZEN.json').read_text())
        assert digest(out/'FROZEN.json')==frozen['case_manifest_sha256'][name];verify(out,record)
        assert digest(src/'FROZEN.json')==record['source_manifest_sha256']
        verify(src,json.loads((src/'FROZEN.json').read_text()))
        support=tifffile.imread(src/'tissue.native.tif')>0
        truth=tifffile.imread(originals/'cases'/name/'reference.native.tif');valid=truth>0
        assert digest(originals/'cases'/name/'reference.native.tif')==row['sample']['reference_native_sha256']
        assert int(valid.sum())==row['coverage']['annotated_pixels']
        assert int((support&~valid).sum())==row['coverage']['unannotated_tissue_pixels']
        for m in METHODS:
            assert digest(src/m/'representation.png')==record['input_rgb_sha256'][m]
            legacy=tifffile.imread(src/m/'bora.native.tif')
            for v in VARIANTS:
                folder=out/m/v;pred=tifffile.imread(folder/'bora.native.tif');assert pred.shape==truth.shape
                assert np.all(pred[support]>0) and np.all(pred[~support]==0)
                assert np.array_equal(np.array(Image.open(folder/'mask.png').convert('RGBA'))[...,3]>0,support)
                diff=np.array(Image.open(folder/'difference.mask.png').convert('RGBA'))[...,3]>0;assert np.array_equal(diff,pred!=legacy)
                s=row['methods'][m][v]['score'];assert s['valid_pixels']==int(valid.sum())
                assert set(s['per_class'])==set(map(str,np.unique(truth[valid])))
                assert record['diagnostics'][m][v]['changed_native_pixels_vs_current']==int(diff.sum())
                if v=='median9': assert np.array_equal(pred,legacy);baseline+=1
                masks+=1
    for name,sha in json.loads((a.root/'VISUAL_MANIFEST.json').read_text())['files_sha256'].items():
        assert digest(a.root/name)==sha;visuals+=1
    save_json(a.root/'AUDIT.json',dict(cases=len(results['cases']),native_masks=masks,pixel_identical_legacy_baselines=baseline,
        verified_visuals=visuals,all_detected_tissue_assigned=True,no_background_labels=True,all_difference_maps_exact=True,
        original_quality_support_preserved=True,converted_rgb_unchanged=True,all_checks_passed=True))
    print(json.dumps(json.loads((a.root/'AUDIT.json').read_text()),indent=2))


if __name__=='__main__':main()
