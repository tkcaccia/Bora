"""Full native coverage, provenance and visualization integrity audit."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import tifffile
from experiments.virchow2_inverse.common import digest,save_json
from experiments.virchow2_smoothing.run import verify
from .registry import STRATEGIES


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    results=json.loads((a.root/'results.json').read_text());protocol=results['protocol'];frozen=json.loads((a.root/'ALL_PREDICTIONS_FROZEN.json').read_text())
    assert digest(a.root/'PROTOCOL.json')==frozen['protocol_sha256']
    assert digest(a.source/'ALL_PREDICTIONS_FROZEN.json')==protocol['source_frozen_sha256']
    for name,sha in protocol['predictor_sources_sha256'].items():assert digest(Path(__file__).with_name(name))==sha
    originals=a.source.parent/protocol['original_folder'];masks=baselines=0
    assert len(results['cases'])==49 and len(protocol['conversions'])==7 and len(STRATEGIES)==20
    for row in results['cases']:
        case=row['case'];out=a.root/'cases'/case;src=a.source/'cases'/case;orig=originals/'cases'/case
        verify(src,json.loads((src/'FROZEN.json').read_text()))
        support=tifffile.imread(src/'tissue.native.tif')>0;truth=tifffile.imread(orig/'reference.native.tif');valid=truth>0
        assert digest(orig/'reference.native.tif')==row['sample']['reference_native_sha256']
        assert int(valid.sum())==row['coverage']['annotated_pixels']
        assert json.loads((src/'encoder_coverage.json').read_text())['all_original_tokens_covered']
        coarse=tifffile.imread(orig/'coarse.tif');allowed=set(map(int,np.unique(coarse)))|{0}
        for conversion in protocol['conversions']:
            folder=out/conversion;record=json.loads((folder/'FROZEN.json').read_text())
            assert digest(folder/'FROZEN.json')==frozen['conversion_manifest_sha256'][case+'/'+conversion];verify(folder,record)
            assert digest(src/'FROZEN.json')==record['source_case_manifest_sha256']
            assert digest(src/conversion/'representation.png')==record['rgb_sha256']
            assert digest(orig/'coarse.tif')==record['source_coarse_sha256']
            current=tifffile.imread(src/conversion/'bora.native.tif')
            for strategy in STRATEGIES:
                target=folder/strategy;mask=tifffile.imread(target/'bora.native.tif')
                assert mask.shape==truth.shape and np.all(mask[support]>0) and np.all(mask[~support]==0)
                assert set(map(int,np.unique(mask)))<=allowed
                assert np.array_equal(np.array(Image.open(target/'mask.png').convert('RGBA'))[...,3]>0,support)
                difference=np.array(Image.open(target/'difference.mask.png').convert('RGBA'))[...,3]>0
                assert np.array_equal(difference,mask!=current)
                score=row['conversions'][conversion]['strategies'][strategy]['score']
                assert score['valid_pixels']==int(valid.sum())
                assert set(score['per_class'])==set(map(str,np.unique(truth[valid])))
                if strategy=='B01':assert np.array_equal(mask,current);baselines+=1
                masks+=1
        print(json.dumps(dict(audited=case)),flush=True)
    visuals=json.loads((a.root/'VISUAL_MANIFEST.json').read_text())['files_sha256']
    for name,sha in visuals.items():assert digest(a.root/name)==sha
    audit=dict(cases=49,conversions=7,strategies=20,native_masks=masks,pixel_identical_current_baselines=baselines,
        all_detected_tissue_assigned=True,no_final_background_labels=True,original_quality_support_preserved=True,
        all_reference_classes_scored=True,exact_difference_maps=True,all_frozen_hashes_verified=True,
        source_inputs_unchanged=True,predictor_source_hashes_verified=True,verified_visuals=len(visuals),all_checks_passed=True)
    save_json(a.root/'AUDIT.json',audit);print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
