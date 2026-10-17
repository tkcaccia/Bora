"""Regenerate B17 on its true sample coordinates; preserve all other masks."""
import argparse,ast,json,time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import tifffile
from threadpoolctl import threadpool_limits
from experiments.virchow2_inverse.common import digest,save_json
from experiments.virchow2_smoothing.run import verify
from experiments.bora_strategies.context import Context,resize
from experiments.bora_strategies.fields import nonlocal_vote
from experiments.bora_strategies import fields


def correct_case(source,root,case,conversions):
    out=root/'cases'/case;sample=json.loads((out/'SAMPLE.json').read_text());src=source/'cases'/case;orig=source.parent/'virchow2-multicase-2026-10-16/cases'/case
    he=np.array(Image.open(orig/'original.png').convert('RGB'));known=tifffile.imread(orig/'coarse.tif');completed=tifffile.imread(src/'coarse.completed.tif')
    work_support=tifffile.imread(src/'tissue.working.tif')>0;support=tifffile.imread(src/'tissue.native.tif')>0;manifests={};cv2.setNumThreads(2)
    with threadpool_limits(limits=2):
        for conversion in conversions:
            folder=out/conversion;record=json.loads((folder/'FROZEN.json').read_text());verify(folder,record)
            if digest(src/conversion/'representation.png')!=record['rgb_sha256']:raise ValueError('RGB input changed')
            if digest(orig/'coarse.tif')!=record['source_coarse_sha256']:raise ValueError('Coarse input changed')
            rgb=np.array(Image.open(src/conversion/'representation.png').convert('RGB'));c=Context(rgb,he,known,completed,work_support,sample['working_mpp_xy'])
            start=time.monotonic();analysis=c.finish(nonlocal_vote(c));native=resize(analysis,sample['native_shape_yx'],True);native[~support]=0
            assert np.all(native[support]>0) and np.all(native[~support]==0)
            target=folder/'B17';old_native=tifffile.imread(target/'bora.native.tif');baseline=tifffile.imread(folder/'B01/bora.native.tif')
            for name,array in [('mask.analysis.tif',analysis),('bora.native.tif',native)]:tifffile.imwrite(target/name,array,compression='deflate')
            details=dict(seconds=time.monotonic()-start,changed_native_pixels_vs_current=int(np.count_nonzero(native!=baseline)),
                diagnostics=dict(coordinates='probabilities interpolated at actual [::step] sample positions; edge-clamped linear',
                    geometry_correction_changed_native_pixels=int(np.count_nonzero(native!=old_native))),fine_reference_used=False)
            save_json(target/'details.json',details);record['records']['B17']=details
            for name in ['mask.analysis.tif','bora.native.tif','details.json']:record['files_sha256']['B17/'+name]=digest(target/name)
            # Every other file remains guarded by its original immutable hash.
            verify(folder,record);save_json(folder/'FROZEN.json',record);manifests[case+'/'+conversion]=digest(folder/'FROZEN.json')
            print(json.dumps(dict(corrected=case,conversion=conversion,changed_pixels=details['diagnostics']['geometry_correction_changed_native_pixels'])),flush=True)
    return manifests


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);a=p.parse_args()
    old=json.loads((a.archive/'ALL_PREDICTIONS_FROZEN.json').read_text());protocol=json.loads((a.root/'PROTOCOL.json').read_text())
    before=a.archive/'runtime_sources/experiments/bora_strategies/fields.py';after=Path(fields.__file__)
    assert digest(before)==protocol['predictor_sources_sha256']['fields.py']
    for name,sha in protocol['predictor_sources_sha256'].items():
        if name!='fields.py':assert digest(after.with_name(name))==sha
    def functions(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    first=functions(before);second=functions(after)
    for name,value in first.items():
        if name!='nonlocal_vote':assert second[name]==value,f'Unexpected other-method change: {name}'
    assert set(second)-set(first)=={'expand_regular_samples'}
    protocol['predictor_sources_sha256']['fields.py']=digest(after);save_json(a.root/'PROTOCOL.json',protocol)
    manifests={}
    with ProcessPoolExecutor(max_workers=3,mp_context=get_context('spawn')) as pool:
        futures=[pool.submit(correct_case,a.source,a.root,case,protocol['conversions']) for case in protocol['cases']]
        for f in futures:manifests.update(f.result())
    save_json(a.root/'ALL_PREDICTIONS_FROZEN.json',{**old,'protocol_sha256':digest(a.root/'PROTOCOL.json'),'conversion_manifest_sha256':manifests})
    save_json(a.root/'CORRECTION_LINEAGE.json',dict(strategy='B17',reason='regular sample probabilities were expanded with resize centres instead of their actual index-0 sample origin',
        initial_folder=a.archive.name,initial_prediction_manifest_sha256=digest(a.archive/'ALL_PREDICTIONS_FROZEN.json'),
        original_fields_sha256=digest(before),corrected_fields_sha256=digest(after),other_strategy_functions_AST_identical=True,
        other_19_prediction_files_preserved_byte_identically=True,correction_driven_by_synthetic_coordinate_identity_test=True,
        initial_scoring_archived=True,final_predictions_frozen_before_final_scoring=True,fine_reference_read_by_correction=False))


if __name__=='__main__':main()
