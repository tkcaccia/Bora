import json
import cv2
import numpy as np
import pytest

from experiments.bora_cycles.cycle_eleven import predict_native
from experiments.bora_cycles.eleven_manifest import CHILDREN,SUPPORT_METHODS,SURVIVORS
from experiments.bora_cycles.methods import predict
from experiments.bora_cycles.native_eleven import contrast_components
from experiments.bora_cycles.registry import candidates
from experiments.bora_cycles.review import validate_children


def fixture():
    yy,xx=np.indices((96,270));palette=np.array([[200,40,30],[25,45,210],[30,185,65]],np.uint8)
    pattern=np.clip(((xx+10*np.sin(yy/11))/90).astype(int),0,2)
    rgb=palette[pattern];coarse=(xx//90+1).astype(np.uint8);coarse[:12,:12]=0
    support=np.ones(coarse.shape,bool);support[:8]=False
    return rgb,coarse,support,palette


@pytest.mark.parametrize('key',[r['id'] for r in CHILDREN if r['id'] not in SUPPORT_METHODS])
def test_whole_field_candidates_are_finite_deterministic_and_do_not_modify_inputs(key):
    rgb,coarse,support,palette=fixture();before=[a.copy() for a in (rgb,coarse,support)]
    a,report=predict(key,rgb,coarse,support,[1,1],palette,he=rgb)
    b,_=predict(key,rgb,coarse,support,[1,1],palette,he=rgb)
    np.testing.assert_array_equal(a,b)
    assert a.shape==coarse.shape and set(np.unique(a))<={1,2,3} and a.min()>0
    assert report['parent']==candidates(11)[key].parent and not report['fine_reference_used']
    assert not report['hard_anchors'] and not report['fixed_margin']
    json.dumps(report,allow_nan=False)
    for old,now in zip(before,(rgb,coarse,support)):np.testing.assert_array_equal(old,now)


@pytest.mark.parametrize('key',sorted(SUPPORT_METHODS))
def test_native_support_can_restore_tissue_and_retains_parent_class_fields(key):
    rgb,coarse,support,palette=fixture();native=np.repeat(np.repeat(rgb,2,0),2,1)
    labels,own,report=predict_native(key,rgb,coarse,support,[1,1],palette,rgb,native,[.5,.5])
    # The top band is deliberately outside common support yet contains stain.
    assert own[:16].all() and np.all(labels[:16]>0)
    parent,_=predict('J01',rgb,coarse,support,[1,1],palette,he=rgb)
    expected=cv2.resize(parent,(native.shape[1],native.shape[0]),interpolation=cv2.INTER_NEAREST)
    np.testing.assert_array_equal(labels[own],expected[own])
    assert np.all(labels[~own]==0)
    assert report['native_support']['own_native_support'] and not report['native_support']['reference_used']
    json.dumps(report,allow_nan=False)


def test_lineage_has_three_new_mechanisms_per_actual_survivor():
    assert SURVIVORS==('J01','M04','N08','N10','C17') and len(candidates(11))==20
    assert validate_children(dict(survivors=list(SURVIVORS)),CHILDREN)
    old={r.mechanism for cycle in range(1,11) for r in candidates(cycle).values()}
    assert not old.intersection(r['intervention'] for r in CHILDREN)


def test_compiled_stain_partition_matches_independent_hand_graph():
    cost=np.array([.1,.2,.3,.4,.9]);a=np.array([0,1,3,4,2]);b=a+1
    np.testing.assert_array_equal(contrast_components(a,b,cost,6,1.,.3),[0,0,0,1,1,2])
    # Doubling physical pixel area halves the area-normalized merge allowance.
    separate=contrast_components(np.array([0]),np.array([1]),np.array([.2]),2,2.,.3)
    np.testing.assert_array_equal(separate,[0,1])
    with pytest.raises(ValueError,match='Sorted finite'):
        contrast_components(a,b,cost[::-1],6,1.,.3)
    with pytest.raises(ValueError,match='Sorted finite'):
        contrast_components(a,b,cost,6,float('nan'),.3)


@pytest.mark.parametrize('key',sorted(SUPPORT_METHODS))
def test_prediction_freeze_preserves_owned_support_without_reading_fine_mask(tmp_path,key):
    from pathlib import Path
    from PIL import Image
    import tifffile
    from experiments.bora_cycles.run import freeze_case,digest
    rgb,coarse,support,palette=fixture();source=tmp_path/'source';root=tmp_path/'result';root.mkdir()
    case=source/'cases'/'fixture';(case/'ensemble').mkdir(parents=True)
    native=np.repeat(np.repeat(rgb,2,0),2,1);native_support=np.repeat(np.repeat(support,2,0),2,1)
    for name,array in [('original.png',rgb),('original.native.png',native),('ensemble/representation.png',rgb)]:
        Image.fromarray(array).save(case/name)
    for name,array in [('coarse.tif',coarse),('tissue.working.tif',support.astype(np.uint8)),('tissue.native.tif',native_support.astype(np.uint8))]:
        tifffile.imwrite(case/name,array)
    paths=['original.png','original.native.png','ensemble/representation.png','coarse.tif','tissue.working.tif','tissue.native.tif']
    (case/'INPUTS_FROZEN.json').write_text(json.dumps(dict(files_sha256={n:digest(case/n) for n in paths})))
    (case/'reference.native.tif').write_bytes(b'Not a readable mask; prediction must never decode it')
    (root/'PROTOCOL.json').write_text(json.dumps(dict(cycle=11)))
    sample=dict(case='fixture',dataset='fixture',palette=palette.tolist(),native_shape_yx=list(native.shape[:2]),
        working_shape_yx=list(coarse.shape),native_mpp_xy=[.5,.5],working_mpp_xy=[1,1])
    freeze_case((source,root,sample,[key],2))
    record=json.loads((root/'cases/fixture/PREDICTIONS.json').read_text())
    assert record['methods'][key]['status']=='ok',record['methods'][key]
    assert not record['fine_reference_read'] and key+'/support.native.tif' in record['files_sha256']
    assert {'RARITY.npz','RARITY.json'}<=set(record['files_sha256'])
    rarity=json.loads((root/'cases/fixture/RARITY.json').read_text())
    assert rarity['diagnostic_only'] and not rarity['affects_candidate_labels'] and not rarity['fine_reference_used']
    mask=tifffile.imread(root/'cases/fixture'/key/'native.tif')
    own=tifffile.imread(root/'cases/fixture'/key/'support.native.tif')>0
    assert np.all(mask[own]>0) and np.all(mask[~own]==0)
    assert np.all(mask[~native_support]>0)  # Common clip was actually bypassed.
