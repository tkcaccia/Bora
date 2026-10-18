import json
import numpy as np
import pytest
from experiments.bora_cycles.twelve_manifest import CHILDREN,NATIVE_CHILDREN,SURVIVORS
from experiments.bora_cycles.registry import candidates
from experiments.bora_cycles.review import validate_children
from experiments.bora_cycles.methods import predict
from experiments.bora_cycles.cycle_twelve import predict_native
from experiments.bora_cycles.expanded_metrics import owns_native_support,may_remove_nonwhite


def fixture():
    yy,xx=np.indices((96,270));palette=np.array([[200,40,30],[25,45,210],[30,185,65]],np.uint8)
    pattern=np.clip(((xx+10*np.sin(yy/11))/90).astype(int),0,2)
    rgb=palette[pattern];coarse=(xx//90+1).astype(np.uint8);coarse[:12,:12]=0
    support=np.ones(coarse.shape,bool);support[:8]=False
    return rgb,coarse,support,palette


@pytest.mark.parametrize('key',[r['id'] for r in CHILDREN if r['id'] not in NATIVE_CHILDREN])
def test_candidate_finite_deterministic_and_input_immutable(key):
    rgb,coarse,support,palette=fixture();before=[a.copy() for a in (rgb,coarse,support)]
    a,report=predict(key,rgb,coarse,support,[1,1],palette,he=rgb)
    b,_=predict(key,rgb,coarse,support,[1,1],palette,he=rgb)
    np.testing.assert_array_equal(a,b)
    assert a.shape==coarse.shape and set(np.unique(a))<={1,2,3} and a.min()>0
    assert report['parent']==candidates(12)[key].parent and not report['fine_reference_used']
    assert not report['hard_anchors'] and not report['fixed_margin']
    json.dumps(report,allow_nan=False)
    for old,now in zip(before,(rgb,coarse,support)):np.testing.assert_array_equal(old,now)


@pytest.mark.parametrize('key',sorted(NATIVE_CHILDREN))
def test_native_support_labels_follow_exact_parent_class_decision(key):
    import cv2
    rgb,coarse,support,palette=fixture();native=np.repeat(np.repeat(rgb,2,0),2,1)
    labels,own,report=predict_native(key,rgb,coarse,support,[1,1],palette,rgb,native,[.5,.5])
    parent,_=predict('J01',rgb,coarse,support,[1,1],palette,he=rgb)
    expected=cv2.resize(parent,native.shape[1::-1],interpolation=cv2.INTER_NEAREST)
    np.testing.assert_array_equal(labels[own],expected[own]);assert np.all(labels[~own]==0)
    assert owns_native_support(key,report) and may_remove_nonwhite(key,report)==(key=='Q01')
    forged=json.loads(json.dumps(report));forged['mechanism']='unregistered'
    assert not owns_native_support(key,forged) and not may_remove_nonwhite(key,forged)
    json.dumps(report,allow_nan=False)


def test_cycle12_has_exactly_three_distinct_children_per_signed_parent():
    selection=json.load(open('experiments/bora_cycles/CYCLE_ELEVEN_SELECTION.json'))
    assert tuple(selection['survivors'])==SURVIVORS and len(candidates(12))==20
    assert validate_children(selection,CHILDREN)
    old={r.mechanism for cycle in range(1,12) for r in candidates(cycle).values()}
    assert not old.intersection(r['intervention'] for r in CHILDREN)


@pytest.mark.parametrize('key',sorted(NATIVE_CHILDREN))
def test_prediction_freeze_saves_native_support_without_decoding_fine_reference(tmp_path,key):
    from PIL import Image
    import tifffile
    from experiments.bora_cycles.run import freeze_case,digest
    rgb,coarse,support,palette=fixture();source=tmp_path/'source';root=tmp_path/'result';root.mkdir()
    case=source/'cases'/'fixture';(case/'ensemble').mkdir(parents=True)
    native=np.repeat(np.repeat(rgb,2,0),2,1);native_support=np.repeat(np.repeat(support,2,0),2,1)
    for name,array in [('original.png',rgb),('original.native.png',native),('ensemble/representation.png',rgb)]:Image.fromarray(array).save(case/name)
    for name,array in [('coarse.tif',coarse),('tissue.working.tif',support.astype(np.uint8)),('tissue.native.tif',native_support.astype(np.uint8))]:tifffile.imwrite(case/name,array)
    paths=['original.png','original.native.png','ensemble/representation.png','coarse.tif','tissue.working.tif','tissue.native.tif']
    (case/'INPUTS_FROZEN.json').write_text(json.dumps(dict(files_sha256={n:digest(case/n) for n in paths})))
    (case/'reference.native.tif').write_bytes(b'Prediction must not decode this invalid fine mask')
    (root/'PROTOCOL.json').write_text(json.dumps(dict(cycle=12)))
    sample=dict(case='fixture',dataset='fixture',palette=palette.tolist(),native_shape_yx=list(native.shape[:2]),
        working_shape_yx=list(coarse.shape),native_mpp_xy=[.5,.5],working_mpp_xy=[1,1])
    freeze_case((source,root,sample,[key],2))
    record=json.loads((root/'cases/fixture/PREDICTIONS.json').read_text());entry=record['methods'][key]
    assert entry['status']=='ok',entry
    assert not record['fine_reference_read'] and key+'/support.native.tif' in record['files_sha256']
    mask=tifffile.imread(root/'cases/fixture'/key/'native.tif')
    own=tifffile.imread(root/'cases/fixture'/key/'support.native.tif')>0
    assert np.all(mask[own]>0) and np.all(mask[~own]==0)
