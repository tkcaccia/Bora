import numpy as np
import pytest
from experiments.bora_strategies.context import Context,simplex
from experiments.bora_strategies.methods import run
from experiments.bora_strategies.registry import STRATEGIES,CONSENSUS,REJECTED_STRATEGIES


def test_simplex_projection_respects_class_mass():
    a=np.random.default_rng(4).normal(size=(12,10,5));p=simplex(a)
    assert np.all(p>=0) and np.allclose(p.sum(-1),1)


def test_context_never_trains_on_completed_unlabelled_tiles():
    he=np.full((100,180,3),150,np.uint8);original=np.ones((100,180),np.uint8);original[:,90:]=0;original[:50,:45]=3
    completed=original.copy();completed[:,90:]=3
    c=Context(he,he,original,completed,np.ones(original.shape,bool),[.5,.5])
    assert np.all(c.core[c.original==0]==-1) and np.all(c.anchors[c.original==0]==-1)
    assert set(c.ids)=={1,3}


def test_retained_solvers_keep_nonconsecutive_label_geometry_and_cover_the_field():
    rgb=np.zeros((180,270,3),np.uint8);rgb[:,:135]=[210,50,80];rgb[:,135:]=[50,170,80]
    original=np.ones((180,270),np.uint8);original[:,135:]=7
    c=Context(rgb,rgb,original,original,np.ones(original.shape,bool),[.48,.52])
    for key in STRATEGIES:
        result=c.finish(run(key,c))
        assert result.shape==c.shape and set(np.unique(result))=={1,7}
    assert set(CONSENSUS)<=set(STRATEGIES) and set(STRATEGIES).isdisjoint(REJECTED_STRATEGIES)
    for key in REJECTED_STRATEGIES:
        with pytest.raises(ValueError):run(key,c)


def test_correlated_conversions_do_not_inflate_case_bootstrap():
    from experiments.bora_strategies.score import summarize
    cases=[]
    for j in range(6):
        values={k:dict(score=dict(macro_iou=.5+j*.02+(i%3)*.01,boundary_f1_at_um={'8':.7})) for i,k in enumerate(STRATEGIES)}
        cases.append(dict(dataset='a' if j<3 else 'b',tiles=dict(macro_iou=.45+j*.02),conversions={m:dict(strategies=values) for m in ('09','19')}))
    single=summarize(cases,['09']);duplicate=summarize(cases,['09','19'])
    for a,b in zip(single,duplicate):assert a['strategy']==b['strategy'] and a['paired_delta_bootstrap95']==b['paired_delta_bootstrap95']


def test_runner_freezes_every_retained_solver_without_reference_access(tmp_path):
    import json
    from PIL import Image
    import tifffile
    from experiments.bora_strategies.run import freeze_case
    from experiments.bora_strategies.registry import CONVERSIONS
    from experiments.virchow2_inverse.common import digest
    source=tmp_path/'inputs';root=tmp_path/'outputs';src=source/'cases'/'synthetic';src.mkdir(parents=True)
    rgb=np.zeros((180,180,3),np.uint8);rgb[:,:90]=[210,50,80];rgb[:,90:]=[50,170,80]
    coarse=np.ones((180,180),np.uint8);coarse[:,90:]=3
    Image.fromarray(rgb).save(src/'original.png');Image.fromarray(rgb).save(src/'original.native.png')
    files={}
    for name,array in [('coarse.tif',coarse),('coarse.completed.tif',coarse),('tissue.native.tif',np.ones_like(coarse)),('tissue.working.tif',np.ones_like(coarse))]:
        tifffile.imwrite(src/name,array);files[name]=digest(src/name)
    for m in CONVERSIONS:
        (src/m).mkdir();Image.fromarray(rgb).save(src/m/'representation.png');files[m+'/representation.png']=digest(src/m/'representation.png')
    files['original.png']=digest(src/'original.png');(src/'INPUTS_FROZEN.json').write_text(json.dumps(dict(files_sha256=files)))
    sample=dict(case='synthetic',native_shape_yx=[180,180],native_mpp_xy=[.5,.5],working_mpp_xy=[.5,.5],coarse_sha256=files['coarse.tif'])
    manifests=freeze_case(source,root,sample)
    assert len(manifests)==len(CONVERSIONS)
    for m in CONVERSIONS:
        record=json.loads((root/'cases'/'synthetic'/m/'FROZEN.json').read_text())
        assert set(record['records'])==set(STRATEGIES)
        assert record['fine_reference_read'] is False
        for k in STRATEGIES:assert set(np.unique(tifffile.imread(root/'cases'/'synthetic'/m/k/'bora.native.tif')))=={1,3}
    # No reference mask is supplied: runner prediction cannot depend on fine annotations.
    # Supply a fine reference only after freezing, then exercise scoring and rendering.
    from experiments.bora_strategies.score import score_case
    from experiments.bora_strategies.publish import render
    tifffile.imwrite(src/'reference.native.tif',coarse)
    sample.update(dataset='tiger',benchmark_cohort='new_extension',reference_native_sha256=digest(src/'reference.native.tif'),palette=[[210,50,80],[0,0,0],[50,170,80]],class_names={'1':'tumor','3':'stroma'})
    (root/'cases'/'synthetic'/'SAMPLE.json').write_text(json.dumps(sample))
    row,records=score_case('synthetic',root,source,dict(conversions=list(CONVERSIONS)))
    assert len(records)==len(CONVERSIONS)*len(STRATEGIES)
    assert row['tiles']['macro_iou']==1.
    rendered=render(root,source,row)
    assert rendered['display_mpp_xy']==[.5,.5]
    for m in CONVERSIONS:
        assert (root/'cases'/'synthetic'/m/'contact.jpg').exists()
        for k in STRATEGIES:
            pred=tifffile.imread(root/'cases'/'synthetic'/m/k/'bora.native.tif')
            assert np.array_equal(np.array(Image.open(root/'cases'/'synthetic'/m/k/'mask.png')),pred)
