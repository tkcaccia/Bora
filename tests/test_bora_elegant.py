import numpy as np
import pytest
from experiments.bora_elegant.core import Evidence,predict,TITLES


def test_no_margin_or_hard_anchors_and_completed_labels_not_used():
    rgb=np.zeros((180,270,3),np.uint8);rgb[:,:135]=[220,45,65];rgb[:,135:]=[55,165,75]
    coarse=np.ones((180,270),np.uint8);coarse[:,135:]=7;coarse[:,:40]=0
    c=Evidence(rgb,coarse,np.ones(coarse.shape,bool),[.48,.52])
    assert np.all(c.weight[c.known]>0) and np.all(c.weight[~c.known]==0)
    for key in TITLES:
        mask,details=predict(key,rgb,coarse,np.ones(coarse.shape,bool),[.48,.52],palette=[[220,45,65]]+[[0,0,0]]*5+[[55,165,75]])
        assert set(np.unique(mask))=={1,7} and mask.shape==c.shape
        assert np.mean(mask[:,10:c.shape[1]//2-10]==1)>.85
        assert np.mean(mask[:,c.shape[1]//2+10:]==7)>.85
        assert details['hard_anchors'] is False and details['fixed_margin'] is False


def test_whole_field_can_correct_wrong_coarse_tile_without_freezing_center():
    rgb=np.zeros((270,270,3),np.uint8);rgb[:,:180]=[220,45,65];rgb[:,180:]=[55,165,75]
    coarse=np.ones((270,270),np.uint8);coarse[:,180:]=7;coarse[90:180,90:180]=7
    mask,_=predict('E02',rgb,coarse,np.ones(coarse.shape,bool),[.5,.5])
    assert mask[67,67]==1


def test_single_class_keeps_nonconsecutive_id_and_native_geometry():
    rgb=np.full((90,90,3),180,np.uint8);coarse=np.full((90,90),9,np.uint8)
    for key in TITLES:
        mask,_=predict(key,rgb,coarse,np.ones(coarse.shape,bool),[.5,.5])
        assert mask.shape==(45,45) and np.all(mask==9)


def test_known_palette_recovers_identifiable_class_probabilities():
    from experiments.bora_elegant.core import unmix
    colours=np.array([[220,45,65],[55,165,75],[55,110,230]],np.float32)
    rng=np.random.default_rng(2);q=rng.dirichlet([2,2,2],size=(90,90)).astype(np.float32)
    rgb=np.rint(q@colours).astype(np.uint8);coarse=np.repeat(np.arange(1,4,dtype=np.uint8),30)[None,:].repeat(90,axis=0)
    c=Evidence(rgb,coarse,np.ones(coarse.shape,bool),[1.,1.])
    recovered=unmix(c,colours)
    assert np.max(np.abs(recovered-q))<.012
    assert c.details['simplex_identifiable'] is True


def test_full_prediction_score_and_render_contract_without_fine_reference(tmp_path):
    import json
    from PIL import Image
    import tifffile
    from experiments.bora_elegant.full import freeze,score_one,render_one
    from experiments.virchow2_inverse.common import digest
    from experiments.bora_strategies.registry import CONVERSIONS
    source=tmp_path/'inputs';root=tmp_path/'out';src=source/'cases'/'synthetic';src.mkdir(parents=True)
    rgb=np.zeros((180,180,3),np.uint8);rgb[:,:90]=[220,45,65];rgb[:,90:]=[55,165,75]
    coarse=np.ones((180,180),np.uint8);coarse[:,90:]=3;support=np.ones_like(coarse)
    Image.fromarray(rgb).save(src/'original.png');Image.fromarray(rgb).save(src/'original.native.png');files={}
    for name,array in [('coarse.tif',coarse),('coarse.completed.tif',coarse),('tissue.native.tif',support),('tissue.working.tif',support)]:
        tifffile.imwrite(src/name,array);files[name]=digest(src/name)
    for m in CONVERSIONS:
        (src/m).mkdir();Image.fromarray(rgb).save(src/m/'representation.png');files[m+'/representation.png']=digest(src/m/'representation.png')
    (src/'INPUTS_FROZEN.json').write_text(json.dumps(dict(files_sha256=files)))
    palette=[[220,45,65],[0,0,0],[55,165,75]]
    sample=dict(case='synthetic',dataset='tiger',benchmark_cohort='development',native_shape_yx=[180,180],working_shape_yx=[180,180],native_mpp_xy=[.5,.5],working_mpp_xy=[.5,.5],palette=palette,class_names={'1':'tumor','3':'stroma'})
    selection=dict(methods=['B02','B19','E09','E11'])
    manifests=freeze((source,root,sample,selection));assert len(manifests)==5
    assert not (src/'reference.native.tif').exists()
    for m in CONVERSIONS:
        rec=json.loads((root/'cases/synthetic'/m/'FROZEN.json').read_text());assert rec['fine_reference_read'] is False
        assert set(rec['records'])==set(selection['methods'])
        for key in selection['methods']:assert set(np.unique(tifffile.imread(root/'cases/synthetic'/m/key/'bora.native.tif')))=={1,3}
    tifffile.imwrite(src/'reference.native.tif',coarse);sample['reference_native_sha256']=digest(src/'reference.native.tif');(root/'cases/synthetic/SAMPLE.json').write_text(json.dumps(sample))
    protocol=dict(conversions=list(CONVERSIONS),strategies={k:dict(title=k,hypothesis='synthetic') for k in selection['methods']})
    row,records=score_one((source,root,'synthetic',protocol));assert len(records)==20
    render_one((source,root,row,protocol))
    assert (root/'cases/synthetic/ensemble/E11/view.mask.png').exists()


def test_public_cli_preserves_geometry_spacing_and_removes_background(tmp_path):
    import json
    from PIL import Image
    import tifffile
    from bora.cli import main
    rgb=np.zeros((90,180,3),np.uint8);rgb[:,:90]=[220,45,65];rgb[:,90:]=[55,165,75]
    coarse=np.ones((90,180),np.uint8);coarse[:,90:]=3;tissue=np.ones_like(coarse);tissue[:10,:10]=0
    Image.fromarray(rgb).save(tmp_path/'rgb.png');tifffile.imwrite(tmp_path/'coarse.tif',coarse);tifffile.imwrite(tmp_path/'tissue.tif',tissue)
    (tmp_path/'palette.json').write_text(json.dumps([[220,45,65],[0,0,0],[55,165,75]]))
    main(['evidence',str(tmp_path/'rgb.png'),str(tmp_path/'coarse.tif'),'--tissue-mask',str(tmp_path/'tissue.tif'),'--palette',str(tmp_path/'palette.json'),'--mpp-x','.48','--mpp-y','.52','--output',str(tmp_path/'out.ome.tif')])
    with tifffile.TiffFile(tmp_path/'out.ome.tif') as tif:
        mask=tif.asarray();assert mask.shape==coarse.shape and np.all(mask[:10,:10]==0) and np.all(mask[tissue>0]>0)
        assert 'PhysicalSizeX="0.48"' in tif.ome_metadata and 'PhysicalSizeY="0.52"' in tif.ome_metadata
