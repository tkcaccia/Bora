import numpy as np
import pytest

from experiments.gigatime_flash.pca_rgb import class_marker_columns, fit_pca_image, pc_scores_to_rgb


def test_class_winners_are_preserved_and_shared_markers_are_not_repeated_in_primary():
    selection={"per_group":{str(g):{"ranking":[{"channel":c,"minimum_pairwise_direction_free_auc":.8}]}
                             for g,c in [(1,"CD138"),(2,"CD138"),(3,"CD3"),(4,"CD3"),(5,"T-bet")]}}
    assignments,unique=class_marker_columns(selection)
    assert [r["marker"] for r in assignments]==["CD138","CD138","CD3","CD3","T-bet"]
    assert unique==["CD138","CD3","T-bet"]


def test_pca_is_invariant_to_feature_units_and_preserves_the_original_frame():
    rng=np.random.default_rng(41)
    features=rng.normal(size=(13,17,6)) @ np.diag([1,2,3,.1,.03,.2])
    support=np.ones((13,17),bool)
    pcs,meta=fit_pca_image(features,support,["r","g","b","a","c","d"])
    rescaled=features*np.array([255,255,255,1,1,1])+np.array([100,30,20,.5,.2,1])
    other,_=fit_pca_image(rescaled,support,["r","g","b","a","c","d"])
    assert pcs.shape==(13,17,3)
    np.testing.assert_allclose(pcs,other,atol=1e-5)
    components=np.asarray(meta["components_pc_by_feature"])
    np.testing.assert_allclose(components @ components.T,np.eye(3),atol=1e-12)
    for row in components:assert row[np.argmax(abs(row))]>0
    assert np.all(np.diff(meta["explained_variance_ratio"])<=0)
    assert meta["fine_reference_used"] is False


def test_pca_fit_excludes_pixels_outside_coarse_support():
    rng=np.random.default_rng(42)
    features=rng.normal(size=(12,15,4))
    support=np.ones((12,15),bool);support[:3]=False
    pcs,meta=fit_pca_image(features,support,list("abcd"))
    changed=features.copy();changed[:3]*=10000
    other,other_meta=fit_pca_image(changed,support,list("abcd"))
    assert meta["feature_mean"]==other_meta["feature_mean"]
    assert meta["components_pc_by_feature"]==other_meta["components_pc_by_feature"]
    np.testing.assert_array_equal(pcs[support],other[support])


def test_saved_transform_reproduces_native_pc_scores():
    rng=np.random.default_rng(43)
    features=rng.normal(size=(11,12,5))
    pcs,m=fit_pca_image(features,np.ones((11,12),bool),list("abcde"))
    z=(features-np.array(m["feature_mean"]))/np.array(m["feature_scale"])
    expected=(z-np.array(m["pca_center_after_standardization"])) @ np.array(m["components_pc_by_feature"]).T
    np.testing.assert_allclose(expected,pcs,atol=3e-7)


def test_pc_rgb_preserves_component_order_and_handles_negative_and_constant_scores():
    scores=np.zeros((10,11,3),np.float32)
    scores[...,0]=np.arange(11)[None,:]-5
    scores[...,1]=np.arange(10)[:,None]-4.5
    scores[...,2]=2
    rgb,metadata=pc_scores_to_rgb(scores,np.ones((10,11),bool))
    assert rgb.shape==scores.shape and rgb.dtype==np.uint8
    assert rgb[5,0,0]==0 and rgb[5,-1,0]==255
    assert rgb[0,5,1]==0 and rgb[-1,5,1]==255
    assert (rgb[...,2]==0).all()
    assert metadata["mapping"]=={"R":"PC1","G":"PC2","B":"PC3"}


@pytest.mark.parametrize("case",["constant","nonfinite","empty_support"])
def test_pca_rejects_unusable_inputs(case):
    features=np.ones((5,6,3));support=np.ones((5,6),bool)
    if case=="nonfinite":features[0,0,0]=np.nan
    if case=="empty_support":support[:]=False
    with pytest.raises(ValueError):fit_pca_image(features,support,list("rgb"))
