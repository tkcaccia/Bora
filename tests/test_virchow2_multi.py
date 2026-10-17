import numpy as np
import pytest
from experiments.virchow2_multi.plan import coarse_mask,group_key,palette
from experiments.virchow2_multi.methods import Context
from experiments.virchow2_multi.score import Evaluator,ranking


def test_ignore_is_not_restored_as_fine_spatial_support():
    truth=np.ones((90,90),np.uint8);truth[:10]=0
    coarse=coarse_mask(truth)
    assert np.all(coarse==1)
    truth[:60]=0
    assert np.all(coarse_mask(truth)==0)


def test_conservative_tiger_case_group():
    assert group_key(dict(dataset="tiger",slide_id="114S"))==group_key(dict(dataset="tiger",slide_id="114B"))=="114"
    assert group_key(dict(dataset="tiger",slide_id="TC_S01_P000159_C001"))=="TC_S01_P000159"




def test_probability_columns_follow_observed_nonconsecutive_labels():
    rng=np.random.default_rng(6);coarse=np.tile(np.repeat([1,3,7,0],90),(360,1)).astype(np.uint8)
    f=rng.normal(size=(26,26,1280)).astype(np.float16)
    image=np.zeros((360,360,3),np.uint8)
    c=Context(image,coarse,f,palette("tiger",list(range(1,8))))
    assert c.classes.tolist()==[1,3,7]
    assert c.p.shape==(26,26,3)
    assert np.allclose(c.colours,np.array(palette("tiger",[1,3,7])))


def test_native_scores_ignore_zero_and_penalize_missing_true_classes():
    truth=np.ones((20,20),np.uint8);truth[:,10:]=2;truth[0]=0
    pred=truth.copy();pred[0]=2
    evaluator=Evaluator(truth,[.25,.25]);assert evaluator.score(pred)["macro_iou"]==1
    truth[4:7,4:7]=3;pred[4:7,4:7]=1
    result=Evaluator(truth,[.25,.25]).score(pred)
    assert result["per_class"]["3"]["iou"]==0
    assert result["macro_iou"]<2/3


def test_boundaries_use_native_axis_spacing():
    truth=np.ones((30,30),np.uint8);truth[:,15:]=2
    pred=np.ones_like(truth);pred[:,19:]=2
    small=Evaluator(truth,[.25,.25]).score(pred)
    large=Evaluator(truth,[2.,2.]).score(pred)
    assert small["boundary_f1_at_um"]["4"]==1
    assert large["boundary_f1_at_um"]["4"]==0
    assert large["boundary_f1_at_um"]["8"]==1


def test_primary_ranking_weights_datasets_not_image_counts():
    from experiments.virchow2_multi.methods import METHODS
    cases=[]
    for dataset in ("tiger","tiger","bcss"):
        methods={}
        for n in METHODS:
            value=(.5 if dataset=="tiger" else .9) if n==9 else .6 if n==10 else .4
            methods[f"{n:02d}"]=dict(after=dict(macro_iou=value),before=dict(macro_iou=.4),rank=1)
        cases.append(dict(dataset=dataset,methods=methods,controls=dict(he=dict(macro_iou=.5))))
    rows=ranking(cases,bootstrap=50)
    assert rows[0]["id"]==9
    assert rows[0]["dataset_balanced_macro_iou"]==pytest.approx(.7)
    assert rows[0]["image_mean_iou"]==pytest.approx(1.9/3)
    assert rows[0]["dataset_balanced_delta_vs_he"]==pytest.approx(.2)
