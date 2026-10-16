import numpy as np
import pytest
from experiments.virchow2_ensemble.run import mean_evidence
from experiments.virchow2_ensemble.score import summarize


def test_aligns_nonconsecutive_class_ids_before_combination():
    a=np.array([[[.7,.2,.1]]],np.float32)
    b=np.array([[[.2,.6,.2]]],np.float32) # order 7,1,3
    c=np.array([[[.4,.1,.5]]],np.float32) # order 3,7,1
    result=mean_evidence([a,b,c],[[1,3,7],[7,1,3],[3,7,1]])
    assert result[0,0]==pytest.approx([.6,.8/3,.4/3])
    assert result.sum()==pytest.approx(1)


def test_refuses_incompatible_vocabularies_and_invalid_mass():
    p=np.array([[[.4,.6]]],np.float32)
    with pytest.raises(ValueError,match='Unmatched'):
        mean_evidence([p,p,p],[[1,3],[1,3],[1,7]])
    with pytest.raises(ValueError,match='mass'):
        mean_evidence([p,p,p*2],[[1,3]]*3)


def test_two_component_fusion_does_not_require_rejected_third_method():
    a=np.array([[[.8,.2]]],np.float32);b=np.array([[[.6,.4]]],np.float32)
    result=mean_evidence([a,b],[[1,7],[7,1]])
    assert result[0,0]==pytest.approx([.6,.4])


def test_paired_summary_keeps_dataset_weight_and_negative_results():
    def case(dataset,baseline,combined):
        return dict(dataset=dataset,controls={k:dict(macro_iou=.5) for k in ('coarse','he')},
            methods={k:dict(after=dict(macro_iou=v)) for k,v in [('09',baseline),('ensemble',combined)]})
    rows=[case('tiger',.6,.7),case('tiger',.6,.7),case('bcss',.6,.3)]
    result=summarize(rows,bootstrap=50)
    assert result['dataset_balanced']['ensemble']==pytest.approx(.5)
    assert result['dataset_balanced_delta']==pytest.approx(-.1)
    assert (result['wins'],result['losses'],result['ties'])==(2,1,0)
    assert result['paired_delta_bootstrap95']==pytest.approx([-.1,-.1])
