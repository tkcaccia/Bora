import numpy as np

from experiments.bora_extension.tissue_fields import choose_field


CATALOG=dict(labels={1:'other tissue',2:'tumor',3:'rare tissue'},ignore=[0],task='fixture')


def test_unknown_mpp_small_image_cannot_be_resized_to_qualify():
    raw=np.ones((500,500),np.uint8);raw[:,250:]=2
    rgb=np.full((500,500,3),[180,100,150],np.uint8)
    selected,reason=choose_field(rgb,raw,CATALOG,None)
    assert selected is None
    assert reason['window_failure_counts']['insufficient_real_field_at_recorded_sampling']>0


def test_ignore_does_not_supply_second_group_and_rare_fine_class_is_kept():
    raw=np.zeros((1280,1280),np.uint8);raw[:,:640]=1
    rgb=np.full((1280,1280,3),[180,100,150],np.uint8)
    selected,reason=choose_field(rgb,raw,CATALOG,None)
    assert selected is None and reason['reason']=='single_tissue_group'
    raw[:,640:]=2;raw[500,500]=3
    selected,reason=choose_field(rgb,raw,CATALOG,None)
    assert reason is None
    rare=selected['raw_to_compact'][3]
    assert np.count_nonzero(selected['reference']==rare)==1
    assert rare not in selected['eligibility']['observed_groups']


def test_glass_only_negative_cannot_masquerade_as_second_tissue_group():
    raw=np.ones((1280,1280),np.uint8);raw[:,640:]=2
    rgb=np.full((1280,1280,3),[180,100,150],np.uint8);rgb[:,:640]=255
    catalog=dict(CATALOG,valid_negative_raw_label=1)
    selected,reason=choose_field(rgb,raw,catalog,None)
    assert selected is None
    assert reason['window_failure_counts']['fewer_than_two_groups_with_stained_tissue_observations']>0
    rgb[:,:640]=[210,140,170]
    selected,reason=choose_field(rgb,raw,catalog,None)
    assert reason is None and selected['eligibility']['eligible']


def test_coarse_source_cannot_gain_eligibility_by_upsampling():
    raw=np.ones((700,700),np.uint8);raw[:,350:]=2
    rgb=np.full((700,700,3),[180,100,150],np.uint8)
    selected,reason=choose_field(rgb,raw,CATALOG,[1.,1.])
    assert selected is None
    raw=np.ones((1200,1200),np.uint8);raw[:,600:]=2
    rgb=np.full((1200,1200,3),[180,100,150],np.uint8)
    selected,reason=choose_field(rgb,raw,CATALOG,[1.,1.])
    assert reason is None and min(selected['reference'].shape)>=896
    assert min(selected['native_capacity_224_yx'])>=4
