from itertools import combinations

import numpy as np

from bora.boundary import BoundaryConfig, _interior
from experiments.gigatime_flash.group_markers import group_selections, refine_interfaces


def test_group_selection_cannot_be_dominated_by_one_easy_opponent():
    contrasts = {"one_easy": [.99, .05, .05], "all_pairs": [.50, .50, .50], "third": [.30, .30, .30]}
    records = []
    for name, effects in contrasts.items():
        for (a, b), effect in zip([(1,2), (1,3), (1,4)], effects):
            records.append({"channel":name,"group_a":a,"group_b":b,
                            "rank_biserial_a_minus_b":effect,"nominal_p":.01,
                            "bh_adjusted_nominal_p_all_pairs_all_markers":.02})
        for a,b in [(2,3),(2,4),(3,4)]:
            records.append({"channel":name,"group_a":a,"group_b":b,
                            "rank_biserial_a_minus_b":.1,"nominal_p":.2,
                            "bh_adjusted_nominal_p_all_pairs_all_markers":.2})
    result = group_selections({"eligible_channels":list(contrasts), "group_tile_counts":{str(i):10 for i in range(1,5)},"pairwise":records})
    assert result["per_group"]["1"]["selected_channels_rgb"] == ["all_pairs","third","one_easy"]
    assert result["per_pair"]["1-2"]["selected_channels_rgb"][0] == "one_easy"


def test_two_class_wrapper_is_the_same_rgb_wand_when_only_one_interface_exists():
    yy,xx = np.mgrid[:128,:128]
    coarse = np.where(xx < 64, 1, 2).astype(np.uint8)
    truth = np.where(xx < 64+9*np.sin(yy/17),1,2).astype(np.uint8)
    rgb = np.where((truth==1)[...,None],[210,40,30],[30,140,210]).astype(np.uint8)
    config = BoundaryConfig(downsample=1)
    expected = _interior(rgb,coarse,32,config)
    actual,meta = refine_interfaces(coarse,{(1,2):rgb},32,config)
    np.testing.assert_array_equal(actual,expected)
    assert np.mean(actual==truth) > np.mean(coarse==truth)
    assert meta["protected_junction_work_pixels"] == 0


def test_pair_merge_is_order_independent_and_protects_other_labels_and_background():
    yy,xx = np.mgrid[:128,:128]
    coarse=np.where(xx<64,1,np.where(yy<64,2,3)).astype(np.uint8)
    coarse[:6]=0
    rgb=np.full((128,128,3),120,np.uint8)
    images={pair:rgb for pair in combinations([1,2,3],2)}
    config=BoundaryConfig(downsample=1)
    one,meta=refine_interfaces(coarse,images,32,config)
    two,_=refine_interfaces(coarse,dict(reversed(list(images.items()))),32,config)
    np.testing.assert_array_equal(one,two)
    assert set(np.unique(one)) == {0,1,2,3}
    assert (one[:6]==0).all()
    # These junction pixels are equally close to two eligible interfaces.
    np.testing.assert_array_equal(one[63:65,64],coarse[63:65,64])
    assert (one[coarse>0]>0).all()
    assert meta["protected_junction_work_pixels"] > 0
