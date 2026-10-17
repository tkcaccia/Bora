import numpy as np
import pytest
from experiments.bora_cycles.metrics import RegionEvaluator
from experiments.bora_cycles.small_region_atlas import components


def test_absent_class_support_limit_and_coarse_recovery_are_distinct_flags():
    truth = np.zeros((18, 20), np.uint8)
    truth[2:5, 2:6] = 1
    truth[10:13, 3:7] = 2
    truth[10:14, 12:16] = 1
    coarse = np.ones_like(truth)
    support = np.ones_like(truth, bool);support[2:5, 2:6] = False
    prediction = np.ones_like(truth);prediction[~support] = 0
    prediction[10:14, 12:16] = 2
    a = components(truth, support, coarse, dict(test=prediction), [.5, .25])
    rows = a["components"]
    assert len(rows) == 3
    by_box = {tuple(r["native_bbox_xywh"]): r for r in rows}
    removed = by_box[(2, 2, 4, 3)]
    absent = by_box[(3, 10, 4, 3)]
    lost = by_box[(12, 10, 4, 4)]
    assert removed["common_support_cannot_reach_50pct"] and not removed["class_absent_from_original_coarse"]
    assert absent["class_absent_from_original_coarse"] and not absent["common_support_cannot_reach_50pct"]
    assert lost["original_coarse_reaches_50pct"] and lost["all_methods_completely_miss"]
    assert lost["area_um2"] == 2
    assert not lost["common_support_cannot_reach_50pct"] and not lost["class_absent_from_original_coarse"]


def test_small_bin_uses_physical_area_and_agrees_with_saved_component_scores():
    truth = np.ones((21, 21), np.uint8)
    prediction = truth.copy();prediction[:12] = 0
    support = np.ones_like(truth, bool)
    score = RegionEvaluator(truth, [.5, .5]).score(prediction)
    a = components(truth, support, truth, dict(test=prediction), [.5, .5], dict(test=score))
    assert len(a["components"]) == 1
    assert a["components"][0]["area_um2"] == 110.25
    assert a["components"][0]["no_method_reaches_50pct"]
    large = components(truth, support, truth, dict(test=prediction), [1, 1])
    assert large["components"] == []
    score["regions"]["small_le_400_um2"]["completely_missed"] = 1
    with pytest.raises(ValueError, match="differ from saved"):
        components(truth, support, truth, dict(test=prediction), [.5, .5], dict(test=score))
