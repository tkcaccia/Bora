import copy

import numpy as np
import pytest

from experiments.bora_cycles.benchmark_scope import EXPECTED, requires_parent_parity, validate_scope
from experiments.bora_cycles.expanded_metrics import aggregate, native_score
from experiments.bora_cycles.runtime_policy import DEFAULT_POLICY


def contract():
    cases = [f"case{i}" for i in range(349)]
    return dict(cycle=8, cases=cases, input_plan_sha256="plan",
        benchmark_scope=dict(version="expanded349-v1", required_cases=cases[:],
            historical_cases=cases[:149], dataset_counts=EXPECTED.copy(), plan_sha256="plan"))


def test_full_contract_cannot_drop_historical_or_new_cases():
    p = contract()
    validate_scope(p)
    for index in (0, 148, 149, 348):
        incomplete = copy.deepcopy(p)
        incomplete["cases"].pop(index)
        with pytest.raises(ValueError, match="every349"):
            validate_scope(incomplete)
        validate_scope(incomplete, pilot=True)
    for key, value in (("dataset_counts", {}), ("plan_sha256", "changed")):
        changed = copy.deepcopy(p)
        changed["benchmark_scope"][key] = value
        with pytest.raises(ValueError, match="identity"):
            validate_scope(changed)
    with pytest.raises(ValueError, match="expanded"):
        validate_scope(dict(cycle=8, cases=p["cases"][:149]))


def test_lineage_parity_applies_to_all_cases_after_first_expanded_cycle():
    p = contract()
    assert requires_parent_parity(p, "case148")
    assert not requires_parent_parity(p, "case149")
    p["cycle"] = 9
    assert requires_parent_parity(p, "case149")
    validate_scope(dict(cycle=7, cases=list(range(149))))
    with pytest.raises(ValueError, match="149"):
        validate_scope(dict(cycle=7, cases=list(range(148))))


@pytest.mark.parametrize("backend", ["numpy", "cpp"])
def test_rings_removal_counts_correct_negative_but_never_hides_deleted_tumor(backend):
    sample = dict(dataset="rings-tumor", native_mpp_xy=None, pixel_only=True,
        physical_metrics_allowed=False, raw_to_compact={"0": 1, "1": 2})
    truth = np.array([[0, 1, 2], [0, 1, 2]], np.uint8)
    mask = np.array([[2, 0, 0], [1, 1, 2]], np.uint8)
    score, _ = native_score(sample, truth, mask, backend)
    # Outside-tumor IoU=2/3, tumor IoU=1/2: reference0 is ignored.
    assert score["macro_iou"] == pytest.approx((2/3+1/2)/2)
    assert score["valid_pixels"] == 4
    assert score["saved_mask_annotated_unassigned_pixels"] == 2
    assert score["per_class"]["2"]["truth_pixels"] == 2
    assert score["per_class"]["2"]["prediction_pixels"] == 1
    assert "small_le_400_px2" in score["regions"]
    assert "small_le_400_um2" not in score["regions"]
    assert all("area_um2" not in r for r in score["missed_regions"])


def test_nonrings_support_deletions_and_absent_classes_stay_in_accuracy():
    sample = dict(dataset="ocelot", native_mpp_xy=[.5, .6])
    truth = np.array([[0, 1, 2], [0, 1, 2]], np.uint8)
    score, _ = native_score(sample, truth, np.array([[2, 0, 1], [1, 1, 1]], np.uint8))
    assert score["macro_iou"] == pytest.approx(.125)
    assert score["per_class"]["2"]["iou"] == 0
    assert score["annotated_unassigned_pixels"] == 1
    assert score["regions"]["small_le_400_um2"]["completely_missed"] == 1


def test_aggregate_does_not_mix_physical_and_pixel_areas_or_change_penalty():
    rows = []
    for i, (units, iou, seconds) in enumerate((("micrometres", .6, .2), ("pixels", .8, 2.))):
        small = "small_le_400_um2" if i == 0 else "small_le_400_px2"
        score = dict(macro_iou=iou, macro_dice=.75,
            regions={small: dict(recall_at_50pct=.2+.6*i, completely_missed=2-i)})
        rows.append(dict(case=str(i), dataset="bcss" if i == 0 else "rings-tumor",
            historical_subset=i == 0, coordinate_units=units, single_eligible_class=i == 1,
            reference_class_count=2, methods={"M": dict(status="ok", score=score, median_seconds=seconds)}))
    rank = aggregate(rows, ["M"], DEFAULT_POLICY)[0]
    assert rank["mean_iou"] == pytest.approx(.7)
    assert rank["runtime_adjusted_score"] == pytest.approx(.675)
    assert rank["small_region_recall"] == pytest.approx(.2)
    assert rank["pixel_small_region_recall"] == pytest.approx(.8)
    assert rank["strata"]["historical149"]["mean_iou"] == .6
    assert rank["strata"]["additional200"]["mean_iou"] == .8
    assert rank["strata"]["single_reference_class"]["mean_iou"] is None


def test_pixel_small_atlas_matches_rings_scoring_after_background_removal():
    from experiments.bora_cycles.small_region_atlas import components
    sample = dict(dataset='rings-tumor', native_mpp_xy=None, pixel_only=True,
        physical_metrics_allowed=False, raw_to_compact={'0': 1, '1': 2})
    truth = np.array([[1, 1, 0, 2, 2]], np.uint8)
    mask = np.array([[0, 0, 0, 0, 2]], np.uint8)
    support = np.array([[0, 0, 0, 1, 1]], bool)
    score, _ = native_score(sample, truth, mask)
    atlas = components(truth, support, truth, dict(M=mask), [1., 1.], dict(M=score), sample=sample)
    assert atlas['coordinate_units'] == 'pixels'
    assert all('area_px2' in c and 'area_um2' not in c for c in atlas['components'])
    negative = next(c for c in atlas['components'] if c['label'] == 1)
    assert negative['correct_pixels_by_method'] == [2]
    assert not negative['common_support_cannot_reach_50pct']


def test_rings_input_bound_cannot_rule_out_correct_removed_negative():
    from experiments.bora_cycles.input_limits import summarize
    sample = dict(dataset='rings-tumor', raw_to_compact={'0': 1, '1': 2})
    truth = np.array([[1, 2]], np.uint8)
    bound = summarize(truth, np.array([[0, 1]], bool), np.array([[2]], np.uint8), {}, [1, 1], sample=sample)
    assert bound['optimistic_macro_iou_upper_bound'] == 1
