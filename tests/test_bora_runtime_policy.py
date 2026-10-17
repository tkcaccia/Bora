from copy import deepcopy
import json

import pytest

from experiments.bora_cycles.runtime_policy import (
    DEFAULT_POLICY, active_policy, adjusted_score, runtime_ranking)
from experiments.bora_cycles import review
from experiments.bora_cycles.run import save


def sample_result():
    cases = []
    for case, fast, slow in (("a", .2, .8), ("b", .2, 1.2)):
        methods = {"fast": dict(status="ok", median_seconds=fast, score=dict(macro_iou=.7)),
                   "slow": dict(status="ok", median_seconds=slow, score=dict(macro_iou=.702))}
        cases.append(dict(case=case, methods=methods))
    return dict(cases=cases, ranking=[dict(method="fast", complete=True, mean_iou=.7),
                                     dict(method="slow", complete=True, mean_iou=.702)])


def test_runtime_penalty_is_per_image_and_never_rewards_subsecond_methods():
    result = sample_result()
    original = deepcopy(result)
    ranked = runtime_ranking(result)
    assert ranked[0]["method"] == "fast"
    assert ranked[1]["runtime_adjusted_score"] == pytest.approx(.697)
    assert ranked[1]["mean_runtime_penalty"] == pytest.approx(.005)
    assert ranked[1]["images_over_one_second"] == 1
    assert ranked[0]["runtime_adjusted_score"] == .7
    assert adjusted_score(.7, 0) == adjusted_score(.7, 1) == .7
    assert result == original
    # Penalizing the mean runtime would incorrectly give this method no cost.
    assert (.8+1.2)/2 == 1


@pytest.mark.parametrize("seconds", [-1., float("inf"), float("nan")])
def test_invalid_elapsed_time_cannot_get_a_free_score(seconds):
    with pytest.raises(ValueError):
        adjusted_score(.7, seconds)


def test_missing_failed_or_empty_cases_cannot_earn_complete_utility():
    result = sample_result()
    del result["cases"][0]["methods"]["slow"]
    assert runtime_ranking(result)[1]["runtime_adjusted_score"] is None
    result = sample_result()
    result["cases"][0]["methods"]["slow"]["status"] = "failed"
    assert runtime_ranking(result)[1]["complete"] is False
    result["cases"] = []
    assert all(not r["complete"] for r in runtime_ranking(result))


def test_only_frozen_prospective_policy_changes_selection():
    assert active_policy(dict(cycle=5)) is None
    assert active_policy(dict(cycle=4, study_protocol=dict(runtime_penalty=DEFAULT_POLICY))) is None
    policy = active_policy(dict(cycle=5, study_protocol=dict(runtime_penalty=DEFAULT_POLICY)))
    assert policy == DEFAULT_POLICY
    policy["threshold_seconds"] = 8
    assert DEFAULT_POLICY["threshold_seconds"] == 1


def test_utility_is_not_clamped_or_misreported_as_iou():
    assert adjusted_score(.5, 21) == pytest.approx(-.5)
    p = dict(DEFAULT_POLICY, iou_penalty_per_excess_second=.1)
    assert adjusted_score(.5, 2, p) == .4


def test_runner_aggregate_uses_new_utility_only_when_policy_is_provided():
    from experiments.bora_cycles.run import aggregate
    result = sample_result()
    for row in result["cases"]:
        row["dataset"] = "synthetic"
        for entry in row["methods"].values():
            entry["score"].update(macro_dice=.8, regions=dict(
                small_le_400_um2=dict(recall_at_50pct=.5, completely_missed=1)))
    assert aggregate(result["cases"], ["slow", "fast"])[0]["method"] == "slow"
    ranked = aggregate(result["cases"], ["slow", "fast"], DEFAULT_POLICY)
    assert ranked[0]["method"] == "fast"
    assert ranked[1]["runtime_adjusted_score"] == pytest.approx(.697)
    assert ranked[1]["mean_iou"] == .702


def selection_fixture(root, cycle):
    keys = [f"S{i:02d}" for i in range(20)]
    protocol = dict(cycle=cycle, roster={k: {} for k in keys},
                    study_protocol=dict(runtime_penalty=DEFAULT_POLICY))
    ranking, methods = [], {}
    for key in keys:
        iou, seconds = (.9, 10) if key == "S00" else (.8, .1) if key == "S01" else (.6, .1)
        ranking.append(dict(method=key, complete=True, mean_iou=iou))
        methods[key] = dict(status="ok", median_seconds=seconds, score=dict(macro_iou=iou))
    save(root/"RESULTS.json", dict(ranking=ranking, cases=[dict(methods=methods)]))
    save(root/"PROTOCOL.json", protocol)
    save(root/"VISUAL_REVIEW.json", dict(unit_test=True))
    return protocol, {k: "A concrete reason for this numerical selection test." for k in keys}


def test_new_selection_requires_adjusted_leader_not_slow_raw_leader(tmp_path, monkeypatch):
    protocol, reasons = selection_fixture(tmp_path, 5)
    # Full-scope/actual-review validation has independent tests; isolate the
    # policy decision here using synthetic scores, not fictitious study reviews.
    monkeypatch.setattr(review, "require_review", lambda root: protocol)
    with pytest.raises(ValueError, match="selection leader"):
        review.select(tmp_path, ["S00", "S02", "S03", "S04", "S05"], reasons)
    selection = review.select(tmp_path, ["S01", "S02", "S03", "S04", "S05"], reasons)
    assert selection["primary_selection_leaders"] == ["S01"]
    assert selection["primary_accuracy_leaders"] == ["S00"]
    assert selection["primary_selection_metric"] == "runtime_adjusted_score"


def test_historical_cycle_keeps_original_accuracy_gate(tmp_path, monkeypatch):
    protocol, reasons = selection_fixture(tmp_path, 4)
    monkeypatch.setattr(review, "require_review", lambda root: protocol)
    with pytest.raises(ValueError, match="selection leader"):
        review.select(tmp_path, ["S01", "S02", "S03", "S04", "S05"], reasons)
    selection = review.select(tmp_path, ["S00", "S02", "S03", "S04", "S05"], reasons)
    assert selection["primary_selection_metric"] == "mean_iou"
