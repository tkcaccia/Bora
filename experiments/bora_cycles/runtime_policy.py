"""Explicit one-second runtime utility; never overwrite frozen accuracy scores.

Historical timings support a labelled posthoc comparison. New cycles freeze
this policy with their protocol before prediction and use it for selection.
"""
from copy import deepcopy
import math


DEFAULT_POLICY = dict(
    id="one-second-excess-runtime-v1", effective_from_cycle=5,
    threshold_seconds=1., iou_penalty_per_excess_second=.05,
    formula="mean_i(macro_IoU_i - 0.05 * max(seconds_i - 1, 0))",
    timing="Per-image median candidate seconds, including preparation and native projection",
    interpretation="Five IoU percentage points per excess second; no bonus below one second",
    historical_scope="Posthoc display for cycles1-4; signed historical selections unchanged")


def validate_policy(policy):
    threshold = float(policy["threshold_seconds"])
    rate = float(policy["iou_penalty_per_excess_second"])
    if not math.isfinite(threshold) or threshold < 0 or not math.isfinite(rate) or rate < 0:
        raise ValueError("Runtime threshold and penalty must be finite and nonnegative")
    return threshold, rate


def adjusted_score(iou, seconds, policy=DEFAULT_POLICY):
    threshold, rate = validate_policy(policy)
    if not math.isfinite(iou) or not 0 <= iou <= 1:
        raise ValueError("Raw IoU must be a finite fraction in [0,1]")
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Candidate runtime must be finite and nonnegative")
    return iou-rate*max(seconds-threshold, 0.)


def active_policy(protocol):
    """Only a policy frozen in this cycle may change its selection objective."""
    policy = protocol.get("study_protocol", {}).get("runtime_penalty")
    if policy is None or protocol["cycle"] < policy["effective_from_cycle"]:
        return None
    validate_policy(policy)
    return deepcopy(policy)


def runtime_ranking(result, policy=DEFAULT_POLICY):
    """Derive per-image excess before averaging; do not penalize the mean time.

    This distinction penalizes occasional slow images even if a method's mean
    is under one second. A failed or missing case cannot earn a complete score.
    """
    threshold, rate = validate_policy(policy)
    rows = []
    for original in result["ranking"]:
        row = deepcopy(original)
        entries = [case["methods"].get(row["method"]) for case in result["cases"]]
        complete = bool(entries) and row["complete"] and all(
            entry and entry["status"] == "ok" for entry in entries)
        if not complete:
            row.update(complete=False, runtime_adjusted_score=None,
                       mean_excess_seconds=None, mean_runtime_penalty=None,
                       images_over_one_second=None)
        else:
            times = [float(entry["median_seconds"]) for entry in entries]
            values = [float(entry["score"]["macro_iou"]) for entry in entries]
            utility = [adjusted_score(iou, t, policy) for iou, t in zip(values, times)]
            excess = [max(t-threshold, 0.) for t in times]
            row.update(runtime_adjusted_score=math.fsum(utility)/len(utility),
                       mean_excess_seconds=math.fsum(excess)/len(excess),
                       mean_runtime_penalty=rate*math.fsum(excess)/len(excess),
                       images_over_one_second=sum(t > threshold for t in times))
        rows.append(row)
    return sorted(rows, key=lambda r: (
        not r["complete"], -(r["runtime_adjusted_score"] if r["complete"] else 0), r["method"]))
