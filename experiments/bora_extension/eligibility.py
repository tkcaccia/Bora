"""Freeze score-blind multigroup/tile eligibility and a prospective input view.

Never imports a scorer or opens a fine reference. Original signed inputs and
results remain immutable. All rejected cases go to a separate diagnostics view.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from experiments.virchow2_inverse.common import coarse_tile_observations

POLICY_PATH = Path(__file__).with_name("eligibility_policy.json")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_new(path, value):
    path = Path(path)
    data = json.dumps(value, indent=2, allow_nan=False)+"\n"
    if path.exists():
        if path.read_text() != data:
            raise ValueError("Immutable eligibility evidence differs: "+str(path))
    else:
        path.write_text(data)


def inspect(coarse, shape, policy):
    """Use exactly the converter's tile sampling, without loading embeddings."""
    if (coarse.ndim != 2 or list(coarse.shape) != list(shape)
            or coarse.dtype.kind not in "iu" or np.any(coarse < 0)):
        raise ValueError("A categorical coarse mask paired with real RGB is required")
    height, width = map(int, shape)
    patch = policy["token_patch_px"]
    grid = ((height+patch-1)//patch, (width+patch-1)//patch, 1)
    dummy = np.zeros(grid, np.float32)
    try:
        _, observed, _, records = coarse_tile_observations(
            dummy, coarse, policy["coarse_tile_px"])
    except ValueError as exc:
        if str(exc) != "No eligible coarse tile summaries":
            raise
        observed, records = np.empty(0, dtype=int), []
    counts = dict(sorted(Counter(map(int, observed)).items()))
    groups = sorted(int(g) for g in np.unique(coarse) if g > 0)
    capacity = [height//policy["encoder_window_px"], width//policy["encoder_window_px"]]
    reasons = []
    if len(groups) < policy["minimum_input_groups"]:
        reasons.append("fewer_than_two_input_groups")
    if len(counts) < policy["minimum_observed_groups"]:
        reasons.append("fewer_than_two_feature_selection_groups")
    if min(capacity) < policy["minimum_complete_base_fields_per_axis"]:
        reasons.append("fewer_than_four_real_encoder_fields_on_each_axis")
    if len(records) < policy["minimum_usable_coarse_tiles"]:
        reasons.append("fewer_than_sixteen_usable_coarse_tiles")
    return dict(eligible=not reasons, exclusion_reasons=reasons,
        input_groups=groups, observed_groups=list(counts),
        usable_coarse_tiles=len(records), coarse_tiles_by_group={str(g): n for g,n in counts.items()},
        complete_base_fields_yx=capacity,
        complete_base_field_capacity=int(np.prod(capacity)),
        real_unpadded_shape_yx=[height, width], coarse_observations=records)


def audit(source, root):
    source, root = Path(source).resolve(), Path(root).resolve()
    policy = json.loads(POLICY_PATH.read_text())
    plan = json.loads((source/"PLAN.json").read_text())
    frozen = json.loads((source/"ALL_INPUTS_FROZEN.json").read_text())
    if frozen["plan_sha256"] != digest(source/"PLAN.json"):
        raise ValueError("Input plan no longer agrees with the all-input freeze")
    samples = plan["samples"]
    if len({s["case"] for s in samples}) != len(samples) or set(frozen["manifests"]) != {s["case"] for s in samples}:
        raise ValueError("Input freeze and plan membership differ")
    root.mkdir(parents=True, exist_ok=True)
    # Persist the fixed rules and identities before inspecting any case. Scores
    # are neither accepted by this command nor read by any imported function.
    save_new(root/"POLICY.json", policy)
    protocol = dict(version=policy["version"], source=str(source),
        input_plan_sha256=digest(source/"PLAN.json"),
        all_inputs_sha256=digest(source/"ALL_INPUTS_FROZEN.json"),
        policy_sha256=digest(root/"POLICY.json"),
        code_sha256=digest(__file__),
        coarse_sampler_sha256=digest(Path(__file__).parents[1]/"virchow2_inverse/common.py"),
        fine_reference_read=False, predictions_read=False, scores_read=False,
        thresholds_chosen_by_score=False, historical_results_modified=False)
    save_new(root/"PROTOCOL.json", protocol)
    rows = []
    for sample in samples:
        case = sample["case"]
        folder = source/"cases"/case
        manifest_path = folder/"INPUTS_FROZEN.json"
        if digest(manifest_path) != frozen["manifests"][case]:
            raise ValueError("Changed input manifest: "+case)
        manifest = json.loads(manifest_path.read_text())
        identities = {name: digest(folder/name) for name in ("original.png", "coarse.tif")}
        if any(sha != manifest["files_sha256"][name] for name,sha in identities.items()):
            raise ValueError("Changed RGB/coarse input: "+case)
        with Image.open(folder/"original.png") as image:
            shape = [image.height, image.width]
        if shape != sample["working_shape_yx"]:
            raise ValueError("RGB geometry differs from plan: "+case)
        row = inspect(tifffile.imread(folder/"coarse.tif"), shape, policy)
        selection_path = folder/"selection.json"
        if digest(selection_path) != manifest["files_sha256"]["selection.json"]:
            raise ValueError("Changed feature selection: "+case)
        selected = json.loads(selection_path.read_text())
        selected_groups = selected.get("observed_classes", selected.get("classes"))
        if sorted(map(int, selected_groups)) != row["observed_groups"]:
            raise ValueError("Recomputed tile groups disagree with feature selection: "+case)
        rows.append(dict(case=case, dataset=sample["dataset"], group=sample["group"],
            class_names=sample["class_names"], working_mpp_xy=sample["working_mpp_xy"],
            coarse_tile_px=sample.get("coarse_tile_px", 90),
            coarse_tile_um=sample.get("coarse_tile_um"),
            input_manifest_sha256=frozen["manifests"][case], input_sha256=identities,
            feature_selection_sha256=digest(selection_path), **row))
    datasets = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        subset = [r for r in rows if r["dataset"] == dataset]
        datasets[dataset] = dict(total=len(subset), eligible=sum(r["eligible"] for r in subset),
            excluded=sum(not r["eligible"] for r in subset),
            exclusion_reason_counts=dict(Counter(reason for r in subset for reason in r["exclusion_reasons"])),
            input_single_group=sum(len(r["input_groups"]) < 2 for r in subset),
            feature_selection_single_group=sum(len(r["observed_groups"]) < 2 for r in subset))
    report = dict(protocol=protocol, policy=policy, datasets=datasets, cases=rows,
        eligible_cases=[r["case"] for r in rows if r["eligible"]],
        excluded_cases=[r["case"] for r in rows if not r["eligible"]],
        eligible_count=sum(r["eligible"] for r in rows),
        excluded_count=sum(not r["eligible"] for r in rows))
    save_new(root/"ELIGIBILITY.json", report)
    return report


def prepare_view(source, report_root, output):
    """Lightweight versioned benchmark, with diagnostics outside its PLAN."""
    source, report_root, output = map(lambda p: Path(p).resolve(), (source, report_root, output))
    report = json.loads((report_root/"ELIGIBILITY.json").read_text())
    if report["protocol"]["input_plan_sha256"] != digest(source/"PLAN.json"):
        raise ValueError("Eligibility audit belongs to another input plan")
    plan = json.loads((source/"PLAN.json").read_text())
    accepted = set(report["eligible_cases"])
    if not accepted:
        raise ValueError("No eligible benchmark images")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("POLICY.json", "PROTOCOL.json", "ELIGIBILITY.json"):
        save_new(output/("ELIGIBILITY_"+name if name == "PROTOCOL.json" else name),
                 json.loads((report_root/name).read_text()))
    view = dict(plan)
    view.update(samples=[s for s in plan["samples"] if s["case"] in accepted],
        historical_cases=[c for c in plan["historical_cases"] if c in accepted],
        additional_cases=[c for c in plan["additional_cases"] if c in accepted],
        benchmark_scope_version="qualified-multigroup-v1",
        eligibility_sha256=digest(report_root/"ELIGIBILITY.json"),
        eligibility_policy_sha256=digest(report_root/"POLICY.json"),
        eligibility_source_plan_sha256=digest(source/"PLAN.json"),
        eligibility_source=str(source), diagnostic_cases=report["excluded_cases"],
        prospective_from_cycle=11,
        historical_scores_are_not_comparable_without_matched_scope=True)
    save_new(output/"PLAN.json", view)
    for sample in plan["samples"]:
        case = sample["case"]
        path = output/("cases" if case in accepted else "diagnostics")/case
        target = source/"cases"/case
        path.parent.mkdir(exist_ok=True)
        if path.is_symlink():
            if path.resolve() != target:
                raise ValueError("Input view symlink changed")
        elif path.exists():
            raise ValueError("Refuse to overwrite a benchmark input")
        else:
            path.symlink_to(target, target_is_directory=True)
    save_new(output/"ALL_INPUTS_FROZEN.json", dict(cases=len(accepted),
        plan_sha256=digest(output/"PLAN.json"),
        manifests={s["case"]: digest(output/"cases"/s["case"]/"INPUTS_FROZEN.json") for s in view["samples"]},
        eligibility_sha256=digest(output/"ELIGIBILITY.json"),
        fine_reference_read=False, conversions_changed=False,
        original_inputs_modified=False, diagnostic_cases=len(report["excluded_cases"])))
    return view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--view", type=Path)
    args = parser.parse_args()
    report = audit(args.source, args.report)
    if args.view:
        prepare_view(args.source, args.report, args.view)
    print(json.dumps({k: report[k] for k in ("eligible_count", "excluded_count", "datasets")}))


if __name__ == "__main__":
    main()
