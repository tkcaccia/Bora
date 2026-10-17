"""Post-freeze scoring and separate unit/cohort aggregates for the349 study."""
import numpy as np

from experiments.bora_extension.task_adapter import geometry, report_units, scoring_prediction


def native_score(sample, truth, mask, backend="numpy", *, evaluator=None):
    from .metrics import RegionEvaluator
    coordinates = geometry(sample, native=True)
    if evaluator is None:
        evaluator = RegionEvaluator(truth, coordinates["algorithm_spacing_xy"])
    scored, policy = scoring_prediction(sample, mask, zero_is_removed_background=True)
    result = report_units(evaluator.score(scored, backend=backend), coordinates["units"])
    result["saved_mask_annotated_unassigned_pixels"] = int(((truth > 0) & (mask == 0)).sum())
    result["coordinate_units"] = coordinates["units"]
    result["task_policy"] = policy
    return result, evaluator


def score_case(args):
    # Imported only after the entire cycle's predictions are frozen/verified.
    import json
    from pathlib import Path
    import cv2
    from PIL import Image
    import tifffile
    from threadpoolctl import threadpool_limits
    from .run import digest, verify, save, resize
    from .metrics import discovery_score
    from .metrics import RegionEvaluator
    from .diagnostics import white_regions, mask_diagnostics
    source, root, sample, keys, manifests, backend, scope = args
    cv2.setNumThreads(2)
    src, out = Path(source)/"cases"/sample["case"], Path(root)/"cases"/sample["case"]
    if digest(out/"PREDICTIONS.json") != manifests[sample["case"]]:
        raise ValueError("Prediction freeze changed before scoring")
    record = json.loads((out/"PREDICTIONS.json").read_text())
    verify(out, record["files_sha256"])
    if digest(src/"reference.native.tif") != sample["reference_native_sha256"]:
        raise ValueError("Original reference changed")
    truth = tifffile.imread(src/"reference.native.tif")
    he = np.array(Image.open(src/"original.native.png").convert("RGB"))
    support = tifffile.imread(src/"tissue.native.tif") > 0
    coordinates = geometry(sample, native=True)
    evaluator = RegionEvaluator(truth, coordinates["algorithm_spacing_xy"])
    white, exterior, detail = white_regions(he)
    selection = json.loads((src/"selection.json").read_text())
    eligible = selection.get("observed_classes", selection.get("classes"))
    if not eligible:
        raise ValueError("Eligible coarse training classes are not documented")
    row = dict(case=sample["case"], dataset=sample["dataset"], sample=sample,
        white=detail, methods={}, coordinate_units=coordinates["units"],
        eligible_class_count=len(eligible), single_eligible_class=len(eligible) == 1,
        historical_subset=sample["case"] in scope["historical_cases"])
    with threadpool_limits(limits=2):
        for key in keys:
            entry = dict(record["methods"][key])
            if entry["status"] == "ok":
                mask = tifffile.imread(out/key/"native.tif")
                # This is required by the RINGS scoring-only zero semantics.
                glass_only = key == 'L03' and entry['details']['mechanism'] == 'white-space-exterior-bottleneck-connectivity'
                if np.any(mask[~support]) or (not glass_only and np.any(mask[support] == 0)):
                    raise ValueError("The full-supported-tissue invariant is violated")
                entry["score"], _ = native_score(sample, truth, mask, backend, evaluator=evaluator)
                entry["background"] = report_units(mask_diagnostics(mask, support, white, exterior,
                    coordinates["algorithm_spacing_xy"], backend=backend), coordinates["units"])
            row["methods"][key] = entry
    regions = np.load(out/"NOVELTY.npz")["regions"]
    row["novelty"] = discovery_score(resize(regions, truth.shape, True), truth)
    coarse = tifffile.imread(src/"coarse.tif")
    represented = set(np.unique(coarse))-{0}
    reference_classes = set(np.unique(truth))-{0}
    row["reference_classes_absent_from_coarse"] = sorted(int(g) for g in reference_classes-represented)
    row["reference_class_count"] = len(reference_classes)
    save(out/"SCORES.json", row)
    print(json.dumps(dict(scored=sample["case"])), flush=True)
    return row


def aggregate(rows, keys, policy):
    from .runtime_policy import runtime_ranking
    ranks = []
    for key in keys:
        entries = [row["methods"][key] for row in rows]
        failed = [row["case"] for row, entry in zip(rows, entries) if entry["status"] != "ok"]
        if failed:
            ranks.append(dict(method=key, complete=False, failed_cases=failed, cases=len(rows), mean_iou=None))
            continue
        def stats(indices):
            return dict(cases=len(indices), mean_iou=float(np.mean([
                entries[i]["score"]["macro_iou"] for i in indices])) if indices else None)
        units = {}
        for name, region_key in (("physical", "small_le_400_um2"), ("pixel", "small_le_400_px2")):
            small = [e["score"]["regions"][region_key] for e in entries if region_key in e["score"]["regions"]]
            recall = [s["recall_at_50pct"] for s in small if s["recall_at_50pct"] is not None]
            units[name] = dict(mean_recall=float(np.mean(recall)) if recall else None,
                contributing_cases=len(recall), completely_missed=sum(s["completely_missed"] for s in small))
        rank = dict(method=key, complete=True, cases=len(rows),
            mean_iou=float(np.mean([e["score"]["macro_iou"] for e in entries])),
            mean_dice=float(np.mean([e["score"]["macro_dice"] for e in entries])),
            mean_seconds=float(np.mean([e["median_seconds"] for e in entries])),
            small_region_recall=units["physical"]["mean_recall"],
            small_region_cases=units["physical"]["contributing_cases"],
            completely_missed_small_regions=units["physical"]["completely_missed"],
            pixel_small_region_recall=units["pixel"]["mean_recall"],
            pixel_small_region_cases=units["pixel"]["contributing_cases"],
            pixel_completely_missed_small_regions=units["pixel"]["completely_missed"],
            per_dataset={d: stats([i for i, r in enumerate(rows) if r["dataset"] == d])
                         for d in sorted({r["dataset"] for r in rows})},
            strata={"historical149": stats([i for i, r in enumerate(rows) if r["historical_subset"]]),
                "additional200": stats([i for i, r in enumerate(rows) if not r["historical_subset"]]),
                "calibrated": stats([i for i, r in enumerate(rows) if r["coordinate_units"] == "micrometres"]),
                "uncalibrated": stats([i for i, r in enumerate(rows) if r["coordinate_units"] == "pixels"]),
                "single_eligible_class": stats([i for i, r in enumerate(rows) if r["single_eligible_class"]]),
                "multiple_eligible_classes": stats([i for i, r in enumerate(rows) if not r["single_eligible_class"]]),
                "single_reference_class": stats([i for i, r in enumerate(rows) if r["reference_class_count"] == 1]),
                "multiple_reference_classes": stats([i for i, r in enumerate(rows) if r["reference_class_count"] > 1])})
        ranks.append(rank)
    return runtime_ranking(dict(cases=rows, ranking=ranks), policy)
