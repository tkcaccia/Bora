"""Post-freeze scoring and separate unit/cohort aggregates for the349 study."""
import numpy as np

from experiments.bora_extension.task_adapter import geometry, report_units, scoring_prediction


def glass_only_method(key, detail):
    """Only declared glass-only mechanisms may leave supported pixels zero."""
    expected = {
        'L03': 'white-space-exterior-bottleneck-connectivity',
        'M10': 'exterior-shortest-integrated-stain-cost',
        'M11': 'eight-ray-unobstructed-white-exterior-visibility',
        'M12': 'binary-he-glass-unary-perimeter-mincut',
        'N10': 'global-stained-tissue-convex-envelope-glass-state',
        'N11': 'he-marker-constrained-minimum-spanning-forest-glass',
        'N12': 'white-graph-root-biconnected-exterior-state',
    }
    return key in expected and detail.get('mechanism') == expected[key]


def owns_native_support(key,detail):
    from .eleven_manifest import CHILDREN,SUPPORT_METHODS,SUPPORT_CONTROL
    mechanisms={r['id']:r['intervention'] for r in CHILDREN if r['id'] in SUPPORT_METHODS}
    mechanisms['B23']=SUPPORT_CONTROL['mechanism']
    from .twelve_manifest import CHILDREN as CHILDREN_TWELVE,NATIVE_CHILDREN
    mechanisms.update({r['id']:r['intervention'] for r in CHILDREN_TWELVE if r['id'] in NATIVE_CHILDREN})
    return key in mechanisms and detail.get('mechanism')==mechanisms[key] and detail.get('native_support',{}).get('own_native_support') is True


def may_remove_nonwhite(key,detail):
    """Prospective exact mechanism, never permission from an arbitrary flag."""
    from .twelve_manifest import CHILDREN
    spec=next(r for r in CHILDREN if r['id']=='Q01')
    return (key=='Q01' and owns_native_support(key,detail)
        and detail.get('mechanism')==spec['intervention']
        and detail['native_support'].get('photometrically_nonwhite_can_be_removed') is True)


def native_score(sample, truth, mask, backend="numpy", *, evaluator=None,white=None,ocelot_white_negative=False):
    from .metrics import RegionEvaluator
    coordinates = geometry(sample, native=True)
    if evaluator is None:
        evaluator = RegionEvaluator(truth, coordinates["algorithm_spacing_xy"])
    scored, policy = scoring_prediction(sample, mask, zero_is_removed_background=True,white=white,ocelot_white_negative=ocelot_white_negative)
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
    if scope.get('version')=='calibrated-tissues-v1':row['benchmark_scope_version']=scope['version']
    with threadpool_limits(limits=2):
        for key in keys:
            entry = dict(record["methods"][key])
            if entry["status"] == "ok":
                mask = tifffile.imread(out/key/"native.tif")
                # This is required by the RINGS scoring-only zero semantics.
                glass_only = glass_only_method(key, entry['details'])
                owned=owns_native_support(key,entry['details'])
                actual_support=support
                if owned:
                    if key+'/support.native.tif' not in record['files_sha256']:
                        raise ValueError('Native support is not included in prediction freeze')
                    actual_support=tifffile.imread(out/key/'support.native.tif')>0
                    if actual_support.shape!=support.shape or (not may_remove_nonwhite(key,entry['details']) and np.any((~white)&~actual_support)):
                        raise ValueError('Native support removed non-white tissue or changed geometry')
                elif entry['details'].get('native_support',{}).get('own_native_support'):
                    raise ValueError('Unregistered native-support contract')
                if np.any(mask[~actual_support]) or (not glass_only and np.any(mask[actual_support] == 0)):
                    raise ValueError("The full-supported-tissue invariant is violated")
                entry["score"], _ = native_score(sample, truth, mask, backend, evaluator=evaluator,white=white,
                    ocelot_white_negative=scope.get('version')=='calibrated-tissues-v1')
                entry["background"] = report_units(mask_diagnostics(mask, support, white, exterior,
                    coordinates["algorithm_spacing_xy"], backend=backend), coordinates["units"])
                if owned:
                    entry['background']['restored_outside_common_support_pixels']=int((actual_support&~support).sum())
                    entry['background']['removed_inside_common_support_pixels']=int((support&~actual_support).sum())
                    entry['background']['support_contract']='Own frozen native H&E support; common-support differences are diagnostics, not an invariant violation'
            row["methods"][key] = entry
    regions = np.load(out/"NOVELTY.npz")["regions"]
    row["novelty"] = discovery_score(resize(regions, truth.shape, True), truth)
    if 'RARITY.npz' in record['files_sha256']:
        rare=np.load(out/'RARITY.npz')['regions']
        row['rare_colour_proposals']=discovery_score(resize(rare,truth.shape,True),truth)
        row['rare_colour_proposals']['diagnostic_only']=True
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
    strata_names=('previous_cohort','new_tissue_datasets') if rows and rows[0].get('benchmark_scope_version')=='calibrated-tissues-v1' else ('historical149','additional200')
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
            strata={strata_names[0]: stats([i for i, r in enumerate(rows) if r["historical_subset"]]),
                strata_names[1]: stats([i for i, r in enumerate(rows) if not r["historical_subset"]]),
                "calibrated": stats([i for i, r in enumerate(rows) if r["coordinate_units"] == "micrometres"]),
                "uncalibrated": stats([i for i, r in enumerate(rows) if r["coordinate_units"] == "pixels"]),
                "single_eligible_class": stats([i for i, r in enumerate(rows) if r["single_eligible_class"]]),
                "multiple_eligible_classes": stats([i for i, r in enumerate(rows) if not r["single_eligible_class"]]),
                "single_reference_class": stats([i for i, r in enumerate(rows) if r["reference_class_count"] == 1]),
                "multiple_reference_classes": stats([i for i, r in enumerate(rows) if r["reference_class_count"] > 1])})
        ranks.append(rank)
    return runtime_ranking(dict(cases=rows, ranking=ranks), policy)
