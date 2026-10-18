"""Locate every small reference component after the whole cycle is frozen.

This is error analysis, never prediction supervision. Record structural
constraints without claiming an independently established biological cause.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
import tifffile

from bora.evidence import resize
from .metrics import RegionEvaluator
from .run import digest, verify
from experiments.bora_extension.task_adapter import geometry, report_units, scoring_prediction


def components(truth, support, native_coarse, masks, mpp_xy, saved_scores=None, *, sample=None,white=None,ocelot_white_negative=False):
    units = "micrometres" if sample is None else geometry(sample, native=True)["units"]
    evaluator = RegionEvaluator(truth, mpp_xy)
    cc = evaluator.components
    n = len(evaluator.area_pixels)
    small = np.flatnonzero((evaluator.area_um2 > 0) & (evaluator.area_um2 <= 400))
    small = small[small > 0]
    boxes = ndi.find_objects(cc)
    support_count = np.bincount(cc.ravel(), weights=np.asarray(support, bool).ravel(), minlength=n)
    from experiments.bora_extension.task_adapter import binary_negative
    negative=binary_negative(sample,ocelot_white_negative=ocelot_white_negative) if sample is not None else None
    white_outside_count=(np.bincount(cc.ravel(),weights=(np.asarray(white,bool)&~np.asarray(support,bool)).ravel(),minlength=n)
        if negative is not None and white is not None else np.zeros(n))
    tile_correct = np.bincount(cc.ravel(), weights=(native_coarse == truth).ravel(), minlength=n)
    known_ids = set(np.unique(native_coarse))-{0}
    correct = []
    methods = list(masks)
    for method in methods:
        prediction = np.asarray(masks[method])
        if sample is not None:
            prediction, _ = scoring_prediction(sample, prediction, zero_is_removed_background=True,
                white=white,ocelot_white_negative=ocelot_white_negative)
        if prediction.shape != truth.shape:
            raise ValueError("Native geometry mismatch")
        count = np.bincount(cc.ravel(), weights=(prediction == truth).ravel(), minlength=n)
        counts = count[small].astype(np.int64)
        if saved_scores is not None:
            score = saved_scores[method]["regions"]["small_le_400_um2" if units == "micrometres" else "small_le_400_px2"]
            recall = float(np.mean(counts/evaluator.area_pixels[small] >= .5)) if len(small) else None
            if (score["components"] != len(small) or score["completely_missed"] != int((counts == 0).sum())
                    or score["recall_at_50pct"] != recall):
                raise ValueError("Independent small-region counts differ from saved scores: "+method)
        correct.append(counts)
    correct = np.stack(correct, -1) if methods else np.empty((len(small), 0), np.int64)
    rows = []
    for j, i in enumerate(small):
        ys, xs = boxes[i-1]
        pixels = int(evaluator.area_pixels[i])
        label = int(evaluator.component_classes[i])
        count = correct[j]
        outside_negative = (sample is not None and sample["dataset"] == "rings-tumor"
            and str(sample.get("raw_to_compact", {}).get("0")) == str(label))
        reachable=pixels if outside_negative else support_count[i]+(white_outside_count[i] if negative is not None and negative[0]==label else 0)
        rows.append(dict(component=int(i), label=label, pixels=pixels,
            area_um2=float(evaluator.area_um2[i]),
            native_bbox_xywh=[xs.start, ys.start, xs.stop-xs.start, ys.stop-ys.start],
            class_absent_from_original_coarse=label not in known_ids,
            common_support_pixels=int(support_count[i]),
            common_support_cannot_reach_50pct=bool(reachable/pixels < .5),
            valid_background_negative_can_be_recovered_by_zero=bool(outside_negative or (negative is not None and negative[0]==label)),
            original_coarse_correct_pixels=int(tile_correct[i]),
            correct_pixels_by_method=count.tolist(),
            all_methods_completely_miss=bool(np.all(count == 0)),
            no_method_reaches_50pct=bool(np.all(count/pixels < .5)),
            original_coarse_reaches_50pct=bool(tile_correct[i]/pixels >= .5)))
    return report_units(dict(method_order=methods, components=rows, coordinate_units=units,
        independent_numpy_small_region_metric_parity=saved_scores is not None), units)


def export(source, root, output):
    source, root, output = Path(source), Path(root), Path(output)
    if output.exists():
        raise ValueError("Use a fresh atlas output file")
    protocol = json.loads((root/"PROTOCOL.json").read_text())
    frozen = json.loads((root/"ALL_PREDICTIONS_FROZEN.json").read_text())
    if frozen["protocol_sha256"] != digest(root/"PROTOCOL.json"):
        raise ValueError("Protocol differs from full prediction freeze")
    if set(frozen["manifests"]) != set(protocol["cases"]):
        raise ValueError("All predictions must be frozen before error analysis")
    result = json.loads((root/"RESULTS.json").read_text())
    by_case = {r["case"]: r for r in result["cases"]}
    rows, total = [], dict(small_components=0, absent_coarse_class=0,
        support_cannot_reach_50pct=0, no_method_reaches_50pct=0,
        all_methods_completely_miss=0, coarse_recovered_but_all_refinements_below_50pct=0)
    totals_by_units = {}
    for case in protocol["cases"]:
        row = by_case[case]
        src, out = source/"cases"/case, root/"cases"/case
        if digest(out/"PREDICTIONS.json") != frozen["manifests"][case]:
            raise ValueError("Prediction manifest changed: "+case)
        prediction = json.loads((out/"PREDICTIONS.json").read_text())
        verify(out, prediction["files_sha256"])
        sample = row["sample"]
        if digest(src/"reference.native.tif") != sample["reference_native_sha256"]:
            raise ValueError("Reference changed: "+case)
        truth = tifffile.imread(src/"reference.native.tif")
        support = tifffile.imread(src/"tissue.native.tif").astype(bool)
        coarse = resize(tifffile.imread(src/"coarse.tif"), truth.shape, True)
        methods = [key for key in protocol["methods"] if row["methods"][key]["status"] == "ok"]
        if not methods:
            raise ValueError("No successful methods in case: "+case)
        # Read one native mask at a time inside components(), rather than retain
        # all 22 large rasters. Mapping values below are lazy, read-only paths.
        class Masks(dict):
            def __getitem__(self, key):
                return tifffile.imread(super().__getitem__(key))
        masks = Masks({key: out/key/"native.tif" for key in methods})
        coordinates = geometry(sample, native=True)
        from PIL import Image
        from .diagnostics import white_regions
        white=white_regions(np.array(Image.open(src/'original.native.png').convert('RGB')))[0]
        data = components(truth, support, coarse, masks, coordinates["algorithm_spacing_xy"],
                          {key: row["methods"][key]["score"] for key in methods},
                          sample=sample if protocol.get("benchmark_scope") else None,white=white,
                          ocelot_white_negative=protocol.get('benchmark_scope',{}).get('version')=='calibrated-tissues-v1')
        unit_total = totals_by_units.setdefault(coordinates['units'], {key:0 for key in total})
        for item in data["components"]:
            events = dict(small_components=1, absent_coarse_class=item['class_absent_from_original_coarse'],
                support_cannot_reach_50pct=item['common_support_cannot_reach_50pct'],
                no_method_reaches_50pct=item['no_method_reaches_50pct'],
                all_methods_completely_miss=item['all_methods_completely_miss'],
                coarse_recovered_but_all_refinements_below_50pct=(
                    item['original_coarse_reaches_50pct'] and item['no_method_reaches_50pct']))
            for key, value in events.items():
                total[key] += int(value)
                unit_total[key] += int(value)
        rows.append(dict(case=case, native_mpp_xy=sample["native_mpp_xy"],
                         omitted_failed_methods=sorted(set(protocol["methods"])-set(methods)), **data))
        print(json.dumps(dict(small_region_atlas=case, components=len(data["components"]))), flush=True)
    record = dict(cycle=protocol["cycle"], pilot=protocol["pilot"], cases=len(rows), totals=total,
        totals_by_coordinate_units=totals_by_units,
        combined_totals_mix_distinct_small_region_criteria=len(totals_by_units) > 1,
        rows=rows, protocol_sha256=digest(root/"PROTOCOL.json"),
        results_sha256=digest(root/"RESULTS.json"), atlas_source_sha256=digest(Path(__file__)),
        score_policy="Original reference-positive pixels; 8-connected per-class components; <=400 um2 if calibrated, <=400 px2 if unknown. Per-case coordinate units are explicit; RINGS zero follows its valid negative task.",
        interpretation="Constraint flags can overlap. Common-support bounds do not bind the native-support interventions. Broad-negative labels use the same scoring-only white-zero adapter as saved scores. All registered controls are included. No semantic identity is inferred for annotation zero.",
        visual_reviews_created=False, predictions_modified=False, scores_modified=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2))
    print(json.dumps(dict(cases=len(rows), **total)), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.root, args.output)


if __name__ == "__main__":
    main()
