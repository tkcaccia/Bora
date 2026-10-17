"""Independently audit prepared extension geometry, labels, and source overlap."""
import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import tifffile

from experiments.virchow2_inverse.common import digest, save_json
from experiments.virchow2_multi.plan import coarse_mask
from .plan import global_rect, intersects


def audit(data, previous, roots):
    data, previous = Path(data), Path(previous)
    old = json.loads((previous / "PLAN.json").read_text())["samples"]
    used_cases = {s["case"] for s in old}
    used_sources = {s["source"]["image_sha256"] for s in old}
    rectangles = [global_rect(s["source"], s["native_crop_xywh"]) for s in old if s["dataset"] == "tiger"]
    records, datasets, single, groups, physical = [], Counter(), Counter(), {}, Counter()
    for root in map(Path, roots):
        plan = json.loads((root / "PLAN.json").read_text())
        if not plan["all_requested_prepared"]:
            raise ValueError("Incomplete source plan")
        if plan["previous_plan_sha256"] != digest(previous / "PLAN.json"):
            raise ValueError("Historical corpus changed")
        hashes = {}
        for s in plan["samples"]:
            case, row = s["case"], s["source"]
            if case in used_cases or row["image_sha256"] in used_sources:
                raise ValueError("Repeated case or source RGB bytes")
            used_cases.add(case)
            used_sources.add(row["image_sha256"])
            folder = root / "cases" / case
            if json.loads((folder / "sample.json").read_text()) != s:
                # JSON stores integer-keyed ontology maps as strings.
                if json.loads(json.dumps(s)) != json.loads((folder / "sample.json").read_text()):
                    raise ValueError("Per-case metadata differs from plan")
            for name, key in (("original.png", "original_sha256"), ("coarse.tif", "coarse_sha256"),
                              ("reference.native.tif", "reference_native_sha256")):
                if digest(folder / name) != s[key]:
                    raise ValueError("Prepared bytes changed")
            for key in ("image", "mask"):
                if digest(data / row[key]) != row[key + "_sha256"]:
                    raise ValueError("Original source bytes changed")
            x, y, width, height = s["native_crop_xywh"]
            native = np.array(Image.open(folder / "original.native.png").convert("RGB"))
            original = np.array(Image.open(data / row["image"]).convert("RGB"))[y:y+height, x:x+width]
            np.testing.assert_array_equal(native, original)
            fine = tifffile.imread(folder / "reference.native.tif")
            raw = np.array(Image.open(data / row["mask"]))[y:y+height, x:x+width]
            expected = np.zeros(raw.shape, np.uint8)
            for code, label in s["raw_to_compact"].items():
                expected[raw == int(code)] = label
            np.testing.assert_array_equal(fine, expected)
            if fine.shape != tuple(s["native_shape_yx"]) or native.shape[:2] != fine.shape:
                raise ValueError("Native image/mask geometry mismatch")
            working = np.array(Image.open(folder / "original.png").convert("RGB"))
            shape = tuple(s["working_shape_yx"])
            if working.shape[:2] != shape:
                raise ValueError("Working image geometry mismatch")
            work_fine = cv2.resize(fine, tuple(shape[::-1]), interpolation=cv2.INTER_NEAREST)
            np.testing.assert_array_equal(tifffile.imread(folder / "coarse.tif"), coarse_mask(work_fine))
            if s["pixel_only"]:
                if any(s[k] is not None for k in ("native_mpp_xy", "working_mpp_xy", "fov_um_xy", "coarse_tile_um")):
                    raise ValueError("Unverified physical calibration was fabricated")
                np.testing.assert_array_equal(working, native)
            else:
                np.testing.assert_allclose(np.array(s["working_mpp_xy"]) * np.array(shape[::-1]), s["fov_um_xy"], rtol=1e-12)
            if s["dataset"] == "tiger":
                rect = global_rect(row, s["native_crop_xywh"])
                if any(slide == rect[0] and intersects(box, rect[1]) for slide, box in rectangles):
                    raise ValueError("Overlapping TIGER source windows")
                rectangles.append(rect)
                if row["cohort"] != "non-TCGA":
                    raise ValueError("TIGER BCSS-overlap source cohort included")
            if s["dataset"] == "ocelot" and row["split"] != "train":
                raise ValueError("OCELOT held-out partition exposed")
            datasets[s["dataset"]] += 1
            single[s["dataset"]] += s["single_coarse_class"]
            physical["pixel_only" if s["pixel_only"] else "calibrated"] += 1
            groups.setdefault(s["dataset"], set()).add(s["group"])
            hashes[case] = {name: digest(folder / name) for name in
                ("sample.json", "original.png", "original.native.png", "coarse.tif", "reference.native.tif")}
        records.append(dict(root=str(root), plan_sha256=digest(root / "PLAN.json"), cases=hashes))
    if dict(datasets) != {"bcss": 40, "tiger": 40, "wsss4luad": 40, "rings-tumor": 40, "ocelot": 40}:
        raise ValueError(f"Requested expansion incomplete: {datasets}")
    result = dict(passed=True, historical_case_count=len(old), additional_case_count=sum(datasets.values()),
        combined_case_count=len(old)+sum(datasets.values()), additional_counts=dict(datasets),
        source_groups={k: len(v) for k, v in groups.items()}, single_coarse_class=dict(single),
        calibration_counts=dict(physical), historical_plan_sha256=digest(previous / "PLAN.json"),
        source_and_native_pixels_verified=True, coarse_majorities_verified=True,
        tiger_overlap_checked=True, no_duplicate_source_RGB_bytes=True,
        historical_predictions_modified=False, additional_feature_inference_complete=False,
        additional_masks_scored=False, roots=records, audit_code_sha256=digest(__file__))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data_root, args.previous, args.roots)
    save_json(args.output, result)
    print(json.dumps({k: result[k] for k in ("passed", "additional_counts", "combined_case_count", "single_coarse_class")}))
