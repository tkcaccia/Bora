"""Check every assembled input/RGB; optional H&E-plus-five-RGB review sheets."""
import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import tifffile

from experiments.virchow2_inverse.common import digest, save_json


def audit(root, historical, cases=None, render=False):
    root, historical = Path(root), Path(historical)
    plan = json.loads((root/"PLAN.json").read_text())
    rows = {s["case"]: s for s in plan["samples"]}
    keys = plan["conversions"]
    wanted = list(rows) if cases is None else cases
    if not set(wanted) <= set(rows) or len(set(wanted)) != len(wanted):
        raise ValueError("Unknown/duplicate audit cases")
    full = cases is None
    if full:
        frozen = json.loads((root/"ALL_INPUTS_FROZEN.json").read_text())
        if frozen["plan_sha256"] != digest(root/"PLAN.json") or set(frozen["manifests"]) != set(rows):
            raise ValueError("Full input freeze changed")
    counts, records = Counter(), []
    for case in wanted:
        sample, folder = rows[case], root/"cases"/case
        path = folder/"INPUTS_FROZEN.json"
        record = json.loads(path.read_text())
        if full and digest(path) != frozen["manifests"][case]:
            raise ValueError("Case input freeze changed")
        for name, sha in record["files_sha256"].items():
            if digest(folder/name) != sha:
                raise ValueError("Input/RGB file changed: "+case+"/"+name)
            counts["frozen_files"] += 1
        if digest(folder/"reference.native.tif") != sample["reference_native_sha256"]:
            raise ValueError("Transferred original reference bytes changed")
        if case in plan["historical_cases"]:
            if digest(path) != digest(historical/"cases"/case/"INPUTS_FROZEN.json"):
                raise ValueError("Historical conversion or input manifest changed")
            counts["exact_historical_manifests"] += 1
        image = np.array(Image.open(folder/"original.png").convert("RGB"))
        native = Image.open(folder/"original.native.png")
        if list(image.shape[:2]) != sample["working_shape_yx"] or list(native.size[::-1]) != sample["native_shape_yx"]:
            raise ValueError("Original RGB geometry changed")
        coarse = tifffile.imread(folder/"coarse.tif")
        completed = tifffile.imread(folder/"coarse.completed.tif")
        if np.any(completed == 0) or not np.array_equal(completed[coarse > 0], coarse[coarse > 0]):
            raise ValueError("Coarse extension changed known original labels")
        support = tifffile.imread(folder/"tissue.native.tif") > 0
        working_support = tifffile.imread(folder/"tissue.working.tif") > 0
        expected_support = cv2.resize(support.astype(np.float32), tuple(image.shape[:2][::-1]), interpolation=cv2.INTER_AREA) > .2
        if not np.array_equal(working_support, expected_support):
            raise ValueError("Working support geometry differs")
        if not json.loads((folder/"encoder_coverage.json").read_text())["all_original_tokens_covered"]:
            raise ValueError("Incomplete original-field encoding")
        selection = json.loads((folder/"selection.json").read_text())
        eligible = selection.get("observed_classes", selection.get("classes"))
        if not eligible:
            raise ValueError("Missing eligible class identity")
        panels = [("Original H&E", Image.fromarray(image))]
        for key in keys:
            im = Image.open(folder/key/"representation.png")
            values = np.asarray(im)
            if im.mode != "RGB" or values.dtype != np.uint8 or values.shape != image.shape:
                raise ValueError("Conversion geometry/type changed")
            if len(eligible) == 1 and not np.all(values == np.asarray(sample["palette"])[eligible[0]-1]):
                raise ValueError("Single eligible class is not an explicit constant RGB")
            detail = json.loads((folder/key/"details.json").read_text())
            if case not in plan["historical_cases"] and sample["working_mpp_xy"] is None:
                if detail["actual_base_sampling"]["mpp_xy"] is not None:
                    raise ValueError("Uncalibrated conversion invents MPP")
            panels.append((key, im.copy()))
            counts["rgb_conversions"] += 1
        counts["single_eligible_class"] += len(eligible) == 1
        row = dict(case=case, dataset=sample["dataset"], input_manifest_sha256=digest(path), eligible_classes=eligible)
        if render:
            destination = root/"input-review"/(case+".jpg")
            destination.parent.mkdir(exist_ok=True)
            sheet = Image.new("RGB", (960, 700), "white")
            draw = ImageDraw.Draw(sheet)
            for i, (title, panel) in enumerate(panels):
                x, y = (i % 3)*320, (i//3)*350
                panel.thumbnail((310, 310))
                sheet.paste(panel, (x+(320-panel.width)//2, y+30))
                draw.text((x+6, y+8), title, fill="black")
            sheet.save(destination, quality=93)
            row["review_sheet_sha256"] = digest(destination)
        records.append(row)
    return dict(passed=True, cases=len(wanted), full_corpus=full, **dict(counts),
        datasets=dict(Counter(rows[c]["dataset"] for c in wanted)),
        plan_sha256=digest(root/"PLAN.json"), audit_code_sha256=digest(__file__),
        fine_reference_decoded=False, bora_run=False, visual_reviews_created=False,
        actual_review_required_for_generated_sheets=render, records=records)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--historical", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cases", nargs="+")
    p.add_argument("--render", action="store_true")
    a = p.parse_args()
    result = audit(a.root, a.historical, a.cases, a.render)
    save_json(a.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}))
