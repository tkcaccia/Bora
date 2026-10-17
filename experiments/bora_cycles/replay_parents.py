"""Strict cross-runtime replay of selected parents; no fine-reference I/O.

This is an engineering audit, not a cycle, a new candidate or an accuracy test.
It never overwrites stored predictions, alters masks to force agreement, or
relaxes exact equality. The sole comparison target is frozen parent output.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import tifffile
from threadpoolctl import threadpool_limits

from bora.evidence import resize
from .methods import predict
from .review import require_review
from .run import digest, environment_record, save


def replay(source, previous, output, cases=None):
    source, previous, output = map(Path, (source, previous, output))
    if output.exists():
        raise ValueError("Use a fresh engineering audit output file")
    protocol = require_review(previous)
    selection = json.loads((previous/"SELECTION.json").read_text())
    for name, field in (("RESULTS.json", "results_sha256"), ("VISUAL_REVIEW.json", "review_sha256")):
        if digest(previous/name) != selection[field]:
            raise ValueError("Signed parent evidence changed")
    if digest(source/"PLAN.json") != protocol["input_plan_sha256"]:
        raise ValueError("Replay needs the identical frozen input plan")
    plan = json.loads((source/"PLAN.json").read_text())["samples"]
    samples = [s for s in plan if cases is None or s["case"] in cases]
    if cases is not None and set(cases) != {s["case"] for s in samples}:
        raise ValueError("Unknown case")
    keys = selection["survivors"]
    frozen = json.loads((previous/"ALL_PREDICTIONS_FROZEN.json").read_text())
    rows = []
    cv2.setNumThreads(2)
    environment = environment_record()
    with threadpool_limits(limits=2):
        for sample in samples:
            case = sample["case"]
            src, old = source/"cases"/case, previous/"cases"/case
            if digest(old/"PREDICTIONS.json") != frozen["manifests"][case]:
                raise ValueError("Frozen parent prediction record changed")
            record = json.loads((old/"PREDICTIONS.json").read_text())
            if digest(src/"ensemble/representation.png") != record["rgb_sha256"]:
                raise ValueError("Fixed RGB changed")
            rgb = np.asarray(Image.open(src/"ensemble/representation.png").convert("RGB"))
            he = np.asarray(Image.open(src/"original.png").convert("RGB"))
            coarse = tifffile.imread(src/"coarse.tif")
            support = tifffile.imread(src/"tissue.working.tif").astype(bool)
            native_support = tifffile.imread(src/"tissue.native.tif").astype(bool)
            for key in keys:
                start = time.perf_counter()
                analysis, detail = predict(key, rgb, coarse, support,
                    sample["working_mpp_xy"], sample["palette"], he=he)
                native = resize(analysis, sample["native_shape_yx"], True)
                native[~native_support] = 0
                seconds = time.perf_counter()-start
                old_path = old/key/"native.tif"
                old_sha = digest(old_path)
                if old_sha != record["files_sha256"][key+"/native.tif"]:
                    raise ValueError("Frozen parent mask changed")
                expected = tifffile.imread(old_path)
                mismatch = int(np.count_nonzero(native != expected))
                row = dict(case=case, method=key, changed_native_pixels=mismatch,
                    exact=mismatch == 0, seconds=seconds,
                    previous_mask_sha256=old_sha, optimizer_success=detail.get("optimizer_success"))
                rows.append(row)
            print(json.dumps(dict(replayed=case, exact=sum(r["exact"] for r in rows[-len(keys):]),
                                  comparisons=len(keys))), flush=True)
    report = dict(scope="engineering parent-output replay; no accuracy evaluation",
        cases=len(samples), all_149_cases=len(samples) == 149, methods=keys,
        comparisons=len(rows), exact_comparisons=sum(r["exact"] for r in rows),
        all_exact=all(r["exact"] for r in rows), mismatches=[r for r in rows if not r["exact"]],
        rows=rows, environment=environment, fine_reference_read=False,
        predictions_modified=False, scores_modified=False, completed_cycle=False,
        previous_selection_sha256=digest(previous/"SELECTION.json"),
        audit_source_sha256=digest(Path(__file__)),
        completed_at_utc=datetime.now(timezone.utc).isoformat())
    save(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()
    report = replay(args.source, args.previous, args.output, args.cases)
    print(json.dumps({k: report[k] for k in ("cases", "comparisons", "exact_comparisons", "all_exact")}))
    if not report["all_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
