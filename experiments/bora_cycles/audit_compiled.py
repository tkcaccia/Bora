"""Compare every saved first-cycle metric against equivalent compiled counts.

Run only after all SCORES.json files exist. A failed equality is an error,
never a reason to overwrite original scores.
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import tifffile
from threadpoolctl import threadpool_limits
from .run import digest, save, verify
from .metrics import RegionEvaluator
from .diagnostics import white_regions, mask_diagnostics


def audit_case(args):
    source, root, case, expected_manifest = args
    cv2.setNumThreads(2)
    src, out = source/"cases"/case, root/"cases"/case
    if digest(out/"PREDICTIONS.json") != expected_manifest:
        raise RuntimeError("Prediction manifest changed")
    frozen = json.loads((out/"PREDICTIONS.json").read_text())
    verify(out, frozen["files_sha256"])
    saved = json.loads((out/"SCORES.json").read_text())
    sample = saved["sample"]
    if digest(src/"reference.native.tif") != sample["reference_native_sha256"]:
        raise RuntimeError("Reference changed")
    truth = tifffile.imread(src/"reference.native.tif")
    support = tifffile.imread(src/"tissue.native.tif") > 0
    he = np.array(Image.open(src/"original.native.png").convert("RGB"))
    evaluator = RegionEvaluator(truth, sample["native_mpp_xy"])
    white, exterior, _ = white_regions(he)
    records = []
    with threadpool_limits(limits=2):
        for key, row in saved["methods"].items():
            if row["status"] != "ok":
                records.append(dict(method=key, skipped_failed_candidate=True))
                continue
            mask = tifffile.imread(out/key/"native.tif")
            started = time.perf_counter()
            score = evaluator.score(mask, backend="cpp")
            semantic_seconds = time.perf_counter()-started
            started = time.perf_counter()
            background = mask_diagnostics(
                mask, support, white, exterior, sample["native_mpp_xy"], backend="cpp")
            background_seconds = time.perf_counter()-started
            if score != row["score"] or background != row["background"]:
                raise RuntimeError(f"Compiled metric mismatch: {case}/{key}")
            records.append(dict(method=key, all_metric_fields_identical=True,
                                all_background_fields_identical=True,
                                semantic_seconds=semantic_seconds,
                                background_seconds=background_seconds,
                                pixels=int(mask.size)))
    print(json.dumps(dict(compiled_audit=case, masks=len(records))), flush=True)
    return dict(case=case, records=records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    protocol = json.loads((args.root/"PROTOCOL.json").read_text())
    frozen = json.loads((args.root/"ALL_PREDICTIONS_FROZEN.json").read_text())
    for case in protocol["cases"]:
        if not (args.root/"cases"/case/"SCORES.json").is_file():
            raise RuntimeError("Wait for original scoring to finish before parity auditing")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        rows = list(pool.map(audit_case, [
            (args.source, args.root, case, frozen["manifests"][case])
            for case in protocol["cases"]]))
    records = [record for row in rows for record in row["records"]]
    repo = Path(__file__).resolve().parents[2]
    sources = {str(path.relative_to(repo)): digest(path) for path in (
        Path(__file__), Path(__file__).with_name("metrics.py"),
        Path(__file__).with_name("diagnostics.py"), Path(__file__).with_name("native.py"),
        repo/"cpp/evidence/kernels.cpp", repo/"cpp/evidence/libbora_evidence.so")}
    save(args.output, dict(
        cases=len(rows), masks_compared=sum(not r.get("skipped_failed_candidate", False) for r in records),
        all_compared_fields_identical=True, rows=rows, compiled_source_sha256=sources,
        original_protocol_sha256=digest(args.root/"PROTOCOL.json"),
        predictions_modified=False, original_scores_modified=False))


if __name__ == "__main__":
    main()
