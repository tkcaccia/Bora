"""Audit frozen feature arrays, coverage and calibration without a model or GT."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.bora_extension.features import sampling, verify_case
from experiments.virchow2_inverse.common import digest
from experiments.virchow2_tissue.support import encoder_coverage


def audit(root, cases=None):
    root = Path(root)
    protocol = json.loads((root/"PROTOCOL.json").read_text())
    samples = {}
    for source, expected in protocol["source_plans"].items():
        path = Path(source)/"PLAN.json"
        if digest(path) != expected:
            raise ValueError("Source feature plan changed")
        for row in json.loads(path.read_text())["samples"]:
            if row["case"] in samples:
                raise ValueError("Duplicate input case")
            samples[row["case"]] = row
    if set(samples) != set(protocol["cases"]):
        raise ValueError("Protocol/source cases differ")
    wanted = list(cases) if cases is not None else protocol["cases"]
    if len(set(wanted)) != len(wanted) or not set(wanted) <= set(samples):
        raise ValueError("Invalid audit subset")
    full = cases is None
    if full:
        frozen = json.loads((root/"ALL_FEATURES_FROZEN.json").read_text())
        if frozen["cases"] != len(samples) or frozen["protocol_sha256"] != digest(root/"PROTOCOL.json"):
            raise ValueError("Global feature freeze changed")
    rows = []
    for case in wanted:
        sample = samples[case]
        folder = root/"cases"/case
        record = verify_case(folder, sample)
        if full and digest(folder/"FEATURES_FROZEN.json") != frozen["case_manifest_sha256"][case]:
            raise ValueError("Case manifest changed after global freeze")
        source = next(Path(s) for s in protocol["source_plans"]
            if (Path(s)/"cases"/case).is_dir())
        if digest(source/"cases"/case/"original.png") != sample["original_sha256"]:
            raise ValueError("Original image changed")
        if digest(source/"cases"/case/"coarse.tif") != sample["coarse_sha256"]:
            raise ValueError("Coarse input changed")
        selection = json.loads((folder/"FEATURE_SELECTION.json").read_text())
        indices = selection["selection"]["indices"]
        if len(indices) != 64 or len(set(indices)) != 64 or min(indices) < 0 or max(indices) >= 1280:
            raise ValueError("Invalid selected feature dimensions")
        for scale, factor in (("base", 1), ("scale_1", 2), ("scale_2", 4)):
            f = folder/"features"
            meta = json.loads((f/("READY.json" if scale == "base" else scale+".json")).read_text())
            values = np.load(f/(scale+".npy"), mmap_mode="r")
            shape = sample["working_shape_yx"] if factor == 1 else [round(v/factor) for v in sample["working_shape_yx"]]
            if values.ndim != 3 or values.dtype != np.float16 or values.shape[2] != (1280 if factor == 1 else 64):
                raise ValueError("Unexpected feature tensor")
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite features")
            if meta["sampling"] != sampling(sample, shape) or meta["original_field_shape_yx"] != shape:
                raise ValueError("Feature sampling/field geometry changed")
            if meta["selected_dimensions"] != (None if factor == 1 else indices):
                raise ValueError("Variant selected dimensions changed")
            if meta["sha256"] != digest(f/(scale+".npy")):
                raise ValueError("Tensor/metadata hash mismatch")
            coverage = encoder_coverage(meta, values.shape[:2], shape)
            if not coverage["all_original_tokens_covered"]:
                raise ValueError("Incomplete original-field coverage")
            if tuple(values.shape[:2]) != tuple(v//14 for v in meta["padded_shape_yx"]):
                raise ValueError("Token canvas shape differs from padded image")
        rows.append(dict(case=case, dataset=sample["dataset"],
            feature_manifest_sha256=digest(folder/"FEATURES_FROZEN.json"),
            calibrated=record["sampling"]["calibrated_model_sampling"],
            single_coarse_class=record["single_coarse_class"]))
        print(json.dumps(dict(audited=case)), flush=True)
    return dict(passed=True, cases=len(rows), full_corpus=full,
        arrays=3*len(rows), datasets=dict(Counter(r["dataset"] for r in rows)),
        calibrated=sum(r["calibrated"] for r in rows),
        pixel_only=sum(not r["calibrated"] for r in rows),
        single_coarse_class=sum(r["single_coarse_class"] for r in rows),
        all_cache_and_input_hashes_verified=True, all_original_tokens_covered=True,
        selected_dimensions_verified=True, arrays_finite=True,
        protocol_sha256=digest(root/"PROTOCOL.json"),
        audit_code_sha256=digest(__file__), rows=rows,
        fine_reference_read=False, predictions_modified=False, bora_run=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cases", nargs="*")
    args = p.parse_args()
    with threadpool_limits(limits=2):
        result = audit(args.root, args.cases)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
