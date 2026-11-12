#!/usr/bin/env python3
"""Reproduce the accepted BCSS case-001 mask from frozen Chiamaka-local inputs.

Run only after other Bora CPU/GPU bundles have completed. This script never
reads an expert mask, modifies a frozen input, or replaces an existing output.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

CASE = "case-001"
EXPECTED_PIXEL_SHA256 = "0249c9b1b37cfa1dfc0cedd941ec0875cae7c38ce0d9fcdf379c09cee7dedf16"
EXPECTED_HISTORICAL_PNG_SHA256 = "bc8e03ad2a13b197f34ac218349d3aa3973fcd7364fb735f99548d58d577fc8a"
EXPECTED_TRAINING_FEATURE_SHA256 = "67350fe358f8302ac745b4ab561ac6c09628725ee9b9715a564055107b3eb574"
EXPECTED_TRAINING_INDEX_SHA256 = "458996ed744a79e3360737da3767dd99ec277bed905770f79302e94304af4229"
EXPECTED_ROI_FEATURE_SHA256 = "0f7ba45a1ea5714db19bdfd48bd8287371da1062331bcf50a03e630ec4d44e34"
EXPECTED_HISTORICAL_MODEL_SHA256 = "3027497160b131b4a8ca5c2fcd89ce5a3bad13d41abb667ce6c3b9a97ae26053"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def pinned(path: Path, expected: str) -> Path:
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"SHA256 mismatch for {path}: {observed} != {expected}")
    return path


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2) + "\n")


def run(*args: str) -> None:
    subprocess.run(["bora", *args], check=True)


def reproduce(root: Path, out: Path) -> dict:
    if version("bora-segmentation") != "0.3.0":
        raise RuntimeError("This recipe requires bora-segmentation 0.3.0")
    if out.exists():
        raise FileExistsError(f"Use a fresh output directory: {out}")
    out.mkdir(parents=True)
    migration = root / "migration"
    sample = read(migration / "benchmark1-supervised-strategy-search-2026-11-10-v1/SAMPLE_COMPLETE.json")
    ontology = read(migration / "benchmark-label-reduction-2026-11-11-v1/ONTOLOGY_FROZEN.json")
    old = read(migration / "benchmark1-peso-fullmask-head-2026-11-12-v3/DEVELOPMENT_EVALUATION_COMPLETE.json")
    frozen_model = read(migration / "benchmark1-eight-class-head-search-2026-11-12-v1/FINAL_MODEL_FROZEN.json")
    if (frozen_model["model_sha256"] != EXPECTED_HISTORICAL_MODEL_SHA256 or
            frozen_model["selected_channels"] != 203):
        raise ValueError("Historical eight-class model identity changed")
    pinned(Path(frozen_model["model_path"]), EXPECTED_HISTORICAL_MODEL_SHA256)
    case = next(row for row in old["cases"] if row["id"] == CASE)
    if case["dataset"] != "bcss" or case["prediction_sha256"] != EXPECTED_HISTORICAL_PNG_SHA256:
        raise ValueError("Accepted BCSS result identity changed")
    inputs = read(migration / "benchmark1-supervised-virchow2-response-2026-11-10-v1/INPUTS_FROZEN.json")
    roi = next(row for row in inputs["rows"] if row["case"] == case["case"] and row["split"] == "validation")
    geometry = read(root.parent / "Bora-Data/benchmark1-supervised-virchow2-response-2026-11-10-v1/case-001/RESPONSE_COMPLETE.json")
    training_features = pinned(Path(sample["feature_path"]), EXPECTED_TRAINING_FEATURE_SHA256)
    training_index = pinned(Path(sample["index_path"]), EXPECTED_TRAINING_INDEX_SHA256)
    roi_features = pinned(Path(roi["feature_path"]), EXPECTED_ROI_FEATURE_SHA256)
    if (sample["feature_sha256"] != EXPECTED_TRAINING_FEATURE_SHA256 or
            sample["index_sha256"] != EXPECTED_TRAINING_INDEX_SHA256 or
            roi["feature_sha256"] != EXPECTED_ROI_FEATURE_SHA256 or
            roi["feature_shape"] != [128, 192, 1280] or
            geometry["working_size_wh"] != [1897, 1460] or
            geometry["combined_mask"]["display_size_wh"] != [768, 591]):
        raise ValueError("Frozen feature or registration contract changed")

    index = np.load(training_index, allow_pickle=False)
    lut = np.zeros(256, np.uint8)
    for old_id, new_id in ontology["old_to_new"].items():
        if new_id is not None:
            lut[int(old_id)] = int(new_id)
    labels = lut[index["label"]]
    label_path = out / "training-labels.npy"
    np.save(label_path, labels)
    fit_manifest = out / "fit.json"
    write(fit_manifest, {
        "format": "bora-supervised-fit-v1",
        "split": "train",
        "method": "eight-class-ridge",
        "source_full_channels": 1280,
        "features": {"path": str(training_features), "sha256": EXPECTED_TRAINING_FEATURE_SHA256},
        "labels": {"path": label_path.name, "sha256": sha256(label_path)},
    })
    model = out / "model.npz"
    run("fit", str(fit_manifest), "--output", str(model))
    run("verify", str(model))
    with np.load(model, allow_pickle=False) as fitted:
        if not np.array_equal(fitted["selected"], frozen_model["original_channel_indices"]):
            raise ValueError("Packaged model selected different Virchow2 channels")
    predict_manifest = out / "predict.json"
    write(predict_manifest, {
        "format": "bora-supervised-predict-v1",
        "model": {"path": model.name, "sha256": sha256(model)},
        "features": {"path": str(roi_features), "sha256": EXPECTED_ROI_FEATURE_SHA256},
        "roi": {
            "working_size_wh": [1897, 1460],
            "display_size_wh": [768, 591],
            "allowed_classes": [1, 2, 3, 4, 5, 6, 7, 8],
            "polish": True,
        },
    })
    prediction = out / "prediction"
    run("predict", str(predict_manifest), "--output", str(prediction))
    run("verify", str(prediction))
    mask = prediction / "class-mask.png"
    with Image.open(mask) as image:
        if image.size != (768, 591) or image.mode != "L":
            raise ValueError("Unexpected mask frame or dtype")
        pixel_sha = hashlib.sha256(image.tobytes()).hexdigest()
    result = {
        "status": "REPRODUCED" if pixel_sha == EXPECTED_PIXEL_SHA256 else "MISMATCH",
        "package_version": "0.3.0",
        "case": CASE,
        "model_path": str(model),
        "model_sha256": sha256(model),
        "mask_path": str(mask),
        "mask_file_sha256": sha256(mask),
        "mask_pixel_sha256": pixel_sha,
        "expected_mask_pixel_sha256": EXPECTED_PIXEL_SHA256,
        "accepted_historical_mask_file_sha256": EXPECTED_HISTORICAL_PNG_SHA256,
        "expert_mask_opened": False,
        "gpu_used": False,
    }
    write(out / "REPRODUCE_RESULT.json", result)
    if result["status"] != "REPRODUCED":
        raise ValueError("Mask pixels differ from the accepted result; preserve outputs")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bora-root", type=Path, required=True,
                        help="Curated Chiamaka Bora project with frozen migration receipts")
    parser.add_argument("--output", type=Path, required=True,
                        help="Fresh directory for all new files")
    args = parser.parse_args()
    try:
        print(json.dumps(reproduce(args.bora_root.resolve(), args.output.resolve()), indent=2))
    except BaseException as exc:
        if args.output.exists() and not (args.output / "FAILED.json").exists():
            (args.output / "FAILED.json").write_text(json.dumps({
                "status": "FAILED_PRESERVE_PARTIAL", "error": repr(exc)}) + "\n")
        raise


if __name__ == "__main__":
    main()
