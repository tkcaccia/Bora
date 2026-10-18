"""Full-field frozen Virchow2 inference for the 200 audited additions.

This stage never opens a fine reference or runs Bora. Unknown pixel spacing
stays unknown; new inference now requires traceable physical sampling.
Historical exploratory pixel-only caches remain archived outside new runs. The encoder,
normalization, token selection and overlap stitching reuse the fixed study
implementation. Multiscale variants retain its original factors two and four.
"""
import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile

from experiments.virchow2_inverse.common import (
    MODEL_REVISION, MODEL_SHA256, coarse_tile_observations, digest, save_json)
from experiments.virchow2_inverse.selection import select_features
from experiments.virchow2_tissue.support import encoder_coverage


def sampling(sample, shape):
    """Use actual resized axes; never give uncalibrated pixels a nominal MPP."""
    h, w = map(int, shape)
    if h < 1 or w < 1:
        raise ValueError("Empty encoder image")
    spacing = sample["working_mpp_xy"]
    if spacing is None:
        if sample.get("fov_um_xy") is not None:
            raise ValueError("Unknown spacing has a contradictory physical field")
        return dict(status="uncalibrated native-pixel inference", mpp_xy=None,
                    window_fov_um_xy=None, patch_fov_um_xy=None,
                    calibrated_model_sampling=False, numerical_units="pixels",
                    shape_yx=[h, w])
    wh, ww = sample["working_shape_yx"]
    if len(spacing) != 2 or not np.isfinite(spacing).all() or min(spacing) <= 0:
        raise ValueError("Invalid measured sampling")
    axes = [float(spacing[0]*ww/w), float(spacing[1]*wh/h)]
    return dict(status="provider-calibrated working-image axes", mpp_xy=axes,
                window_fov_um_xy=[224*v for v in axes],
                patch_fov_um_xy=[14*v for v in axes],
                calibrated_model_sampling=True, numerical_units="micrometres",
                shape_yx=[h, w])


def choose_dimensions(features, coarse):
    summaries, labels, _, _ = coarse_tile_observations(features, coarse)
    classes = np.unique(labels)
    if len(classes) > 1:
        selection = select_features(summaries, labels, "balanced_mrmr", 64)
    else:
        # No discriminative statistic exists with one observed class. These
        # dimensions only permit the same cache format; RGB will be constant.
        variance = np.var(summaries, axis=0)
        indices = np.argsort(-variance, kind="stable")[:64]
        selection = dict(indices=indices.tolist(),
                         method="single-class variance cache only",
                         discriminative_selection_possible=False)
    selected = np.asarray(selection["indices"], dtype=int)
    if len(selected) != 64 or len(np.unique(selected)) != 64:
        raise ValueError("The fixed conversion requires 64 distinct channels")
    return selected, dict(selection=selection, observed_classes=classes.tolist(),
        single_coarse_class=len(classes) == 1,
        conversion_degeneracy="constant observed-class RGB" if len(classes) == 1 else None,
        training="original coarse tile summaries only; fine reference never read")


def encode_complete(model, image, selected, encode):
    """Reflect-pad small fields, without resizing away original pixel detail.

    The fixed encoder's patch padding alone assumes both axes >=224. New
    small WSSS fields and their scale variants require this explicit extension.
    Token centres remain at the same original top-left coordinates.
    """
    h, w = image.shape[:2]
    pads = ((0, max(0, 224-h)), (0, max(0, 224-w)), (0, 0))
    padded = np.pad(image, pads, mode="reflect") if h < 224 or w < 224 else image
    values, metadata = encode(model, padded, selected)
    metadata = dict(metadata, original_field_shape_yx=[h, w],
        minimum_window_padding_bottom_right_px=[pads[0][1], pads[1][1]],
        extra_padding="reflect at right/bottom only; original pixels unchanged")
    return values, metadata


def verify_case(folder, sample, *, calibration=None):
    record = json.loads((folder/"FEATURES_FROZEN.json").read_text())
    if record["image_sha256"] != sample["original_sha256"] or record["coarse_sha256"] != sample["coarse_sha256"]:
        raise ValueError("Frozen feature input identity changed")
    # Historical evidence can still be inspected. Prospective reuse must be
    # bound to the current physical transform, not just identical PNG bytes.
    if calibration is not None and record.get("physical_calibration") != calibration:
        raise ValueError("Frozen feature physical calibration changed or is missing; use a new cache version")
    for name, expected in record["files_sha256"].items():
        if digest(folder/name) != expected:
            raise ValueError("Frozen feature cache changed: " + name)
    return record


def freeze_case(source, root, sample, model, encode):
    """Encode all original pixels, irrespective of fine masks or tissue support."""
    from .calibration import require
    calibration = require(sample)  # Before GPU inference or reuse of an old cache.
    began = time.monotonic()
    src, dst = source/"cases"/sample["case"], root/"cases"/sample["case"]
    for name, key in (("original.png", "original_sha256"), ("coarse.tif", "coarse_sha256")):
        if digest(src/name) != sample[key]:
            raise ValueError("Source input changed: " + name)
    if (dst/"FEATURES_FROZEN.json").exists():
        return verify_case(dst, sample, calibration=calibration)
    if shutil.disk_usage(root).free < 20*1024**3:
        raise RuntimeError("Keep at least20 GiB free before another feature case")
    dst.mkdir(parents=True, exist_ok=True)
    features = dst/"features"
    features.mkdir(exist_ok=True)
    image = np.array(Image.open(src/"original.png").convert("RGB"))
    coarse = tifffile.imread(src/"coarse.tif")
    if list(image.shape[:2]) != sample["working_shape_yx"] or coarse.shape != image.shape[:2]:
        raise ValueError("Working image/coarse geometry mismatch")
    base, meta = encode_complete(model, image, None, encode)
    coverage = encoder_coverage(meta, base.shape[:2], image.shape[:2])
    np.save(features/"base.npy", base)
    save_json(features/"READY.json", dict(**meta, coverage=coverage,
        sha256=digest(features/"base.npy"), image_sha256=sample["original_sha256"],
        sampling=sampling(sample, image.shape[:2])))
    selected, selection = choose_dimensions(base, coarse)
    save_json(dst/"FEATURE_SELECTION.json", selection)
    del base
    for name, factor in (("scale_1", 2), ("scale_2", 4)):
        scaled = np.array(Image.fromarray(image).resize(
            (round(image.shape[1]/factor), round(image.shape[0]/factor)), Image.Resampling.LANCZOS))
        values, info = encode_complete(model, scaled, selected, encode)
        extra_coverage = encoder_coverage(info, values.shape[:2], scaled.shape[:2])
        np.save(features/(name+".npy"), values)
        save_json(features/(name+".json"), dict(**info, coverage=extra_coverage,
            shape_yx=list(scaled.shape[:2]), sampling=sampling(sample, scaled.shape[:2]),
            factor=factor, sha256=digest(features/(name+".npy"))))
        del values
    files = {str(p.relative_to(dst)): digest(p) for p in sorted(features.iterdir())}
    files["FEATURE_SELECTION.json"] = digest(dst/"FEATURE_SELECTION.json")
    record = dict(case=sample["case"], dataset=sample["dataset"],
        image_sha256=sample["original_sha256"], coarse_sha256=sample["coarse_sha256"],
        encoder_revision=MODEL_REVISION, encoder_checkpoint_sha256=MODEL_SHA256,
        files_sha256=files, sampling=sampling(sample, image.shape[:2]),
        all_original_tokens_covered=True, fine_reference_read=False, physical_calibration=calibration,
        bora_run=False, seconds=time.monotonic()-began,
        single_coarse_class=selection["single_coarse_class"])
    save_json(dst/"FEATURES_FROZEN.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="*")
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    sources = [(s, json.loads((s/"PLAN.json").read_text())) for s in args.sources]
    samples = [(s, row) for s, plan in sources for row in plan["samples"]]
    if len({r["case"] for _, r in samples}) != len(samples):
        raise ValueError("Duplicate extension case identifiers")
    repo = Path(__file__).resolve().parents[2]
    files = [Path(__file__),Path(__file__).with_name("calibration.py"),Path(__file__).with_name("CALIBRATION_POLICY.json")] + [p for package in ("virchow2_multi", "virchow2_inverse", "virchow2_tissue")
                               for p in sorted((repo/"experiments"/package).glob("*.py"))]
    protocol = dict(cases=[r["case"] for _, r in samples],
        source_plans={str(s): digest(s/"PLAN.json") for s, _ in sources},
        code_sha256={str(p.relative_to(repo)): digest(p) for p in files},
        encoder_checkpoint_sha256=MODEL_SHA256, encoder_revision=MODEL_REVISION,
        encoder_config_sha256=digest(args.model_dir/"config.json"),
        conversion_mechanisms_changed=False, multiscale_factors=[1, 2, 4],
        window_px=224, patch_px=14, stride_px=112, batch_size=4,
        small_field_padding="reflect right/bottom to at least224; original top-left pixels unchanged",
        gpu_memory_fraction=.4, torch_threads=2, nice_expected=12,
        entire_original_RGB_encoded=True, fine_references_read=False,
        uncalibrated_images="blocked before GPU inference; retained only in acquisition inventory",
        single_class="encode full image; variance-only cache; constant observed-class RGB downstream")
    if (root/"PROTOCOL.json").exists() and json.loads((root/"PROTOCOL.json").read_text()) != protocol:
        raise ValueError("Frozen feature protocol changed")
    save_json(root/"PROTOCOL.json", protocol)
    wanted = set(args.cases) if args.cases else set(protocol["cases"])
    if not wanted <= set(protocol["cases"]):
        raise ValueError("Requested unknown case")
    pending = [(s, r) for s, r in samples if r["case"] in wanted]
    # Reject unknown/incorrect physical sampling before importing GPU packages.
    from .calibration import require
    for _, sample in samples:
        require(sample)
    # Import GPU packages only in the executable inference path.
    import torch
    from threadpoolctl import threadpool_limits
    from experiments.virchow2_inverse.gpu_ideas import load_model
    from experiments.virchow2_multi.run import encode
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.4)
    model = None
    with threadpool_limits(limits=2):
        for index, (source, sample) in enumerate(pending):
            if not (root/"cases"/sample["case"]/"FEATURES_FROZEN.json").exists() and model is None:
                model = load_model(args.model_dir)
            record = freeze_case(source, root, sample, model, encode)
            gc.collect()
            print(json.dumps(dict(case=sample["case"], complete=True,
                requested_completed=index+1, requested_total=len(pending),
                seconds=record["seconds"], sampling=record["sampling"]["status"])), flush=True)
    if all((root/"cases"/r["case"]/"FEATURES_FROZEN.json").exists() for _, r in samples):
        records = [verify_case(root/"cases"/r["case"], r, calibration=require(r)) for _, r in samples]
        save_json(root/"ALL_FEATURES_FROZEN.json", dict(cases=len(records),
            protocol_sha256=digest(root/"PROTOCOL.json"),
            case_manifest_sha256={r["case"]: digest(root/"cases"/r["case"]/"FEATURES_FROZEN.json") for r in records},
            all_original_tokens_covered=True, fine_references_read=False, bora_run=False))


if __name__ == "__main__":
    main()
