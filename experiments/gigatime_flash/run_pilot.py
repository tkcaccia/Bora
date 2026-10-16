#!/usr/bin/env python3
"""Resolution-aware, coarse-label-only virtual-marker Bora pilot on TIGER 114S.

Run from the repository root with PYTHONPATH=src. No fine reference mask is
loaded until all inference, marker selection and refinement have finished.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import tifffile
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.gigatime_flash.model import GigaTIMEFlash
from bora.boundary import BoundaryConfig, smooth_tile_boundaries
from bora.virtual_markers import sampling_plan, select_markers, standardize_markers

REVISION = "47a5ee847d77cbe0e75f2f67f0c95d45e9bd7ca2"
CHECKPOINT_SHA256 = "ac93f0ea65df2cc6de388415ec5f5c074feab4754aabab450216ef69fe5dd411"
CNN_REVISION = "a6556030bdb8071857559e12b921620aa1bbac5b"
CNN_SHA256 = "d200308302c7fefd13b259c62de546f2c859c28d1079e9f67dc8808915d95d0c"
CHANNELS = ["DAPI", "TRITC", "Cy5", "PD-1", "CD14", "CD4", "T-bet", "CD34", "CD68",
            "CD16", "CD11c", "CD138", "CD20", "CD3", "CD8", "PD-L1", "CK", "Ki67",
            "Tryptase", "Actin-D", "Caspase3-D", "PHH3-B", "Transgelin"]
SAMPLE = "tissue-cells--114S_[33188, 13944, 34412, 15110]"
CLASS_NAMES = {1: "invasive tumor", 2: "tumor-associated stroma", 3: "in-situ tumor",
               4: "healthy glands", 5: "necrosis", 6: "inflamed stroma", 7: "rest"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def starts(length, window=256, stride=128):
    last = max(0, length-window)
    return sorted(set([*range(0, last+1, stride), last]))


def load_model(weights, family="flash"):
    if family not in {"cnn", "flash"}:
        raise ValueError("Unknown GigaTIME model family")
    expected_sha = CNN_SHA256 if family == "cnn" else CHECKPOINT_SHA256
    if digest(weights) != expected_sha:
        raise ValueError(f"Checkpoint hash differs from the pinned released {family} weights")
    if family == "cnn":
        from experiments.gigatime_flash.original_cnn import gigatime
        model = gigatime(num_classes=23)
    else:
        model = GigaTIMEFlash(num_classes=23)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    # Fail closed. Includes the patch-projection convolution's LoRA adapters,
    # which the official fallback notebook silently omits.
    model.load_state_dict(checkpoint, strict=True)
    count = sum(p.numel() for p in model.parameters())
    if family == "flash" and count != 23_806_559:
        raise ValueError(f"Unexpected Flash parameter count: {count}")
    model.eval().requires_grad_(False)
    return model, {"model_family": family,
                   "model_name": "GigaTIME CNN" if family == "cnn" else "GigaTIME-Flash",
                   "model_repository": "prov-gigatime/GigaTIME" if family == "cnn" else "prov-gigatime/gigatime-flash",
                   "model_revision": CNN_REVISION if family == "cnn" else REVISION,
                   "checkpoint_sha256": expected_sha,
                   "parameters": count, "state_tensors": len(checkpoint),
                   "strict_load": True, "missing_tensors": [], "unexpected_tensors": []}


def predict_maps(rgb, model, plan, config, folder, batch_size, device, model_info=None):
    """Overlap-add 256px windows; retain full maps and all partial image edges."""
    folder.mkdir(parents=True, exist_ok=True)
    nh, nw = plan["inference_shape_yx"]
    h, w = rgb.shape[:2]
    checkpoint_sha = model_info["checkpoint_sha256"] if model_info else CHECKPOINT_SHA256
    family = model_info.get("model_family", "flash") if model_info else "flash"
    fingerprint_data = {"rgb_sha256": hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),
        "sampling": plan, "config": config, "checkpoint_sha256": checkpoint_sha,
        "inference_source": inspect.getsource(predict_maps), "torch": torch.__version__,
        "architecture_source_sha256": digest(Path(__file__).with_name(
            "original_cnn.py" if family == "cnn" else "model.py"))}
    fingerprint = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest()
    native_path = folder / "virtual_markers.native.npy"
    meta_path = folder / "inference.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old["sampling"] != plan or old["checkpoint_sha256"] != checkpoint_sha:
            raise ValueError("Output directory contains incompatible cached inference")
        if old.get("cache_fingerprint") == fingerprint:
            if digest(native_path) != old["native_maps_sha256"]:
                raise ValueError("Cached marker maps failed checksum")
            return np.load(native_path, mmap_mode="r"), old
    began = time.perf_counter()
    if (nh, nw) == (h, w):
        scaled = rgb
    else:
        scaled = cv2.resize(rgb, (nw, nh), interpolation=(
            cv2.INTER_CUBIC if nh > h or nw > w else cv2.INTER_AREA))
    ph, pw = max(nh, 256), max(nw, 256)
    padded = np.pad(scaled, ((0, ph-nh), (0, pw-nw), (0, 0)), mode="reflect")
    coordinates = [(y, x) for y in starts(ph) for x in starts(pw)]
    # A nonzero Hann floor retains the outermost pixels.
    hann = np.maximum(np.hanning(256).astype(np.float32), 0.01)
    blend = hann[:, None] * hann[None, :]
    accum = np.zeros((23, ph, pw), np.float32)
    mass = np.zeros((ph, pw), np.float32)
    mean = np.asarray(config["normalization"]["mean"], np.float32)
    std = np.asarray(config["normalization"]["std"], np.float32)
    model.to(device)
    with torch.inference_mode():
        for k in range(0, len(coordinates), batch_size):
            batch_coords = coordinates[k:k+batch_size]
            tiles = np.stack([padded[y:y+256, x:x+256] for y, x in batch_coords])
            values = ((tiles.astype(np.float32)/255 - mean)/std).transpose(0, 3, 1, 2)
            tensor = torch.from_numpy(np.ascontiguousarray(values)).to(device)
            logits = model(tensor)
            if tuple(logits.shape[1:]) != (23, 256, 256):
                raise ValueError("Flash output dimensions differ from the model configuration")
            probabilities = logits.sigmoid().cpu().numpy()
            if not np.isfinite(probabilities).all():
                raise ValueError("Nonfinite virtual activations")
            for (y, x), maps in zip(batch_coords, probabilities):
                accum[:, y:y+256, x:x+256] += maps * blend
                mass[y:y+256, x:x+256] += blend
    if not np.all(mass > 0):
        raise RuntimeError("Uncovered inference pixels")
    model_grid_path = folder / "virtual_markers.model_grid.npy"
    model_grid = np.lib.format.open_memmap(model_grid_path, mode="w+", dtype=np.float16,
                                         shape=(23, nh, nw))
    native = np.lib.format.open_memmap(native_path, mode="w+", dtype=np.float16,
                                      shape=(23, h, w))
    for c in range(23):
        plane = accum[c, :nh, :nw] / mass[:nh, :nw]
        model_grid[c] = plane
        native[c] = (plane if (nh, nw) == (h, w) else
                     cv2.resize(plane, (w, h), interpolation=cv2.INTER_LINEAR))
    model_grid.flush()
    native.flush()
    elapsed = time.perf_counter() - began
    info = {"sampling": plan, "checkpoint_sha256": checkpoint_sha, "model_family": family,
            "cache_fingerprint": fingerprint,
            "input_window_px": [256, 256], "window_field_of_view_um_xy":
                [256*x for x in plan["effective_inference_mpp_xy"]],
            "vit_patch_px": [16, 16] if family == "flash" else None,
            "token_grid_yx": [16, 16] if family == "flash" else None,
            "output_window_px": [256, 256], "window_stride_px": 128,
            "windows": len(coordinates), "batch_size": batch_size, "precision": "float32",
            "input_interpolation": "none" if (nh, nw) == (h, w) else "bicubic",
            "output_alignment": "bilinear to original pixel-centre frame, align_corners=False convention",
            "stitching": "probability-space overlap-add; separable Hann with 0.01 axis floor",
            "edge_policy": "last window anchored at image edge; reflect pad only when dimension <256",
            "all_image_pixels_covered": True, "channel_order": config["channels"],
            "channels_are_measured_stains": False, "saved_dtype": "float16",
            "inference_seconds": elapsed, "native_maps_sha256": digest(native_path),
            "model_grid_maps_sha256": digest(model_grid_path),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if device == "cuda" else None}
    save_json(meta_path, info)
    print(json.dumps({"inference_complete_mpp": plan["requested_mpp"],
                      "windows": len(coordinates), "seconds": elapsed}), flush=True)
    return native, info


def internal_border(labels):
    border = np.zeros(labels.shape, bool)
    for a, b, axis in [(labels[:-1], labels[1:], 0), (labels[:, :-1], labels[:, 1:], 1)]:
        diff = (a != b) & (a > 0) & (b > 0)
        if axis == 0:
            border[:-1] |= diff
            border[1:] |= diff
        else:
            border[:, :-1] |= diff
            border[:, 1:] |= diff
    return border


def score(pred, truth, mpp_xy, names):
    valid = truth > 0
    per_class = {}
    for value in np.unique(truth[valid]):
        p, t = (pred == value) & valid, truth == value
        inter = np.count_nonzero(p & t)
        total = np.count_nonzero(p) + np.count_nonzero(t)
        union = np.count_nonzero(p | t)
        per_class[str(value)] = {"name": names[str(value)],
            "iou": float(inter/union), "dice": float(2*inter/total),
            "truth_pixels": int(t.sum()), "prediction_pixels": int(p.sum())}
    pb, tb = internal_border(pred) & valid, internal_border(truth)
    boundary = {"f1_at_um": {}, "assd_um": None, "hd95_um": None}
    if pb.any() and tb.any():
        mx, my = mpp_xy
        dp = ndi.distance_transform_edt(~tb, sampling=(my, mx))[pb]
        dt = ndi.distance_transform_edt(~pb, sampling=(my, mx))[tb]
        for tol in [4, 8, 16]:
            precision, recall = float(np.mean(dp <= tol)), float(np.mean(dt <= tol))
            boundary["f1_at_um"][str(tol)] = (2*precision*recall/(precision+recall)
                                                     if precision+recall else 0.0)
        boundary["assd_um"] = float((dp.mean()+dt.mean())/2)
        boundary["hd95_um"] = float(np.percentile(np.r_[dp, dt], 95))
    return {"pixel_accuracy": float(np.mean(pred[valid] == truth[valid])),
            "macro_iou": float(np.mean([v["iou"] for v in per_class.values()])),
            "macro_dice": float(np.mean([v["dice"] for v in per_class.values()])),
            "per_class": per_class, "internal_boundary": boundary,
            "evaluation_shape_yx": list(truth.shape), "reference_ignore_label": 0,
            "valid_pixels": int(valid.sum())}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--data-root", type=Path, default=Path("/mnt/sata_ssd/Bora-benchmark"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--model-family", choices=["cnn", "flash"], default="cnn")
    p.add_argument("--dapi-validation-report", type=Path, required=True,
                   help="Paired-reference audit confirmation.json; no unvalidated default run")
    p.add_argument("--channel-validation-report", type=Path,
                   help="Required for CNN: channel_validation.json from validate_channels.py")
    p.add_argument("--tile-sizes", nargs="+", type=int, default=[56, 90, 216])
    p.add_argument("--sampling", nargs="+", type=float, default=[0.25])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    torch.manual_seed(20261016)
    np.random.seed(20261016)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    row = next(json.loads(line) for line in (args.data_root/"prepared/tiger.jsonl").open()
               if json.loads(line)["id"] == SAMPLE)
    image_path = args.data_root/row["image"]
    if digest(image_path) != row["image_sha256"]:
        raise ValueError("Source RGB checksum mismatch")
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    mpp_xy = [float(row["mpp_um"])]*2
    existing = args.data_root/"results/bora-pilot-independent-2026-10-16/tiger"
    provenance = json.loads((existing/"sample.json").read_text())
    if provenance["inventory"]["id"] != SAMPLE:
        raise ValueError("Existing coarse masks belong to another image")
    mapping = {int(k): int(v) for k, v in provenance["label_remap"].items()}
    names = {str(v): CLASS_NAMES[k] for k, v in mapping.items()}
    config = json.loads(args.config.read_text())
    if args.model_family == "cnn":
        if config.get("architecture") != "gigatime" or config.get("num_classes") != 23:
            raise ValueError("Unexpected original GigaTIME CNN configuration")
        config = {"num_channels": 23, "input_shape": [3, 256, 256],
                  "normalization": {k: config["pretrained_cfg"][k] for k in ["mean", "std"]},
                  "channels": CHANNELS}
    if config["num_channels"] != 23 or config["input_shape"] != [3, 256, 256]:
        raise ValueError("Unexpected model configuration")
    boundary_config = BoundaryConfig()
    model, model_info = load_model(args.weights, args.model_family)
    validation = json.loads(args.dapi_validation_report.read_text())
    expected_validation_sha = validation.get("cnn_sha256") if args.model_family == "cnn" else validation["model"]["checkpoint_sha256"]
    if expected_validation_sha != model_info["checkpoint_sha256"]:
        raise ValueError("DAPI reference audit used a different model checkpoint")
    audit_key = "original_cnn_raw" if args.model_family == "cnn" else "flash_raw"
    reference_scores = validation["summary"][audit_key]
    if (validation["n_confirmation"] < 20 or reference_scores["auc"]["mean"] < .85 or
            reference_scores["pearson_8px"]["mean"] < .60):
        raise ValueError("Reference DAPI sanity check failed; do not run marker refinement")
    channel_validation = None
    excluded_channels = ["DAPI", "TRITC", "Cy5"]
    if args.model_family == "cnn" and not args.channel_validation_report:
        raise ValueError("CNN refinement requires the paired-channel reference screen")
    if args.channel_validation_report:
        channel_validation = json.loads(args.channel_validation_report.read_text())
        if channel_validation["model"]["checkpoint_sha256"] != model_info["checkpoint_sha256"]:
            raise ValueError("Channel reference screen used a different checkpoint")
        eligible = channel_validation["eligible_channels"]
        if not eligible or not set(eligible).issubset(config["channels"]):
            raise ValueError("Empty or incompatible eligible channel list")
        excluded_channels += [name for name in config["channels"] if name not in eligible]
    maps_by_scale, inference = {}, {}
    for sampling in args.sampling:
        key = f"{sampling:.6f}"
        plan = sampling_plan(rgb.shape[:2], mpp_xy, sampling)
        maps_by_scale[key], inference[key] = predict_maps(rgb, model, plan, config,
                            args.output/f"inference_{key}", args.batch_size, device, model_info)
    model.cpu()
    if device == "cuda":
        torch.cuda.empty_cache()
    predictions, run_info = {}, []
    for size in args.tile_sizes:
        case = args.output/f"tile_{size}"
        case.mkdir(exist_ok=True)
        coarse_path = existing/f"error_0/tile_{size}/coarse.tif"
        coarse = tifffile.imread(coarse_path)
        if coarse.shape != rgb.shape[:2]:
            raise ValueError("Coarse labels are not in the native image frame")
        tifffile.imwrite(case/"coarse.tif", coarse, compression="deflate")
        predictions[size] = {"coarse": str(case/"coarse.tif")}
        began = time.perf_counter()
        rgb_result = smooth_tile_boundaries(rgb, coarse, size, boundary_config)
        rgb_path = case/"bora_rgb.tif"
        tifffile.imwrite(rgb_path, rgb_result, compression="deflate")
        predictions[size]["bora_rgb"] = str(rgb_path)
        run_info.append({"tile_size": size, "method": "bora_rgb",
                         "seconds": time.perf_counter()-began,
                         "coarse_sha256": digest(coarse_path)})
        for scale, maps in maps_by_scale.items():
            began = time.perf_counter()
            hwc = np.moveaxis(maps, 0, -1).astype(np.float32)
            selection = select_markers(hwc, coarse, size, config["channels"],
                                       excluded=tuple(dict.fromkeys(excluded_channels)))
            if channel_validation:
                selection["reference_eligibility_rule"] = channel_validation["rule"]
            save_json(case/f"marker_selection_{scale}.json", selection)
            columns = list(selection["comparisons"][0])
            with (case/f"marker_contrasts_{scale}.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                writer.writerows(selection["comparisons"])
            selected, scaling = standardize_markers(hwc[..., selection["selected_indices"]], coarse > 0)
            del hwc
            result = smooth_tile_boundaries(rgb, coarse, size, boundary_config, feature_image=selected)
            del selected
            key = f"bora_markers_{scale}"
            dest = case/f"{key}.tif"
            tifffile.imwrite(dest, result, compression="deflate")
            predictions[size][key] = str(dest)
            run_info.append({"tile_size": size, "method": key, "scaling": scaling,
                             "selected_channels": selection["selected_channels"],
                             "seconds": time.perf_counter()-began,
                             "coarse_sha256": digest(coarse_path)})
            print(json.dumps({"refined": key, "tile_size": size,
                              "channels": selection["selected_channels"]}), flush=True)
    # Fine expert labels enter only here, after every prediction is fixed.
    mask_path = args.data_root/row["mask"]
    if digest(mask_path) != row["mask_sha256"]:
        raise ValueError("Source expert mask checksum mismatch")
    raw_truth = np.asarray(Image.open(mask_path))
    truth = np.zeros(raw_truth.shape, np.uint8)
    for original, compact in mapping.items():
        truth[raw_truth == original] = compact
    tifffile.imwrite(args.output/"expert_reference.compact.tif", truth, compression="deflate")
    scores = []
    for size, paths in predictions.items():
        for method, path in paths.items():
            result = score(tifffile.imread(path), truth, mpp_xy, names)
            scores.append({"tile_size": size, "method": method, **result})
    report = {"sample": row, "image_type": "ROI, not a full WSI", "native_mpp_status":
              "provider-reported approximate; scanner, objective NA and true optical resolving power unknown",
              "microscope_objective_magnification": None, "microscope_optical_resolution_um": None,
              "model": model_info, "inference": inference,
              "reference_scale_um_per_pixel": 128/512,
              "reference_scale_status": "author-clarified: 128 um / 556 px registration patches resized to 512 px before 256 px windows",
              "dapi_reference_validation": {"report_sha256": digest(args.dapi_validation_report),
                  "method": audit_key, "n_patches": validation["n_confirmation"], "summary": reference_scores,
                  "status": "passes reference sanity check; not clinical or target-domain marker validation",
                  "target_has_measured_marker_ground_truth": False},
              "eligible_marker_channels": channel_validation["eligible_channels"] if channel_validation else None,
              "channel_reference_validation": {"report_sha256": digest(args.channel_validation_report),
                  "rule": channel_validation["rule"], "summary": channel_validation["summary"]} if channel_validation else None,
              "sources": {"original_paper": "https://doi.org/10.1016/j.cell.2025.11.016",
                "flash_paper": "https://arxiv.org/html/2607.18218v1",
                "preprocessing_clarification": "https://github.com/prov-gigatime/GigaTIME/issues/4",
                "channel_order_clarification": "https://github.com/prov-gigatime/GigaTIME/issues/8",
                "tiger": "https://tiger.grand-challenge.org/Data/"},
              "coarse_protocol": "reference-derived C++ tile-majority labels, 0% injected border error; controlled reconstruction, not independently inferred groups",
              "coarse_generation": {k: v for k, v in provenance.items() if k != "inventory"},
              "reference_mask_used_after_predictions_fixed": True,
              "boundary_config": asdict(boundary_config), "working_downsample": boundary_config.downsample,
              "bora_working_mpp_xy": [x*boundary_config.downsample for x in mpp_xy],
              "native_label_remap": mapping, "compact_class_names": names,
              "runs": run_info, "scores": scores,
              "n_images": 1, "results_are_exploratory": True,
              "environment": {"python": sys.version, "platform": platform.platform(),
                "device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                "packages": {k: importlib.metadata.version(k) for k in ["torch", "torchvision", "timm", "numpy", "opencv-python", "scipy", "Pillow", "tifffile"]}},
              "source_sha256": {"image": digest(image_path), "mask": digest(mask_path),
                  "run_pilot.py": digest(__file__), "model.py": digest(Path(__file__).with_name("model.py")),
                  "original_cnn.py": digest(Path(__file__).with_name("original_cnn.py")),
                  "boundary.py": digest(Path(__file__).resolve().parents[2]/"src/bora/boundary.py"),
                  "virtual_markers.py": digest(Path(__file__).resolve().parents[2]/"src/bora/virtual_markers.py")}}
    save_json(args.output/"results.json", report)
    with (args.output/"scores.csv").open("w", newline="") as f:
        columns = ["tile_size", "method", "pixel_accuracy", "macro_iou", "macro_dice", "boundary_f1_4um", "assd_um", "hd95_um"]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in scores:
            writer.writerow({**{k: r[k] for k in columns[:5]},
                "boundary_f1_4um": r["internal_boundary"]["f1_at_um"].get("4"),
                "assd_um": r["internal_boundary"]["assd_um"], "hd95_um": r["internal_boundary"]["hd95_um"]})
    from experiments.gigatime_flash.plot_overlays import draw
    draw(args.output, args.data_root, tile_size=90 if 90 in predictions else min(predictions))
    print(json.dumps({"completed": str(args.output), "scores": str(args.output/"scores.csv")}), flush=True)


if __name__ == "__main__":
    main()
