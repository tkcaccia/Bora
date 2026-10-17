"""Prepare additional image/coarse/reference triples, without running a model.

The fine reference is allowed only at this acquisition/planning stage to select
boundary-containing fields and create controlled coarse inputs. Never selects
fields by an algorithm's prediction or score. Historical inputs stay immutable.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image
import tifffile

from experiments.virchow2_inverse.common import digest, save_json, coarse_tile_observations
from experiments.virchow2_multi.plan import (LABELS, palette, coarse_mask,
    candidate_windows, group_key, mpp, RED, GREEN)

ADDITIONS = {"bcss": 40, "tiger": 40, "wsss4luad": 40, "rings-tumor": 40}
EXTRA_LABELS = {"wsss4luad": {0: "tumor epithelium", 1: "tumor-associated stroma", 2: "normal tissue"},
    "rings-tumor": {0: "outside tumor target (not an expert stroma label)", 1: "tumor"}}


def global_rect(row, crop):
    """TIGER bbox is level-ROI pixel geometry; preserve source slide identity."""
    if row["dataset"] != "tiger":
        return None
    match = re.search(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", row["id"])
    if match is None:
        raise ValueError("Cannot audit TIGER overlap without a source bbox")
    x0, y0, x1, y1 = map(int, match.groups())
    if (x1-x0, y1-y0) != (row["width"], row["height"]):
        raise ValueError("TIGER source bbox differs from received dimensions")
    x, y, width, height = crop
    return row["slide_id"], (x0+x, y0+y, x0+x+width, y0+y+height)


def intersects(first, second):
    return (first[0] < second[2] and second[0] < first[2]
            and first[1] < second[3] and second[1] < first[3])


def pixel_windows(truth, side=1280):
    """Native pixel crops for uncalibrated files; never invent micrometres."""
    height, width = truth.shape
    ch, cw = min(height, side), min(width, side)
    positions = [(y, x) for y in sorted({0, (height-ch)//2, height-ch})
                 for x in sorted({0, (width-cw)//2, width-cw})]
    ranked = []
    for y, x in positions:
        a = truth[y:y+ch:8, x:x+cw:8]
        _, count = np.unique(a[a > 0], return_counts=True)
        p = count / max(count.sum(), 1)
        entropy = -float(np.sum(p * np.log(np.maximum(p, 1e-30))))
        ranked.append((entropy, -y, -x, y, x, ch, cw))
    return [(r[-4], r[-3], r[-2], r[-1]) for r in sorted(ranked, reverse=True)]


def read_rows(data, dataset):
    path = data / "prepared" / (dataset + ".jsonl")
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def prepare(data, previous, root, seed=8819):
    data, previous, root = map(Path, (data, previous, root))
    if root.exists():
        raise FileExistsError("Choose a new extension root; existing plans are immutable")
    old = json.loads((previous / "PLAN.json").read_text())["samples"]
    used_ids = {(s["dataset"], s["source"]["id"]) for s in old}
    used_hashes = {s["source"]["image_sha256"] for s in old}
    groups = {d: {s["group"] for s in old if s["dataset"] == d} for d in ADDITIONS}
    rectangles = []
    for s in old:
        rect = global_rect(s["source"], s["native_crop_xywh"])
        if rect is not None:
            rectangles.append(rect)
    rng = np.random.default_rng(seed)
    accepted, excluded = [], []
    root.mkdir(parents=True)
    for dataset, quota in ADDITIONS.items():
        candidates = sorted(read_rows(data, dataset), key=lambda r: r["id"])
        rng.shuffle(candidates)
        if dataset == "tiger":
            candidates = [r for r in candidates if r.get("cohort") == "non-TCGA" and group_key(r) != "114"]
            # Prefer groups not yet represented, then spread additions over groups.
            candidates.sort(key=lambda r: group_key(r) in groups[dataset])
        count, new_groups = 0, Counter()
        while candidates and count < quota:
            if dataset == "tiger":
                candidates.sort(key=lambda r: (group_key(r) in groups[dataset], new_groups[group_key(r)]))
            row = candidates.pop(0)
            identity = (dataset, row["id"])
            if identity in used_ids or row["image_sha256"] in used_hashes:
                continue
            group = group_key(row) if dataset == "tiger" else row["group_id"]
            if dataset in ("bcss", "ocelot") and group in groups[dataset]:
                continue
            calibrated = dataset in ("bcss", "tiger", "ocelot")
            spacing, resolution = mpp(row, data) if calibrated else (None,
                dict(status="distributed pixel spacing unverified", physical_metrics_allowed=False))
            image_path, mask_path = data / row["image"], data / row["mask"]
            if digest(image_path) != row["image_sha256"] or digest(mask_path) != row["mask_sha256"]:
                raise ValueError("Downloaded source changed")
            raw = np.array(Image.open(mask_path))
            if raw.ndim != 2 or raw.shape != (row["height"], row["width"]):
                raise ValueError("Source mask is not a spatially paired categorical image")
            catalog = LABELS[dataset] if calibrated else EXTRA_LABELS[dataset]
            ignore = set(row["ignore_labels"])
            if set(np.unique(raw)) - set(catalog) - ignore:
                raise ValueError("Unreconciled source labels")
            # Raw zero is tumor in WSSS4LUAD and a valid negative target in RINGS.
            codes = sorted(int(g) for g in np.unique(raw) if g not in ignore)
            mapping = {g: i+1 for i, g in enumerate(codes)}
            mapped = np.zeros(raw.shape, np.uint8)
            for g, label in mapping.items():
                mapped[raw == g] = label
            windows = ([r[-4:] for r in candidate_windows(mapped, spacing)]
                       if calibrated else pixel_windows(mapped))
            chosen = None
            for y, x, ch, cw in windows:
                crop = [x, y, cw, ch]
                rect = global_rect(row, crop)
                if rect and any(slide == rect[0] and intersects(box, rect[1]) for slide, box in rectangles):
                    continue
                shape = [round(ch*spacing[1]/.5), round(cw*spacing[0]/.5)] if calibrated else [ch, cw]
                truth = mapped[y:y+ch, x:x+cw]
                work = cv2.resize(truth, tuple(shape[::-1]), interpolation=cv2.INTER_NEAREST)
                coarse = coarse_mask(work)
                dummy = np.zeros((int(np.ceil(shape[0]/14)), int(np.ceil(shape[1]/14)), 1), np.float32)
                try:
                    _, labels, _, _ = coarse_tile_observations(dummy, coarse)
                except ValueError:
                    continue
                if calibrated and len(np.unique(labels)) < 2:
                    continue
                chosen = (crop, shape, truth, coarse, rect, np.unique(labels).tolist())
                break
            if chosen is None:
                excluded.append(dict(dataset=dataset, id=row["id"], group=group,
                    reason="No eligible non-overlapping field/coarse summaries under frozen geometry rules"))
                continue
            crop, shape, truth, coarse, rect, coarse_classes = chosen
            x, y, cw, ch = crop
            # Compact within the selected crop, while retaining all fine classes.
            present = sorted(int(g) for g in np.unique(truth) if g > 0)
            old_to_new = {g: i+1 for i, g in enumerate(present)}
            fine = np.zeros(truth.shape, np.uint8)
            compact = np.zeros(coarse.shape, np.uint8)
            for g, label in old_to_new.items():
                fine[truth == g] = label
                compact[coarse == g] = label
            raw_to_compact = {g: old_to_new[label] for g, label in mapping.items() if label in old_to_new}
            count += 1
            number = sum(s["dataset"] == dataset for s in old) + count
            case = f"{dataset}_{number:02d}"
            folder = root / "cases" / case
            folder.mkdir(parents=True)
            rgb = np.array(Image.open(image_path).convert("RGB"))[y:y+ch, x:x+cw]
            working = cv2.resize(rgb, tuple(shape[::-1]), interpolation=cv2.INTER_AREA) if calibrated else rgb
            Image.fromarray(rgb).save(folder / "original.native.png")
            Image.fromarray(working).save(folder / "original.png")
            tifffile.imwrite(folder / "reference.native.tif", fine, compression="deflate")
            tifffile.imwrite(folder / "coarse.tif", compact, compression="deflate")
            colours = (palette(dataset, sorted(raw_to_compact)) if calibrated else
                ([RED, GREEN, [55, 110, 230]] if dataset == "wsss4luad" else [GREEN, RED]))
            if not calibrated:
                full_codes = sorted(EXTRA_LABELS[dataset])
                colours = [colours[full_codes.index(g)] for g in sorted(raw_to_compact)]
            sample = dict(case=case, dataset=dataset, group=group, source=row,
                benchmark_cohort="extension-2026-10-18-development", source_region_type="ROI",
                native_crop_xywh=crop, native_shape_yx=[ch, cw], working_shape_yx=shape,
                native_mpp_xy=spacing,
                working_mpp_xy=[cw*spacing[0]/shape[1], ch*spacing[1]/shape[0]] if calibrated else None,
                fov_um_xy=[cw*spacing[0], ch*spacing[1]] if calibrated else None,
                resolution=resolution, pixel_only=not calibrated, physical_metrics_allowed=calibrated,
                raw_to_compact=raw_to_compact,
                class_names={raw_to_compact[g]: catalog[g] for g in sorted(raw_to_compact)}, palette=colours,
                raw_ignore_codes=sorted(ignore), coarse_reference_derived=True, coarse_tile_px=90,
                coarse_tile_um=45 if calibrated else None,
                single_coarse_class=len(np.unique(compact[compact > 0])) == 1,
                original_sha256=digest(folder/"original.png"),
                coarse_sha256=digest(folder/"coarse.tif"),
                reference_native_sha256=digest(folder/"reference.native.tif"))
            save_json(folder / "sample.json", sample)
            accepted.append(sample)
            used_ids.add(identity)
            used_hashes.add(row["image_sha256"])
            groups[dataset].add(group)
            new_groups[group] += 1
            if rect:
                rectangles.append(rect)
            print(json.dumps(dict(prepared=case, coarse_classes=coarse_classes, pixel_only=not calibrated)), flush=True)
    counts = dict(Counter(s["dataset"] for s in accepted))
    record = dict(seed=seed, requested_additions=ADDITIONS, accepted_counts=counts,
        all_requested_prepared=counts == ADDITIONS, historical_case_count=len(old),
        previous_plan_sha256=digest(previous/"PLAN.json"), samples=accepted, exclusions=excluded,
        historical_inputs_modified=False, feature_inference_started=False,
        selection="Seeded source order; class-entropy fields, not algorithm-score selection; no overlapping TIGER source rectangles",
        spacing_policy="Unverified WSSS4LUAD/RINGS remain native pixels; physical spacing and areas are null, never inferred from magnification",
        additional_organs_pending="OCELOT tissue cohort acquired and audited separately",
        code_sha256=digest(__file__))
    save_json(root / "PLAN.json", record)
    if counts != ADDITIONS:
        raise RuntimeError(f"Incomplete requested extension, inspect explicit exclusions: {counts}")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.data_root, args.previous, args.root)
