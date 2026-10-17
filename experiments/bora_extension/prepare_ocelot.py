"""Inventory original OCELOT tissue pairs and select forty development ROIs.

Uses only provider training data, one TCGA case per selected image, and keeps
cell point annotations separate. Does not fit an encoder or a Bora model.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.virchow2_inverse.common import digest
from experiments.virchow2_multi.plan import LABELS, FIXED, GREEN, RED
from . import plan

QUOTAS = {"bladder": 7, "endometrium": 7, "head-and-neck": 6,
          "kidney": 7, "prostate": 6, "stomach": 7}


def tissue_spacing(sample, shape=(1024, 1024)):
    region = sample["tissue"]
    spacing = [region["resized_mpp_x"], region["resized_mpp_y"]]
    expected = [(region["x_end"]-region["x_start"]) * sample["mpp_x"] / shape[1],
                (region["y_end"]-region["y_start"]) * sample["mpp_y"] / shape[0]]
    if (not all(np.isfinite(v) and v > 0 for v in spacing)
            or not np.allclose(spacing, expected, rtol=1e-9, atol=1e-12)):
        raise ValueError("Reported resized spacing and native source geometry disagree")
    return spacing


def inventory(data):
    data = Path(data)
    folder = data / "raw/ocelot/extracted/ocelot2023_v1.0.1"
    metadata_path = folder / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    expected = {"1": "BG (Background)", "2": "CA (Cancer Area)", "255": "UNK (Unknown)"}
    if metadata["label_info"]["tissue"] != expected or metadata["general_info"]["version"] != "1.0.1":
        raise ValueError("Unexpected OCELOT ontology/release")
    rows = []
    for identity, sample in sorted(metadata["sample_pairs"].items()):
        split = sample["subset"]
        image_path = folder / f"images/{split}/tissue/{identity}.jpg"
        mask_path = folder / f"annotations/{split}/tissue/{identity}.png"
        image = Image.open(image_path)
        mask = np.array(Image.open(mask_path))
        if image.size != (1024, 1024) or mask.shape != (1024, 1024):
            raise ValueError("Original OCELOT tissue geometry mismatch")
        if not set(np.unique(mask)) <= {1, 2, 255}:
            raise ValueError("Unexpected tissue mask values")
        region = sample["tissue"]
        spacing = tissue_spacing(sample, mask.shape)
        slide = sample["slide_name"]
        case = "-".join(slide.split("-")[:3])
        rows.append(dict(dataset="ocelot", id=identity, group_id=case,
            group_level="provider_TCGA_case", patient_id=case, slide_id=slide,
            source_subset=split, split=split, organ=sample["organ"],
            source_record="https://zenodo.org/records/8417503", source_version="1.0.1",
            image=str(image_path.relative_to(data)), mask=str(mask_path.relative_to(data)),
            image_sha256=digest(image_path), mask_sha256=digest(mask_path),
            height=1024, width=1024, mpp_x_um=spacing[0], mpp_y_um=spacing[1],
            source_mpp_xy=[sample["mpp_x"], sample["mpp_y"]],
            source_coordinates=region, raw_labels=[int(g) for g in np.unique(mask)],
            ignore_labels=[255], label_mode="semantic", region_type="ROI",
            annotation="provider expert tissue annotation; non-cancer BG is not a stroma label",
            resolution_status="provider per-axis resized tissue-patch metadata, cross-checked against source geometry",
            license="CC BY-NC 4.0", metadata_sha256=digest(metadata_path)))
    if len(rows) != 663:
        raise ValueError("OCELOT 1.0.1 should have 663 sample pairs")
    path = data / "prepared/ocelot.jsonl"
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    if path.exists() and path.read_text() != payload:
        raise ValueError("OCELOT inventory changed")
    path.write_text(payload)
    return rows


def prepare(data, previous, root):
    data, previous, root = map(Path, (data, previous, root))
    rows = inventory(data)
    old = json.loads((previous / "PLAN.json").read_text())["samples"]
    old_groups = {s["source"].get("patient_id", s["source"]["group_id"]) for s in old}
    rng = np.random.default_rng(8819)
    # Shuffle all training candidates within each organ, preserving a round-robin
    # order so a rejected/one-class field can be replaced within the same organ.
    by_organ = defaultdict(list)
    for row in rows:
        if row["split"] == "train" and row["group_id"] not in old_groups:
            by_organ[row["organ"]].append(row)
    for values in by_organ.values():
        rng.shuffle(values)
    # Use the same field builder, separately per organ with the accumulated plan
    # as prior evidence, guaranteeing unique case IDs/patients across the40.
    LABELS["ocelot"] = {1: "non-cancer background (not expert stroma)", 2: "cancer area"}
    FIXED["ocelot"] = {1: GREEN, 2: RED}
    plan.EXTRA_LABELS["ocelot"] = LABELS["ocelot"]
    plan.ADDITIONS = {"ocelot": 40}
    original_read = plan.read_rows
    # Precompute eligibility using the same field/coarse criteria before quota
    # sampling. Each accepted source is independent of any model score.
    selected, used = [], set(old_groups)
    for organ, quota in QUOTAS.items():
        for row in by_organ[organ]:
            if row["group_id"] in used:
                continue
            mask = np.array(Image.open(data / row["mask"]))
            truth = np.where(mask == 255, 0, mask).astype(np.uint8)
            spacing = [row["mpp_x_um"], row["mpp_y_um"]]
            good = False
            for window in plan.candidate_windows(truth, spacing):
                y, x, ch, cw = window[-4:]
                shape = [round(ch*spacing[1]/.5), round(cw*spacing[0]/.5)]
                working = plan.cv2.resize(truth[y:y+ch, x:x+cw], tuple(shape[::-1]), interpolation=plan.cv2.INTER_NEAREST)
                coarse = plan.coarse_mask(working)
                dummy = np.zeros((int(np.ceil(shape[0]/14)), int(np.ceil(shape[1]/14)), 1), np.float32)
                try:
                    _, labels, _, _ = plan.coarse_tile_observations(dummy, coarse)
                except ValueError:
                    continue
                if len(np.unique(labels)) == 2:
                    good = True
                    break
            if good:
                selected.append(row)
                used.add(row["group_id"])
                if sum(r["organ"] == organ for r in selected) == quota:
                    break
        if sum(r["organ"] == organ for r in selected) != quota:
            raise ValueError(f"Not enough eligible independent training cases for {organ}")
    plan.read_rows = lambda ignored, dataset: selected if dataset == "ocelot" else original_read(ignored, dataset)
    result = plan.prepare(data, previous, root)
    result["organ_quotas"] = QUOTAS
    result["organ_counts"] = dict(Counter(s["source"]["organ"] for s in result["samples"]))
    result["provider_partition"] = "train only; val/test untouched"
    result["inventory_sha256"] = digest(data / "prepared/ocelot.jsonl")
    result["additional_organs_pending"] = None
    result["ocelot_preparation_code_sha256"] = digest(__file__)
    plan.save_json(root / "PLAN.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.data_root, args.previous, args.root)
    print(json.dumps({k: result[k] for k in ("accepted_counts", "organ_counts", "provider_partition")}))
