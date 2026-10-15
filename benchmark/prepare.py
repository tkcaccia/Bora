"""Validate image/mask pairs and build immutable, hashed input manifests."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image

from .download import DEFAULT_ROOT, digest


def write_once(path, data):
    path = Path(path)
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f"Existing output differs: {path}; use a new root or manifest name")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)


def entry(image, mask, root, dataset, sample, group, split, mode, ignore, mpp):
    with Image.open(image) as im, Image.open(mask) as mk:
        rgb, labels = np.asarray(im), np.asarray(mk)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or labels.ndim != 2 or rgb.shape[:2] != labels.shape:
            raise ValueError(f"Bad RGB/categorical mask geometry: {sample}")
        if rgb.dtype != np.uint8 or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"Unsupported image or mask dtype: {sample}")
        values = list(map(int, np.unique(labels)))
        if dataset == "bcss" and any(v < 0 or v > 21 for v in values):
            raise ValueError(f"Unexpected BCSS class: {sample}")
        if dataset == "tiger" and any(v < 0 or v > 7 for v in values):
            raise ValueError(f"Unexpected TIGER class: {sample}")
        width, height = im.size
    return {"id": sample, "dataset": dataset, "group_id": group, "split": split,
            "image": str(image.resolve().relative_to(root)), "mask": str(mask.resolve().relative_to(root)),
            "image_sha256": digest(image), "mask_sha256": digest(mask), "width": width, "height": height,
            "raw_labels": values, "label_mode": mode, "ignore_labels": ignore, "mpp_um": mpp}


def bcss(root):
    import pyarrow.parquet as pq
    raw = root / "raw" / "bcss"
    rows = []
    for shard in sorted((raw / "data").glob("*.parquet")):
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=1):
            item = batch.to_pylist()[0]
            sample = item["image_id"]
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", sample):
                raise ValueError("Unsafe image ID")
            out = root / "prepared" / "bcss" / "pairs"
            image, mask = out / (sample + ".png"), out / (sample + "_mask.png")
            for key, path in [("image", image), ("mask", mask)]:
                payload = item[key]["bytes"]
                if not payload:
                    raise ValueError("Expected embedded image bytes in pinned BCSS mirror")
                write_once(path, payload)
            row = entry(image, mask, root, "bcss", sample, item["patient_id"], "unsplit",
                        "semantic", [0, 7, 15], 0.25)
            row.update(xmin=item["xmin"], ymin=item["ymin"], preprocessing="mirror_color_normalized")
            rows.append(row)
    if len(rows) != 151:
        raise ValueError(f"Expected all 151 BCSS pairs, observed {len(rows)}")
    return rows


def glas(root):
    raw = root / "raw" / "glas" / "extracted"
    grade_files = list(raw.rglob("Grade.csv"))
    grades = {}
    if len(grade_files) == 1:
        with grade_files[0].open(newline="") as handle:
            for row in csv.DictReader(handle, skipinitialspace=True):
                grades[row["name"]] = row
    rows = []
    for mask in sorted(raw.rglob("*_anno.bmp")):
        sample = mask.stem.removesuffix("_anno")
        image = mask.with_name(sample + ".bmp")
        grade = grades.get(sample)
        if grade is None:
            raise ValueError(f"Missing publisher patient mapping for {sample}")
        row = entry(image, mask, root, "glas", sample, "glas-patient-" + grade["patient ID"],
                    sample.split("_")[0], "gland_instances", [], None)
        row["grade"] = grade["grade (GlaS)"]
        rows.append(row)
    if Counter(r["split"] for r in rows) != {"train": 85, "testA": 60, "testB": 20}:
        raise ValueError("GlaS requires 85 train, 60 testA and 20 testB pairs")
    return rows


def crag(root):
    raw = root / "raw" / "crag" / "extracted"
    rows = []
    for mask in sorted(raw.rglob("*.png")):
        if mask.parent.name.lower() not in ("annotation", "annotations"):
            continue
        parent = mask.parent.parent
        image_dir = next((p for p in parent.iterdir() if p.is_dir() and p.name.lower() == "images"), None)
        if image_dir is None:
            raise ValueError("Missing CRAG Images directory")
        split = parent.name.lower()
        if split not in ("train", "test"):
            raise ValueError("Unrecognized CRAG split")
        # No invented patient mapping: group at image level until the provider supplies it.
        row = entry(image_dir / mask.name, mask, root, "crag", split + "-" + mask.stem,
                    "unknown-patient-" + split + "-" + mask.stem, split, "gland_instances", [], None)
        row["group_level"] = "image_only_patient_unavailable"
        rows.append(row)
    if Counter(r["split"] for r in rows) != {"train": 173, "test": 40}:
        raise ValueError("Expected CRAG v2: 173 train and 40 test pairs; reject legacy extra masks")
    return rows


def crag_candidate(root):
    raw = root / "raw" / "crag-candidate" / "extracted"
    rows = []
    for mask in sorted(raw.rglob("*.png")):
        if mask.parent.name != "mask":
            continue
        split = mask.parent.parent.name
        if split not in ("train_sup_16", "train_unsup_137", "val", "test"):
            raise ValueError("Unexpected CRAG candidate partition")
        image = mask.parent.parent / "image" / mask.name
        row = entry(image, mask, root, "crag-candidate", split + "--" + mask.stem,
                    "unknown-patient--" + mask.stem, split, "gland_instances_unverified", [], None)
        row.update(group_level="image_only_patient_unavailable", source_status="candidate_provenance_unverified")
        rows.append(row)
    expected = {"train_sup_16": 16, "train_unsup_137": 137, "val": 20, "test": 40}
    if Counter(r["split"] for r in rows) != expected:
        raise ValueError("Candidate release must contain 16/137/20/40 paired images in its four partitions")
    return rows


def tiger(root):
    raw = root / "raw" / "tiger" / "extracted"
    rows = []
    for mask in sorted(raw.rglob("*.png")):
        if mask.parent.name.lower() != "masks":
            continue
        image_dir = mask.parent.parent / "images"
        image = image_dir / mask.name
        subset = mask.parent.parent.name
        if subset not in ("tissue-bcss", "tissue-cells"):
            raise ValueError("Unrecognized TIGER annotation subset")
        original_id = mask.stem
        slide = original_id.split("_[")[0]
        if slide == original_id:
            raise ValueError("TIGER ROI coordinates missing from filename")
        cohort = "TCGA" if slide.startswith("TCGA-") else "non-TCGA"
        group = "-".join(slide.split("-")[:3]) if cohort == "TCGA" else slide
        row = entry(image, mask, root, "tiger", subset + "--" + original_id, group, "unsplit",
                    "semantic", [0], 0.5)
        row.update(cohort=cohort, slide_id=slide, source_subset=subset,
                   group_level="patient" if cohort == "TCGA" else "slide_patient_mapping_unverified")
        rows.append(row)
    if Counter(r["source_subset"] for r in rows) != {"tissue-bcss": 151, "tissue-cells": 1879}:
        raise ValueError("Pinned TIGER archive requires 151 tissue-bcss and 1879 tissue-cells image/mask pairs")
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", choices=("bcss", "tiger", "glas", "crag", "crag-candidate"))
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    a = p.parse_args(argv)
    root = a.root.resolve()
    rows = globals()[a.dataset.replace("-", "_")](root)
    rows.sort(key=lambda r: r["id"])
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate sample IDs")
    manifest = root / "prepared" / f"{a.dataset}.jsonl"
    write_once(manifest, ("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n").encode())
    groups = {}
    for row in rows:
        groups.setdefault(row["group_id"], set()).add(row["split"])
    report = {"dataset": a.dataset, "samples": len(rows), "groups": len(groups),
              "splits": dict(Counter(r["split"] for r in rows)), "manifest_sha256": digest(manifest),
              "groups_crossing_released_partitions": {k: sorted(v) for k, v in groups.items() if len(v) > 1},
              "split_policy": "Preserve released partitions; do not assign evaluation splits"}
    if a.dataset == "tiger":
        report.update(subsets=dict(Counter(r["source_subset"] for r in rows)),
                      cohorts=dict(Counter(r["cohort"] for r in rows)),
                      source_slides=len({r["slide_id"] for r in rows}))
    if a.dataset == "crag-candidate":
        report["use_status"] = "Candidate mirror; original provenance, license and annotation fidelity unverified"
    path = root / "metadata" / f"{a.dataset}-prepared.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
