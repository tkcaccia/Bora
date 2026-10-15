"""Inventory additional source releases without inference, resizing or relabelling."""
from __future__ import annotations

import argparse
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image

from .download import DEFAULT_ROOT, digest
from .prepare import entry, write_once


def validate_wsss_pair(image_size, mask):
    """Reject mispaired masks; class zero is tumor, never an ignore value."""
    if mask.ndim != 2 or (mask.shape[1], mask.shape[0]) != image_size:
        return "image_mask_shape_mismatch"
    if not set(map(int, np.unique(mask))).issubset({0, 1, 2, 3}):
        return "unexpected_mask_values"
    return None


def wsss4luad(root):
    import pyarrow.parquet as pq
    raw = root / "raw" / "wsss4luad"
    rows, audit, weak = [], [], []
    for shard in sorted(raw.glob("*.parquet")):
        split = shard.stem
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=1):
            item = batch.to_pylist()[0]
            sample = item["image_id"]
            if not re.fullmatch(r"[A-Za-z0-9_., \[\]-]+", sample):
                raise ValueError("Unsafe source ID")
            image_bytes = item["image"]["bytes"]
            with Image.open(BytesIO(image_bytes)) as im:
                size = im.size
            if size != (item["width"], item["height"]):
                raise ValueError("Mirror dimensions disagree with embedded RGB")
            report = {"id": sample, "split": split, "width": size[0], "height": size[1]}
            if split == "train":
                weak.append({**report, "labels": {k: item[k] for k in
                             ("label_tumor", "label_stroma", "label_normal")},
                             "annotation_level": "image_only", "source_shard": str(shard.relative_to(root))})
                continue  # No dense masks; do not invent pseudo ground truth.
            base = root / "prepared" / "wsss4luad" / "released" / split
            image = base / "images" / (sample + ".png")
            write_once(image, image_bytes)
            mask = base / "masks" / (sample + ".png")
            supplied = item.get("mask")
            if not supplied or not supplied.get("bytes"):
                report["status"] = "missing_dense_mask"
            else:
                write_once(mask, supplied["bytes"])
                with Image.open(mask) as mk:
                    values = np.asarray(mk)
                report["mask_shape"] = list(values.shape)
                report["status"] = validate_wsss_pair(size, values) or "accepted"
            background = item.get("background_mask")
            if background and background.get("bytes"):
                bg = base / "background_masks" / (sample + ".png")
                write_once(bg, background["bytes"])
                report["background_mask"] = str(bg.relative_to(root))
                report["background_mask_sha256"] = digest(bg)
                with Image.open(bg) as bm:
                    bg_values = np.asarray(bm)
                report["background_matches_image_shape"] = (bg_values.shape == (size[1], size[0]))
                if report["status"] == "accepted" and not np.array_equal(values == 3, bg_values != 0):
                    report["status"] = "background_mask_disagreement"
            report.update(image=str(image.relative_to(root)), image_sha256=digest(image))
            if mask.exists():
                report.update(mask=str(mask.relative_to(root)), mask_sha256=digest(mask))
            audit.append(report)
            if report["status"] != "accepted":
                continue
            row = entry(image, mask, root, "wsss4luad", split + "--" + sample,
                        "unknown-slide--" + split + "--" + sample, split, "semantic", [3], None)
            row.update(group_level="roi_only_slide_patient_mapping_unavailable",
                       source_status="geometry_checked_mirror", calibration_status="10x_reported_mpp_unverified")
            rows.append(row)
    if len(weak) != 10091 or Counter(x["split"] for x in audit) != {"test": 80, "validation": 40}:
        raise ValueError("Unexpected WSSS4LUAD release count")
    metadata = root / "metadata"
    write_once(metadata / "wsss4luad-mask-audit.json", (json.dumps(audit, indent=2) + "\n").encode())
    write_once(metadata / "wsss4luad-image-labels.jsonl",
               ("\n".join(json.dumps(r, sort_keys=True) for r in weak) + "\n").encode())
    return rows, {"weak_training_patches": len(weak), "dense_release_audit": dict(Counter(x["status"] for x in audit)),
                  "test_use_status": "Rejected: missing masks or image/mask shape mismatch; never repair by resizing"}


def cocahis_vista(root):
    raw = root / "raw" / "cocahis-vista" / "extracted" / "CoCaHis"
    rows = []
    images = sorted((raw / "images" / "train").glob("*.png"))
    masks = sorted((raw / "masks" / "train").glob("*.png"))
    if {p.name for p in images} != {p.name for p in masks} or len(images) != 302:
        raise ValueError("Expected all 302 author-released processed CoCaHis pairs")
    for image in images:
        mask = raw / "masks" / "train" / image.name
        row = entry(image, mask, root, "cocahis-vista", image.stem,
                    "source-image--" + image.stem.split("-")[0], "train", "binary_target_unverified", [], None)
        if not set(row["raw_labels"]).issubset({0, 1}):
            raise ValueError("Unexpected processed CoCaHis mask values")
        row.update(group_level="source_image_prefix_patient_mapping_unavailable",
                   source_status="processed_train_subset_semantic_mapping_unreconciled",
                   preprocessing="Provider supplied 1024px patches; exact tiling/resampling and annotator aggregation unavailable")
        rows.append(row)
    return rows, {"use_status": "Downloaded processed training subset; verify foreground meaning, original license and calibration before manuscript evaluation",
                  "original_release_acquired": False, "independent_heldout_release": False}


def rings(root):
    raw = root / "raw" / "rings" / "extracted"
    rows = []
    for split, expected in (("TRAIN", 1000), ("TEST", 500)):
        # Multi-archive extraction keeps a separate outer folder for each ZIP.
        base = raw / split / split
        images = sorted((base / "IMAGES").glob("*.png"))
        if len(images) != expected:
            raise ValueError(f"Expected {expected} RINGS {split} RGBs, got {len(images)}")
        for folder in ("MANUAL GLANDS", "MANUAL TUMOR"):
            if {p.name for p in images} != {p.name for p in (base / folder).glob("*.png")}:
                raise ValueError("RINGS masks must match all image filenames")
        for image in images:
            for dataset, folder in (("rings-glands", "MANUAL GLANDS"), ("rings-tumor", "MANUAL TUMOR")):
                row = entry(image, base / folder / image.name, root, dataset, split.lower() + "--" + image.stem,
                            split.lower() + "--source-prefix--" + "_".join(image.stem.split("_")[:2]),
                            split.lower(), "binary_semantic", [], None)
                if not set(row["raw_labels"]).issubset({0, 1}):
                    raise ValueError("Unexpected RINGS binary mask values")
                with Image.open(base / folder / image.name) as mask_image:
                    positive_pixels = int(np.count_nonzero(np.asarray(mask_image)))
                row.update(source_dataset="rings", group_level="filename_prefix_within_partition_patient_mapping_unverified",
                           source_status="original_publisher_release", target="glands" if dataset.endswith("glands") else "tumor",
                           positive_pixels=positive_pixels)
                rows.append(row)
    return rows, {"unique_images": 1500, "paired_annotation_targets": 3000,
                  "use_status": "Two overlapping binary targets retained separately; zero in tumor mask is not a stroma label"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", choices=("wsss4luad", "rings", "cocahis-vista"))
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    a = p.parse_args(argv)
    root = a.root.resolve()
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    rows, details = globals()[a.dataset.replace("-", "_")](root)
    datasets = sorted({r["dataset"] for r in rows})
    inventories = {}
    for dataset in datasets:
        subset = sorted((r for r in rows if r["dataset"] == dataset), key=lambda r: r["id"])
        if len({r["id"] for r in subset}) != len(subset):
            raise ValueError("Duplicate sample ID")
        path = root / "prepared" / (dataset + ".jsonl")
        write_once(path, ("\n".join(json.dumps(r, sort_keys=True) for r in subset) + "\n").encode())
        inventories[dataset] = {"pairs": len(subset), "sha256": digest(path),
                                "splits": dict(Counter(r["split"] for r in subset)),
                                "size_min": [min(r[k] for r in subset) for k in ("width", "height")],
                                "size_max": [max(r[k] for r in subset) for k in ("width", "height")]}
    report = {"dataset": a.dataset, "inventories": inventories, **details,
              "split_policy": "Preserve supplied partitions; no new split, inference or coarse labels generated"}
    (root / "metadata" / (a.dataset + "-prepared.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
