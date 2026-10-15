"""Classify inventoried images by acquired extent, without changing source inventories."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from .download import DEFAULT_ROOT


# Classification follows the acquired release, never dimensions, suffix or dataset title.
# "ROI" includes small patches/tiles; it does not imply a manually selected tissue region.
ROI_RELEASES = {
    "bcss": ("tissue_region", "WSI", "Distributed BCSS image/mask regions; no complete source slides acquired"),
    "tiger": ("tissue_region", "WSI", "TIGER roi-level-annotations archive; source-slide counts are not acquired WSI counts"),
    "glas": ("tissue_patch", "WSI", "Distributed GlaS colorectal image/mask patches, not complete scanned slides"),
    "crag": ("tissue_patch", "WSI", "Distributed canonical CRAG patches, not complete scanned slides"),
    "crag-candidate": ("tissue_patch", "unknown", "Candidate mirror contains cropped gland images; original processing remains unverified"),
    "sicapv2": ("tissue_patch", "WSI", "Original SICAPv2 archive contains 512px patches; its slide tables do not supply full WSI images"),
    "wsss4luad": ("tissue_patch", "WSI", "Pinned WSSS4LUAD mirror distributes patches/regions, including weakly labelled training patches"),
    "rings-glands": ("tissue_patch", "unknown", "Original RINGS release supplies 1500px image fields with gland masks; no complete WSI provenance established"),
    "rings-tumor": ("tissue_patch", "unknown", "The same original RINGS image fields with a separate tumor annotation target"),
    "cocahis-vista": ("processed_patch", "unknown", "Author-released VISTA derivative contains 1024px processed CoCaHis patches"),
    **{f"segpath-{target}": ("tissue_patch", "WSI", "Native SegPath release supplies 984px H&E patches, not complete source slides")
       for target in ("panck", "asma", "cd235a", "cd45rb", "cd3cd20", "erg", "mist1", "mnda")},
}

TYPE_FIELDS = ["image_type", "image_subtype", "image_type_basis", "parent_image_type", "parent_slide_id", "parent_image"]
CSV_FIELDS = ["dataset", "id", "record_scope", *TYPE_FIELDS, "image", "mask", "image_sha256", "mask_sha256",
              "source_shard", "width", "height", "split", "group_id", "annotation_status", "source_status",
              "crop_x", "crop_y", "crop_level"]


def classify_image(row, *, whole_slide=False):
    """Return provenance-based extent metadata for a supported acquired image."""
    dataset = row["dataset"]
    if whole_slide:
        if dataset != "camelyon16-subset" or any(k in row for k in ("crop_level", "crop_x", "crop_y")):
            raise ValueError("Full WSI classification requires an explicit complete-slide record")
        levels = row.get("image_levels", [])
        if not levels or list(levels[0]) != [row["width"], row["height"]]:
            raise ValueError("Full WSI classification requires verified source level-0 geometry")
        return dict(image_type="WSI", image_subtype="full_slide",
                    image_type_basis="Complete original CAMELYON16 TIFF with verified OpenSlide pyramid and level-0 geometry",
                    parent_image_type="not_applicable", parent_slide_id="", parent_image="")
    if dataset == "camelyon16-subset":
        if not all(k in row for k in ("crop_x", "crop_y", "crop_level")):
            raise ValueError("CAMELYON ROI classification requires recorded crop coordinates")
        slide = row.get("slide_id") or row["group_id"]
        return dict(image_type="ROI", image_subtype="native_wsi_window",
                    image_type_basis="Recorded native window extracted from a complete CAMELYON16 WSI; the window is an ROI",
                    parent_image_type="WSI", parent_slide_id=slide,
                    parent_image=f"raw/camelyon16-subset/images/{slide}.tif")
    if dataset not in ROI_RELEASES:
        raise ValueError(f"No verified acquired-image classification for {dataset}")
    subtype, parent_type, basis = ROI_RELEASES[dataset]
    return dict(image_type="ROI", image_subtype=subtype, image_type_basis=basis,
                parent_image_type=parent_type, parent_slide_id=row.get("slide_id", ""), parent_image="")


def whole_slide_records(root):
    path = root / "metadata" / "camelyon16-subset-wsi.json"
    if not path.exists():
        return []
    return [{**item, "dataset": "camelyon16-subset", "id": item["slide"],
             "group_id": item["slide"], "split": "train"} for item in json.loads(path.read_text())]


def classification_records(root):
    """Cover paired inputs, complete WSIs, weak images and quarantined WSSS images."""
    paired_images = set()
    for path in sorted((root / "prepared").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            paired_images.add(row["image"])
            yield {**row, **classify_image(row), "record_scope": "prepared_annotation_pair",
                   "annotation_status": row.get("source_status", "supplied_mask_pair")}
    for row in whole_slide_records(root):
        yield {**row, **classify_image(row, whole_slide=True), "record_scope": "complete_source_wsi",
               "annotation_status": "expert_derived_tumor_mask_and_automatically_detected_normal_tissue"}
    weak = root / "metadata" / "wsss4luad-image-labels.jsonl"
    if weak.exists():
        for line in weak.read_text().splitlines():
            row = {**json.loads(line), "dataset": "wsss4luad"}
            yield {**row, **classify_image(row), "record_scope": "image_level_only",
                   "annotation_status": "image_level_labels_only_no_dense_mask"}
    audit = root / "metadata" / "wsss4luad-mask-audit.json"
    if audit.exists():
        for item in json.loads(audit.read_text()):
            if item["image"] in paired_images:
                continue
            row = {**item, "dataset": "wsss4luad"}
            yield {**row, **classify_image(row), "record_scope": "quarantined_dense_pair",
                   "annotation_status": row["status"]}


def image_reference(row):
    """RINGS targets share one RGB; embedded Parquet images use shard/split/ID."""
    if row.get("image"):
        return ("file", row["image"])
    if row.get("source_shard"):
        return ("embedded", row["source_shard"], row["split"], row["id"])
    raise ValueError("An acquired image needs a file path or embedded shard identity")


def summarize(rows):
    unique, datasets = {}, defaultdict(list)
    for row in rows:
        reference = image_reference(row)
        if reference in unique and unique[reference]["image_type"] != row["image_type"]:
            raise ValueError("Conflicting image classifications for the same source image")
        unique[reference] = row
        datasets[row["dataset"]].append(row)
    return {
        "classification_rule": "Acquired image extent from release provenance; ROI includes patches/tiles; no pixel-size heuristic",
        "scope": "Inventoried image inputs, complete acquired source WSIs, weak WSSS training images embedded in Parquet, and quarantined WSSS images; masks and example copies are not extra images",
        "counting_note": "Counts are source image references, not independent slides/patients or a content-hash deduplication. RINGS annotation targets share RGB images; CAMELYON crops and their parent WSIs are separate inputs from the same slides.",
        "records": len(rows), "unique_image_references": len(unique),
        "records_by_type": dict(sorted(Counter(r["image_type"] for r in rows).items())),
        "unique_image_references_by_type": dict(sorted(Counter(r["image_type"] for r in unique.values()).items())),
        "records_by_scope": dict(sorted(Counter(r["record_scope"] for r in rows).items())),
        "datasets": {name: {"records": len(items), "unique_image_references": len({image_reference(r) for r in items}),
                            "records_by_type": dict(sorted(Counter(r["image_type"] for r in items).items()))}
                     for name, items in sorted(datasets.items())},
    }


def export_classifications(root, examples=None):
    rows = list(classification_records(root))
    summary = summarize(rows)
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    with (metadata / "image-types.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (metadata / "image-types-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if examples:
        import shutil
        examples.mkdir(parents=True, exist_ok=True)
        for name in ("image-types.csv", "image-types-summary.json"):
            shutil.copy2(metadata / name, examples / name)
        slides = [{**row, **classify_image(row, whole_slide=True)} for row in whole_slide_records(root)]
        if slides:
            target = examples / "camelyon16-subset" / "whole-slide-information.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(slides, indent=2) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--examples", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(export_classifications(args.root, args.examples), indent=2))


if __name__ == "__main__":
    main()
