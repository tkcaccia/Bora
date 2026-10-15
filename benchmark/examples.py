"""Copy original example images and matching masks from validated inventories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from .download import DEFAULT_ROOT, digest


def select_examples(rows, count):
    """Retain stable examples, with tissue-scale TIGER regions rather than cell crops."""
    chosen, seen = [], set()
    if rows and rows[0]["dataset"] == "wsss4luad":
        rows = [r for r in rows if min(r["width"], r["height"]) >= 1000 and {0, 1}.issubset(r["raw_labels"])]
    if rows and rows[0]["dataset"] in ("rings-glands", "rings-tumor"):
        rows = [r for r in rows if .01 < r["positive_pixels"] / (r["width"] * r["height"]) < .95]
    if rows and rows[0]["dataset"] == "tiger":
        rows = [r for r in rows if min(r["width"], r["height"]) >= 1000]
        # Keep one large BCSS-derived ROI, then two independent non-TCGA ROIs.
        for cohort in ("TCGA", "non-TCGA"):
            for row in rows:
                if row["cohort"] == cohort and row["group_id"] not in seen:
                    chosen.append(row)
                    seen.add(row["group_id"])
                    break
    for row in rows:
        if len(chosen) >= count:
            break
        # For additional TIGER examples prefer a visible tumor/stroma interface.
        if row["dataset"] == "tiger" and (row["cohort"] != "non-TCGA" or not {1, 2}.issubset(row["raw_labels"])):
            continue
        if row["group_id"] not in seen:
            chosen.append(row)
            seen.add(row["group_id"])
    return chosen[:count]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--count", type=int, default=3)
    p.add_argument("--replace-changed", action="store_true", help="Replace previously generated example copies if selection changed")
    a = p.parse_args(argv)
    if a.count < 1:
        p.error("--count must be positive")
    records = []
    for manifest in sorted((a.root / "prepared").glob("*.jsonl")):
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        chosen = select_examples(rows, a.count)
        for index, row in enumerate(chosen, 1):
            record = {"dataset": row["dataset"], "sample_id": row["id"], "group_id": row["group_id"],
                      "raw_labels": row["raw_labels"], "label_mode": row["label_mode"],
                      "ignore_labels": row["ignore_labels"], "mpp_um": row["mpp_um"]}
            record.update(width=row["width"], height=row["height"])
            record.update({key: row[key] for key in ("split", "group_level", "preprocessing") if key in row})
            if "grade" in row:
                record["image_level_grade"] = row["grade"]
            if "source_status" in row:
                record["source_status"] = row["source_status"]
            for key in ("image", "mask"):
                source = a.root / row[key]
                target = a.destination / row["dataset"] / f"{index:02}-{key}{source.suffix}"
                if digest(source) != row[key + "_sha256"]:
                    raise ValueError(f"Source checksum mismatch: {source}")
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if digest(target) != row[key + "_sha256"]:
                        if not a.replace_changed:
                            raise ValueError(f"Existing example differs: {target}; use --replace-changed for generated copies")
                        shutil.copy2(source, target)
                else:
                    shutil.copy2(source, target)
                record[key] = str(target.relative_to(a.destination))
                record[key + "_sha256"] = digest(target)
            records.append(record)
    if not records:
        raise ValueError("No validated dataset manifests found")
    (a.destination / "examples.json").write_text(json.dumps(records, indent=2) + "\n")
    text = ["# Dataset examples", "", "Original image bytes and paired raw masks, copied without resizing or running Bora.",
            "Mask grayscale values are class/instance IDs, not display intensities; consult the dataset report.",
            "These examples are a convenience sample, not a representative evaluation cohort.", "",
            "| Dataset | Example | Original sample |", "|---|---|---|"]
    for row in records:
        text.append(f"| {row['dataset']} | [RGB]({row['image']}) · [mask]({row['mask']}) | {row['sample_id']} |")
    (a.destination / "README.md").write_text("\n".join(text) + "\n")
    print(f"Copied {len(records)} original image/mask pairs to {a.destination}")


if __name__ == "__main__":
    main()
