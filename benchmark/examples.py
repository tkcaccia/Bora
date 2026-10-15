"""Copy original example images and matching masks from validated inventories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from .download import DEFAULT_ROOT, digest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--count", type=int, default=3)
    a = p.parse_args(argv)
    if a.count < 1:
        p.error("--count must be positive")
    records = []
    for manifest in sorted((a.root / "prepared").glob("*.jsonl")):
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        chosen, seen = [], set()
        if rows and rows[0]["dataset"] == "tiger":
            strata = set()
            for row in rows:
                stratum = (row["source_subset"], row["cohort"])
                if stratum not in strata:
                    chosen.append(row)
                    strata.add(stratum)
                    seen.add(row["group_id"])
                if len(chosen) == a.count:
                    break
        for row in rows:
            if len(chosen) == a.count:
                break
            if row["group_id"] not in seen:
                chosen.append(row)
                seen.add(row["group_id"])
            if len(chosen) == a.count:
                break
        for index, row in enumerate(chosen, 1):
            record = {"dataset": row["dataset"], "sample_id": row["id"], "group_id": row["group_id"],
                      "raw_labels": row["raw_labels"], "label_mode": row["label_mode"],
                      "ignore_labels": row["ignore_labels"], "mpp_um": row["mpp_um"]}
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
                        raise ValueError(f"Existing example differs: {target}")
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
