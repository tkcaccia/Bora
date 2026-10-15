"""Report exact RGB-byte overlap and unique-image counts for the additions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from .download import DEFAULT_ROOT

ADDITIONS = {"wsss4luad", "cocahis-vista", "rings-glands", "rings-tumor"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    a = p.parse_args(argv)
    root = a.root.resolve()
    rows = [json.loads(line) for path in sorted((root / "prepared").glob("*.jsonl"))
            for line in path.read_text().splitlines() if line.strip()]
    new = [row for row in rows if row["dataset"] in ADDITIONS]
    by_hash = defaultdict(dict)
    for row in new:
        by_hash[row["image_sha256"]][row["image"]] = {
            "image": row["image"], "split": row["split"], "dataset": row["dataset"]}
    conflicts = {key: list(values.values()) for key, values in by_hash.items()
                 if len({item["split"] for item in values.values()}) > 1}
    report = {
        "new_unique_image_paths": len({row["image"] for row in new}),
        "new_unique_rgb_byte_hashes": len(by_hash),
        "new_annotation_pairs": len(new),
        "new_datasets_by_target": dict(Counter(row["dataset"] for row in new)),
        "exact_rgb_hashes_crossing_released_partitions": conflicts,
        "scope": "Byte-identical RGB overlap only; not patient/slide/stain-variant independence",
        "all_annotation_pairs": len(rows),
        "all_unique_image_paths": len({row["image"] for row in rows}),
        "bora_executed": False,
    }
    (root / "metadata" / "additional-content-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
