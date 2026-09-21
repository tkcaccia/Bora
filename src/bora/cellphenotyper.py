"""Adapters around source snapshots copied verbatim from CellPhenotyper."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def mask_to_geojson(mask, output, *, page=-1, max_page_side=2048,
                    min_area=500.0, smooth_buffer=10.0, smooth_passes=3,
                    simplify=6.0, group_prefix="group_",
                    dissolve_by_value=True, fill_holes=True,
                    preserve_topology=True, polygon_backend="auto"):
    """Run CellPhenotyper's post-refinement converter without modifying it."""
    script = Path(__file__).with_name("cellphenotyper_mask_to_geojson.py")
    command = [
        sys.executable, str(script), "--mask", str(mask), "--page", str(page),
        "--max-page-side", str(max_page_side), "--out", str(output),
        "--min-area", str(min_area), "--smooth-buffer", str(smooth_buffer),
        "--smooth-passes", str(smooth_passes), "--simplify", str(simplify),
        "--group-prefix", str(group_prefix), "--polygon-backend", polygon_backend,
    ]
    if dissolve_by_value:
        command.append("--dissolve-by-value")
    if fill_holes:
        command.append("--fill-holes")
    if preserve_topology:
        command.append("--preserve-topology")
    subprocess.run(command, check=True)
    with Path(output).open(encoding="utf-8") as handle:
        return len(json.load(handle).get("features", []))
