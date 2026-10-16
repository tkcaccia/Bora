#!/usr/bin/env python3
"""Resume ten serial 216-pixel foreground-island cleanup experiments."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/mnt/sata_ssd/benchmarks/Bora")
TILING = DATA / "output-tilling-mask"
SOURCE = DATA / "baseline-pre-island-cleanup"
PYTHON = sys.executable
EXPERIMENTS = [
    (1, 128, 0.95, 0.0),
    (2, 256, 0.95, 0.0),
    (3, 500, 0.95, 0.0),
    (4, 750, 0.95, 0.0),
    (5, 1000, 0.95, 0.0),
    (6, 1250, 0.95, 0.0),
    (7, 1500, 0.95, 0.0),
    (8, 1000, 0.95, 0.65),
    (9, 1000, 0.95, 0.75),
    (10, 1500, 0.95, 0.75),
]


def call(*arguments: object) -> None:
    command = [PYTHON, *map(str, arguments)]
    print("RUN", " ".join(command), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    incumbent_number = 0
    for number, island_area, surround, circularity in EXPERIMENTS:
        root = DATA / f"output-bora-cycle-{number:02d}"
        evaluation = DATA / f"output-bora-cycle-{number:02d}-vector-eval"
        incumbent = (SOURCE if incumbent_number == 0 else
                     DATA / f"output-bora-cycle-{incumbent_number:02d}")
        incumbent_eval = (DATA / "baseline-pre-island-cleanup-vector-eval"
                          if incumbent_number == 0 else
                          DATA / f"output-bora-cycle-{incumbent_number:02d}-vector-eval")
        decision_path = root / "216" / "cycle_decision.json"
        if not decision_path.is_file():
            call(ROOT / "scripts/run_cleanup_cycle.py", "--source-root", SOURCE,
                 "--output", root, "--tiling-root", TILING, "--tile-size", 216,
                 "--block-size", 2048, "--sigma", 0, "--band-radius", 0,
                 "--max-hole-area", 0, "--max-island-area", island_area,
                 "--surround-fraction", surround,
                 "--min-circularity", circularity, "--geojson-workers", 1)
            call(ROOT / "scripts/score_bora_geojson.py", "--bora-root", root,
                 "--tiling-root", TILING, "--evaluation-root", evaluation,
                 "--tile-size", 216)
            call(ROOT / "scripts/score_216_borders.py", "--tiling-root", TILING,
                 "--vector-root", evaluation, "--radius", 64, "--block-size", 2048,
                 "--internal-only")
            call(ROOT / "scripts/evaluate_216_cycle.py", "--candidate-root", root,
                 "--candidate-vector-root", evaluation, "--incumbent-root", incumbent,
                 "--incumbent-vector-root", incumbent_eval, "--border-radius", 64,
                 "--internal-only", "--output", decision_path)
        decision = json.loads(decision_path.read_text())
        manifest = {"cycle": number, "max_island_area": island_area,
                    "surround_fraction": surround, "min_circularity": circularity,
                    "accepted": decision["accepted"],
                    "incumbent_before": incumbent_number,
                    "internal_border_decision": str(decision_path)}
        (root / "216" / "cycle_manifest.json").write_text(json.dumps(manifest, indent=2))
        if decision["accepted"]:
            incumbent_number = number
        print("DECISION", json.dumps(manifest), flush=True)
    summary = {"baseline": str(SOURCE), "experiments": len(EXPERIMENTS),
               "selected_cycle": incumbent_number,
               "cycles": [json.loads((DATA / f"output-bora-cycle-{n:02d}" / "216" /
                                       "cycle_manifest.json").read_text())
                          for n, *_ in EXPERIMENTS]}
    (DATA / "bora_216_island_ten_cycle_summary.json").write_text(json.dumps(summary, indent=2))
    print("COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
