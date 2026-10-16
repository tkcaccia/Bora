#!/usr/bin/env python3
"""Score a geometry-first plus pairwise 216-pixel cycle on all slides."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/mnt/sata_ssd/benchmarks/Bora")
TILING = DATA / "output-tilling-mask"


def call(*args: object) -> None:
    cmd = [sys.executable, *map(str, args)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", type=int, required=True)
    parser.add_argument("--source-cycle", type=int, required=True)
    parser.add_argument("--incumbent-cycle", type=int, default=7)
    parser.add_argument("--core-distance", type=int, default=40)
    parser.add_argument("--search-radius", type=int, default=64)
    args = parser.parse_args()
    source = DATA / f"output-bora-cycle-{args.source_cycle:02d}"
    incumbent = DATA / f"output-bora-cycle-{args.incumbent_cycle:02d}"
    incumbent_eval = DATA / f"output-bora-cycle-{args.incumbent_cycle:02d}-vector-eval"
    output = DATA / f"output-bora-cycle-{args.number:02d}"
    evaluation = DATA / f"output-bora-cycle-{args.number:02d}-vector-eval"
    decision = output / "216" / "cycle_decision.json"
    call(ROOT / "scripts/run_pairwise_cycle.py", "--source-root", source,
         "--input-root", DATA / "input", "--output", output,
         "--tiling-root", TILING, "--block-size", 1024,
         "--core-distance", args.core_distance,
         "--search-radius", args.search_radius)
    call(ROOT / "scripts/score_bora_geojson.py", "--bora-root", output,
         "--tiling-root", TILING, "--evaluation-root", evaluation,
         "--tile-size", 216)
    call(ROOT / "scripts/score_216_borders.py", "--tiling-root", TILING,
         "--vector-root", evaluation, "--radius", 64,
         "--block-size", 2048, "--internal-only")
    call(ROOT / "scripts/evaluate_216_cycle.py", "--candidate-root", output,
         "--candidate-vector-root", evaluation,
         "--incumbent-root", incumbent,
         "--incumbent-vector-root", incumbent_eval,
         "--border-radius", 64, "--internal-only", "--smoothing-gate",
         "--output", decision)
    verdict = json.loads(decision.read_text())
    manifest = {"cycle": args.number, "method": "geometry_then_pairwise_grabcut",
                "source_cycle": args.source_cycle,
                "incumbent_cycle": args.incumbent_cycle,
                "core_distance": args.core_distance,
                "search_radius": args.search_radius,
                "accepted": verdict["accepted"], "decision": str(decision)}
    (output / "216" / "cycle_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("COMPLETE", json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
