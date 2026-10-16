#!/usr/bin/env python3
"""Build and score a numbered geometry-first 216-pixel contour cycle."""
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
    parser.add_argument("--source-cycle", type=int, default=7)
    parser.add_argument("--incumbent-cycle", type=int,
                        help="accepted cycle to beat; defaults to source cycle")
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--band-radius", type=int, required=True)
    args = parser.parse_args()
    source = DATA / f"output-bora-cycle-{args.source_cycle:02d}"
    incumbent = args.incumbent_cycle or args.source_cycle
    incumbent_root = DATA / f"output-bora-cycle-{incumbent:02d}"
    incumbent_eval = DATA / f"output-bora-cycle-{incumbent:02d}-vector-eval"
    output = DATA / f"output-bora-cycle-{args.number:02d}"
    evaluation = DATA / f"output-bora-cycle-{args.number:02d}-vector-eval"
    decision = output / "216" / "cycle_decision.json"
    # Large Gaussian halos make 2048-pixel blocks use excessive RAM. The
    # overlap is still complete at smaller block sizes, so the contour rule
    # is unchanged while peak memory stays bounded.
    halo = args.band_radius + int(5*args.sigma + 0.999)
    block_size = 1024 if halo <= 750 else 768
    # Keep the long 216-pixel search at one slide process. Two processes
    # exhausted RAM under concurrent desktop workloads during cycle 34.
    slide_workers = 1
    call(ROOT / "scripts/run_cleanup_cycle.py", "--source-root", source,
         "--output", output, "--tiling-root", TILING, "--tile-size", 216,
         "--block-size", block_size, "--sigma", args.sigma,
         "--band-radius", args.band_radius, "--max-hole-area", 0,
         "--max-island-area", 0, "--geojson-workers", 1,
         "--slide-workers", slide_workers)
    call(ROOT / "scripts/score_bora_geojson.py", "--bora-root", output,
         "--tiling-root", TILING, "--evaluation-root", evaluation,
         "--tile-size", 216)
    call(ROOT / "scripts/score_216_borders.py", "--tiling-root", TILING,
         "--vector-root", evaluation, "--radius", 64, "--block-size", 2048,
         "--internal-only")
    call(ROOT / "scripts/evaluate_216_cycle.py", "--candidate-root", output,
         "--candidate-vector-root", evaluation,
         "--incumbent-root", incumbent_root,
         "--incumbent-vector-root", incumbent_eval,
         "--border-radius", 64, "--internal-only", "--smoothing-gate",
         "--output", decision)
    result = json.loads(decision.read_text())
    manifest = {"cycle": args.number, "method": "symmetric_multilabel_contour_competition",
                "source_cycle": args.source_cycle, "incumbent_cycle": incumbent,
                "sigma": args.sigma,
                "band_radius": args.band_radius, "block_size": block_size,
                "slide_workers": slide_workers,
                "accepted": result["accepted"],
                "decision": str(decision)}
    (output / "216" / "cycle_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("COMPLETE", json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
