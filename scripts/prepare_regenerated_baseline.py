#!/usr/bin/env python3
"""Rebuild three-size watershed and accepted smoothing baselines from new tiles."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/mnt/sata_ssd/benchmarks/Bora")
BENCHMARK = DATA / "benchmark-regenerated-2026-10-15"
TILING = DATA / "output-tilling-mask-regenerated-2026-10-15"
PYTHON = sys.executable
MIN_FREE_BYTES = 20 * 1024**3
MIN_AVAILABLE_MEMORY_KIB = 10 * 1024**2


def wait_for_space() -> None:
    while True:
        free_disk = shutil.disk_usage(BENCHMARK).free
        memory = dict(line.split(":", 1) for line in
                      Path("/proc/meminfo").read_text().splitlines())
        available_kib = int(memory["MemAvailable"].strip().split()[0])
        if free_disk >= MIN_FREE_BYTES and available_kib >= MIN_AVAILABLE_MEMORY_KIB:
            return
        print("Waiting for 20 GiB disk and 10 GiB available RAM before next WSI stage",
              flush=True)
        time.sleep(300)


def call(cmd: list[str], log: Path) -> None:
    wait_for_space()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "2"
    env["VIPS_CONCURRENCY"] = "2"
    with log.open("a") as stream:
        stream.write("COMMAND " + " ".join(cmd) + "\n")
        stream.flush()
        subprocess.run(cmd, cwd=ROOT, env=env,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    BENCHMARK.mkdir(parents=True, exist_ok=True)
    log = BENCHMARK / "prepare_baseline.log"
    original = BENCHMARK / "baseline"
    gentle = BENCHMARK / "baseline-gentle"
    pre_strong = BENCHMARK / "baseline-pre-strong"
    accepted = BENCHMARK / "baseline-accepted"
    for size in (56, 90, 216):
        call([PYTHON, str(ROOT / "scripts/run_bora_from_tile_grids.py"),
              "--images", str(DATA / "input"), "--tiles", str(TILING),
              "--output", str(original / str(size)), "--tile-size", str(size),
              "--block-size", "1024"], log)
        call([PYTHON, str(ROOT / "scripts/run_cleanup_cycle.py"),
              "--source-root", str(original), "--output", str(gentle),
              "--tiling-root", str(TILING), "--tile-size", str(size),
              "--block-size", "1024", "--sigma", "12",
              "--band-radius", "28", "--max-hole-area", "1500",
              "--max-island-area", "0", "--geojson-workers", "2"], log)
        call([PYTHON, str(ROOT / "scripts/run_cleanup_cycle.py"),
              "--source-root", str(gentle), "--output", str(pre_strong),
              "--tiling-root", str(TILING), "--tile-size", str(size),
              "--block-size", "1024", "--sigma", "0",
              "--band-radius", "0", "--max-hole-area", "0",
              "--max-island-area", "1500", "--surround-fraction", "0.95",
              "--geojson-workers", "2"], log)
        call([PYTHON, str(ROOT / "scripts/run_cleanup_cycle.py"),
              "--source-root", str(pre_strong), "--output", str(accepted),
              "--tiling-root", str(TILING), "--tile-size", str(size),
              "--block-size", "768", "--sigma", "80",
              "--band-radius", "216", "--max-hole-area", "0",
              "--max-island-area", "0", "--geojson-workers", "2"], log)
        for root in (original, accepted):
            eval_root = Path(str(root)+"-vector-eval")
            call([PYTHON, str(ROOT / "scripts/score_bora_geojson.py"),
                  "--bora-root", str(root), "--tiling-root", str(TILING),
                  "--evaluation-root", str(eval_root),
                  "--tile-size", str(size)], log)
            call([PYTHON, str(ROOT / "scripts/score_multisize_borders.py"),
                  "--tiling-root", str(TILING), "--vector-root", str(eval_root),
                  "--tile-size", str(size)], log)
        pre_eval = Path(str(pre_strong)+"-vector-eval")
        call([PYTHON, str(ROOT / "scripts/score_bora_geojson.py"),
              "--bora-root", str(pre_strong), "--tiling-root", str(TILING),
              "--evaluation-root", str(pre_eval), "--tile-size", str(size),
              "--sample", "Visium_HD_11mm_Human_Pancreas_tissue_image",
              "--skip-vector-raster-check"], log)
        print(f"BASELINE SIZE {size} COMPLETE", flush=True)
    (BENCHMARK / "baseline_manifest.json").write_text(json.dumps({
        "tiling_root": str(TILING),
        "ground_truth_snapshot": str(BENCHMARK / "ground_truth"),
        "original": str(original), "gentle": str(gentle),
        "pre_strong": str(pre_strong), "accepted": str(accepted),
        "accepted_method": "native watershed; sigma=12/band=28/small-hole=1500; enclosed-island=1500; sigma=80/band=216",
        "tile_sizes": [56, 90, 216], "samples_per_size": 4,
        "note": "New annotations differ from the preceding cycle-12/50 benchmark; scores are not directly comparable."
    }, indent=2))
    print("REGENERATED BASELINES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
