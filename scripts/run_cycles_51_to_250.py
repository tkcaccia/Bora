#!/usr/bin/env python3
"""Resumable 200-cycle, 12-slide full-resolution contour search.

Every cycle exports all 12 final GeoJSONs and scores them against the frozen,
regenerated ground truth. Candidates are independent alternatives to the
accepted predecessor, starting from the same native watershed source.
"""
from __future__ import annotations

import argparse
import fcntl
import itertools
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/mnt/sata_ssd/benchmarks/Bora")
BENCHMARK = DATA / "benchmark-regenerated-2026-10-15"
TILING = DATA / "output-tilling-mask-regenerated-2026-10-15"
INPUT = DATA / "input"
SOURCE = BENCHMARK / "baseline-pre-strong"
BASELINE = BENCHMARK / "baseline-accepted"
STATE = BENCHMARK / "cycles_51_to_250.json"
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


def specifications() -> list[dict]:
    first = (64, 112, 216, 0.20, 16)
    combinations = list(itertools.product(
        (24, 32, 48, 64, 80, 96, 128),
        (32, 48, 64, 80, 112, 144),
        (64, 96, 128, 176, 216),
        (0.0, 0.05, 0.10, 0.20, 0.40),
        (0, 8, 16, 32),
    ))
    combinations.remove(first)
    random.Random(20261015).shuffle(combinations)
    values = [first, *combinations[:199]]
    return [{"mesh_spacing": m, "support_radius": s,
             "band_radius": b, "color_weight": c,
             "post_smooth_radius": p} for m, s, b, c, p in values]


def call(cmd: list[str], log: Path) -> None:
    wait_for_space()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "2"
    with log.open("a") as stream:
        stream.write("COMMAND " + " ".join(cmd) + "\n")
        stream.flush()
        subprocess.run(cmd, cwd=ROOT, env=environment,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def save_state(records: list[dict], incumbent: int | str,
               current: int | None) -> None:
    payload = {"completed_through": max((r["cycle"] for r in records), default=50),
               "current_cycle": current, "incumbent": incumbent,
               "finished": current is None and len(records) == 200,
               "updated_utc": datetime.now(timezone.utc).isoformat(),
               "benchmark_root": str(BENCHMARK),
               "tiling_root": str(TILING), "cycles": records}
    STATE.write_text(json.dumps(payload, indent=2))


def prerequisites() -> None:
    previous = json.loads((DATA / "bora_cycles_13_to_50.json").read_text())
    if not previous.get("finished"):
        raise RuntimeError("The preceding search has not completed cycle 50")
    if not (ROOT / "cpp/libbora_tri_mesh.so").is_file():
        raise FileNotFoundError("Build cpp/libbora_tri_mesh.so with CMake first")
    if not (BENCHMARK / "ground_truth_manifest.json").is_file():
        raise FileNotFoundError("Frozen regenerated ground truth is missing")
    if not (BENCHMARK / "regenerated_mask_manifest.json").is_file():
        raise FileNotFoundError("Regenerated tile-mask provenance is missing")
    for base in (SOURCE, BASELINE):
        for size in (56, 90, 216):
            if len(list((base / str(size)).glob("*/bora_refined.geojson"))) != 4:
                raise RuntimeError(f"{base}/{size}: expected four baseline GeoJSONs")
    if len(list((Path(str(BASELINE)+"-vector-eval")).glob(
            "*/**/border_comparison_r64.json"))) != 12:
        raise RuntimeError("Accepted baseline lacks 12 border scores")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--kernel-threads", type=int, default=2)
    parser.add_argument("--geojson-workers", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=1536)
    args = parser.parse_args()
    prerequisites()
    BENCHMARK.mkdir(parents=True, exist_ok=True)
    specs = specifications()
    with (BENCHMARK / ".cycles_51_to_250.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records: list[dict] = []
        incumbent: int | str = "baseline"
        incumbent_root = BASELINE
        for cycle in range(51, 251):
            spec = specs[cycle-51]
            output = BENCHMARK / f"output-bora-cycle-{cycle:03d}"
            evaluation = BENCHMARK / f"output-bora-cycle-{cycle:03d}-vector-eval"
            decision = output / "cycle_decision.json"
            manifest = output / "cycle_manifest.json"
            log = BENCHMARK / f"cycle_{cycle:03d}.log"
            if decision.is_file() and manifest.is_file():
                prior = json.loads(manifest.read_text())
                if prior["incumbent"] != incumbent:
                    raise RuntimeError(f"Cycle {cycle} was compared to {prior['incumbent']}, expected {incumbent}")
                verdict = json.loads(decision.read_text())
            else:
                save_state(records, incumbent, cycle)
                print(f"START {cycle}: {spec}; incumbent={incumbent}", flush=True)
                cmd = [PYTHON, str(ROOT / "scripts/run_tri_mesh_cycle.py"),
                       "--source-root", str(SOURCE), "--output", str(output),
                       "--tiling-root", str(TILING), "--images", str(INPUT),
                       "--workers", str(args.workers),
                       "--kernel-threads", str(args.kernel_threads),
                       "--geojson-workers", str(args.geojson_workers),
                       "--adaptive-by-tile-size",
                       "--block-size", str(args.block_size)]
                for name, value in spec.items():
                    cmd += ["--"+name.replace("_", "-"), str(value)]
                call(cmd, log)
                call([PYTHON, str(ROOT / "scripts/score_bora_geojson.py"),
                      "--bora-root", str(output), "--tiling-root", str(TILING),
                      "--evaluation-root", str(evaluation),
                      "--skip-vector-raster-check",
                      "--reuse-unchanged-vector-root",
                      str(Path(str(SOURCE)+"-vector-eval")),
                      "--tile-size", "56", "--tile-size", "90",
                      "--tile-size", "216"], log)
                call([PYTHON, str(ROOT / "scripts/score_multisize_borders.py"),
                      "--tiling-root", str(TILING), "--vector-root", str(evaluation)], log)
                incumbent_eval = Path(str(incumbent_root)+"-vector-eval")
                call([PYTHON, str(ROOT / "scripts/evaluate_multisize_cycle.py"),
                      "--candidate", str(output),
                      "--candidate-eval", str(evaluation),
                      "--incumbent", str(incumbent_root),
                      "--incumbent-eval", str(incumbent_eval),
                      "--output", str(decision)], log)
                verdict = json.loads(decision.read_text())
                manifest.write_text(json.dumps({"cycle": cycle,
                    "specification": spec, "incumbent": incumbent,
                    "accepted": verdict["accepted"],
                    "evaluation": str(evaluation)}, indent=2))
            vectors = list(output.glob("*/**/bora_refined.geojson"))
            if len(vectors) != 12:
                raise RuntimeError(f"Cycle {cycle} has {len(vectors)} GeoJSONs, expected 12")
            records.append({"cycle": cycle, "specification": spec,
                            "accepted": verdict["accepted"],
                            "decision": str(decision), "geojson_count": len(vectors)})
            if verdict["accepted"]:
                incumbent = cycle
                incumbent_root = output
            save_state(records, incumbent, cycle+1 if cycle < 250 else None)
            print(f"END {cycle}: accepted={verdict['accepted']}, incumbent={incumbent}",
                  flush=True)


if __name__ == "__main__":
    main()
