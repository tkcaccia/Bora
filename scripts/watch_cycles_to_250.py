#!/usr/bin/env python3
"""Local recovery process for the regenerated 200-cycle benchmark."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/mnt/sata_ssd/benchmarks/Bora")
BENCHMARK = DATA / "benchmark-regenerated-2026-10-15"
PREVIOUS = DATA / "bora_cycles_13_to_50.json"
STATE = BENCHMARK / "cycles_51_to_250.json"


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def run(script: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    with (BENCHMARK / f"{script}.log").open("a") as log:
        log.write(f"START {script}\n")
        log.flush()
        subprocess.run([sys.executable, str(ROOT / "scripts" / (script+".py"))],
                       cwd=ROOT, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)


def main() -> None:
    BENCHMARK.mkdir(parents=True, exist_ok=True)
    while not read_json(PREVIOUS).get("finished"):
        time.sleep(60)
    incumbent = read_json(PREVIOUS).get("quantitative_incumbent")
    if incumbent != 12:
        raise RuntimeError(f"Cycle 50 accepted incumbent {incumbent}; reapply that "
                           "method to the regenerated masks before starting the new search")
    for script, completion in (
        ("prepare_regenerated_baseline", BENCHMARK / "baseline_manifest.json"),
        ("run_cycles_51_to_250", STATE),
    ):
        if script == "run_cycles_51_to_250" and read_json(STATE).get("finished"):
            continue
        if script == "prepare_regenerated_baseline" and completion.is_file():
            continue
        for attempt in range(1, 4):
            try:
                run(script)
            except subprocess.CalledProcessError as error:
                print(f"{script} failed attempt {attempt}: {error}", flush=True)
                if attempt == 3:
                    raise
                time.sleep(120)
            else:
                break
    if not read_json(STATE).get("finished"):
        raise RuntimeError("The 200-cycle search exited without finishing cycle 250")
    print("COMPLETE THROUGH CYCLE 250", flush=True)


if __name__ == "__main__":
    main()
