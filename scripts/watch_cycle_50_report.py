#!/usr/bin/env python3
"""Build the final cycle PDF when the unattended search reaches cycle 50."""
from __future__ import annotations

import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/mnt/sata_ssd/benchmarks/Bora")
STATE = DATA / "bora_cycles_13_to_50.json"
LOCK = DATA / ".bora_cycles_13_to_50.lock"
PYTHON = Path("/home/chiamaka/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3")
PDF = ROOT / "output/pdf/Bora_216_cycles_to_50_report.pdf"


def main() -> None:
    restarts = 0
    while True:
        try:
            state = json.loads(STATE.read_text())
        except (OSError, json.JSONDecodeError):
            time.sleep(10)
            continue
        if state["finished"]:
            break
        with LOCK.open("r") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)
                if restarts >= 3:
                    raise RuntimeError("Cycle search stopped before cycle 50 after three restarts")
                restarts += 1
                print(f"Restarting cycle search after interruption ({restarts}/3)",
                      flush=True)
                with (DATA / "bora_cycles_13_to_50.log").open("a") as log:
                    subprocess.run([sys.executable,
                                    str(ROOT / "scripts/run_cycles_to_50.py")],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        time.sleep(60)
    subprocess.run([str(PYTHON), str(ROOT / "scripts/build_cycle_50_report.py")],
                   cwd=ROOT, check=True)
    if not PDF.is_file() or PDF.stat().st_size < 1000:
        raise RuntimeError("Final cycle PDF missing or empty")
    subprocess.run(["/usr/bin/pdfinfo", str(PDF)], check=True,
                   stdout=(DATA / "bora_cycle_50_pdfinfo.txt").open("w"))
    print(f"FINAL REPORT {PDF}", flush=True)


if __name__ == "__main__":
    main()
