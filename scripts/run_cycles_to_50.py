#!/usr/bin/env python3
"""Resume the numbered 216-pixel contour search and stop after cycle 50.

Every candidate is scored from its final GeoJSON on all four slides. Only a
candidate passing the smoothing-focused internal-border gate replaces the
quantitative incumbent. Narrow-tolerance contour F1 is reported separately.
"""
from __future__ import annotations

import fcntl
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/mnt/sata_ssd/benchmarks/Bora")
PYTHON = sys.executable
STATE = DATA / "bora_cycles_13_to_50.json"

# Search broad displacement/smoothing scales before fine-tuning around the
# strongest source. Every tuple is (Gaussian sigma, symmetric editable radius).
GEOMETRY = [
    (5, 24), (8, 32), (12, 48), (16, 64), (20, 80), (24, 96),
    (28, 112), (32, 128), (36, 144), (44, 176), (52, 208),
    (60, 216), (68, 216), (76, 216), (88, 216), (100, 216),
    (112, 216), (124, 216), (136, 216), (148, 216),
    (10, 80), (14, 112), (18, 144), (22, 176), (26, 208),
    (34, 216), (42, 216), (50, 216), (58, 216), (66, 216),
]
PAIRWISE = [(24, 48), (32, 64), (48, 64), (32, 96),
            (48, 96), (64, 96), (48, 128)]


def specification(number: int) -> dict:
    if number == 13:
        return {"method": "pairwise", "core_distance": 40,
                "search_radius": 64, "source": "visual_12"}
    if 14 <= number <= 43:
        sigma, radius = GEOMETRY[number - 14]
        return {"method": "geometry", "sigma": sigma,
                "band_radius": radius, "source": "incumbent"}
    if 44 <= number <= 50:
        core, radius = PAIRWISE[number - 44]
        return {"method": "pairwise", "core_distance": core,
                "search_radius": radius, "source": "incumbent"}
    raise ValueError(f"Outside planned cycle range: {number}")


def save_state(completed: list[dict], incumbent: int,
               current: int | None) -> None:
    payload = {"completed_through": max((r["cycle"] for r in completed), default=12),
               "current_cycle": current, "quantitative_incumbent": incumbent,
               "finished": current is None and len(completed) == 38,
               "updated_utc": datetime.now(timezone.utc).isoformat(),
               "cycles": completed}
    STATE.write_text(json.dumps(payload, indent=2))


def main() -> None:
    lock = DATA / ".bora_cycles_13_to_50.lock"
    with lock.open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        incumbent = 12
        completed = []
        for number in range(13, 51):
            spec = specification(number)
            output = DATA / f"output-bora-cycle-{number:02d}"
            decision = output / "216" / "cycle_decision.json"
            manifest = output / "216" / "cycle_manifest.json"
            if manifest.is_file() and decision.is_file():
                verdict = json.loads(decision.read_text())
                prior = json.loads(manifest.read_text())
                if int(prior.get("incumbent_cycle", -1)) != incumbent:
                    raise RuntimeError(f"Cycle {number} was scored against "
                                       f"{prior.get('incumbent_cycle')}, expected {incumbent}")
            else:
                source = 12 if spec["source"] == "visual_12" else incumbent
                entry = ROOT / "scripts" / ("run_geometry_cycle.py"
                                             if spec["method"] == "geometry"
                                             else "run_pairwise_experiment.py")
                cmd = [PYTHON, str(entry), "--number", str(number),
                       "--source-cycle", str(source),
                       "--incumbent-cycle", str(incumbent)]
                if spec["method"] == "geometry":
                    cmd += ["--sigma", str(spec["sigma"]),
                            "--band-radius", str(spec["band_radius"])]
                else:
                    cmd += ["--core-distance", str(spec["core_distance"]),
                            "--search-radius", str(spec["search_radius"])]
                save_state(completed, incumbent, number)
                print(f"[cycle search] START {number}: {spec}, source={source}, "
                      f"incumbent={incumbent}", flush=True)
                with (DATA / f"bora_cycle_{number:02d}.log").open("a") as log:
                    log.write("COMMAND " + " ".join(cmd) + "\n")
                    log.flush()
                    subprocess.run(cmd, cwd=ROOT, stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
                verdict = json.loads(decision.read_text())
            accepted = bool(verdict["accepted"])
            completed.append({"cycle": number, "method": spec["method"],
                              "specification": spec, "accepted": accepted,
                              "decision": str(decision),
                              "geojson_count": len(list((output / "216").glob(
                                  "*/bora_refined.geojson")))})
            if completed[-1]["geojson_count"] != 4:
                raise RuntimeError(f"Cycle {number} lacks four GeoJSON outputs")
            if accepted:
                incumbent = number
            save_state(completed, incumbent, number + 1 if number < 50 else None)
            print(f"[cycle search] END {number}: accepted={accepted}, "
                  f"incumbent={incumbent}", flush=True)
        print("[cycle search] COMPLETE at cycle 50", flush=True)


if __name__ == "__main__":
    main()
