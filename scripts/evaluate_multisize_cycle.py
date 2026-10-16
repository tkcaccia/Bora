#!/usr/bin/env python3
"""Conservative acceptance rule for the regenerated four-slide, three-size set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(candidate: Path, candidate_eval: Path,
             incumbent: Path, incumbent_eval: Path) -> dict:
    rows = []
    for size in (56, 90, 216):
        for folder in sorted((candidate / str(size)).iterdir()):
            if not folder.is_dir():
                continue
            sample = folder.name
            new = json.loads((candidate_eval / str(size) / sample /
                              "border_comparison_r64.json").read_text())["internal"]
            old = json.loads((incumbent_eval / str(size) / sample /
                              "border_comparison_r64.json").read_text())["internal"]
            new_time = json.loads((folder / "comparison.json").read_text())[
                "refinement_seconds"]
            old_time = json.loads((incumbent / str(size) / sample /
                                   "comparison.json").read_text())[
                "refinement_seconds"]
            metrics = {}
            for name, sign in (("macro_iou", 1), ("accuracy", 1),
                               ("boundary_f1_8px", 1),
                               ("boundary_f1_16px", 1),
                               ("boundary_f1_32px", 1),
                               ("false_tile_imprint_edge_pixels", -1),
                               ("off_band_internal_edge_pixels", -1),
                               ("assd_pixels", -1),
                               ("hausdorff95_pixels", -1)):
                a, b = old["bora"].get(name), new["bora"].get(name)
                metrics[name] = {"incumbent": a, "candidate": b,
                                 "improvement": sign*(b-a)
                                 if a is not None and b is not None else None}
            classes_present = all(
                r["tp"] > 0 for r in json.loads((candidate_eval / str(size) /
                    sample / "vector_comparison.json").read_text())[
                        "vector_vs_ground_truth"]["per_class"]
                if r["support"] > 0)
            # Accuracy is the hard constraint. Other metrics are measured so
            # a contour that merely becomes shorter cannot be called better.
            accuracy_ok = all(metrics[m]["improvement"] is None or
                              metrics[m]["improvement"] >= -1e-9
                              for m in ("macro_iou", "accuracy"))
            false_tile_ok = (metrics["false_tile_imprint_edge_pixels"]["improvement"]
                             is None or
                             metrics["false_tile_imprint_edge_pixels"]["improvement"] >= 0)
            off_band_ok = (metrics["off_band_internal_edge_pixels"]["improvement"]
                           is None or
                           metrics["off_band_internal_edge_pixels"]["improvement"] >= 0)
            rows.append({"sample": sample, "tile_size": size,
                         "classes_present": classes_present,
                         "accuracy_non_decreasing": accuracy_ok,
                         "false_tile_imprint_non_increasing": false_tile_ok,
                         "off_band_internal_edges_non_increasing": off_band_ok,
                         "accepted": classes_present and accuracy_ok and
                         false_tile_ok and off_band_ok,
                         "metrics": metrics,
                         "incumbent_seconds": old_time,
                         "candidate_seconds": new_time,
                         "speedup": old_time/new_time if new_time else None})
    if len(rows) != 12:
        raise ValueError(f"Expected 12 tissue/size evaluations, found {len(rows)}")
    improved = any(any(row["metrics"][name]["improvement"] is not None and
                       row["metrics"][name]["improvement"] > 1e-9
                       for name in ("macro_iou", "accuracy",
                                    "false_tile_imprint_edge_pixels"))
                   for row in rows)
    total_old = sum(r["incumbent_seconds"] for r in rows)
    total_new = sum(r["candidate_seconds"] for r in rows)
    return {"accepted": all(r["accepted"] for r in rows) and
            (improved or total_new < total_old - 1e-6),
            "accuracy_scope": "64-pixel band around internal GT or tile interfaces",
            "quality_rule": "all 12 nondecreasing border macro IoU and accuracy, nonincreasing false tile-edge and off-band internal-edge pixels, all GT classes retained; at least one strict quality or speed improvement",
            "incumbent_total_seconds": total_old,
            "candidate_total_seconds": total_new,
            "total_speedup": total_old/total_new if total_new else None,
            "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-eval", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--incumbent-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.candidate, args.candidate_eval,
                      args.incumbent, args.incumbent_eval)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"accepted": result["accepted"],
                      "speedup": result["total_speedup"]}), flush=True)


if __name__ == "__main__":
    main()
