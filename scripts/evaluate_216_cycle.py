#!/usr/bin/env python3
"""Accept a 216-pixel candidate only if every tissue clears accuracy gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ("macro_iou", "accuracy", "foreground_iou")


def _results(root: Path, vector_root: Path | None,
             border_radius: int | None) -> dict[str, dict]:
    folders = sorted((root / "216").glob("*/comparison.json"))
    if len(folders) != 4:
        raise ValueError(f"Expected four completed 216-pixel runs in {root}; found {len(folders)}")
    found = {}
    for path in folders:
        result = json.loads(path.read_text())
        sample = result["sample"]
        metrics = result["bora"]
        if vector_root is not None:
            vector = vector_root / "216" / sample / "vector_comparison.json"
            if not vector.is_file():
                raise ValueError(f"Missing final GeoJSON score: {vector}")
            metrics = json.loads(vector.read_text())["vector_vs_ground_truth"]
        borders = None
        if border_radius is not None:
            if vector_root is None:
                raise ValueError("Boundary scoring requires a vector evaluation root")
            border_file = (vector_root / "216" / sample /
                           f"border_comparison_r{border_radius}.json")
            if not border_file.is_file():
                raise ValueError(f"Missing boundary score: {border_file}")
            borders = json.loads(border_file.read_text())
        found[sample] = {"metrics": metrics, "baseline": result["baseline"],
                         "borders": borders,
                         "seconds": result["refinement_seconds"],
                         "report": str(path)}
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--candidate-vector-root", type=Path)
    parser.add_argument("--incumbent-root", type=Path)
    parser.add_argument("--incumbent-vector-root", type=Path)
    parser.add_argument("--border-radius", type=int,
                        help="gate only fixed internal/external boundary bands")
    parser.add_argument("--internal-only", action="store_true",
                        help="gate only borders between foreground tissue classes")
    parser.add_argument("--contour-quality-gate", action="store_true",
                        help="also require 8px F1 and false tile imprint to improve")
    parser.add_argument("--smoothing-gate", action="store_true",
                        help="require border overlap/accuracy and absolute false tile-edge count; report narrow F1 separately")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.contour_quality_gate and args.smoothing_gate:
        parser.error("Choose one contour acceptance gate")
    candidate = _results(args.candidate_root, args.candidate_vector_root,
                         args.border_radius)
    incumbent = (_results(args.incumbent_root, args.incumbent_vector_root,
                          args.border_radius)
                 if args.incumbent_root else None)
    decisions = []
    for sample, item in candidate.items():
        if args.border_radius is not None:
            borders = item["borders"]
            reference_borders = (incumbent[sample]["borders"] if incumbent
                                 else borders)
            changes = {}
            diagnostics = {}
            for region in (("internal",) if args.internal_only else ("internal", "external")):
                for metric in ("macro_iou", "accuracy"):
                    key = f"{region}_{metric}"
                    current_value = borders[region]["bora"][metric]
                    reference_value = reference_borders[region][
                        "bora" if incumbent else "tile"][metric]
                    if current_value is not None and reference_value is not None:
                        changes[key] = {
                            "reference": reference_value,
                            "candidate": current_value,
                            "change": current_value-reference_value,
                        }
            if args.contour_quality_gate:
                for metric, sign in (("boundary_f1_8px", 1),
                                     ("false_tile_imprint_fraction", -1)):
                    current_value = borders["internal"]["bora"].get(metric)
                    reference_value = reference_borders["internal"][
                        "bora" if incumbent else "tile"].get(metric)
                    if current_value is not None and reference_value is not None:
                        changes[f"internal_{metric}"] = {
                            "reference": reference_value,
                            "candidate": current_value,
                            "change": sign * (current_value-reference_value),
                            "direction": "higher" if sign == 1 else "lower",
                        }
            if args.smoothing_gate:
                current = borders["internal"]["bora"]
                reference = reference_borders["internal"]["bora" if incumbent else "tile"]
                name = "false_tile_imprint_edge_pixels"
                if current.get(name) is not None and reference.get(name) is not None:
                    changes[f"internal_{name}"] = {
                        "reference": reference[name], "candidate": current[name],
                        "change": reference[name] - current[name],
                        "direction": "lower",
                    }
                name = "boundary_f1_8px"
                if current.get(name) is not None and reference.get(name) is not None:
                    diagnostics[f"internal_{name}"] = {
                        "reference": reference[name], "candidate": current[name],
                        "change": current[name] - reference[name],
                    }
            all_truth_classes_present = all(
                entry["tp"] > 0 for entry in item["metrics"]["per_class"]
                if entry["support"] > 0)
            not_applicable = args.internal_only and not changes
            accepted = (not_applicable or (bool(changes) and all(
                entry["change"] >= -1e-9 for entry in changes.values())))
            accepted &= all_truth_classes_present
            decisions.append({
                "sample": sample, "accepted": bool(accepted),
                "not_applicable_no_internal_border": bool(not_applicable),
                "improved": any(entry["change"] > 1e-9
                                for entry in changes.values()),
                "all_truth_classes_present": all_truth_classes_present,
                "boundary_metrics": changes,
                "diagnostic_boundary_metrics": diagnostics,
                "refinement_seconds": item["seconds"],
                "comparison": item["report"],
            })
            continue
        reference = (incumbent[sample]["metrics"] if incumbent
                     else item["baseline"])
        current = item["metrics"]
        deltas = {name: current[name] - reference[name] for name in METRICS}
        all_truth_classes_present = all(
            entry["tp"] > 0 for entry in current["per_class"]
            if entry["support"] > 0)
        accepted = all(value >= -1e-9 for value in deltas.values())
        accepted &= all_truth_classes_present
        decisions.append({
            "sample": sample, "accepted": bool(accepted),
            "improved": any(value > 1e-9 for value in deltas.values()),
            "all_truth_classes_present": all_truth_classes_present,
            "reference": {name: reference[name] for name in METRICS},
            "candidate": {name: current[name] for name in METRICS},
            "change": deltas,
            "refinement_seconds": item["seconds"],
            "comparison": item["report"],
        })
    result = {
        "tile_size": 216,
        "comparison_basis": ("final_geojson_boundary_bands" if args.border_radius
                             else "final_geojson" if args.candidate_vector_root
                             else "refined_raster"),
        "boundary_radius": args.border_radius,
        "internal_only": bool(args.internal_only),
        "quality_gate": ("smoothing" if args.smoothing_gate else
                         "strict_contour" if args.contour_quality_gate else "overlap"),
        "candidate_root": str(args.candidate_root),
        "incumbent_root": str(args.incumbent_root) if incumbent else None,
        "accepted": (all(row["accepted"] for row in decisions) and
                     any(row["improved"] for row in decisions)),
        "per_tissue": decisions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
