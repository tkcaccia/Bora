#!/usr/bin/env python3
"""Finish a numbered Bora sample whose refined pyramid already exists."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from time import perf_counter

from bora.benchmark_pipeline import compare_masks
from bora.geojson_compat import enrich_wsitools_geojson
from bora.multicpu_geojson import convert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiling-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--tile-size", type=int, default=216)
    parser.add_argument("--geojson-workers", type=int, default=2)
    parser.add_argument("--geojson-seconds", type=float,
                        help="measured conversion time if GeoJSON already exists")
    args = parser.parse_args()
    folder = args.output_root / str(args.tile_size) / args.sample
    mask = folder / "bora_refined.pyramidal.ome.tif"
    vector = folder / "bora_refined.geojson"
    if not mask.is_file():
        raise FileNotFoundError(mask)
    tiling = args.tiling_root / args.sample
    source = tiling / f"mask_tile_grid_{args.tile_size}x{args.tile_size}_cpp.tif"
    truth = tiling / "mask_pixel_cpp.tif"
    metadata = tiling / "metrics_cpp.json"
    info = json.loads(metadata.read_text())
    started = perf_counter()
    conversion = None
    if not vector.is_file():
        conversion = convert(mask, vector, page=0, min_area=500.0,
                             simplify=2.0, workers=args.geojson_workers)
    annotations = enrich_wsitools_geojson(vector, metadata, info["width"], info["height"])
    geojson_seconds = (args.geojson_seconds if args.geojson_seconds is not None
                       else perf_counter() - started)
    started = perf_counter()
    baseline = compare_masks(source, truth, block_size=2048)
    bora = compare_masks(mask, truth, block_size=2048)
    scoring_seconds = perf_counter() - started
    folder_birth = int(subprocess.check_output(["stat", "-c", "%W", str(folder)], text=True))
    mask_modified = int(mask.stat().st_mtime)
    estimated_boundary_and_pyramid = max(0, mask_modified - folder_birth)
    report = {
        "mode": "seeded_grabcut_and_balanced_local_color_wand",
        "recovered_after_export_interruption": True,
        "timing_estimated_from_file_timestamps": True,
        "boundary_and_pyramid_seconds_estimate": estimated_boundary_and_pyramid,
        "geojson_seconds": geojson_seconds,
        "refinement_seconds": estimated_boundary_and_pyramid + geojson_seconds,
        "geojson_conversion": conversion,
        "annotation_metadata": annotations,
        "output": str(mask), "geojson": str(vector),
    }
    result = {
        "sample": args.sample, "tile_size": args.tile_size,
        "tile_grid": str(source), "truth_pixel_mask": str(truth),
        "refined_mask": str(mask), "refined_geojson": str(vector),
        "refinement_seconds": report["refinement_seconds"],
        "boundary_seconds": estimated_boundary_and_pyramid,
        "pyramid_seconds": None,
        "geojson_seconds": geojson_seconds,
        "scoring_seconds": scoring_seconds,
        "timing_estimated_from_file_timestamps": True,
        "baseline": baseline, "bora": bora,
        "improvement": {key: bora[key]-baseline[key] for key in
                        ("accuracy", "foreground_dice", "foreground_iou",
                         "macro_dice", "macro_iou", "micro_dice", "micro_iou")},
    }
    (folder / "bora_refined.report.json").write_text(json.dumps(report, indent=2))
    (folder / "comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"sample": args.sample, "macro_iou": bora["macro_iou"],
                      "seconds_estimated": report["refinement_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
