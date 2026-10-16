#!/usr/bin/env python3
"""Expand tiling-mask grids, run Bora, and score against pixel masks."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import rasterio

from bora.benchmark_pipeline import compare_masks, expand_tile_grid


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--tiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=56)
    parser.add_argument("--sample", action="append",
                        help="process only the named tissue sample; repeatable")
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for tile_dir in sorted(path for path in args.tiles.iterdir() if path.is_dir()):
        sample = tile_dir.name
        if args.sample and sample not in args.sample:
            continue
        image = args.images / f"{sample}.btf"
        grid = tile_dir / f"mask_tile_grid_{args.tile_size}x{args.tile_size}_cpp.tif"
        truth = tile_dir / "mask_pixel_cpp.tif"
        missing = [path for path in (image, grid, truth) if not path.is_file()]
        if missing:
            raise SystemExit(f"{sample}: missing {', '.join(map(str, missing))}")
        destination = args.output / sample
        destination.mkdir(parents=True, exist_ok=True)
        coarse = destination / f"bora_input_tile_{args.tile_size}x{args.tile_size}.tif"
        refined = destination / "bora_refined.pyramidal.ome.tif"
        geojson = destination / "bora_refined.geojson"
        report = destination / "bora_refined.report.json"
        log = destination / "bora.log"
        started = perf_counter()
        with rasterio.open(image) as image_source, rasterio.open(grid) as mask_source:
            exact_frame = ((image_source.height, image_source.width) ==
                           (mask_source.height, mask_source.width))
        if exact_frame:
            # Fixed tiling-mask output is already Bora-ready; use the exact
            # artifact directly so provenance remains unambiguous.
            coarse = grid
        elif not (args.resume and coarse.is_file()):
            expand_tile_grid(grid, image, coarse, args.tile_size, args.tile_size)
        expansion_seconds = perf_counter() - started
        refine_started = perf_counter()
        if not (args.resume and refined.is_file() and geojson.is_file()):
            command = [
                sys.executable, "-m", "bora.cli", "refine", str(image), str(coarse),
                "--output", str(refined), "--geojson", str(geojson),
                "--report", str(report), "--backend", "watershed",
                "--stream", "--block-size", str(args.block_size),
                "--watershed-resolution", "native", "--geojson-profile", "accurate",
            ]
            with log.open("w", encoding="utf-8") as stream:
                stream.write("$ " + " ".join(command) + "\n\n")
                stream.flush()
                environment = os.environ.copy()
                converter_bins = [
                    "/home/chiamaka/miniconda3/share/bioformats2raw-0.9.4-0/bin",
                    "/home/chiamaka/miniconda3/share/raw2ometiff-0.7.1-0/bin",
                ]
                environment["PATH"] = os.pathsep.join(
                    converter_bins + [environment.get("PATH", "")])
                subprocess.run(command, check=True, stdout=stream,
                               stderr=subprocess.STDOUT, env=environment)
        refinement_seconds = perf_counter() - refine_started
        score_started = perf_counter()
        baseline = compare_masks(coarse, truth, block_size=args.block_size)
        bora = compare_masks(refined, truth, block_size=args.block_size)
        scoring_seconds = perf_counter() - score_started
        result = {
            "sample": sample, "image": str(image), "tile_grid": str(grid),
            "bora_input": str(coarse), "input_already_full_resolution": exact_frame,
            "truth_pixel_mask": str(truth),
            "refined_mask": str(refined), "refined_geojson": str(geojson),
            "expansion_seconds": expansion_seconds,
            "refinement_seconds": refinement_seconds,
            "scoring_seconds": scoring_seconds,
            "baseline": baseline, "bora": bora,
            "improvement": {key: bora[key] - baseline[key] for key in
                            ("accuracy", "foreground_dice", "foreground_iou",
                             "macro_dice", "macro_iou", "micro_dice", "micro_iou")},
        }
        (destination / "comparison.json").write_text(json.dumps(result, indent=2))
        results.append(result)
        print(json.dumps({"sample": sample, "baseline_macro_iou": baseline["macro_iou"],
                          "bora_macro_iou": bora["macro_iou"],
                          "refinement_seconds": refinement_seconds}), flush=True)
    (args.output / "summary.json").write_text(json.dumps({
        "schema_version": "bora.tile_grid_run.v1", "tile_size": args.tile_size,
        "images": str(args.images), "tiles": str(args.tiles), "samples": results,
    }, indent=2))


if __name__ == "__main__":
    main()
