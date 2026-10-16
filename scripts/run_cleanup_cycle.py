#!/usr/bin/env python3
"""Build a numbered smoothing cycle from saved refined 216-pixel masks."""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

from bora.geojson_compat import enrich_wsitools_geojson
from bora.multicpu_geojson import convert
from bora.postprocess import postprocess_streaming
from bora.pyramid import pyramidize_mask


def _pyramid_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    import tifffile
    try:
        with tifffile.TiffFile(path) as tif:
            return len(tif.series[0].levels) > 1
    except (OSError, ValueError, IndexError):
        return False


def run_one(source_root: Path, output_root: Path, tiling_root: Path,
            sample: str, size: int, block_size: int, sigma: float,
            band_radius: int, max_hole_area: int, workers: int,
            max_island_area: int = 0, surround_fraction: float = 0.9,
            min_circularity: float = 0.0) -> dict:
    source = source_root / str(size) / sample / "bora_refined.pyramidal.ome.tif"
    if not source.is_file():
        raise FileNotFoundError(source)
    folder = output_root / str(size) / sample
    folder.mkdir(parents=True, exist_ok=True)
    refined = folder / "bora_refined.pyramidal.ome.tif"
    vector = folder / "bora_refined.geojson"
    comparison = folder / "comparison.json"
    if comparison.is_file() and refined.is_file() and vector.is_file():
        return json.loads(comparison.read_text())
    stage_path = folder / "stages.json"
    stages = json.loads(stage_path.read_text()) if stage_path.is_file() else {}
    if not refined.is_file() or "cleanup" not in stages:
        if not _pyramid_complete(refined):
            started = perf_counter()
            cleanup = postprocess_streaming(source, refined, block_size=block_size,
                                            sigma=sigma, band_radius=band_radius,
                                            max_hole_area=max_hole_area,
                                            max_island_area=max_island_area,
                                            surround_fraction=surround_fraction,
                                            min_circularity=min_circularity)
            stages["cleanup"] = {"result": cleanup, "seconds": perf_counter()-started}
            stage_path.write_text(json.dumps(stages, indent=2))
    if not _pyramid_complete(refined):
        started = perf_counter()
        pyramidize_mask(refined, "LZW", 4)
        stages["pyramid_seconds"] = perf_counter()-started
        stage_path.write_text(json.dumps(stages, indent=2))
    if not vector.is_file() or "geojson" not in stages:
        started = perf_counter()
        converted = convert(refined, vector, page=0, min_area=500.0,
                            simplify=2.0, workers=workers)
        stages["geojson"] = {"result": converted, "seconds": perf_counter()-started}
        stage_path.write_text(json.dumps(stages, indent=2))
    metadata_path = tiling_root / sample / "metrics_cpp.json"
    metadata = json.loads(metadata_path.read_text())
    annotated = enrich_wsitools_geojson(vector, metadata_path,
                                        metadata["width"], metadata["height"])
    cleanup = stages.get("cleanup", {}).get("result", {})
    cleanup_seconds = stages.get("cleanup", {}).get("seconds", 0.0)
    pyramid_seconds = stages.get("pyramid_seconds", 0.0)
    converted = stages.get("geojson", {}).get("result", {})
    geojson_seconds = stages.get("geojson", {}).get("seconds", 0.0)
    incremental_seconds = cleanup_seconds + pyramid_seconds + geojson_seconds
    source_comparison_path = source_root / str(size) / sample / "comparison.json"
    source_comparison = json.loads(source_comparison_path.read_text())
    # The initial watershed benchmark stores only total refinement time;
    # later numbered cycles additionally store a boundary-stage time.
    original_boundary_seconds = source_comparison.get(
        "boundary_seconds", source_comparison["refinement_seconds"])
    refinement_seconds = original_boundary_seconds + incremental_seconds
    tile = tiling_root / sample / f"mask_tile_grid_{size}x{size}_cpp.tif"
    truth = tiling_root / sample / "mask_pixel_cpp.tif"
    # Acceptance uses the rasterized final GeoJSON in a fixed internal-border
    # band. Whole-mask scoring here would repeat two billion-pixel scans.
    baseline = source_comparison["baseline"]
    bora = None
    scoring_seconds = 0.0
    report = {"mode": ("small_foreground_island_cleanup" if max_island_area
                       else "balanced_wand_plus_hole_filling_and_contour_smoothing"),
              "source_mask": str(source), "output": str(refined),
              "geojson": str(vector), "cleanup": cleanup,
              "annotation_metadata": annotated,
              "geojson_conversion": converted,
              "cleanup_seconds": cleanup_seconds,
              "pyramid_seconds": pyramid_seconds,
              "geojson_seconds": geojson_seconds,
              "incremental_seconds": incremental_seconds,
              "original_boundary_seconds": original_boundary_seconds,
              "refinement_seconds": refinement_seconds,
              "timing_estimated_from_file_timestamps": source_comparison.get(
                  "timing_estimated_from_file_timestamps", False)}
    result = {"sample": sample, "tile_size": size,
              "tile_grid": str(tile), "truth_pixel_mask": str(truth),
              "refined_mask": str(refined), "refined_geojson": str(vector),
              "source_refined_mask": str(source),
              "refinement_seconds": refinement_seconds,
              "boundary_seconds": original_boundary_seconds + cleanup_seconds,
              "cleanup_seconds": cleanup_seconds,
              "incremental_seconds": incremental_seconds,
              "pyramid_seconds": pyramid_seconds,
              "geojson_seconds": geojson_seconds,
              "scoring_seconds": scoring_seconds,
              "timing_estimated_from_file_timestamps": source_comparison.get(
                  "timing_estimated_from_file_timestamps", False),
              "baseline": baseline, "bora": bora,
              "improvement": None,
              "accuracy_scope": "final GeoJSON internal tissue-border band; see vector evaluation"}
    (folder / "bora_refined.report.json").write_text(json.dumps(report, indent=2))
    comparison.write_text(json.dumps(result, indent=2))
    print(json.dumps({"sample": sample, "macro_iou": None,
                      "refinement_seconds": refinement_seconds}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tiling-root", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=216)
    parser.add_argument("--sample", action="append")
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--sigma", type=float, default=8.0)
    parser.add_argument("--band-radius", type=int, default=20)
    parser.add_argument("--max-hole-area", type=int, default=1500)
    parser.add_argument("--max-island-area", type=int, default=0)
    parser.add_argument("--surround-fraction", type=float, default=0.9)
    parser.add_argument("--min-circularity", type=float, default=0.0)
    parser.add_argument("--geojson-workers", type=int, default=2)
    parser.add_argument("--slide-workers", type=int, default=1,
                        help="run independent tissue slides in separate CPU processes")
    args = parser.parse_args()
    converter_bins = [
        "/home/chiamaka/miniconda3/share/bioformats2raw-0.9.4-0/bin",
        "/home/chiamaka/miniconda3/share/raw2ometiff-0.7.1-0/bin",
    ]
    os.environ["PATH"] = os.pathsep.join(converter_bins + [os.environ.get("PATH", "")])
    samples = args.sample or [item.name for item in sorted((args.source_root / str(args.tile_size)).iterdir())
                              if item.is_dir()]
    def inputs(sample):
        return (args.source_root, args.output, args.tiling_root,
                sample, args.tile_size, args.block_size, args.sigma,
                args.band_radius, args.max_hole_area,
                args.geojson_workers, args.max_island_area,
                args.surround_fraction, args.min_circularity)

    if args.slide_workers > 1 and len(samples) > 1:
        with ProcessPoolExecutor(max_workers=args.slide_workers,
                                 mp_context=get_context("spawn")) as pool:
            futures = [pool.submit(run_one, *inputs(sample)) for sample in samples]
            results = [future.result() for future in futures]
    else:
        results = []
        for sample in samples:
            results.append(run_one(*inputs(sample)))
            gc.collect()
            ctypes.CDLL("libc.so.6").malloc_trim(0)
    (args.output / str(args.tile_size) / "summary.json").write_text(
        json.dumps({"tile_size": args.tile_size, "results": results}, indent=2))


if __name__ == "__main__":
    main()
