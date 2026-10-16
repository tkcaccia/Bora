#!/usr/bin/env python3
"""Run fast image-guided Bora boundary refinement on existing tiling masks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from bora.benchmark_pipeline import compare_masks
from bora.boundary import BoundaryConfig, smooth_boundaries_streaming
from bora.multicpu_geojson import convert
from bora.geojson_compat import enrich_wsitools_geojson
from bora.pyramid import pyramidize_mask


def run_one(image: Path, tiling: Path, output: Path, size: int,
            block_size: int, config: BoundaryConfig,
            geojson_workers: int = 2) -> dict:
    sample = image.stem
    tile = tiling / sample / f"mask_tile_grid_{size}x{size}_cpp.tif"
    truth = tiling / sample / "mask_pixel_cpp.tif"
    if not tile.is_file() or not truth.is_file():
        raise FileNotFoundError(f"Missing existing tiling-mask output for {sample}, {size}")
    folder = output / str(size) / sample
    folder.mkdir(parents=True, exist_ok=True)
    refined = folder / "bora_refined.pyramidal.ome.tif"
    vector = folder / "bora_refined.geojson"
    comparison = folder / "comparison.json"
    report_path = folder / "bora_refined.report.json"
    if comparison.is_file() and refined.is_file() and vector.is_file():
        print(f"{sample} {size}: already complete", flush=True)
        return json.loads(comparison.read_text())
    start = perf_counter()
    boundary_report = smooth_boundaries_streaming(
        image, tile, refined, size, block_size, config)
    boundary_seconds = perf_counter() - start
    start = perf_counter()
    pyramidize_mask(refined, "LZW", 8)
    pyramid_seconds = perf_counter() - start
    start = perf_counter()
    geojson_report = convert(refined, vector, page=0, min_area=500.0,
                             simplify=2.0, workers=geojson_workers)
    tiling_metrics = json.loads((tile.parent / "metrics_cpp.json").read_text())
    annotation_report = enrich_wsitools_geojson(
        vector, tile.parent / "metrics_cpp.json",
        tiling_metrics["width"], tiling_metrics["height"])
    geojson_seconds = perf_counter() - start
    refinement_seconds = boundary_seconds + pyramid_seconds + geojson_seconds
    start = perf_counter()
    baseline = compare_masks(tile, truth, block_size=block_size)
    bora = compare_masks(refined, truth, block_size=block_size)
    scoring_seconds = perf_counter() - start
    report = {
        **boundary_report, "backend": "adaptive_boundary_wand",
        "image": str(image), "mask": str(tile), "output": str(refined),
        "geojson": str(vector), "pyramidal": True,
        "boundary_seconds": boundary_seconds,
        "pyramid_seconds": pyramid_seconds,
        "geojson_seconds": geojson_seconds,
        "geojson_conversion": geojson_report,
        "annotation_metadata": annotation_report,
        "refinement_seconds": refinement_seconds,
    }
    result = {
        "sample": sample, "tile_size": size,
        "image": str(image), "tile_grid": str(tile),
        "truth_pixel_mask": str(truth),
        "refined_mask": str(refined), "refined_geojson": str(vector),
        "refinement_seconds": refinement_seconds,
        "boundary_seconds": boundary_seconds,
        "pyramid_seconds": pyramid_seconds,
        "geojson_seconds": geojson_seconds,
        "scoring_seconds": scoring_seconds,
        "baseline": baseline, "bora": bora,
        "improvement": {key: bora[key] - baseline[key] for key in
                        ("accuracy", "foreground_dice", "foreground_iou",
                         "macro_dice", "macro_iou", "micro_dice", "micro_iou")},
    }
    report_path.write_text(json.dumps(report, indent=2))
    comparison.write_text(json.dumps(result, indent=2))
    print(json.dumps({"sample": sample, "tile_size": size,
                      "baseline_macro_iou": baseline["macro_iou"],
                      "bora_macro_iou": bora["macro_iou"],
                      "improvement": result["improvement"]["macro_iou"],
                      "refinement_seconds": refinement_seconds}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--tiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, action="append", choices=(56, 90, 216))
    parser.add_argument("--sample", action="append")
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--geojson-workers", type=int, default=2)
    parser.add_argument("--external-core-fraction", type=float, default=0.37)
    parser.add_argument("--adaptive-external-core", action="store_true")
    parser.add_argument("--internal-core-fraction", type=float, default=0.28)
    parser.add_argument("--grabcut-iterations", type=int, default=1)
    parser.add_argument("--watershed-compactness", type=float, default=0.1)
    parser.add_argument("--median-size", type=int, default=5)
    parser.add_argument("--repair-block-seams", action="store_true")
    parser.add_argument("--min-seam-run", type=int, default=24)
    parser.add_argument("--seam-roi-tiles", type=int, default=2)
    parser.add_argument("--repair-external-only", action="store_true")
    parser.add_argument("--protect-internal-band", type=int, default=0)
    parser.add_argument("--internal-search-fraction", type=float, default=0.28)
    parser.add_argument("--local-color-fraction", type=float, default=0.70)
    parser.add_argument("--internal-spatial-weight", type=float, default=1.5)
    parser.add_argument("--internal-median-size", type=int, default=9)
    args = parser.parse_args()
    converter_bins = [
        "/home/chiamaka/miniconda3/share/bioformats2raw-0.9.4-0/bin",
        "/home/chiamaka/miniconda3/share/raw2ometiff-0.7.1-0/bin",
    ]
    os.environ["PATH"] = os.pathsep.join(converter_bins + [os.environ.get("PATH", "")])
    config = BoundaryConfig(external_core_fraction=args.external_core_fraction,
                            adaptive_external_core=args.adaptive_external_core,
                            internal_core_fraction=args.internal_core_fraction,
                            grabcut_iterations=args.grabcut_iterations,
                            watershed_compactness=args.watershed_compactness,
                            median_size=args.median_size,
                            repair_block_seams=args.repair_block_seams,
                            min_seam_run=args.min_seam_run,
                            seam_roi_tiles=args.seam_roi_tiles,
                            repair_external_only=args.repair_external_only,
                            protect_internal_band=args.protect_internal_band,
                            internal_search_fraction=args.internal_search_fraction,
                            local_color_fraction=args.local_color_fraction,
                            internal_spatial_weight=args.internal_spatial_weight,
                            internal_median_size=args.internal_median_size)
    images = sorted(args.images.glob("*.btf"))
    if args.sample:
        images = [image for image in images if image.stem in args.sample]
    if not images:
        raise SystemExit("No matching BTF images")
    results = []
    for size in (args.tile_size or (56, 90, 216)):
        for image in images:
            results.append(run_one(image, args.tiles, args.output,
                                   size, args.block_size, config,
                                   args.geojson_workers))
        (args.output / str(size) / "summary.json").write_text(json.dumps({
            "tile_size": size,
            "results": [item for item in results if item["tile_size"] == size],
        }, indent=2))


if __name__ == "__main__":
    main()
