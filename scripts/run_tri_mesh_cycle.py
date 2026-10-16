#!/usr/bin/env python3
"""Run the C++ triangular contour stage and export a GeoJSON for each slide."""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import shutil
from pathlib import Path
from time import perf_counter

import tifffile

from bora.geojson_compat import enrich_wsitools_geojson
from bora.multicpu_geojson import convert
from bora.pyramid import pyramidize_mask
from bora.tri_mesh import (TriMeshConfig, refine_tri_mesh_streaming,
                           scale_for_tile_size)


def _pyramid_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with tifffile.TiffFile(path) as tif:
            return len(tif.series[0].levels) > 1
    except (OSError, ValueError, IndexError):
        return False


def run_one(source_root: Path, output_root: Path, tiling_root: Path,
            image_root: Path, sample: str, size: int, config: TriMeshConfig,
            block_size: int, workers: int, geojson_workers: int) -> dict:
    source = source_root / str(size) / sample / "bora_refined.pyramidal.ome.tif"
    if not source.is_file():
        raise FileNotFoundError(source)
    folder = output_root / str(size) / sample
    folder.mkdir(parents=True, exist_ok=True)
    refined = folder / "bora_refined.pyramidal.ome.tif"
    vector = folder / "bora_refined.geojson"
    comparison = folder / "comparison.json"
    if comparison.is_file() and _pyramid_complete(refined) and vector.is_file():
        return json.loads(comparison.read_text())
    stages_file = folder / "stages.json"
    stages = json.loads(stages_file.read_text()) if stages_file.is_file() else {}
    metadata_path = tiling_root / sample / "metrics_cpp.json"
    metadata = json.loads(metadata_path.read_text())
    one_class = sum(int(x["value"]) > 0 for x in metadata["classes"]) == 1
    if one_class:
        # This stage only edits tissue-to-tissue interfaces. A one-class slide
        # must be bit-identical; reuse the source without rescanning it.
        if not refined.is_file():
            os.link(source, refined)
        if not vector.is_file():
            shutil.copy2(source_root / str(size) / sample / "bora_refined.geojson", vector)
        stages.setdefault("tri_mesh", {"result": {
            "shape": [metadata["height"], metadata["width"]],
            "changed_pixels": 0, "blocks": 0, "workers": 0,
            "parameters": vars(config), "runtime_seconds": 0.0,
            "skipped_no_internal_boundary": True}, "seconds": 0.0})
        stages.setdefault("pyramid_seconds", 0.0)
        stages.setdefault("geojson", {"result": {"copied_unchanged": True},
                                       "seconds": 0.0})
        stages_file.write_text(json.dumps(stages, indent=2))
    if "tri_mesh" not in stages:
        started = perf_counter()
        result = refine_tri_mesh_streaming(
            source, refined, image_root / f"{sample}.btf" if config.color_weight else None,
            config=config, block_size=block_size, workers=workers)
        stages["tri_mesh"] = {"result": result, "seconds": perf_counter()-started}
        stages_file.write_text(json.dumps(stages, indent=2))
    if not _pyramid_complete(refined):
        started = perf_counter()
        pyramidize_mask(refined, "LZW", 4)
        stages["pyramid_seconds"] = perf_counter()-started
        stages_file.write_text(json.dumps(stages, indent=2))
    if not vector.is_file() or "geojson" not in stages:
        started = perf_counter()
        converted = convert(refined, vector, page=0, min_area=500.0,
                            simplify=2.0, workers=geojson_workers)
        stages["geojson"] = {"result": converted, "seconds": perf_counter()-started}
        stages_file.write_text(json.dumps(stages, indent=2))
    annotation = enrich_wsitools_geojson(vector, metadata_path,
                                         metadata["width"], metadata["height"])
    source_comparison = json.loads((source_root / str(size) / sample /
                                    "comparison.json").read_text())
    stage_seconds = sum(stages.get(key, {}).get("seconds", 0)
                        for key in ("tri_mesh", "geojson")) + stages.get("pyramid_seconds", 0)
    # The source refinement is shared by competing contour candidates. Both
    # incremental and complete pipeline times are retained for fair reporting.
    total_seconds = source_comparison["refinement_seconds"] + stage_seconds
    report = {"mode": "triangular_mesh_contour_competition_cpp",
              "source_mask": str(source), "output": str(refined),
              "geojson": str(vector), "parameters": vars(config),
              "tri_mesh": stages["tri_mesh"], "geojson": stages["geojson"],
              "annotation_metadata": annotation,
              "incremental_seconds": stage_seconds,
              "complete_pipeline_seconds": total_seconds}
    (folder / "bora_refined.report.json").write_text(json.dumps(report, indent=2))
    row = {"sample": sample, "tile_size": size,
           "tile_grid": str(tiling_root / sample / f"mask_tile_grid_{size}x{size}_cpp.tif"),
           "truth_pixel_mask": str(tiling_root / sample / "mask_pixel_cpp.tif"),
           "refined_mask": str(refined), "refined_geojson": str(vector),
           "source_refined_mask": str(source),
           "refinement_seconds": total_seconds,
           "incremental_seconds": stage_seconds,
           "tri_mesh_seconds": stages["tri_mesh"]["seconds"],
           "pyramid_seconds": stages.get("pyramid_seconds", 0),
           "geojson_seconds": stages["geojson"]["seconds"],
           "baseline": source_comparison["baseline"], "bora": None,
           "accuracy_scope": "final GeoJSON internal border band; see vector evaluation"}
    comparison.write_text(json.dumps(row, indent=2))
    print(json.dumps({"sample": sample, "tile_size": size,
                      "tri_mesh_seconds": row["tri_mesh_seconds"],
                      "complete_seconds": total_seconds}), flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tiling-root", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, action="append", choices=(56, 90, 216))
    parser.add_argument("--sample", action="append")
    parser.add_argument("--mesh-spacing", type=int, default=64)
    parser.add_argument("--support-radius", type=int, default=80)
    parser.add_argument("--band-radius", type=int, default=216)
    parser.add_argument("--color-weight", type=float, default=0.0)
    parser.add_argument("--post-smooth-radius", type=int, default=0)
    parser.add_argument("--adaptive-by-tile-size", action="store_true")
    parser.add_argument("--kernel-threads", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--geojson-workers", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=2048)
    args = parser.parse_args()
    config = TriMeshConfig(args.mesh_spacing, args.support_radius,
                           args.band_radius, args.color_weight,
                           args.kernel_threads, args.post_smooth_radius)
    converter_bins = [
        "/home/chiamaka/miniconda3/share/bioformats2raw-0.9.4-0/bin",
        "/home/chiamaka/miniconda3/share/raw2ometiff-0.7.1-0/bin",
    ]
    os.environ["PATH"] = os.pathsep.join(converter_bins + [os.environ.get("PATH", "")])
    for size in args.tile_size or (56, 90, 216):
        size_config = scale_for_tile_size(config, size) if args.adaptive_by_tile_size else config
        rows = []
        for folder in sorted((args.source_root / str(size)).iterdir()):
            if not folder.is_dir() or (args.sample and folder.name not in args.sample):
                continue
            rows.append(run_one(args.source_root, args.output, args.tiling_root,
                                args.images, folder.name, size, size_config,
                                args.block_size, args.workers, args.geojson_workers))
            gc.collect()
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        summary = args.output / str(size) / "summary.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps({"tile_size": size, "results": rows}, indent=2))


if __name__ == "__main__":
    main()
