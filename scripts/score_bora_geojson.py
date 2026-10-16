#!/usr/bin/env python3
"""Rasterize Bora GeoJSON at native resolution and score against GeoJSON truth."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import box, mapping, shape
from shapely.strtree import STRtree

from bora.benchmark_pipeline import compare_masks


def vector_parts(path: Path):
    data = json.loads(path.read_text())
    geometries, labels = [], []
    for feature in data["features"]:
        label = int(feature["properties"]["value"])
        if label <= 0:
            continue
        geometry = shape(feature["geometry"])
        parts = geometry.geoms if geometry.geom_type == "MultiPolygon" else [geometry]
        for part in parts:
            if not part.is_empty:
                geometries.append(part)
                labels.append(label)
    return geometries, labels


def rasterize_vector(path: Path, reference: Path, output: Path,
                     block_size: int = 2048) -> int:
    geometries, labels = vector_parts(path)
    tree = STRtree(geometries) if geometries else None
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(reference) as truth:
        profile = truth.profile.copy()
        profile.update(driver="GTiff", count=1, dtype="uint16", nodata=0,
                       tiled=True, blockxsize=512, blockysize=512,
                       compress="deflate", BIGTIFF="IF_SAFER")
        with rasterio.open(output, "w", **profile) as destination:
            for row in range(0, truth.height, block_size):
                for col in range(0, truth.width, block_size):
                    height = min(block_size, truth.height - row)
                    width = min(block_size, truth.width - col)
                    indices = (tree.query(box(col, row, col + width, row + height))
                               if tree is not None else [])
                    selected = [(mapping(geometries[int(i)]), labels[int(i)])
                                for i in indices]
                    values = rasterize(selected, out_shape=(height, width),
                                       transform=rasterio.Affine.translation(col, row),
                                       fill=0, dtype=np.uint16)
                    destination.write(values, 1, window=Window(col, row, width, height))
    return len(geometries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bora-root", type=Path, required=True)
    parser.add_argument("--tiling-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path,
                        help="optional separate directory for rasterized vectors and scores")
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--tile-size", type=int, action="append", choices=(56, 90, 216),
                        help="score only this tile size; repeatable")
    parser.add_argument("--sample", action="append", help="score only this sample; repeatable")
    parser.add_argument("--skip-vector-raster-check", action="store_true",
                        help="omit the optional GeoJSON-versus-refined-TIFF QA scan")
    parser.add_argument("--reuse-unchanged-vector-root", type=Path,
                        help="reuse scored single-class vectors when internal-only refinement left them unchanged")
    args = parser.parse_args()
    for size in (args.tile_size or (56, 90, 216)):
        for directory in sorted((args.bora_root / str(size)).iterdir()):
            if not directory.is_dir() or (args.sample and directory.name not in args.sample):
                continue
            sample = directory.name
            vector = directory / "bora_refined.geojson"
            truth = args.tiling_root / sample / "mask_pixel_cpp.tif"
            refined = directory / "bora_refined.pyramidal.ome.tif"
            evaluation = (args.evaluation_root / str(size) / sample
                          if args.evaluation_root else directory)
            raster = evaluation / "bora_refined_geojson_rasterized.tif"
            comparison_path = evaluation / "vector_comparison.json"
            if comparison_path.is_file() and raster.is_file():
                print(f"{sample} {size}: vector comparison already complete", flush=True)
                continue
            if args.reuse_unchanged_vector_root is not None:
                metadata = json.loads((args.tiling_root / sample / "metrics_cpp.json").read_text())
                one_class = sum(int(x["value"]) > 0 for x in metadata["classes"]) == 1
                prior = args.reuse_unchanged_vector_root / str(size) / sample
                if one_class and (prior / "vector_comparison.json").is_file() and (
                        prior / "bora_refined_geojson_rasterized.tif").is_file():
                    evaluation.mkdir(parents=True, exist_ok=True)
                    if not raster.is_file():
                        os.link(prior / "bora_refined_geojson_rasterized.tif", raster)
                    result = json.loads((prior / "vector_comparison.json").read_text())
                    result.update({"vector": str(vector),
                                   "rasterized_vector": str(raster),
                                   "vector_vs_refined_raster": None,
                                   "vector_scoring_seconds": 0.0,
                                   "reused_unchanged_single_class_vector": True})
                    comparison_path.write_text(json.dumps(result, indent=2))
                    print(f"{sample} {size}: reused unchanged one-class vector", flush=True)
                    continue
            started = perf_counter()
            part_count = rasterize_vector(vector, truth, raster, args.block_size)
            result = {
                "sample": sample, "tile_size": size,
                "reference": str(truth), "vector": str(vector),
                "rasterized_vector": str(raster), "polygon_parts": part_count,
                "vector_vs_ground_truth": compare_masks(raster, truth,
                                                         block_size=args.block_size),
                "vector_vs_refined_raster": (None if args.skip_vector_raster_check
                                              else compare_masks(raster, refined,
                                                                 block_size=args.block_size)),
            }
            result["vector_scoring_seconds"] = perf_counter() - started
            comparison_path.write_text(json.dumps(result, indent=2))
            print(json.dumps({"sample": sample, "tile_size": size,
                              "vector_macro_iou": result["vector_vs_ground_truth"]["macro_iou"],
                              "vector_raster_accuracy": (result["vector_vs_refined_raster"]["accuracy"]
                                                         if result["vector_vs_refined_raster"] else None)}),
                  flush=True)


if __name__ == "__main__":
    main()
