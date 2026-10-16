#!/usr/bin/env python3
"""Run a numbered 216-pixel pairwise contour candidate on four slides."""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
from pathlib import Path
from time import perf_counter

from bora.geojson_compat import enrich_wsitools_geojson
from bora.multicpu_geojson import convert
from bora.pairwise import PairwiseConfig, pairwise_refine_streaming
from bora.pyramid import pyramidize_mask


def run_one(source_root: Path, input_root: Path, output_root: Path,
            tiling_root: Path, sample: str, block_size: int,
            config: PairwiseConfig) -> None:
    source_folder = source_root / "216" / sample
    source = source_folder / "bora_refined.pyramidal.ome.tif"
    image = input_root / f"{sample}.btf"
    folder = output_root / "216" / sample
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "bora_refined.pyramidal.ome.tif"
    vector = folder / "bora_refined.geojson"
    stages_path = folder / "stages.json"
    stages = json.loads(stages_path.read_text()) if stages_path.exists() else {}
    if "pairwise" not in stages:
        stages["pairwise"] = pairwise_refine_streaming(
            image, source, output, block_size, config)
        stages_path.write_text(json.dumps(stages, indent=2))
    if "pyramid_seconds" not in stages:
        started = perf_counter()
        pyramidize_mask(output, "LZW", 4)
        stages["pyramid_seconds"] = perf_counter() - started
        stages_path.write_text(json.dumps(stages, indent=2))
    if "geojson" not in stages:
        started = perf_counter()
        stages["geojson"] = convert(output, vector, page=0,
                                     min_area=500.0, simplify=2.0, workers=1)
        stages["geojson_seconds"] = perf_counter() - started
        stages_path.write_text(json.dumps(stages, indent=2))
    metadata_path = tiling_root / sample / "metrics_cpp.json"
    metadata = json.loads(metadata_path.read_text())
    annotated = enrich_wsitools_geojson(vector, metadata_path,
                                        metadata["width"], metadata["height"])
    previous = json.loads((source_folder / "comparison.json").read_text())
    incremental = (stages["pairwise"]["runtime_seconds"] +
                   stages["pyramid_seconds"] + stages["geojson_seconds"])
    seconds = previous["refinement_seconds"] + incremental
    result = {**previous, "source_refined_mask": str(source),
              "refined_mask": str(output), "refined_geojson": str(vector),
              "refinement_seconds": seconds,
              "pairwise_seconds": stages["pairwise"]["runtime_seconds"],
              "incremental_seconds": incremental,
              "bora": None, "improvement": None,
              "accuracy_scope": "final GeoJSON, internal tissue-border band"}
    (folder / "comparison.json").write_text(json.dumps(result, indent=2))
    (folder / "bora_refined.report.json").write_text(json.dumps(
        {"mode": "pairwise_seeded_grabcut", "source_mask": str(source),
         "output": str(output), "geojson": str(vector),
         "pairwise": stages["pairwise"], "annotation_metadata": annotated,
         "refinement_seconds": seconds}, indent=2))
    print(json.dumps({"sample": sample, "refinement_seconds": seconds,
                      "pairwise_changed_pixels": stages["pairwise"]["changed_between_tissue_labels_pixels"]}),
          flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tiling-root", type=Path, required=True)
    parser.add_argument("--sample", action="append")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--core-distance", type=int, default=40)
    parser.add_argument("--search-radius", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=2)
    args = parser.parse_args()
    converter_bins = [
        "/home/chiamaka/miniconda3/share/bioformats2raw-0.9.4-0/bin",
        "/home/chiamaka/miniconda3/share/raw2ometiff-0.7.1-0/bin",
    ]
    os.environ["PATH"] = os.pathsep.join(converter_bins + [os.environ.get("PATH", "")])
    config = PairwiseConfig(core_distance=args.core_distance,
                            search_radius=args.search_radius,
                            iterations=args.iterations)
    samples = args.sample or [p.name for p in sorted((args.source_root / "216").iterdir())
                              if p.is_dir()]
    for sample in samples:
        run_one(args.source_root, args.input_root, args.output, args.tiling_root,
                sample, args.block_size, config)
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)


if __name__ == "__main__":
    main()
