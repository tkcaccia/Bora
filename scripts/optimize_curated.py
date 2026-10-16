#!/usr/bin/env python3
"""Calibrate Bora on curated WSI GeoJSONs and emit refined masks/GeoJSONs."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np

from bora.curated_training import (
    CuratedConfig, appearance_features, fit_curated_model, geojson_properties,
    predict_curated, predict_curated_tiled, rasterize_geojson, read_rgb_downsampled,
    regularize_prediction, segmentation_metrics, write_full_pyramidal_mask,
    write_scaled_geojson,
)
from bora.curated_training import _spatial_shape


def locate(root, good_path):
    sample = good_path.name.split("_ome_image_annotations")[0]
    return sample, (
        root / sample / "01_input" / sample / f"{sample}.ome.tif",
        root / sample / "15_cluster_geojson" / sample /
        f"{sample}_standard_grown_mask_smooth_class.geojson",
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--downsample", type=int, default=16)
    parser.add_argument("--sample", action="append", default=[],
                        help="process only this exact sample name; repeatable")
    parser.add_argument("--probability-sigma", type=float,
                        help="override probability smoothing (default: validated per tissue)")
    parser.add_argument("--no-full-mask", action="store_true")
    args = parser.parse_args(argv)
    curated = next((path for path in (args.root / "good geoson",
                                      args.root / "good geojson") if path.is_dir()), None)
    if curated is None:
        raise SystemExit("missing 'good geoson' or 'good geojson' directory")
    args.output.mkdir(parents=True, exist_ok=True)
    run_started = perf_counter()
    reports = []
    for good_path in sorted(curated.glob("*.geojson")):
        sample, (image_path, coarse_path) = locate(args.root, good_path)
        if args.sample and sample not in args.sample:
            continue
        started = perf_counter()
        sample_dir = args.output / sample
        sample_dir.mkdir(parents=True, exist_ok=True)
        # The pancreas rare-structure task benefits from extra capacity; the
        # lighter setting is faster and validated equally or better elsewhere.
        hard = "Pancreas" in sample
        policy = ({"probability_sigma": .5, "min_component": 256, "radius": 0}
                  if hard else
                  {"probability_sigma": .5, "min_component": 128, "radius": 0}
                  if "Lymph_Node" in sample else
                  {"probability_sigma": .5, "min_component": 128, "radius": 4}
                  if "Tonsil" in sample else
                  {"probability_sigma": .5, "min_component": 0, "radius": 2})
        if args.probability_sigma is not None:
            policy["probability_sigma"] = args.probability_sigma
        config = CuratedConfig(
            downsample=args.downsample,
            samples_per_class=50000 if hard else 20000,
            max_iter=220 if hard else 120,
            max_leaf_nodes=127 if hard else 63,
            learning_rate=.06 if hard else .08,
        )
        rgb = read_rgb_downsampled(image_path, config.downsample)
        target, names = rasterize_geojson(good_path, rgb.shape[:2],
                                          config.downsample, target=True)
        coarse, _ = rasterize_geojson(coarse_path, rgb.shape[:2],
                                      config.downsample, target=False)
        features = appearance_features(rgb, coarse)
        model, fit_report = fit_curated_model(features, target, config)
        if policy["probability_sigma"]:
            del features
            prediction, prediction_seconds = predict_curated_tiled(
                model, rgb, coarse, tile_size=512, halo=96,
                probability_sigma=policy["probability_sigma"])
        else:
            prediction, prediction_seconds = predict_curated(model, features)
        raw_metrics = segmentation_metrics(target, prediction)
        regularized = regularize_prediction(
            prediction, min_component=policy["min_component"],
            radius=policy["radius"])
        regularized_metrics = segmentation_metrics(target, regularized)
        if regularized_metrics["macro_iou"] >= raw_metrics["macro_iou"]:
            selected, selected_metrics, postprocess = regularized, regularized_metrics, "median_components"
        else:
            selected, selected_metrics, postprocess = prediction, raw_metrics, "none"
        properties = geojson_properties(good_path)
        geojson_path = sample_dir / f"{sample}_bora_refined.geojson"
        feature_count = write_scaled_geojson(selected, geojson_path,
                                             config.downsample, properties)
        mask_path = sample_dir / f"{sample}_bora_refined.pyramidal.ome.tif"
        if not args.no_full_mask:
            write_full_pyramidal_mask(selected, mask_path,
                                      _spatial_shape(image_path),
                                      config.downsample)
        np.savez_compressed(sample_dir / "calibration_prediction.npz",
                            prediction=selected)
        joblib.dump({"model": model, "config": asdict(config),
                     "properties": properties, "class_names": names},
                    sample_dir / "bora_curated_model.joblib")
        report = {
            "sample": sample, "image": str(image_path), "coarse_geojson": str(coarse_path),
            "curated_target": str(good_path), "config": asdict(config),
            "fit": fit_report, "prediction_seconds": prediction_seconds,
            "inference_policy": policy,
            "raw_metrics": raw_metrics, "selected_metrics": selected_metrics,
            "postprocess": postprocess, "geojson": str(geojson_path),
            "pyramidal_mask": None if args.no_full_mask else str(mask_path),
            "geojson_feature_count": feature_count,
            "runtime_seconds": perf_counter() - started,
        }
        (sample_dir / "optimization_report.json").write_text(json.dumps(report, indent=2))
        reports.append(report)
        print(json.dumps({"sample": sample, "metrics": selected_metrics,
                          "runtime_seconds": report["runtime_seconds"]}), flush=True)
    summary = {"schema_version": "bora.curated_optimization.v1",
               "runtime_seconds": perf_counter() - run_started, "samples": reports}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
