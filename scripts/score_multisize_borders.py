#!/usr/bin/env python3
"""Score final GeoJSON contours on every 56/90/216-pixel benchmark.

The region of interest is a fixed band around ground-truth or coarse tile
interfaces, so a large unchanged interior cannot hide a poor contour.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window

from score_216_borders import edges, summarize, update


TOLERANCES = (8, 16, 32)


def empty_internal_score() -> dict:
    """A one-class slide has no tissue-to-tissue contour to evaluate."""
    scores = {}
    for model in ("tile", "bora"):
        item = summarize({})
        item.update({"boundary_edge_pixels": 0,
                     "truth_boundary_edge_pixels": 0,
                     "contour_length_ratio": None,
                     "false_tile_imprint_edge_pixels": 0,
                     "off_band_internal_edge_pixels": 0,
                     "false_tile_imprint_fraction": None,
                     "assd_pixels": None,
                     "hausdorff95_pixels": None})
        for tolerance in TOLERANCES:
            for metric in ("precision", "recall", "f1"):
                item[f"boundary_{metric}_{tolerance}px"] = None
        scores[model] = item
    return {**scores, "changed_between_tissue_labels_pixels": 0,
            "score_seconds": 0.0}


def _distance(edge: np.ndarray) -> np.ndarray:
    if edge.any():
        return cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 3)
    return np.full(edge.shape, 257, np.float32)


def _percentile(hist: np.ndarray, fraction: float) -> int | None:
    total = int(hist.sum())
    if not total:
        return None
    return int(np.searchsorted(np.cumsum(hist), fraction*total))


def score(truth_path: Path, tile_path: Path, vector_path: Path,
          radius: int = 64, block_size: int = 2048) -> dict:
    started = perf_counter()
    confusion = {m: {} for m in ("tile", "bora")}
    records = {m: {"prediction": 0, "truth": 0,
                   "prediction_distance": np.zeros(258, np.int64),
                   "truth_distance": np.zeros(258, np.int64),
                   "false_tile_imprint": 0,
                   "off_band_internal_edge": 0,
                   "matched_prediction": {t: 0 for t in TOLERANCES},
                   "matched_truth": {t: 0 for t in TOLERANCES}}
               for m in ("tile", "bora")}
    changed = 0
    with (rasterio.open(truth_path) as truth,
          rasterio.open(tile_path) as tile,
          rasterio.open(vector_path) as vector):
        shape = truth.height, truth.width
        if shape != (tile.height, tile.width) or shape != (vector.height, vector.width):
            raise ValueError("Ground truth, tile mask, and final GeoJSON raster differ in size")
        for row in range(0, truth.height, block_size):
            for col in range(0, truth.width, block_size):
                h, w = min(block_size, truth.height-row), min(block_size, truth.width-col)
                halo = radius + max(TOLERANCES) + 4
                top, left = max(0, row-halo), max(0, col-halo)
                bottom = min(truth.height, row+h+halo)
                right = min(truth.width, col+w+halo)
                window = Window(left, top, right-left, bottom-top)
                t, b, v = (reader.read(1, window=window)
                           for reader in (truth, tile, vector))
                ys, xs = slice(row-top, row-top+h), slice(col-left, col-left+w)
                te, _ = edges(t); be, _ = edges(b); ve, _ = edges(v)
                dtruth, dtile, dbora = map(_distance, (te, be, ve))
                outside = ((dtruth[ys, xs] > radius) &
                           (dtile[ys, xs] > radius) & (t[ys, xs] > 0))
                records["bora"]["off_band_internal_edge"] += int(np.count_nonzero(
                    ve[ys, xs] & outside))
                roi = ((dtruth[ys, xs] <= radius) |
                       (dtile[ys, xs] <= radius)) & (t[ys, xs] > 0)
                if not roi.any():
                    continue
                changed += int(np.count_nonzero((b[ys, xs] != v[ys, xs]) &
                                                  (b[ys, xs] > 0) &
                                                  (v[ys, xs] > 0) & roi))
                reference = te[ys, xs] & roi
                for name, prediction, distance_to_prediction in (
                    ("tile", be[ys, xs] & roi, dtile),
                    ("bora", ve[ys, xs] & roi, dbora),
                ):
                    update(confusion[name], t[ys, xs],
                           (b if name == "tile" else v)[ys, xs], roi)
                    record = records[name]
                    record["prediction"] += int(prediction.sum())
                    record["truth"] += int(reference.sum())
                    prediction_distance = dtruth[ys, xs][prediction]
                    truth_distance = distance_to_prediction[ys, xs][reference]
                    record["prediction_distance"] += np.bincount(
                        np.minimum(prediction_distance.astype(np.int32), 257),
                        minlength=258)
                    record["truth_distance"] += np.bincount(
                        np.minimum(truth_distance.astype(np.int32), 257),
                        minlength=258)
                    for tolerance in TOLERANCES:
                        record["matched_prediction"][tolerance] += int(np.count_nonzero(
                            prediction_distance <= tolerance))
                        record["matched_truth"][tolerance] += int(np.count_nonzero(
                            truth_distance <= tolerance))
                    record["false_tile_imprint"] += int(np.count_nonzero(
                        prediction & (dtile[ys, xs] <= 2) &
                        (dtruth[ys, xs] > 8)))
    result = {name: summarize(confusion[name]) for name in records}
    for name, record in records.items():
        item = result[name]
        p, t = record["prediction"], record["truth"]
        item["boundary_edge_pixels"] = p
        item["truth_boundary_edge_pixels"] = t
        item["contour_length_ratio"] = p/t if t else None
        item["false_tile_imprint_edge_pixels"] = record["false_tile_imprint"]
        item["off_band_internal_edge_pixels"] = record["off_band_internal_edge"]
        item["false_tile_imprint_fraction"] = record["false_tile_imprint"]/p if p else None
        for tolerance in TOLERANCES:
            precision = record["matched_prediction"][tolerance]/p if p else None
            recall = record["matched_truth"][tolerance]/t if t else None
            f1 = (2*precision*recall/(precision+recall)
                  if precision is not None and recall is not None and precision+recall
                  else None)
            item[f"boundary_precision_{tolerance}px"] = precision
            item[f"boundary_recall_{tolerance}px"] = recall
            item[f"boundary_f1_{tolerance}px"] = f1
        ph, th = record["prediction_distance"], record["truth_distance"]
        item["assd_pixels"] = (float(np.dot(np.arange(258), ph+th))/(p+t)
                               if p+t else None)
        values = (_percentile(ph, .95), _percentile(th, .95))
        item["hausdorff95_pixels"] = max(x for x in values if x is not None) if any(
            x is not None for x in values) else None
    result["changed_between_tissue_labels_pixels"] = changed
    result["score_seconds"] = perf_counter()-started
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiling-root", type=Path, required=True)
    parser.add_argument("--vector-root", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, action="append", choices=(56, 90, 216))
    parser.add_argument("--sample", action="append")
    parser.add_argument("--radius", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    for size in args.tile_size or (56, 90, 216):
        folders = sorted((args.vector_root / str(size)).glob("*/vector_comparison.json"))
        for path in folders:
            sample = path.parent.name
            if args.sample and sample not in args.sample:
                continue
            output = path.parent / f"border_comparison_r{args.radius}.json"
            if output.exists() and not args.refresh:
                continue
            tile = args.tiling_root / sample
            metadata = json.loads((tile / "metrics_cpp.json").read_text())
            one_class = sum(int(x["value"]) > 0 for x in metadata["classes"]) == 1
            metrics = (empty_internal_score() if one_class else
                       score(tile / "mask_pixel_cpp.tif",
                             tile / f"mask_tile_grid_{size}x{size}_cpp.tif",
                             path.parent / "bora_refined_geojson_rasterized.tif",
                             args.radius, args.block_size))
            result = {"sample": sample, "tile_size": size, "radius": args.radius,
                      "region_definition": "within radius of GT or tile internal boundaries; excludes GT background",
                      "internal": metrics}
            output.write_text(json.dumps(result, indent=2))
            print(json.dumps({"sample": sample, "tile_size": size,
                              "internal_macro_iou": metrics["bora"]["macro_iou"],
                              "boundary_f1_8px": metrics["bora"]["boundary_f1_8px"],
                              "seconds": metrics["score_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
