#!/usr/bin/env python3
"""Score native-resolution annotation boundaries in fixed GT/tile bands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window


def edges(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    internal = np.zeros(labels.shape, bool)
    external = np.zeros(labels.shape, bool)
    for first, second, first_slice, second_slice in (
        (labels[:, :-1], labels[:, 1:], (slice(None), slice(None, -1)),
         (slice(None), slice(1, None))),
        (labels[:-1, :], labels[1:, :], (slice(None, -1), slice(None)),
         (slice(1, None), slice(None))),
    ):
        different = first != second
        inside = different & (first > 0) & (second > 0)
        outside = different & ((first == 0) != (second == 0))
        internal[first_slice] |= inside
        internal[second_slice] |= inside
        external[first_slice] |= outside
        external[second_slice] |= outside
    return internal, external


def band(edge: np.ndarray, radius: int) -> np.ndarray:
    if not edge.any():
        return np.zeros(edge.shape, bool)
    distance = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 3)
    return distance <= radius


def update(pairs: dict[tuple[int, int], int], truth: np.ndarray,
           prediction: np.ndarray, selected: np.ndarray) -> None:
    if not selected.any():
        return
    t = truth[selected].astype(np.uint64, copy=False)
    p = prediction[selected].astype(np.uint64, copy=False)
    encoded, counts = np.unique((t << np.uint64(32)) | p,
                                return_counts=True)
    for value, count in zip(encoded, counts):
        value = int(value)
        key = (value >> 32, value & 0xFFFFFFFF)
        pairs[key] = pairs.get(key, 0) + int(count)


def summarize(pairs: dict[tuple[int, int], int]) -> dict:
    total = sum(pairs.values())
    labels = sorted({label for pair in pairs for label in pair if label > 0})
    per_class = []
    tp_sum = fp_sum = fn_sum = 0
    for label in labels:
        tp = pairs.get((label, label), 0)
        fn = sum(n for (t, p), n in pairs.items() if t == label and p != label)
        fp = sum(n for (t, p), n in pairs.items() if p == label and t != label)
        union = tp + fp + fn
        per_class.append({"label": label, "support": tp+fn, "tp": tp,
                          "fp": fp, "fn": fn,
                          "iou": tp/union if union else 1.0})
        tp_sum += tp; fp_sum += fp; fn_sum += fn
    return {"pixels": total,
            "accuracy": sum(n for (t, p), n in pairs.items() if t == p)/total
                if total else None,
            "macro_iou": float(np.mean([x["iou"] for x in per_class]))
                if per_class else None,
            "micro_iou": tp_sum/(tp_sum+fp_sum+fn_sum)
                if tp_sum+fp_sum+fn_sum else None,
            "per_class": per_class}


def score(truth_path: Path, tile_path: Path, vector_raster: Path,
          radius: int, block_size: int, internal_only: bool = False) -> dict:
    regions = ("internal",) if internal_only else ("internal", "external")
    counts = {region: {model: {} for model in ("tile", "bora")}
              for region in regions}
    changed_between_labels = 0
    boundary_counts = {region: {model: {"matched_prediction": 0,
                                         "prediction": 0,
                                         "matched_truth": 0,
                                         "truth": 0,
                                         "false_tile_imprint": 0}
                                for model in ("tile", "bora")}
                       for region in regions}
    with (rasterio.open(truth_path) as truth,
          rasterio.open(tile_path) as tile,
          rasterio.open(vector_raster) as vector):
        shape = (truth.height, truth.width)
        if shape != (tile.height, tile.width) or shape != (vector.height, vector.width):
            raise ValueError("Mask dimensions do not match")
        for row in range(0, truth.height, block_size):
            for col in range(0, truth.width, block_size):
                h = min(block_size, truth.height-row)
                w = min(block_size, truth.width-col)
                top, left = max(0, row-radius-2), max(0, col-radius-2)
                bottom = min(truth.height, row+h+radius+2)
                right = min(truth.width, col+w+radius+2)
                context = Window(left, top, right-left, bottom-top)
                t = truth.read(1, window=context)
                b = tile.read(1, window=context)
                vcontext = vector.read(1, window=context)
                v = vcontext[row-top:row-top+h, col-left:col-left+w]
                ys, xs = slice(row-top, row-top+h), slice(col-left, col-left+w)
                te, tx = edges(t)
                be, bx = edges(b)
                ve, vx = edges(vcontext)
                changed_between_labels += int(np.count_nonzero(
                    (b[ys, xs] != v) & (b[ys, xs] > 0) & (v > 0)))
                for name, edge in (("internal", te | be),
                                   ("external", tx | bx)):
                    if name not in regions:
                        continue
                    roi = band(edge, radius)[ys, xs]
                    if name == "internal":
                        roi &= t[ys, xs] > 0
                    update(counts[name]["tile"], t[ys, xs], b[ys, xs], roi)
                    update(counts[name]["bora"], t[ys, xs], v, roi)
                    reference_edge = te if name == "internal" else tx
                    near_reference_8 = band(reference_edge, 8)[ys, xs]
                    near_tile_2 = band(be, 2)[ys, xs] if name == "internal" else None
                    for model, predicted_edge in (("tile", be if name == "internal" else bx),
                                                  ("bora", ve if name == "internal" else vx)):
                        record = boundary_counts[name][model]
                        prediction = predicted_edge[ys, xs] & roi
                        reference = reference_edge[ys, xs] & roi
                        # An 8-pixel tolerance measures contour placement,
                        # independently of the 64-pixel region-overlap band.
                        record["prediction"] += int(prediction.sum())
                        record["truth"] += int(reference.sum())
                        record["matched_prediction"] += int(np.count_nonzero(
                            prediction & near_reference_8))
                        record["matched_truth"] += int(np.count_nonzero(
                            reference & band(predicted_edge, 8)[ys, xs]))
                        if name == "internal":
                            # Count predicted class edges that still sit on a
                            # tile edge more than eight pixels from the true
                            # interface: visible residual tile imprint.
                            record["false_tile_imprint"] += int(np.count_nonzero(
                                prediction & near_tile_2 & ~near_reference_8))
    result = {name: {model: summarize(pairs) for model, pairs in models.items()}
              for name, models in counts.items()}
    for name, models in boundary_counts.items():
        for model, record in models.items():
            precision = record["matched_prediction"] / record["prediction"] if record["prediction"] else None
            recall = record["matched_truth"] / record["truth"] if record["truth"] else None
            f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
            result[name][model]["boundary_precision_8px"] = precision
            result[name][model]["boundary_recall_8px"] = recall
            result[name][model]["boundary_f1_8px"] = f1
            if name == "internal":
                result[name][model]["false_tile_imprint_edge_pixels"] = record["false_tile_imprint"]
                result[name][model]["false_tile_imprint_fraction"] = (
                    record["false_tile_imprint"] / record["prediction"]
                    if record["prediction"] else None)
    result["changed_between_tissue_labels_pixels"] = changed_between_labels
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiling-root", type=Path, required=True)
    parser.add_argument("--vector-root", type=Path, required=True)
    parser.add_argument("--sample", action="append")
    parser.add_argument("--radius", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--internal-only", action="store_true")
    parser.add_argument("--refresh", action="store_true",
                        help="recompute metrics after a scoring-method update")
    args = parser.parse_args()
    folders = sorted((args.vector_root / "216").glob("*/vector_comparison.json"))
    for path in folders:
        sample = path.parent.name
        if args.sample and sample not in args.sample:
            continue
        out = path.parent / f"border_comparison_r{args.radius}.json"
        if out.exists() and not args.refresh:
            print(f"{sample}: already complete", flush=True)
            continue
        tile = args.tiling_root / sample
        metrics = score(tile / "mask_pixel_cpp.tif",
                        tile / "mask_tile_grid_216x216_cpp.tif",
                        path.parent / "bora_refined_geojson_rasterized.tif",
                        args.radius, args.block_size, args.internal_only)
        result = {"sample": sample, "tile_size": 216, "radius": args.radius,
                  "region_definition": "within radius of GT or original tile boundaries; internal excludes GT background",
                  **metrics}
        out.write_text(json.dumps(result, indent=2))
        print(json.dumps({"sample": sample,
                          "internal_macro_iou": metrics["internal"]["bora"]["macro_iou"],
                          "external_macro_iou": (metrics["external"]["bora"]["macro_iou"]
                                                 if "external" in metrics else None)}),
              flush=True)


if __name__ == "__main__":
    main()
