#!/usr/bin/env python3
"""Measure GeoJSON fidelity and complexity against a pyramidal label mask."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import tifffile
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely import STRtree
from shapely.geometry import box, shape


def polygon_stats(geometry):
    polygons = ([geometry] if geometry.geom_type == "Polygon" else
                list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [])
    return {
        "polygon_parts": len(polygons),
        "holes": sum(len(p.interiors) for p in polygons),
        "vertices": sum(len(p.exterior.coords) + sum(len(r.coords) for r in p.interiors)
                        for p in polygons),
    }


def evaluate(mask_path, geojson_path, level=2):
    started = perf_counter()
    with tifffile.TiffFile(mask_path) as tif:
        series = tif.series[0]
        full_h, full_w = series.levels[0].shape[-2:]
        reference = np.asarray(series.levels[level].asarray())
    with Path(geojson_path).open(encoding="utf-8") as handle:
        collection = json.load(handle)
    parsed = []
    totals = {"polygon_parts": 0, "holes": 0, "vertices": 0}
    valid = True
    for feature in collection.get("features", []):
        geometry = shape(feature["geometry"])
        value = int(feature["properties"].get("value", feature["properties"].get("label", 0)))
        parsed.append((geometry, value))
        valid &= geometry.is_valid and not geometry.is_empty
        for key, count in polygon_stats(geometry).items():
            totals[key] += count
    scale_x, scale_y = full_w / reference.shape[1], full_h / reference.shape[0]
    predicted = rasterize(
        [(geometry, value) for geometry, value in parsed], out_shape=reference.shape,
        transform=Affine(scale_x, 0, 0, 0, scale_y, 0), fill=0,
        dtype=reference.dtype, all_touched=False)
    labels = sorted(int(v) for v in np.unique(reference) if int(v) > 0)
    iou = {}
    for value in labels:
        union = np.count_nonzero((reference == value) | (predicted == value))
        iou[str(value)] = (float(np.count_nonzero((reference == value) &
                                                  (predicted == value))) / union
                           if union else 1.0)
    fg_union = np.count_nonzero((reference > 0) | (predicted > 0))
    result = {
        "geojson": str(geojson_path), "evaluation_level": int(level),
        "evaluation_shape_yx": list(map(int, reference.shape)),
        "features": len(parsed), **totals, "all_valid_nonempty": bool(valid),
        "bytes": Path(geojson_path).stat().st_size,
        "pixel_accuracy": float(np.mean(reference == predicted)),
        "foreground_iou": (float(np.count_nonzero((reference > 0) &
                                                   (predicted > 0))) / fg_union
                           if fg_union else 1.0),
        "per_label_iou": iou,
        "macro_label_iou": float(np.mean(list(iou.values()))) if iou else 1.0,
        "evaluation_seconds": perf_counter() - started,
    }
    return result


def evaluate_level0_streaming(mask_path, geojson_path, block_size=2048):
    """Exact level-0 comparison with bounded memory."""
    import pyvips

    started = perf_counter()
    with Path(geojson_path).open(encoding="utf-8") as handle:
        collection = json.load(handle)
    features, parts, part_values = [], [], []
    totals = {"polygon_parts": 0, "holes": 0, "vertices": 0}
    valid = True
    for feature in collection.get("features", []):
        geometry = shape(feature["geometry"])
        value = int(feature["properties"].get("value", feature["properties"].get("label", 0)))
        features.append((geometry, value))
        valid &= geometry.is_valid and not geometry.is_empty
        for key, count in polygon_stats(geometry).items():
            totals[key] += count
        polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
        parts.extend(polygons)
        part_values.extend([value] * len(polygons))
    index = STRtree(parts)
    image = pyvips.Image.new_from_file(str(mask_path), access="random")
    maximum = max(part_values, default=0)
    confusion = np.zeros((maximum + 1, maximum + 1), dtype=np.int64)
    blocks = 0
    for y in range(0, image.height, block_size):
        for x in range(0, image.width, block_size):
            h, w = min(block_size, image.height-y), min(block_size, image.width-x)
            crop = image.crop(x, y, w, h)
            dtype = {"uchar":np.uint8, "ushort":np.uint16, "uint":np.uint32}[crop.format]
            reference = np.ndarray(buffer=crop.write_to_memory(), dtype=dtype,
                                   shape=(h, w)).copy()
            hits = index.query(box(x, y, x+w, y+h))
            predicted = rasterize(
                [(parts[int(i)], part_values[int(i)]) for i in hits],
                out_shape=(h, w), transform=Affine(1, 0, x, 0, 1, y),
                fill=0, dtype=dtype)
            size = maximum + 1
            confusion += np.bincount(
                reference.astype(np.int64).ravel() * size +
                predicted.astype(np.int64).ravel(), minlength=size*size).reshape(size, size)
            blocks += 1
    labels = list(range(1, maximum + 1))
    total = int(confusion.sum())
    iou = {}
    for value in labels:
        union = confusion[value, :].sum() + confusion[:, value].sum() - confusion[value, value]
        iou[str(value)] = float(confusion[value, value] / union) if union else 1.0
    fg_intersection = confusion[1:, 1:].sum()
    fg_union = total - confusion[0, 0]
    return {
        "geojson": str(geojson_path), "evaluation_level": 0,
        "evaluation_shape_yx": [image.height, image.width], "blocks": blocks,
        "features": len(features), **totals, "all_valid_nonempty": bool(valid),
        "bytes": Path(geojson_path).stat().st_size,
        "pixel_accuracy": float(np.trace(confusion) / total),
        "foreground_iou": float(fg_intersection / fg_union),
        "per_label_iou": iou,
        "macro_label_iou": float(np.mean(list(iou.values()))),
        "confusion_matrix": confusion.tolist(),
        "evaluation_seconds": perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mask")
    parser.add_argument("geojson", nargs="+")
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--stream-level0", action="store_true")
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--out")
    args = parser.parse_args()
    evaluator = ((lambda mask, geo: evaluate_level0_streaming(
        mask, geo, args.block_size)) if args.stream_level0 else
        (lambda mask, geo: evaluate(mask, geo, args.level)))
    results = [evaluator(args.mask, path) for path in args.geojson]
    text = json.dumps(results, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
