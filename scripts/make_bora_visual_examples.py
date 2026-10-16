#!/usr/bin/env python3
"""Create paired image/mask crops where Bora changes benchmark accuracy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from rasterio.enums import Resampling
from rasterio.windows import Window
from scipy.ndimage import uniform_filter

LABELS = {
    "Visium_HD_11mm_Human_Pancreas_tissue_image": "Pancreas",
    "Visium_HD_3prime_Human_Lymph_Node_tissue_image": "Lymph node",
    "Visium_HD_3prime_Human_Tonsil_tissue_image": "Tonsil",
    "Visium_HD_6p5mm_Human_Brain_Cancer_tissue_image": "Brain cancer",
}


def candidate(sample, size, inputs, tiles, evaluations, crop=1536, thumb_scale=64):
    image = inputs / f"{sample}.btf"
    tile = tiles / sample / f"mask_tile_grid_{size}x{size}_cpp.tif"
    truth = tiles / sample / "mask_pixel_cpp.tif"
    vector = evaluations / str(size) / sample / "bora_refined_geojson_rasterized.tif"
    with rasterio.open(truth) as gt, rasterio.open(tile) as coarse, rasterio.open(vector) as bora:
        small_shape = (max(1, gt.height // thumb_scale), max(1, gt.width // thumb_scale))
        t = gt.read(1, out_shape=small_shape, resampling=Resampling.nearest)
        c = coarse.read(1, out_shape=small_shape, resampling=Resampling.nearest)
        b = bora.read(1, out_shape=small_shape, resampling=Resampling.nearest)
        span = max(3, crop // thumb_scale)
        gain = uniform_filter((c != t).astype(np.float32) -
                              (b != t).astype(np.float32), size=span)
        changed = uniform_filter((c != b).astype(np.float32), size=span)
        tissue = uniform_filter((t > 0).astype(np.float32), size=span)
        eligible = (changed > 0.01) & (tissue > 0.05)
        border = span // 2 + 1
        eligible[:border] = False; eligible[-border:] = False
        eligible[:, :border] = False; eligible[:, -border:] = False
        if not eligible.any():
            return []
        results = []
        for direction in (1, -1):
            scores = gain * direction
            scores[~eligible] = -np.inf
            row, col = np.unravel_index(np.argmax(scores), scores.shape)
            if not np.isfinite(scores[row, col]) or scores[row, col] <= 0:
                continue
            x = min(max(0, col * thumb_scale - crop // 2), gt.width - crop)
            y = min(max(0, row * thumb_scale - crop // 2), gt.height - crop)
            results.append((float(scores[row, col]), direction, sample, size,
                            x, y, image, tile, vector, truth))
        return results


def panel_image(values, rgb, colors):
    mask = np.zeros((*values.shape, 3), np.uint8)
    for label, color in colors.items():
        mask[values == label] = color
    background = values == 0
    overlay = (0.45 * rgb + 0.55 * mask).astype(np.uint8)
    overlay[background] = rgb[background]
    return Image.fromarray(overlay)


def make_figure(record, tiling_root, output_path, crop=1536):
    score, direction, sample, size, x, y, image, coarse, vector, truth = record
    window = Window(x, y, crop, crop)
    with rasterio.open(image) as src:
        rgb = np.moveaxis(src.read([1, 2, 3], window=window), 0, -1)
    masks = []
    for path in (coarse, vector, truth):
        with rasterio.open(path) as src:
            masks.append(src.read(1, window=window))
    metrics = json.loads((tiling_root / sample / "metrics_cpp.json").read_text())
    colors = {int(item["value"]): np.array(item["color"], np.uint8)
              for item in metrics["classes"] if int(item["value"]) > 0}
    base_accuracy = float(np.mean(masks[0] == masks[2]))
    bora_accuracy = float(np.mean(masks[1] == masks[2]))
    shrink = max(1, crop // 512)
    rgb = rgb[::shrink, ::shrink]
    masks = [mask[::shrink, ::shrink] for mask in masks]
    images = [Image.fromarray(rgb)] + [panel_image(mask, rgb, colors) for mask in masks]
    width, height = images[0].size
    canvas = Image.new("RGB", (2 * width + 36, 2 * height + 215), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        title_font = label_font = ImageFont.load_default()
    label = "improvement" if direction == 1 else "regression"
    draw.text((18, 10), f"{LABELS.get(sample, sample)} | {size} px | {label}",
              fill="#182537", font=title_font)
    draw.text((18, 38), f"ROI x={x}, y={y}, {crop} x {crop} pixels; "
              f"tile accuracy {base_accuracy:.3f}; Bora GeoJSON {bora_accuracy:.3f}",
              fill="#415366", font=label_font)
    names = ("H&E image", "Tile mask", "Bora GeoJSON", "Ground truth GeoJSON")
    for index, (name, picture) in enumerate(zip(names, images)):
        column, row = index % 2, index // 2
        px, py = 12 + column * (width + 12), 78 + row * (height + 18)
        canvas.paste(picture, (px, py))
        draw.text((px + 3, py + height + 1), name, fill="#182537", font=label_font)
    legend_y = 2 * height + 165
    draw.text((14, legend_y - 21), "Class colors", fill="#182537", font=label_font)
    cursor = 14
    for item in metrics["classes"]:
        if int(item["value"]) <= 0:
            continue
        name = str(item["name"])
        color = tuple(map(int, item["color"]))
        draw.rectangle((cursor, legend_y, cursor + 15, legend_y + 15), fill=color)
        draw.text((cursor + 21, legend_y - 1), name, fill="#182537", font=label_font)
        cursor += min(205, 32 + 10 * len(name))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return {"sample": sample, "tile_size": size, "x": x, "y": y,
            "crop": crop, "direction": label, "tile_accuracy": base_accuracy,
            "bora_vector_accuracy": bora_accuracy, "file": str(output_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tiling-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for size in (56, 90, 216):
        for sample_dir in sorted(args.tiling_root.iterdir()):
            if sample_dir.is_dir():
                records += candidate(sample_dir.name, size, args.input,
                                     args.tiling_root, args.evaluation_root)
    selected = []
    for direction, slug in ((1, "improvement"), (-1, "regression")):
        options = [item for item in records if item[1] == direction]
        if direction == -1:
            multiclass = [item for item in options if "Pancreas" not in item[2]]
            options = multiclass or options
        if options:
            best = max(options, key=lambda item: item[0])
            selected.append(make_figure(best, args.tiling_root,
                                        args.output / f"bora_{slug}_roi.png"))
    (args.output / "visual_examples.json").write_text(json.dumps(selected, indent=2))
    for item in selected:
        print(json.dumps(item))


if __name__ == "__main__":
    main()
