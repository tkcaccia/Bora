"""End-to-end GeoJSON -> tile mask -> Bora -> evaluation pipeline."""
from __future__ import annotations

import csv
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
import tifffile
from rasterio.windows import Window


ANNOTATION_SUFFIXES = (
    "_standard_grown_mask_smooth_class",
    "_grown_mask_smooth_class",
    "_annotations",
)
IMAGE_SUFFIXES = (".ome.tiff", ".ome.tif", ".tiff", ".tif", ".btf")


@dataclass(frozen=True)
class Sample:
    name: str
    image: Path
    geojson: Path


def _strip_image_suffix(name: str) -> str:
    lower = name.lower()
    for suffix in IMAGE_SUFFIXES:
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _annotation_stem(path: Path) -> str:
    name = path.name[:-8] if path.name.lower().endswith(".geojson") else path.stem
    for suffix in ANNOTATION_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def discover_samples(input_dir: str | Path, image_dir: str | Path | None = None) -> list[Sample]:
    """Pair images and GeoJSON files by their normalized sample names."""
    root = Path(input_dir)
    if not root.is_dir():
        raise ValueError(f"Input directory does not exist: {root}")
    image_root = Path(image_dir) if image_dir is not None else root
    if not image_root.is_dir():
        raise ValueError(f"Image directory does not exist: {image_root}")
    images: dict[str, list[Path]] = {}
    for path in image_root.iterdir():
        if any(path.name.lower().endswith(suffix) for suffix in IMAGE_SUFFIXES):
            images.setdefault(_strip_image_suffix(path.name), []).append(path)
    annotations = {_annotation_stem(path): path for path in root.glob("*.geojson")}
    errors, samples = [], []
    for name, geojson in sorted(annotations.items()):
        matches = images.get(name, [])
        if not matches:
            errors.append(f"{name}: no matching TIFF image")
            continue
        if len(matches) > 1:
            errors.append(f"{name}: multiple matching images: {', '.join(map(str, matches))}")
            continue
        image = matches[0]
        if image.is_symlink() and not image.exists():
            errors.append(f"{name}: broken image symlink: {image} -> {os.readlink(image)}")
            continue
        samples.append(Sample(name, image, geojson))
    if errors:
        raise ValueError("Invalid benchmark inputs:\n- " + "\n- ".join(errors))
    if not samples:
        raise ValueError(f"No image/GeoJSON pairs found in {root}")
    return samples


def run_checked(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + shlex.join(command) + "\n\n")
        log.flush()
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    if process.returncode:
        raise RuntimeError(f"Command failed ({process.returncode}); see {log_path}")


def expand_tile_grid(grid_path: str | Path, reference_path: str | Path,
                     output_path: str | Path, tile_width: int, tile_height: int) -> Path:
    """Expand a tile grid to a flat, memory-mappable full-resolution TIFF."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(reference_path) as ref, rasterio.open(grid_path) as grid:
        expected = (math.ceil(ref.height / tile_height), math.ceil(ref.width / tile_width))
        if (grid.height, grid.width) != expected:
            raise ValueError(
                f"Tile grid shape {(grid.height, grid.width)} != expected {expected}")
        # Bora's WSI path uses tifffile.memmap, so this intermediate must be an
        # uncompressed contiguous plane rather than a compressed/tiled TIFF.
        dst = tifffile.memmap(output, shape=(ref.height, ref.width),
                              dtype=np.dtype(grid.dtypes[0]), bigtiff=True,
                              photometric="minisblack", metadata={"axes": "YX"})
        try:
            block = 2048
            for row0 in range(0, ref.height, block):
              for col0 in range(0, ref.width, block):
                row1, col1 = min(ref.height, row0+block), min(ref.width, col0+block)
                gr0, gc0 = row0 // tile_height, col0 // tile_width
                gr1, gc1 = math.ceil(row1 / tile_height), math.ceil(col1 / tile_width)
                small = grid.read(1, window=Window(gc0, gr0, gc1-gc0, gr1-gr0))
                large = np.repeat(np.repeat(small, tile_height, axis=0), tile_width, axis=1)
                oy, ox = row0 - gr0 * tile_height, col0 - gc0 * tile_width
                dst[row0:row1, col0:col1] = large[oy:oy+row1-row0, ox:ox+col1-col0]
            dst.flush()
        finally:
            del dst
    return output


def prepare_coarse_mask(grid_path: str | Path, reference_path: str | Path,
                        output_path: str | Path, tile_width: int,
                        tile_height: int, *, resume: bool = True) -> Path:
    """Use a full-resolution grid directly or expand a compact tile index grid."""
    grid_path = Path(grid_path)
    output_path = Path(output_path)
    with rasterio.open(reference_path) as image, rasterio.open(grid_path) as grid:
        if (grid.height, grid.width) == (image.height, image.width):
            return grid_path
    if not (resume and output_path.exists()):
        expand_tile_grid(grid_path, reference_path, output_path,
                         tile_width, tile_height)
    return output_path


def compare_masks(prediction_path: str | Path, truth_path: str | Path,
                  *, block_size: int = 2048) -> dict:
    """Compute exact multiclass metrics using bounded-memory windows."""
    pairs: dict[tuple[int, int], int] = {}
    total = correct = 0
    with rasterio.open(prediction_path) as pred, rasterio.open(truth_path) as truth:
        if (pred.height, pred.width) != (truth.height, truth.width):
            raise ValueError(f"Mask shape mismatch: {(pred.height, pred.width)} vs "
                             f"{(truth.height, truth.width)}")
        for row in range(0, truth.height, block_size):
            for col in range(0, truth.width, block_size):
                window = Window(col, row, min(block_size, truth.width-col),
                                min(block_size, truth.height-row))
                p = pred.read(1, window=window).astype(np.uint64, copy=False).ravel()
                t = truth.read(1, window=window).astype(np.uint64, copy=False).ravel()
                total += t.size
                correct += int(np.count_nonzero(p == t))
                # Pack each truth/prediction pair into one scalar. A 1-D
                # integer unique is markedly faster than sorting a structured
                # two-column array on gigapixel masks.
                encoded = (t << np.uint64(32)) | p
                unique, counts = np.unique(encoded, return_counts=True)
                for value, count in zip(unique, counts):
                    ivalue = int(value)
                    key = (ivalue >> 32, ivalue & 0xFFFFFFFF)
                    pairs[key] = pairs.get(key, 0) + int(count)
    labels = sorted({x for pair in pairs for x in pair if x != 0})
    per_class = []
    sum_tp = sum_fp = sum_fn = 0
    for label in labels:
        tp = pairs.get((label, label), 0)
        fn = sum(n for (t, p), n in pairs.items() if t == label and p != label)
        fp = sum(n for (t, p), n in pairs.items() if p == label and t != label)
        denom_dice, denom_iou = 2*tp + fp + fn, tp + fp + fn
        dice = 2*tp / denom_dice if denom_dice else 1.0
        iou = tp / denom_iou if denom_iou else 1.0
        per_class.append({"label": label, "support": tp+fn, "tp": tp, "fp": fp,
                          "fn": fn, "dice": dice, "iou": iou})
        sum_tp += tp; sum_fp += fp; sum_fn += fn
    foreground_intersection = sum(n for (t, p), n in pairs.items() if t > 0 and p > 0)
    truth_fg = sum(n for (t, _), n in pairs.items() if t > 0)
    pred_fg = sum(n for (_, p), n in pairs.items() if p > 0)
    fg_denom = truth_fg + pred_fg
    return {
        "pixels": total,
        "accuracy": correct / total if total else 1.0,
        "foreground_dice": 2*foreground_intersection / fg_denom if fg_denom else 1.0,
        "foreground_iou": foreground_intersection /
            (truth_fg + pred_fg - foreground_intersection)
            if truth_fg + pred_fg - foreground_intersection else 1.0,
        "macro_dice": float(np.mean([x["dice"] for x in per_class])) if per_class else 1.0,
        "macro_iou": float(np.mean([x["iou"] for x in per_class])) if per_class else 1.0,
        "micro_dice": 2*sum_tp / (2*sum_tp + sum_fp + sum_fn)
            if 2*sum_tp + sum_fp + sum_fn else 1.0,
        "micro_iou": sum_tp / (sum_tp + sum_fp + sum_fn)
            if sum_tp + sum_fp + sum_fn else 1.0,
        "per_class": per_class,
    }


def _tile_slug(tile_size: str) -> tuple[str, int, int]:
    text = str(tile_size).lower()
    parts = text.split("x", 1)
    width = int(parts[0]); height = int(parts[-1])
    if width <= 0 or height <= 0:
        raise ValueError("tile size must be positive")
    return f"{width}x{height}", width, height


def run_pipeline(input_dir: str | Path, output_dir: str | Path, *, image_dir=None, tile_size="224",
                 tiling_mask_command="tiling-mask", tiling_mask_backend="python", coordinates="pixel",
                 class_property=None, workers=4, backend="watershed", checkpoint=None,
                 repo_dir=None, device="cuda", block_size=2048, resume=True,
                 extra_bora_args: Iterable[str] = ()) -> dict:
    samples = discover_samples(input_dir, image_dir=image_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    slug, tile_width, tile_height = _tile_slug(tile_size)
    results = []
    for sample in samples:
        sample_dir = output_root / sample.name
        tiling_dir = sample_dir / "tiling_mask"
        logs = sample_dir / "logs"
        if tiling_mask_backend == "cpp":
            truth = tiling_dir / "mask_pixel_cpp.tif"
            grid = tiling_dir / f"mask_tile_grid_{slug}_cpp.tif"
        elif tiling_mask_backend == "python":
            truth = tiling_dir / "mask_pixel.tif"
            grid = tiling_dir / f"mask_tile_grid_{slug}.tif"
        else:
            raise ValueError("tiling_mask_backend must be 'python' or 'cpp'")
        coarse = sample_dir / f"coarse_{slug}.tif"
        refined = sample_dir / "bora_refined.ome.tif"
        refined_geojson = sample_dir / "bora_refined.geojson"
        if not (resume and truth.exists() and grid.exists()):
            command = shlex.split(tiling_mask_command) + [str(sample.image), str(sample.geojson),
                       "--tile-size", str(tile_size), "--output", str(tiling_dir),
                       "--workers", str(workers)]
            if tiling_mask_backend == "python":
                command += ["--coordinates", coordinates]
            if class_property and tiling_mask_backend == "python":
                command += ["--class-property", class_property]
            elif class_property:
                raise ValueError("The C++ tiling-mask backend does not support --class-property")
            run_checked(command, logs / "tiling-mask.log")
        coarse = prepare_coarse_mask(grid, sample.image, coarse,
                                     tile_width, tile_height, resume=resume)
        if not (resume and refined.exists() and refined_geojson.exists()):
            command = [sys.executable, "-m", "bora.cli", "refine", str(sample.image),
                       str(coarse), "--output", str(refined), "--geojson",
                       str(refined_geojson), "--backend", backend, "--device", device,
                       "--block-size", str(block_size)]
            if backend == "adaptive-wand":
                if tile_width != tile_height:
                    raise ValueError("adaptive-wand currently requires square source tiles")
                command += ["--source-tile-size", str(tile_width)]
            if checkpoint:
                command += ["--checkpoint", str(checkpoint)]
            if repo_dir:
                command += ["--repo-dir", str(repo_dir)]
            command += list(extra_bora_args)
            run_checked(command, logs / "bora.log")
        baseline = compare_masks(coarse, truth, block_size=block_size)
        bora = compare_masks(refined, truth, block_size=block_size)
        result = {
            "sample": sample.name, "image": str(sample.image),
            "source_geojson": str(sample.geojson), "truth_mask": str(truth),
            "coarse_mask": str(coarse), "refined_mask": str(refined),
            "refined_geojson": str(refined_geojson), "baseline": baseline, "bora": bora,
            "improvement": {key: bora[key] - baseline[key] for key in
                            ("accuracy", "foreground_dice", "foreground_iou",
                             "macro_dice", "macro_iou", "micro_dice", "micro_iou")},
        }
        (sample_dir / "comparison.json").write_text(json.dumps(result, indent=2))
        results.append(result)
    summary = {"input_dir": str(Path(input_dir)),
               "image_dir": str(Path(image_dir)) if image_dir else str(Path(input_dir)),
               "output_dir": str(output_root),
               "tile_size": slug, "tiling_mask_backend": tiling_mask_backend,
               "backend": backend, "samples": results}
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2))
    fields = ["sample", "stage", "accuracy", "foreground_dice", "foreground_iou",
              "macro_dice", "macro_iou", "micro_dice", "micro_iou"]
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for result in results:
            for stage in ("baseline", "bora", "improvement"):
                writer.writerow({"sample": result["sample"], "stage": stage,
                                 **{key: result[stage][key] for key in fields[2:]}})
    return summary
