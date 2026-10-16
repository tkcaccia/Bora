"""Small-hole cleanup and symmetric smoothing of categorical tissue borders."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from math import pi

import cv2
import numpy as np
import rasterio
import tifffile
from rasterio.windows import Window


def _fill_small_background_holes(labels: np.ndarray, max_area: int) -> np.ndarray:
    result = labels.copy()
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        (result == 0).astype(np.uint8), 8)
    height, width = result.shape
    for component_id in range(1, count):
        x, y, w, h, area = map(int, stats[component_id])
        if area > max_area or x == 0 or y == 0 or x+w == width or y+h == height:
            continue
        x0, y0, x1, y1 = x-1, y-1, x+w+1, y+h+1
        part = components[y0:y1, x0:x1] == component_id
        ring = cv2.dilate(part.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        neighbors = result[y0:y1, x0:x1][ring & ~part]
        neighbors = neighbors[neighbors > 0]
        if neighbors.size == 0:
            continue
        ids, amounts = np.unique(neighbors, return_counts=True)
        result[y0:y1, x0:x1][part] = ids[np.argmax(amounts)]
    return result


def _remove_small_foreground_islands(labels: np.ndarray, max_area: int,
                                     surround_fraction: float = 0.9,
                                     min_circularity: float = 0.0) -> tuple[np.ndarray, int]:
    """Absorb tiny enclosed class islands into their dominant touching class."""
    result = labels.copy()
    changed = 0
    height, width = result.shape
    for raw_label in np.unique(labels):
        if raw_label == 0:
            continue
        count, components, stats, _ = cv2.connectedComponentsWithStats(
            (labels == raw_label).astype(np.uint8), 8)
        for component_id in range(1, count):
            x, y, w, h, area = map(int, stats[component_id])
            if area > max_area or x == 0 or y == 0 or x+w == width or y+h == height:
                continue
            x0, y0, x1, y1 = x-1, y-1, x+w+1, y+h+1
            part = components[y0:y1, x0:x1] == component_id
            if min_circularity:
                contours, _ = cv2.findContours(part.astype(np.uint8),
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                perimeter = cv2.arcLength(max(contours, key=cv2.contourArea), True)
                circularity = 4*pi*area/(perimeter*perimeter) if perimeter else 0
                if circularity < min_circularity:
                    continue
            ring = cv2.dilate(part.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            neighbors = result[y0:y1, x0:x1][ring & ~part]
            if neighbors.size == 0 or np.any(neighbors == 0):
                continue
            ids, amounts = np.unique(neighbors, return_counts=True)
            winner = int(np.argmax(amounts))
            if ids[winner] == raw_label or amounts[winner] / neighbors.size < surround_fraction:
                continue
            result[y0:y1, x0:x1][part] = ids[winner]
            changed += area
    return result, changed


def smooth_and_fill(labels: np.ndarray, *, sigma: float = 8.0,
                    band_radius: int = 20, max_hole_area: int = 1500,
                    max_island_area: int = 0,
                    surround_fraction: float = 0.9,
                    min_circularity: float = 0.0) -> np.ndarray:
    """Fill tiny enclosed zero regions, then smooth only class-to-class edges."""
    source = np.asarray(labels)
    if (source.ndim != 2 or sigma < 0 or band_radius < 0 or max_hole_area < 0
            or max_island_area < 0 or not 0 < surround_fraction <= 1):
        raise ValueError("Expected a 2-D mask and nonnegative cleanup settings")
    if min_circularity < 0 or min_circularity > 1:
        raise ValueError("min_circularity must be between zero and one")
    filled = _fill_small_background_holes(source, max_hole_area) if max_hole_area else source.copy()
    if max_island_area:
        filled, _ = _remove_small_foreground_islands(
            filled, max_island_area, surround_fraction, min_circularity)
    if sigma == 0 or band_radius == 0:
        return filled
    foreground = filled > 0
    edge = np.zeros(source.shape, bool)
    horizontal = (filled[:, :-1] != filled[:, 1:]) & foreground[:, :-1] & foreground[:, 1:]
    edge[:, :-1] |= horizontal
    edge[:, 1:] |= horizontal
    vertical = (filled[:-1] != filled[1:]) & foreground[:-1] & foreground[1:]
    edge[:-1] |= vertical
    edge[1:] |= vertical
    if not edge.any():
        return filled
    distance = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 3)
    editable = foreground & (distance <= band_radius)
    best = np.zeros(source.shape, np.float32)
    answer = filled.copy()
    for raw_id in np.unique(filled[foreground]):
        score = cv2.GaussianBlur((filled == raw_id).astype(np.float32), (0, 0), sigma)
        selected = editable & (score > best)
        answer[selected] = raw_id
        best[selected] = score[selected]
    return answer


def postprocess_streaming(input_path: Path, output_path: Path, *, block_size: int = 2048,
                          sigma: float = 8.0, band_radius: int = 20,
                          max_hole_area: int = 1500,
                          max_island_area: int = 0,
                          surround_fraction: float = 0.9,
                          min_circularity: float = 0.0) -> dict:
    """Apply the same bounded cleanup to every native-resolution WSI block."""
    started = perf_counter()
    halo = max(64, band_radius + int(np.ceil(5*sigma)),
               int(np.ceil(np.sqrt(max_island_area))) + 2)
    with rasterio.open(input_path) as source:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        destination = tifffile.memmap(output_path, shape=(source.height, source.width),
                                      dtype=np.dtype(source.dtypes[0]),
                                      photometric="minisblack", metadata={"axes": "YX"},
                                      bigtiff=True)
        changed = holes_filled = class_changed = 0
        for y in range(0, source.height, block_size):
            for x in range(0, source.width, block_size):
                h, w = min(block_size, source.height-y), min(block_size, source.width-x)
                y0, x0 = max(0, y-halo), max(0, x-halo)
                y1, x1 = min(source.height, y+h+halo), min(source.width, x+w+halo)
                crop = source.read(1, window=Window(x0, y0, x1-x0, y1-y0))
                if np.unique(crop).size < 2:
                    central = crop[y-y0:y-y0+h, x-x0:x-x0+w]
                    destination[y:y+h, x:x+w] = central
                    continue
                updated = smooth_and_fill(crop, sigma=sigma,
                                          band_radius=band_radius,
                                          max_hole_area=max_hole_area,
                                          max_island_area=max_island_area,
                                          surround_fraction=surround_fraction,
                                          min_circularity=min_circularity)
                original = crop[y-y0:y-y0+h, x-x0:x-x0+w]
                selected = updated[y-y0:y-y0+h, x-x0:x-x0+w]
                destination[y:y+h, x:x+w] = selected
                changed += int(np.count_nonzero(selected != original))
                holes_filled += int(np.count_nonzero((original == 0) & (selected > 0)))
                class_changed += int(np.count_nonzero((original > 0) & (selected > 0) &
                                                       (original != selected)))
            destination.flush()
            print(f"[Bora cleanup] row {min(y+block_size, source.height)}/{source.height}", flush=True)
        destination.flush()
        shape = [source.height, source.width]
    return {"shape": shape, "changed_pixels": changed,
            "filled_background_pixels": holes_filled,
            "changed_between_tissue_labels_pixels": class_changed,
            "sigma": sigma, "band_radius": band_radius,
            "max_hole_area": max_hole_area,
            "max_island_area": max_island_area,
            "surround_fraction": surround_fraction,
            "min_circularity": min_circularity,
            "runtime_seconds": perf_counter()-started}
