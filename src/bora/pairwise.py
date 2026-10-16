"""Seeded two-tissue contour competition within a symmetric boundary band."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import rasterio
import tifffile
from rasterio.windows import Window

from .streaming import _vips_crop


@dataclass(frozen=True)
class PairwiseConfig:
    downsample: int = 2
    core_distance: int = 40
    search_radius: int = 64
    iterations: int = 2
    min_seed_pixels: int = 128


def compete_pair(image: np.ndarray, labels: np.ndarray,
                 config: PairwiseConfig = PairwiseConfig()) -> np.ndarray:
    """Use two protected interiors as equal GrabCut seeds for one interface.

    Only pixels inside the symmetric search band can change. Blocks with a
    background or a third class are left untouched, preserving junctions and
    tissue-background boundaries until a separate multi-label method exists.
    """
    if image.shape[:2] != labels.shape or image.ndim != 3:
        raise ValueError("RGB image and categorical mask dimensions differ")
    if (config.downsample < 1 or config.core_distance < 1 or
            config.search_radius < 1 or config.iterations < 1):
        raise ValueError("Pairwise distances and iterations must be positive")
    ids = np.unique(labels)
    if ids.size != 2 or ids[0] == 0:
        return labels.copy()
    first, second = map(int, ids)
    factor = config.downsample
    height, width = labels.shape
    size = (max(1, int(np.ceil(width/factor))),
            max(1, int(np.ceil(height/factor))))
    rgb = cv2.resize(np.asarray(image[..., :3], dtype=np.uint8), size,
                     interpolation=cv2.INTER_AREA)
    coarse = cv2.resize(labels, size, interpolation=cv2.INTER_NEAREST)
    one = (coarse == first).astype(np.uint8)
    two = (coarse == second).astype(np.uint8)
    distance_one = cv2.distanceTransform(one, cv2.DIST_L2, 3)
    distance_two = cv2.distanceTransform(two, cv2.DIST_L2, 3)
    core = max(1, int(np.ceil(config.core_distance/factor)))
    first_seed = distance_one > core
    second_seed = distance_two > core
    if min(int(first_seed.sum()), int(second_seed.sum())) < config.min_seed_pixels:
        return labels.copy()
    markers = np.where(one, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    markers[first_seed] = cv2.GC_FGD
    markers[second_seed] = cv2.GC_BGD
    background = np.zeros((1, 65), np.float64)
    foreground = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(rgb, markers, None, background, foreground,
                    config.iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return labels.copy()
    chosen = ((markers == cv2.GC_FGD) |
              (markers == cv2.GC_PR_FGD))
    candidate = cv2.resize(np.where(chosen, first, second).astype(labels.dtype),
                           (width, height), interpolation=cv2.INTER_NEAREST)
    # Both labels have the same opportunity to claim a pixel. The original
    # label provides only the fixed cores and the width of the search region.
    edge = np.zeros(labels.shape, bool)
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[:, :-1] |= labels[:, 1:] != labels[:, :-1]
    edge[1:] |= labels[1:] != labels[:-1]
    edge[:-1] |= labels[1:] != labels[:-1]
    distance = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 3)
    editable = distance <= config.search_radius
    answer = labels.copy()
    answer[editable] = candidate[editable]
    return answer


def pairwise_refine_streaming(image_path: Path, mask_path: Path,
                              output_path: Path, block_size: int = 2048,
                              config: PairwiseConfig = PairwiseConfig()) -> dict:
    """Apply pairwise competition to suitable WSI windows with bounded memory."""
    import pyvips

    started = perf_counter()
    with rasterio.open(mask_path) as source:
        image = pyvips.Image.new_from_file(str(image_path), access="random")
        if image.bands == 1 and image.get_typeof("n-pages") and image.get("n-pages") in (3, 4):
            pages = [pyvips.Image.new_from_file(str(image_path), page=i,
                                                access="random") for i in range(3)]
            image = pages[0].bandjoin(pages[1:])
        if (image.height, image.width) != (source.height, source.width):
            raise ValueError("Image and mask dimensions differ")
        halo = max(config.search_radius + config.core_distance + 16, 128)
        halo = int(np.ceil(halo/config.downsample))*config.downsample
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        destination = tifffile.memmap(output_path, shape=(source.height, source.width),
                                      dtype=np.dtype(source.dtypes[0]),
                                      photometric="minisblack", metadata={"axes": "YX"},
                                      bigtiff=True)
        changed = refined = skipped = 0
        for y in range(0, source.height, block_size):
            for x in range(0, source.width, block_size):
                h, w = min(block_size, source.height-y), min(block_size, source.width-x)
                y0, x0 = max(0, y-halo), max(0, x-halo)
                y1, x1 = min(source.height, y+h+halo), min(source.width, x+w+halo)
                labels = source.read(1, window=Window(x0, y0, x1-x0, y1-y0))
                center = labels[y-y0:y-y0+h, x-x0:x-x0+w]
                ids = np.unique(labels)
                if len(ids) != 2 or ids[0] == 0:
                    destination[y:y+h, x:x+w] = center
                    skipped += 1
                    continue
                rgb = _vips_crop(image, x0, y0, x1-x0, y1-y0)
                updated = compete_pair(rgb, labels, config)
                chosen = updated[y-y0:y-y0+h, x-x0:x-x0+w]
                changed += int(np.count_nonzero(chosen != center))
                destination[y:y+h, x:x+w] = chosen
                refined += 1
            destination.flush()
            print(f"[Bora pairwise] row {min(y+block_size, source.height)}/{source.height} "
                  f"refined={refined} skipped={skipped}", flush=True)
        destination.flush()
        shape = [source.height, source.width]
    return {"mode": "pairwise_seeded_grabcut", "shape": shape,
            "parameters": asdict(config), "block_size": block_size,
            "refined_blocks": refined, "skipped_blocks": skipped,
            "changed_between_tissue_labels_pixels": changed,
            "runtime_seconds": perf_counter()-started}
