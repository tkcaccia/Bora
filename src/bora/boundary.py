"""Image-guided refinement of every interface in a categorical tile mask."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import rasterio
import tifffile
from rasterio.enums import Resampling
from rasterio.windows import Window
from scipy import ndimage as ndi

from .streaming import _vips_crop


@dataclass(frozen=True)
class BoundaryConfig:
    downsample: int = 2
    external_core_fraction: float = 0.37
    adaptive_external_core: bool = False
    ambiguous_external_core_fraction: float = 0.20
    color_separation_threshold: float = 1.2
    internal_core_fraction: float = 0.28
    grabcut_iterations: int = 1
    gradient_sigma: float = 2.0
    watershed_compactness: float = 0.1
    median_size: int = 5
    min_island_area: int = 32
    repair_block_seams: bool = False
    min_seam_run: int = 24
    seam_roi_tiles: int = 2
    repair_external_only: bool = False
    protect_internal_band: int = 0
    internal_method: str = "balanced_wand"
    internal_search_fraction: float = 0.28
    local_color_fraction: float = 0.70
    internal_spatial_weight: float = 1.5
    internal_median_size: int = 9
    internal_regularizer: str = "median"
    internal_regularizer_weight: float = 0.35
    internal_regularizer_iterations: int = 6
    internal_edge_sigma: float = 12.0


def _regularize_classes(answer, editable, ids, lab, config, costs=None):
    """Optional categorical postfilters; the default preserves legacy output.

    Majority ties keep the current label. Edge-aware red/black ICM minimizes
    the original wand unary plus a four-neighbour contrast-weighted Potts cost.
    Label values are identities, never ordered measurements, in both options.
    """
    mode = config.internal_regularizer
    if mode == "none":
        return answer
    if mode == "median":
        if config.internal_median_size > 1 and answer.max() <= 255:
            smoothed = cv2.medianBlur(answer.astype(np.uint8), config.internal_median_size)
            answer[editable] = smoothed[editable]
        return answer
    if mode == "majority":
        size = config.internal_median_size
        counts = np.stack([cv2.boxFilter((answer == g).astype(np.float32), -1,
                          (size, size), normalize=False, borderType=cv2.BORDER_REPLICATE)
                           for g in ids])
        best = counts.max(axis=0)
        current = np.zeros(answer.shape, np.float32)
        for i, g in enumerate(ids):
            current[answer == g] = counts[i][answer == g]
        # Preserve ties rather than preferring the smallest numeric class ID.
        unique = (counts == best).sum(axis=0) == 1
        chosen = ids[counts.argmax(axis=0)]
        update = editable & unique & (best > current)
        answer[update] = chosen[update]
        return answer
    if mode != "edge_potts" or costs is None:
        raise ValueError(f"Unknown or incomplete internal regularizer: {mode}")
    if config.internal_regularizer_weight < 0 or config.internal_edge_sigma <= 0:
        raise ValueError("Regularizer weight must be nonnegative and edge sigma positive")
    sigma2 = 2 * config.internal_edge_sigma ** 2
    horizontal = np.exp(-np.sum(np.diff(lab, axis=1) ** 2, axis=2) / sigma2)
    vertical = np.exp(-np.sum(np.diff(lab, axis=0) ** 2, axis=2) / sigma2)
    yy, xx = np.indices(answer.shape)
    for _ in range(config.internal_regularizer_iterations):
        changed = 0
        for parity in (0, 1):
            energy = costs.copy()
            for i, g in enumerate(ids):
                penalty = np.zeros(answer.shape, np.float32)
                penalty[:, 1:] += horizontal * (answer[:, :-1] != g)
                penalty[:, :-1] += horizontal * (answer[:, 1:] != g)
                penalty[1:] += vertical * (answer[:-1] != g)
                penalty[:-1] += vertical * (answer[1:] != g)
                energy[i] += config.internal_regularizer_weight * penalty
            minimum = energy.min(axis=0)
            current = np.full(answer.shape, np.inf, np.float32)
            for i, g in enumerate(ids):
                current[answer == g] = energy[i][answer == g]
            selected = ids[energy.argmin(axis=0)]
            update = editable & ((yy + xx) % 2 == parity) & (minimum < current - 1e-7)
            changed += int(update.sum())
            answer[update] = selected[update]
        if changed == 0:
            break
    return answer


def _connected_to_seeds(candidate: np.ndarray, seeds: np.ndarray,
                        min_area: int) -> np.ndarray:
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), 8)
    if count <= 1:
        return candidate
    keep = np.zeros(count, bool)
    keep[np.unique(components[seeds])] = True
    keep[0] = False
    keep &= stats[:, cv2.CC_STAT_AREA] >= min_area
    return keep[components]


def estimate_slide_models(image_path: Path, mask_path: Path,
                          source_tile_size: int,
                          config: BoundaryConfig,
                          scale: int = 32) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit a shared foreground/background colour model on a WSI thumbnail."""
    with rasterio.open(mask_path) as masks, rasterio.open(image_path) as image:
        shape = (max(1, masks.height // scale), max(1, masks.width // scale))
        labels = masks.read(1, out_shape=shape, resampling=Resampling.nearest)
        rgb = np.moveaxis(image.read([1, 2, 3], out_shape=(3, *shape),
                                      resampling=Resampling.average), 0, -1)
    foreground = labels > 0
    if not foreground.any() or foreground.all():
        return None
    inside = cv2.distanceTransform(foreground.astype(np.uint8), cv2.DIST_L2, 3)
    outside = cv2.distanceTransform((~foreground).astype(np.uint8), cv2.DIST_L2, 3)
    core = max(2.0, source_tile_size / scale * config.external_core_fraction)
    markers = np.where(foreground, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    markers[inside > core], markers[outside > core] = cv2.GC_FGD, cv2.GC_BGD
    if np.count_nonzero(markers == cv2.GC_FGD) < 32 or np.count_nonzero(markers == cv2.GC_BGD) < 32:
        return None
    background = np.zeros((1, 65), np.float64)
    tissue = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(rgb, markers, None, background, tissue, 5,
                    cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    return background, tissue


def _exterior(image: np.ndarray, labels: np.ndarray, tile: float,
              config: BoundaryConfig,
              slide_models: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    foreground = labels > 0
    if foreground.all() or not foreground.any():
        return foreground
    inside = cv2.distanceTransform(foreground.astype(np.uint8), cv2.DIST_L2, 3)
    outside = cv2.distanceTransform((~foreground).astype(np.uint8), cv2.DIST_L2, 3)
    fraction = config.external_core_fraction
    if config.adaptive_external_core:
        near = (inside > max(2.0, tile * 0.2)) & (inside < tile * 1.5)
        far = (outside > max(2.0, tile * 0.2)) & (outside < tile * 1.5)
        sampled_inside, sampled_outside = near[::4, ::4], far[::4, ::4]
        if sampled_inside.sum() >= 64 and sampled_outside.sum() >= 64:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[::4, ::4].astype(np.float32)
            a, b = lab[sampled_inside], lab[sampled_outside]
            delta = a.mean(axis=0) - b.mean(axis=0)
            variance = (a.var(axis=0) + b.var(axis=0)) / 2
            separation = float(np.sqrt(np.sum(delta * delta / (variance + 1))))
            if separation < config.color_separation_threshold:
                fraction = config.ambiguous_external_core_fraction
    core = max(3.0, tile * fraction)
    certain_inside, certain_outside = inside > core, outside > core
    if not certain_inside.any():
        certain_inside = inside >= max(1.0, 0.6 * float(inside.max()))
    if not certain_outside.any():
        certain_outside = outside >= max(1.0, 0.6 * float(outside.max()))
    markers = np.where(foreground, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    markers[certain_inside], markers[certain_outside] = cv2.GC_FGD, cv2.GC_BGD
    try:
        if slide_models is None:
            background, tissue = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            iterations, mode = max(3, config.grabcut_iterations), cv2.GC_INIT_WITH_MASK
        else:
            background, tissue = (slide_models[0].copy(), slide_models[1].copy())
            iterations, mode = config.grabcut_iterations, cv2.GC_EVAL
        cv2.grabCut(image, markers, None, background, tissue, iterations, mode)
        selected = (markers == cv2.GC_FGD) | (markers == cv2.GC_PR_FGD)
    except cv2.error:
        selected = foreground.copy()
    selected = cv2.medianBlur(selected.astype(np.uint8), config.median_size) > 0
    selected |= certain_inside
    selected &= ~certain_outside
    selected = _connected_to_seeds(selected, certain_inside,
                                   config.min_island_area)
    holes, count = ndi.label(~selected)
    if count:
        areas = np.bincount(holes.ravel())
        edge_ids = np.unique(np.concatenate((holes[0], holes[-1],
                                             holes[:, 0], holes[:, -1])))
        areas[edge_ids] = 0
        fill = (areas > 0) & (areas < config.min_island_area)
        fill[0] = False
        selected |= fill[holes]
    return selected


def _gradient(image: np.ndarray, sigma: float) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab = cv2.GaussianBlur(lab, (0, 0), sigma)
    energy = np.zeros(image.shape[:2], np.float32)
    for channel in cv2.split(lab):
        gx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        energy += gx * gx + gy * gy
    energy = np.sqrt(energy)
    scale = float(np.quantile(energy[::4, ::4], 0.95))
    return np.clip(energy / max(scale, 1e-6), 0, 3)


def _interior(image: np.ndarray, labels: np.ndarray, tile: float,
              config: BoundaryConfig) -> np.ndarray:
    """Compete neighboring classes using symmetric, local H&E color models.

    Each class gets a color distribution from its own reliable pixels near the
    interface.  The editable band is the same width on both sides.  A weak
    distance prior prevents distant classes from jumping across a region, and
    the final small median removes isolated assignments.
    """
    ids = np.unique(labels[labels > 0])
    if len(ids) == 1:
        return np.full(labels.shape, ids[0], labels.dtype)
    if config.internal_method != "balanced_wand":
        raise ValueError(f"Unknown internal method: {config.internal_method}")
    foreground = labels > 0
    internal = np.zeros(labels.shape, bool)
    pair = (labels[:, 1:] != labels[:, :-1]) & foreground[:, 1:] & foreground[:, :-1]
    internal[:, 1:] |= pair
    internal[:, :-1] |= pair
    pair = (labels[1:] != labels[:-1]) & foreground[1:] & foreground[:-1]
    internal[1:] |= pair
    internal[:-1] |= pair
    if not internal.any():
        return labels.copy()
    radius = max(2, int(round(tile * config.internal_search_fraction)))
    distance_from_edge = cv2.distanceTransform((~internal).astype(np.uint8), cv2.DIST_L2, 3)
    editable = foreground & (distance_from_edge <= radius)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab = cv2.GaussianBlur(lab, (0, 0), 1.4)
    # OpenCV LAB has channel ranges 0..255; scaling makes the three channels
    # comparable and keeps variance floors meaningful for H&E images.
    lab[..., 0] *= 0.55
    lab[..., 1:] *= 0.8
    sigma = max(3.0, tile * config.local_color_fraction)
    best = np.full(labels.shape, np.inf, np.float32)
    answer = labels.copy()
    costs = [] if config.internal_regularizer == "edge_potts" else None
    for raw_id in ids:
        region = labels == raw_id
        inside = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 3)
        # Pixels safely inside the class estimate its nearby color, even when
        # the tile boundary itself contains colors from the adjacent class.
        core = region & (inside >= max(2.0, tile * config.internal_core_fraction))
        if core.sum() < 32:
            core = region & (inside >= max(1.0, float(inside.max()) * 0.4))
        sample = core.astype(np.float32)
        mass = cv2.GaussianBlur(sample, (0, 0), sigma)
        global_values = lab[core]
        if global_values.size == 0:
            if costs is not None:
                costs.append(np.full(labels.shape, np.inf, np.float32))
            continue
        global_mean = np.mean(global_values, axis=0)
        global_var = np.var(global_values, axis=0)
        cost = np.zeros(labels.shape, np.float32)
        for channel in range(3):
            values = lab[..., channel]
            mean = cv2.GaussianBlur(values * sample, (0, 0), sigma) / np.maximum(mass, 1e-4)
            second = cv2.GaussianBlur(values * values * sample, (0, 0), sigma) / np.maximum(mass, 1e-4)
            mean = np.where(mass > 1e-4, mean, global_mean[channel])
            variance = np.where(mass > 1e-4, second - mean * mean, global_var[channel])
            variance = np.maximum(variance, 64.0)
            cost += (values - mean) ** 2 / variance
        cost /= 3.0
        # Symmetric wand range: only labels within the same radius can claim
        # a pixel. A modest spatial cost stabilizes genuinely ambiguous color.
        outside = cv2.distanceTransform((~region).astype(np.uint8), cv2.DIST_L2, 3)
        cost += config.internal_spatial_weight * (outside / radius) ** 2
        cost[outside > radius + 1] = np.inf
        if costs is not None:
            costs.append(cost)
        chosen = editable & (cost < best)
        answer[chosen] = raw_id
        best[chosen] = cost[chosen]
    answer = _regularize_classes(answer, editable, ids, lab, config,
                                  np.stack(costs) if costs is not None else None)
    # Color alone can select isolated pixels in heterogeneous H&E. Keep only
    # regions connected to a reliable seed of the same original class.
    for raw_id in ids:
        region = labels == raw_id
        inside = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 3)
        seeds = region & (inside >= max(2.0, tile * config.internal_core_fraction))
        if not seeds.any():
            seeds = region & (inside >= max(1.0, float(inside.max()) * 0.4))
        count, components = cv2.connectedComponents((answer == raw_id).astype(np.uint8), 8)
        if count <= 1:
            continue
        anchored = np.zeros(count, bool)
        anchored[np.unique(components[seeds])] = True
        anchored[0] = False
        detached = (answer == raw_id) & ~anchored[components]
        answer[detached] = labels[detached]
    answer[~foreground] = 0
    return answer


def smooth_tile_boundaries(image: np.ndarray | None, labels: np.ndarray,
                           source_tile_size: int,
                           config: BoundaryConfig = BoundaryConfig(),
                           slide_models: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """Move exterior and class boundaries toward local H&E contours."""
    labels = np.asarray(labels)
    if labels.ndim != 2 or source_tile_size <= 0 or config.downsample <= 0:
        raise ValueError("Expected a 2-D label mask and positive tile/downsample sizes")
    if (config.internal_search_fraction <= 0 or config.local_color_fraction <= 0 or
            config.internal_spatial_weight < 0 or config.internal_median_size < 1 or
            config.internal_median_size % 2 != 1):
        raise ValueError("Balanced wand fractions must be positive and median size odd")
    if image is not None and image.shape[:2] != labels.shape:
        raise ValueError("Image and mask dimensions differ")
    if np.unique(labels).size <= 1:
        return labels.copy()
    if image is None:
        raise ValueError("Image-guided refinement requires an RGB image")
    height, width = labels.shape
    factor = config.downsample
    shape = (max(1, int(np.ceil(width / factor))),
             max(1, int(np.ceil(height / factor))))
    rgb = cv2.resize(np.asarray(image[..., :3], dtype=np.uint8), shape,
                     interpolation=cv2.INTER_AREA)
    coarse = cv2.resize(labels, shape, interpolation=cv2.INTER_NEAREST)
    exterior = _exterior(rgb, coarse, source_tile_size / factor,
                         config, slide_models)
    interior = _interior(rgb, coarse, source_tile_size / factor, config)
    result = np.where(exterior, interior, 0).astype(labels.dtype)
    return cv2.resize(result, (width, height), interpolation=cv2.INTER_NEAREST)


def _repair_block_seams(image, source, destination, source_tile_size: int,
                        block_size: int, config: BoundaryConfig,
                        slide_models) -> tuple[int, int]:
    """Re-evaluate straight joins while preserving the surrounding labels."""
    height, width = destination.shape
    repaired_runs = changed_pixels = 0
    for vertical in (True, False):
        length = width if vertical else height
        for seam in range(block_size, length, block_size):
            first = np.asarray(destination[:, seam-1] if vertical else
                               destination[seam-1, :])
            second = np.asarray(destination[:, seam] if vertical else
                                destination[seam, :])
            a = ((first > 0) != (second > 0) if config.repair_external_only
                 else first != second)
            transitions = np.diff(np.pad(a.astype(np.int8), (1, 1)))
            starts = np.flatnonzero(transitions == 1)
            stops = np.flatnonzero(transitions == -1)
            for start, stop in zip(starts, stops):
                if stop - start < config.min_seam_run:
                    continue
                across = config.seam_roi_tiles * source_tile_size
                along = source_tile_size
                if vertical:
                    x0, x1 = max(0, seam-across), min(width, seam+across)
                    y0, y1 = max(0, start-along), min(height, stop+along)
                else:
                    x0, x1 = max(0, start-along), min(width, stop+along)
                    y0, y1 = max(0, seam-across), min(height, seam+across)
                labels = source.read(1, window=Window(x0, y0, x1-x0, y1-y0))
                if np.unique(labels).size <= 1:
                    continue
                rgb = _vips_crop(image, x0, y0, x1-x0, y1-y0)
                old = np.asarray(destination[y0:y1, x0:x1]).copy()
                candidate = smooth_tile_boundaries(rgb, labels,
                                                   source_tile_size, config,
                                                   slide_models)
                difference = (candidate != old).astype(np.uint8)
                count, components, stats, _ = cv2.connectedComponentsWithStats(
                    difference, 8)
                if count <= 1:
                    continue
                if vertical:
                    span = components[start-y0:stop-y0,
                                      seam-x0-1:seam-x0+1]
                else:
                    span = components[seam-y0-1:seam-y0+1,
                                      start-x0:stop-x0]
                selected_ids = np.unique(span)
                keep = np.zeros(count, bool)
                keep[selected_ids] = True
                keep[0] = False
                keep &= stats[:, cv2.CC_STAT_AREA] <= 4*source_tile_size**2
                border_ids = np.unique(np.concatenate((
                    components[0], components[-1],
                    components[:, 0], components[:, -1])))
                keep[border_ids] = False
                chosen = keep[components]
                if config.repair_external_only:
                    chosen &= (old == 0) | (candidate == 0)
                if config.protect_internal_band:
                    inside_edge = np.zeros(labels.shape, bool)
                    xedge = ((labels[:, 1:] != labels[:, :-1]) &
                             (labels[:, 1:] > 0) & (labels[:, :-1] > 0))
                    yedge = ((labels[1:, :] != labels[:-1, :]) &
                             (labels[1:, :] > 0) & (labels[:-1, :] > 0))
                    inside_edge[:, 1:] |= xedge
                    inside_edge[:, :-1] |= xedge
                    inside_edge[1:, :] |= yedge
                    inside_edge[:-1, :] |= yedge
                    if inside_edge.any():
                        distance = cv2.distanceTransform(
                            (~inside_edge).astype(np.uint8), cv2.DIST_L2, 3)
                        chosen &= distance > config.protect_internal_band
                if not chosen.any():
                    continue
                patched = old.copy()
                patched[chosen] = candidate[chosen]
                if vertical:
                    new = patched[start-y0:stop-y0,
                                  seam-x0-1:seam-x0+1]
                else:
                    new = patched[seam-y0-1:seam-y0+1,
                                  start-x0:stop-x0]
                pair_a, pair_b = ((new[:, 0], new[:, 1]) if vertical
                                  else (new[0], new[1]))
                remaining = np.count_nonzero(
                    ((pair_a > 0) != (pair_b > 0))
                    if config.repair_external_only else (pair_a != pair_b))
                if remaining > (stop-start)*0.75:
                    continue
                destination[y0:y1, x0:x1] = patched
                repaired_runs += 1
                changed_pixels += int(chosen.sum())
        destination.flush()
    return repaired_runs, changed_pixels


def smooth_boundaries_streaming(image_path: Path, mask_path: Path, output_path: Path,
                                source_tile_size: int, block_size: int = 2048,
                                config: BoundaryConfig = BoundaryConfig()) -> dict:
    """Refine a whole slide with overlapped windows and bounded memory."""
    import pyvips

    if (config.min_seam_run < 1 or config.seam_roi_tiles < 1 or
            config.protect_internal_band < 0):
        raise ValueError("Seam settings must be positive (protection may be zero)")
    started = perf_counter()
    with rasterio.open(mask_path) as source:
        image = pyvips.Image.new_from_file(str(image_path), access="random")
        if image.bands == 1 and image.get_typeof("n-pages") and image.get("n-pages") in (3, 4):
            pages = [pyvips.Image.new_from_file(str(image_path), page=i,
                                                access="random") for i in range(3)]
            image = pages[0].bandjoin(pages[1:])
        if (image.height, image.width) != (source.height, source.width):
            raise ValueError("Image and mask dimensions differ")
        slide_models = estimate_slide_models(image_path, mask_path,
                                             source_tile_size, config)
        halo = max(48, int(np.ceil(source_tile_size * 0.65)) + 16)
        halo = int(np.ceil(halo / config.downsample)) * config.downsample
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        destination = tifffile.memmap(output_path, shape=(source.height, source.width),
                                      dtype=np.dtype(source.dtypes[0]),
                                      photometric="minisblack", metadata={"axes": "YX"},
                                      bigtiff=True)
        changed = internal_changed = refined = copied = 0
        for y in range(0, source.height, block_size):
            for x in range(0, source.width, block_size):
                h, w = min(block_size, source.height-y), min(block_size, source.width-x)
                y0, x0 = max(0, y-halo), max(0, x-halo)
                y1, x1 = min(source.height, y+h+halo), min(source.width, x+w+halo)
                labels = source.read(1, window=Window(x0, y0, x1-x0, y1-y0))
                central = labels[y-y0:y-y0+h, x-x0:x-x0+w]
                if np.unique(labels).size == 1:
                    destination[y:y+h, x:x+w] = central
                    copied += 1
                    continue
                rgb = _vips_crop(image, x0, y0, x1-x0, y1-y0)
                updated = smooth_tile_boundaries(rgb, labels, source_tile_size,
                                                 config, slide_models)
                selected = updated[y-y0:y-y0+h, x-x0:x-x0+w]
                changed += int(np.count_nonzero(selected != central))
                internal_changed += int(np.count_nonzero(
                    (selected != central) & (selected > 0) & (central > 0)))
                destination[y:y+h, x:x+w] = selected
                refined += 1
            destination.flush()
            print(f"[Bora boundary] row {min(y+block_size, source.height)}/{source.height} "
                  f"refined={refined} copied={copied}", flush=True)
        seam_runs = seam_pixels = 0
        if config.repair_block_seams:
            seam_runs, seam_pixels = _repair_block_seams(
                image, source, destination, source_tile_size, block_size,
                config, slide_models)
            changed += seam_pixels
            print(f"[Bora boundary] repaired_block_seams={seam_runs} "
                  f"changed_pixels={seam_pixels}", flush=True)
        destination.flush()
        height, width = source.height, source.width
    return {"mode": "seeded_grabcut_and_balanced_local_color_wand",
            "shape": [height, width], "source_tile_size": source_tile_size,
            "block_size": block_size, "refined_blocks": refined,
            "copied_blocks": copied, "changed_pixels": changed,
            "changed_between_tissue_labels_pixels": internal_changed,
            "repaired_block_seams": seam_runs,
            "seam_repair_changed_pixels": seam_pixels,
            "parameters": asdict(config),
            "shared_slide_color_model": slide_models is not None,
            "runtime_seconds": perf_counter()-started}
