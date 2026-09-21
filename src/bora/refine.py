from dataclasses import asdict, dataclass
from time import perf_counter
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology


@dataclass(frozen=True)
class RefineConfig:
    core_erosion: int = 16
    outer_dilation: int = 24
    tile_size: int = 2048
    tile_overlap: int = 256
    min_area: int = 32
    smooth_radius: int = 2


def bbox(mask, margin=0):
    yy, xx = np.nonzero(mask)
    if not len(yy): return 0, 0, 0, 0
    return (max(0, int(xx.min())-margin), max(0, int(yy.min())-margin),
            min(mask.shape[1], int(xx.max())+1+margin), min(mask.shape[0], int(yy.max())+1+margin))


def starts(lo, hi, size, overlap):
    if hi-lo <= size: return [max(0, hi-size)]
    values = list(range(lo, max(lo+1, hi-size+1), max(1, size-overlap)))
    return sorted(set(values + [max(lo, hi-size)]))


def refine_labels(image, labels, backend, config=RefineConfig()):
    labels = np.asarray(labels)
    if image.shape[:2] != labels.shape:
        raise ValueError(f"Image/mask shape mismatch: {image.shape[:2]} vs {labels.shape}")
    t0 = perf_counter()
    output = np.zeros(labels.shape, np.uint32)
    score = np.full(labels.shape, -np.inf, np.float32)
    ids = np.unique(labels); ids = ids[ids > 0]
    processed = 0
    for raw_id in ids:
        lid, coarse = int(raw_id), labels == raw_id
        if int(coarse.sum()) < config.min_area:
            output[coarse], score[coarse] = lid, 2
            continue
        core = morphology.binary_erosion(coarse, morphology.disk(config.core_erosion)) if config.core_erosion else coarse.copy()
        if not core.any(): core = coarse.copy()
        outer = morphology.binary_dilation(coarse, morphology.disk(config.outer_dilation)) if config.outer_dilation else coarse.copy()
        x0, y0, x1, y1 = bbox(outer)
        for ty in starts(y0, y1, config.tile_size, config.tile_overlap):
            for tx in starts(x0, x1, config.tile_size, config.tile_overlap):
                yy1, xx1 = min(labels.shape[0], ty+config.tile_size), min(labels.shape[1], tx+config.tile_size)
                sub_outer, sub_core = outer[ty:yy1, tx:xx1], core[ty:yy1, tx:xx1]
                if not sub_outer.any(): continue
                box = bbox(coarse[ty:yy1, tx:xx1], 4)
                pred, conf = backend.predict(image[ty:yy1, tx:xx1], sub_core, sub_outer, box)
                candidate = np.asarray(pred, bool) & sub_outer & ~sub_core
                dst_s, dst_l = score[ty:yy1, tx:xx1], output[ty:yy1, tx:xx1]
                update = candidate & (conf > dst_s)
                dst_l[update], dst_s[update] = lid, conf[update]
                dst_l[sub_core], dst_s[sub_core] = lid, 2
                processed += 1
    # Retain only components connected to the coarse seed; restore rejected labels.
    for raw_id in ids:
        lid, seed = int(raw_id), labels == raw_id
        current = output == lid
        if not current.any(): output[seed] = lid; continue
        cc, _ = ndi.label(current)
        keep = np.unique(cc[seed]); keep = keep[keep > 0]
        output[current & ~np.isin(cc, keep)] = 0
    if config.smooth_radius:
        for raw_id in ids:
            lid = int(raw_id)
            region = morphology.binary_closing(output == lid, morphology.disk(config.smooth_radius))
            output[region & (output == 0)] = lid
    return output, {"parameters": asdict(config), "input_label_count": int(len(ids)),
        "output_label_count": int(len(np.unique(output[output > 0]))), "processed_tiles": processed,
        "runtime_seconds": perf_counter()-t0}
