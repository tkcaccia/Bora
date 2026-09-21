"""CellPhenotyper-compatible wrapper for annealed-wand competition."""
from __future__ import annotations

import numpy as np
from PIL import Image

from .annealed_wand import annealed_wand_boundary_competition


def run_annealed_wand(image, labels, tissue=None, *, downsample=4, **kwargs):
    """Run competition on a BOX-downsampled grid and project labels back."""
    source = np.asarray(labels)
    tissue = source > 0 if tissue is None else np.asarray(tissue, dtype=bool)
    step = max(1, int(downsample))
    if step == 1:
        return annealed_wand_boundary_competition(
            np.asarray(image), source, tissue, **kwargs)

    height = int(np.ceil(source.shape[0] / step))
    width = int(np.ceil(source.shape[1] / step))
    work_image = np.asarray(
        Image.fromarray(np.asarray(image, dtype=np.uint8)).resize(
            (width, height), Image.Resampling.BOX))
    work_labels = source[::step, ::step]
    work_tissue = tissue[::step, ::step]
    work_kwargs = dict(kwargs)
    radius = int(work_kwargs.get("boundary_radius", 64))
    work_kwargs["boundary_radius"] = max(1, int(np.ceil(radius / step)))
    work_result, metadata = annealed_wand_boundary_competition(
        work_image, work_labels, work_tissue, **work_kwargs)

    y_index = np.minimum(np.arange(source.shape[0]) // step, work_result.shape[0] - 1)
    x_index = np.minimum(np.arange(source.shape[1]) // step, work_result.shape[1] - 1)
    expanded = work_result[np.ix_(y_index, x_index)]
    change = (expanded != source) & (expanded > 0) & (source > 0) & tissue
    result = source.copy()
    result[change] = expanded[change].astype(source.dtype, copy=False)
    result[~tissue] = 0
    metadata.update(
        working_downsample=step,
        working_shape_yx=list(map(int, work_result.shape)),
        changed_pixels_working=int(metadata.get("changed_pixels", 0)),
        changed_pixels=int(np.count_nonzero(result != source)),
        maximum_boundary_displacement_px=radius + step - 1,
        foreground_footprint_invariant=True,
    )
    return result.astype(source.dtype, copy=False), metadata
