"""Annotation-free white-space and colour-novelty diagnostics."""
import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


def white_regions(he):
    x = np.asarray(he, np.float32)/255
    maximum, minimum = x.max(-1), x.min(-1)
    saturation = (maximum-minimum)/np.maximum(maximum, 1e-6)
    bright = (minimum > .90) & (saturation < .12)
    components, count = ndi.label(bright, np.ones((3, 3)))
    border = np.unique(np.concatenate([
        components[0], components[-1], components[:, 0], components[:, -1]]))
    border = border[border > 0]
    exterior = np.isin(components, border) & bright
    return bright, exterior, dict(
        threshold_min_rgb=.90, threshold_saturation_max=.12,
        white_components=count, exterior_white_components=len(border),
        caveat="Border connection is an image diagnostic, not proof of glass; enclosed lumina are preserved.")


def novelty(rgb, coarse, support, mpp_xy):
    """Distance from colours observed in originally labelled tissue.

    Outlier regions can be mixed tissue or an image/model artefact. A novelty
    map cannot establish a biological class.
    """
    x = np.asarray(rgb, np.float32)/255
    known = (np.asarray(coarse) > 0) & support
    if not known.any():
        raise ValueError("No known coarse tissue for novelty estimation")
    bins = 32
    lattice = np.minimum((x*bins).astype(int), bins-1)
    index = (lattice[..., 0]*bins+lattice[..., 1])*bins+lattice[..., 2]
    counts = np.bincount(index[known], minlength=bins**3)
    observed = np.flatnonzero(counts > 0)
    coordinates = np.stack([observed//bins**2, observed//bins % bins,
                            observed % bins], -1)/bins+.5/bins
    tree = cKDTree(coordinates)
    distance = tree.query(x.reshape(-1, 3), workers=2)[0].reshape(x.shape[:2])
    threshold = max(.04, float(np.quantile(distance[known], .995)))
    selected = support & (distance > threshold)
    regions, n = ndi.label(selected, np.ones((3, 3)))
    area = np.bincount(regions.ravel())[1:]*float(np.prod(mpp_xy))
    return distance.astype(np.float32), regions.astype(np.int32), dict(
        distance_threshold=threshold, training_colour_cells=len(observed),
        proposed_regions=n, proposed_pixels=int(selected.sum()),
        area_um2=area.tolist(), fine_reference_used=False,
        identity="unresolved colour novelty; no biological names inferred")


def rare_colors(rgb, support, mpp_xy):
    """Rare observed colours, including minority colours in known coarse tiles.

    This complements unseen-colour distance. The latter cannot flag a colour
    merely for being rare if that colour was present in the training domain.
    No annotation, coarse label or predicted semantic class enters this rule.
    """
    x = np.asarray(rgb, np.float32)/255
    support = np.asarray(support, bool)
    if not support.any():
        raise ValueError("No tissue for colour rarity estimation")
    bins = 32
    cell = np.minimum((x*bins).astype(int), bins-1)
    index = (cell[..., 0]*bins+cell[..., 1])*bins+cell[..., 2]
    count = np.bincount(index[support], minlength=bins**3).reshape((bins,)*3)
    density = ndi.gaussian_filter(count.astype(float), .7, mode="nearest")
    density += max(density.sum(), 1)*1e-9
    density /= density.sum()
    score = -np.log(density.ravel()[index])
    threshold = float(np.quantile(score[support], .995))
    selected = support & (score > threshold)
    regions, n = ndi.label(selected, np.ones((3, 3)))
    areas = np.bincount(regions.ravel())[1:]*float(np.prod(mpp_xy))
    return score.astype(np.float32), regions.astype(np.int32), dict(
        rule="upper 0.5% of observed tissue negative log colour density",
        quantization_bins=bins, density_smoothing_bins=.7,
        rarity_threshold=threshold, proposed_regions=n,
        proposed_pixels=int(selected.sum()), area_um2=areas.tolist(),
        fine_reference_used=False, coarse_labels_used=False,
        identity="rare-colour region candidates, not established new tissue classes")


def mask_diagnostics(labels, support, white, exterior, mpp_xy, backend="numpy"):
    positive = labels > 0
    radius = max(1, round(8/np.sqrt(np.prod(mpp_xy))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2*radius+1, 2*radius+1))
    if backend == "numpy":
        nearby = np.zeros(labels.shape, np.uint16)
        for label in np.unique(labels[positive]):
            nearby += cv2.dilate((labels == label).astype(np.uint8), kernel)
        interface_white = white & (nearby >= 2)
    elif backend == "cpp":
        from .native import white_diversity
        interface_white = np.zeros(labels.shape, bool)
        if white.any():
            # max != min among positive labels is equivalent to >=2 classes.
            # A rectangle overestimates the ellipse, so this prefilter cannot
            # discard an actual qualifying ellipse. Query exact offsets only
            # where a white pixel could see multiple classes.
            dtype = np.uint8 if labels.max() < 255 else np.uint16
            sentinel = np.iinfo(dtype).max
            field = labels.astype(dtype)
            rectangular = np.ones(kernel.shape, np.uint8)
            maximum = cv2.dilate(field, rectangular, borderType=cv2.BORDER_CONSTANT,
                                  borderValue=0)
            nonzero = np.where(positive, labels, sentinel).astype(dtype)
            minimum = cv2.erode(nonzero, rectangular, borderType=cv2.BORDER_CONSTANT,
                                borderValue=int(sentinel))
            points = np.flatnonzero(white & (maximum > minimum))
            offsets = np.argwhere(kernel > 0)-radius
            order = np.argsort(np.sum(offsets**2, axis=1), kind="stable")
            found = white_diversity(labels, points, offsets[order])
            interface_white.ravel()[points] = found
    else:
        raise ValueError("Unknown diagnostic implementation")
    return dict(
        unassigned_supported_pixels=int((support & ~positive).sum()),
        assigned_outside_frozen_support=int((~support & positive).sum()),
        assigned_exterior_white_pixels=int((exterior & positive).sum()),
        assigned_enclosed_white_pixels=int((white & ~exterior & positive).sum()),
        white_near_multiple_classes=int(interface_white.sum()),
        unassigned_white_near_multiple_classes=int((interface_white & ~positive).sum()),
        diagnostic_radius_um=8,
        interpretation="White-area counts are diagnostics, not expert background accuracy.")
