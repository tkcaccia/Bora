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


def mask_diagnostics(labels, support, white, exterior, mpp_xy):
    positive = labels > 0
    radius = max(1, round(8/np.sqrt(np.prod(mpp_xy))))
    nearby = np.zeros(labels.shape, np.uint16)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2*radius+1, 2*radius+1))
    for label in np.unique(labels[positive]):
        nearby += cv2.dilate((labels == label).astype(np.uint8), kernel)
    interface_white = white & (nearby >= 2)
    return dict(
        unassigned_supported_pixels=int((support & ~positive).sum()),
        assigned_outside_frozen_support=int((~support & positive).sum()),
        assigned_exterior_white_pixels=int((exterior & positive).sum()),
        assigned_enclosed_white_pixels=int((white & ~exterior & positive).sum()),
        white_near_multiple_classes=int(interface_white.sum()),
        unassigned_white_near_multiple_classes=int((interface_white & ~positive).sum()),
        diagnostic_radius_um=8,
        interpretation="White-area counts are diagnostics, not expert background accuracy.")
