"""Native annotation metrics. This module is never imported by prediction."""
import numpy as np
from scipy import ndimage as ndi


class RegionEvaluator:
    def __init__(self, truth, mpp_xy):
        self.truth = np.asarray(truth)
        self.valid = self.truth > 0
        self.ids = np.unique(self.truth[self.valid])
        if not self.ids.size:
            raise ValueError("Reference has no annotated pixels")
        spacing = np.asarray(mpp_xy, float)
        if spacing.shape != (2,) or (spacing <= 0).any():
            raise ValueError("Positive per-axis native spacing required")
        self.pixel_area = float(spacing.prod())
        self.components = np.zeros(truth.shape, np.int32)
        areas, classes = [0], [0]
        offset = 0
        for label in self.ids:
            cc, n = ndi.label(self.truth == label, np.ones((3, 3)))
            nonzero = cc > 0
            self.components[nonzero] = cc[nonzero] + offset
            areas.extend(np.bincount(cc.ravel(), minlength=n+1)[1:].tolist())
            classes.extend([int(label)]*n)
            offset += n
        self.area_pixels = np.asarray(areas)
        self.area_um2 = self.area_pixels*self.pixel_area
        self.component_classes = np.asarray(classes)

    def score(self, prediction, backend="numpy"):
        prediction = np.asarray(prediction)
        if prediction.shape != self.truth.shape:
            raise ValueError("Native geometry mismatch")
        if backend not in ("numpy", "cpp"):
            raise ValueError("Unknown metric implementation")
        confusion, native_correct = None, None
        if backend == "cpp":
            from .native import region_counts
            confusion, native_correct = region_counts(
                self.truth, prediction, self.components, len(self.area_pixels))
        per_class = {}
        for label in self.ids:
            if confusion is None:
                target = self.truth == label
                proposed = (prediction == label) & self.valid
                intersection = int(np.count_nonzero(target & proposed))
                target_pixels, proposed_pixels = int(target.sum()), int(proposed.sum())
            else:
                intersection = int(confusion[label, label])
                target_pixels = int(confusion[label].sum())
                proposed_pixels = int(confusion[:, label].sum())
            union = target_pixels+proposed_pixels-intersection
            total = target_pixels+proposed_pixels
            per_class[str(int(label))] = dict(
                iou=intersection/union, dice=2*intersection/total,
                truth_pixels=target_pixels, prediction_pixels=proposed_pixels)
        correct = native_correct if native_correct is not None else np.bincount(
            self.components.ravel(),
            weights=(prediction == self.truth).ravel(),
            minlength=len(self.area_pixels))
        coverage = correct/np.maximum(self.area_pixels, 1)
        strata = {}
        for name, low, high in (
                ("small_le_400_um2", 0, 400),
                ("medium_400_2500_um2", 400, 2500),
                ("large_gt_2500_um2", 2500, np.inf),
                ("all", 0, np.inf)):
            take = (self.area_um2 > low) & (self.area_um2 <= high)
            take[0] = False
            n = int(take.sum())
            strata[name] = dict(
                components=n,
                mean_correct_coverage=float(coverage[take].mean()) if n else None,
                recall_at_50pct=float((coverage[take] >= .5).mean()) if n else None,
                completely_missed=int((correct[take] == 0).sum()),
                missed_at_50pct=int((coverage[take] < .5).sum()),
                total_area_um2=float(self.area_um2[take].sum()))
        missed = np.flatnonzero((coverage < .5) & (self.area_pixels > 0))
        missed = missed[missed > 0]
        return dict(
            macro_iou=float(np.mean([r["iou"] for r in per_class.values()])),
            macro_dice=float(np.mean([r["dice"] for r in per_class.values()])),
            pixel_accuracy=float(np.trace(confusion)/confusion.sum()) if confusion is not None
            else float((prediction[self.valid] == self.truth[self.valid]).mean()),
            annotated_unassigned_pixels=int(confusion[:, 0].sum()) if confusion is not None
            else int(np.count_nonzero(self.valid & (prediction == 0))),
            valid_pixels=int(self.valid.sum()), per_class=per_class, regions=strata,
            missed_regions=[
                dict(component=int(i), label=int(self.component_classes[i]),
                     area_um2=float(self.area_um2[i]), correct_coverage=float(coverage[i]))
                for i in missed])


def discovery_score(regions, truth):
    """Class-agnostic proposal overlap, kept separate from semantic accuracy.

    No assignment of a biological class is inferred from this score.
    """
    regions = np.asarray(regions)
    truth = np.asarray(truth)
    valid = truth > 0
    counts = np.bincount(regions[valid].ravel())
    annotated = np.unique(regions[valid & (regions > 0)])
    return dict(proposed_components=int(regions.max()),
                components_intersecting_annotations=int(len(annotated)),
                annotated_proposed_pixels=int(counts[1:].sum()),
                semantic_identity="unknown; proposals are not new named tissue classes")
