"""Explicit numerical coordinates and task semantics for expanded scoring.

These functions do not infer calibration, read reference masks, or alter saved
predictions. Integration into the expanded runner is a separate required step.
"""
import math

import numpy as np


def geometry(sample, *, native=False):
    key = "native_mpp_xy" if native else "working_mpp_xy"
    spacing = sample[key]
    if spacing is None:
        if not sample.get("pixel_only") or sample.get("physical_metrics_allowed") is not False:
            raise ValueError("Uncalibrated input must explicitly forbid physical metrics")
        return dict(algorithm_spacing_xy=[1., 1.], units="pixels",
            physical_spacing_xy=None, physical_metrics_allowed=False,
            notice="Unit numerical pixel coordinates; not an assumed microscopy resolution")
    if len(spacing) != 2 or not all(math.isfinite(v) and v > 0 for v in spacing):
        raise ValueError("Finite positive measured axis spacing required")
    if sample.get("pixel_only") or sample.get("physical_metrics_allowed") is False:
        raise ValueError("Measured spacing contradicts the pixel-only declaration")
    return dict(algorithm_spacing_xy=list(spacing), units="micrometres",
        physical_spacing_xy=list(spacing), physical_metrics_allowed=True,
        notice="Provider-reported image pixel spacing, not optical resolving power")


def report_units(value, units):
    """Report pixel-based lengths/areas honestly when a numerical API uses _um.

    IoU, Dice, class IDs, probabilities, times and dimensionless parameters are
    unchanged. Physical and pixel-based small-region aggregates stay separate.
    """
    if units == "micrometres":
        return value
    if units != "pixels":
        raise ValueError("Unknown coordinate units")
    if isinstance(value, dict):
        return {k.replace("mpp_xy", "pixel_step_xy").replace("_um2", "_px2").replace("_um", "_px"):
                report_units(v, units) for k, v in value.items()}
    if isinstance(value, list):
        return [report_units(v, units) for v in value]
    return value


def binary_negative(sample, *, ocelot_white_negative=False):
    """Return the verified broad-negative target; this is not a stroma label."""
    targets={'peso':(1,'non-epithelial target'),'orca':(127,'non-carcinoma tissue target')}
    if ocelot_white_negative and sample['dataset']=='ocelot':
        if sample.get('source',{}).get('source_version')!='1.0.1':
            raise ValueError('OCELOT background adapter requires the verified1.0.1 ontology')
        targets['ocelot']=(1,'non-cancer background target; not expert stroma')
    if sample['dataset'] in targets:
        raw,name=targets[sample['dataset']]
        mapping={str(k):int(v) for k,v in sample['raw_to_compact'].items()}
        if (sample['dataset']!='ocelot' and sample.get('valid_negative_raw_label')!=raw) or str(raw) not in mapping:
            raise ValueError('Binary target lacks the declared negative-class contract')
        return mapping[str(raw)],name
    return None


def scoring_prediction(sample, prediction, *, zero_is_removed_background, white=None,ocelot_white_negative=False):
    """Map saved glass-zero to a declared negative only for numerical scoring.

    PESO/ORCA and prospectively OCELOT require RGB-white pixels. Historical
    RINGS retains its original all-zero outside-tumor adapter. Positive target
    deletions remain false negatives; saved predictions never change.
    """
    negative=binary_negative(sample,ocelot_white_negative=ocelot_white_negative)
    if negative is not None:
        label,name=negative
        if not zero_is_removed_background or white is None or np.asarray(white).shape!=prediction.shape:
            raise ValueError('Binary background scoring requires the RGB-only white mask and support invariant')
        result=np.array(prediction,copy=True);selected=(result==0)&np.asarray(white,bool)
        result[selected]=label
        return result,dict(prediction_modified_for_scoring=True,saved_prediction_modified=False,
            scoring_only_zero_to_class=label,converted_pixels=int(selected.sum()),
            task=name,conversion_scope='Only photometrically white saved-zero pixels; non-white unassigned pixels remain errors',
            positive_tumour_or_epithelium_deleted_remains_false_negative=True)
    if sample["dataset"] != "rings-tumor":
        return prediction, dict(prediction_modified_for_scoring=False,
            policy="All positive reference pixels scored; prediction zero remains incorrect")
    if not zero_is_removed_background:
        raise ValueError("RINGS adapter requires the all-supported-tissue-assigned invariant")
    mapping = sample["raw_to_compact"]
    # An all-tumor reference crop can lack raw0 entirely. A virtual negative ID
    # still represents deleted tumor correctly; it is never a predicted class.
    outside = int(mapping["0"]) if "0" in mapping else max(map(int, mapping.values()))+1
    result = np.array(prediction, copy=True)
    if result.dtype.kind not in "iu" or outside > np.iinfo(result.dtype).max:
        raise ValueError("Categorical prediction dtype cannot encode outside-tumor")
    count = int((result == 0).sum())
    result[result == 0] = outside
    return result, dict(prediction_modified_for_scoring=True,
        saved_prediction_modified=False, scoring_only_zero_to_class=outside,
        virtual_negative_class="0" not in mapping, converted_pixels=count,
        task="tumor versus outside tumor; outside tumor is not expert stroma",
        positive_tumor_deleted_by_support_remains_false_negative=True)
