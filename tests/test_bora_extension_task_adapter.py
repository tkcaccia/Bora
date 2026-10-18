import numpy as np
import pytest

from experiments.bora_extension.task_adapter import geometry, report_units, scoring_prediction


@pytest.mark.parametrize('dataset,raw,positive',[('peso',1,2),('orca',127,255)])
def test_new_negative_targets_only_forgive_removed_white_not_missing_stained_tissue(dataset,raw,positive):
    sample=dict(dataset=dataset,valid_negative_raw_label=raw,raw_to_compact={str(raw):1,str(positive):2})
    saved=np.array([[0,0,2]],np.uint8);white=np.array([[True,False,True]])
    scored,policy=scoring_prediction(sample,saved,zero_is_removed_background=True,white=white)
    np.testing.assert_array_equal(scored,[[1,0,2]])
    np.testing.assert_array_equal(saved,[[0,0,2]])
    assert policy['converted_pixels']==1 and not policy['saved_prediction_modified']
    # If the white deleted pixel is positive tumour/epithelium, negative1
    # still disagrees with positive2; it is not removed from evaluation.
    assert scored[0,0]!=2
    with pytest.raises(ValueError,match='RGB-only white'):
        scoring_prediction(sample,saved,zero_is_removed_background=True)


def test_ocelot_white_background_adapter_is_prospective_and_version_bound():
    sample=dict(dataset='ocelot',source=dict(source_version='1.0.1'),raw_to_compact={'1':1,'2':2})
    saved=np.array([[0,0,2]],np.uint8);white=np.array([[True,False,False]])
    historical,_=scoring_prediction(sample,saved,zero_is_removed_background=True,white=white)
    np.testing.assert_array_equal(historical,saved)
    prospective,_=scoring_prediction(sample,saved,zero_is_removed_background=True,white=white,ocelot_white_negative=True)
    np.testing.assert_array_equal(prospective,[[1,0,2]])
    sample['source']['source_version']='unknown'
    with pytest.raises(ValueError,match='verified1.0.1'):
        scoring_prediction(sample,saved,zero_is_removed_background=True,white=white,ocelot_white_negative=True)


def test_unknown_calibration_requires_explicit_pixel_only_contract():
    with pytest.raises(ValueError, match="explicitly"):
        geometry(dict(working_mpp_xy=None))
    result = geometry(dict(working_mpp_xy=None, pixel_only=True, physical_metrics_allowed=False))
    assert result["physical_spacing_xy"] is None
    assert result["units"] == "pixels"
    assert result["algorithm_spacing_xy"] == [1., 1.]
    measured = geometry(dict(working_mpp_xy=[.49, .51]))
    assert measured["algorithm_spacing_xy"] == [.49, .51]
    assert measured["physical_metrics_allowed"]
    with pytest.raises(ValueError, match="contradicts"):
        geometry(dict(working_mpp_xy=[.5, .5], pixel_only=True, physical_metrics_allowed=False))


def test_pixel_reporting_cannot_publish_physical_area_or_length_keys():
    raw = dict(analysis_mpp_xy=[1, 1], diffusion_length_um=16,
        regions={"small_le_400_um2": {"total_area_um2": 345}}, macro_iou=.6, seconds=.2)
    result = report_units(raw, "pixels")
    assert result["regions"]["small_le_400_px2"]["total_area_px2"] == 345
    assert result["diffusion_length_px"] == 16
    assert result["analysis_pixel_step_xy"] == [1, 1]
    assert result["macro_iou"] == raw["macro_iou"] and result["seconds"] == .2
    assert "diffusion_length_um" in raw  # Source diagnostics remain unchanged.


def test_rings_correct_background_and_deleted_tumor_have_different_outcomes():
    row = dict(dataset="rings-tumor", raw_to_compact={"0": 1, "1": 2})
    truth = np.array([[1, 2], [1, 2]], np.uint8)
    saved = np.array([[0, 0], [1, 2]], np.uint8)
    scored, policy = scoring_prediction(row, saved, zero_is_removed_background=True)
    assert np.array_equal(saved, [[0, 0], [1, 2]])
    assert np.array_equal(scored, [[1, 1], [1, 2]])
    assert (scored == truth).sum() == 3  # Deleted true tumor is still wrong.
    assert policy["converted_pixels"] == 2 and not policy["saved_prediction_modified"]
    with pytest.raises(ValueError, match="invariant"):
        scoring_prediction(row, saved, zero_is_removed_background=False)
    unchanged, _ = scoring_prediction(dict(dataset="bcss"), saved, zero_is_removed_background=True)
    assert np.array_equal(unchanged, saved)  # No general support/GT intersection.


def test_rings_all_tumor_crop_cannot_make_deleted_tumor_correct():
    row = dict(dataset="rings-tumor", raw_to_compact={"1": 1})
    saved = np.array([[0, 1]], np.uint8)
    scored, policy = scoring_prediction(row, saved, zero_is_removed_background=True)
    assert scored.tolist() == [[2, 1]]
    assert policy["virtual_negative_class"]
