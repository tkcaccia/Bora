import numpy as np
import pytest

from bora.boundary import BoundaryConfig, smooth_tile_boundaries
from bora.virtual_markers import sampling_plan, select_markers, standardize_markers


def test_sampling_preserves_physical_field_with_integer_pixel_grid():
    plan = sampling_plan((1166, 1224), (0.5, 0.5), 128/556)
    nh, nw = plan["inference_shape_yx"]
    mx, my = plan["effective_inference_mpp_xy"]
    assert nw*mx == pytest.approx(612)
    assert nh*my == pytest.approx(583)
    assert plan["optical_resolution_improved_by_resampling"] is False
    assert plan["native_shape_yx"] == [1166, 1224]


@pytest.mark.parametrize("mpp", [0, -1, float("nan")])
def test_sampling_rejects_unknown_or_invalid_scale(mpp):
    with pytest.raises(ValueError):
        sampling_plan((100, 100), (0.5, 0.5), mpp)


def test_coarse_tile_contrasts_exclude_background_and_nuclear_channel():
    coarse = np.ones((80, 80), np.uint8)
    coarse[:, 40:] = 2
    maps = np.zeros((80, 80, 5), np.float32)
    maps[coarse == 1] = [1, 1, 1, .9, .5]
    maps[coarse == 2] = [0, 0, 0, .1, .5]
    result = select_markers(maps, coarse, 20,
                           ["DAPI", "TRITC", "Cy5", "CK", "flat"], top_k=2)
    assert result["selected_channels"] == ["CK"]
    assert result["group_tile_counts"] == {"1": 8, "2": 8}
    assert result["reference_mask_used_for_selection"] is False
    assert len(result["tiles"]) == 16


def test_marker_wand_uses_channels_when_rgb_has_no_interface():
    yy, xx = np.mgrid[:192, :192]
    truth = np.where(xx < 96+13*np.sin(yy/20), 1, 2).astype(np.uint8)
    coarse = np.where(xx < 96, 1, 2).astype(np.uint8)
    rgb = np.full((192, 192, 3), 180, np.uint8)
    maps = np.where(truth[..., None] == 1, .9, .1).astype(np.float32)
    features, _ = standardize_markers(maps, coarse > 0)
    result = smooth_tile_boundaries(rgb, coarse, 48, feature_image=features)
    assert np.mean(result == truth) > np.mean(coarse == truth) + .02
    assert np.all(result[:, :40] == 1)
    assert np.all(result[:, 150:] == 2)
    assert set(np.unique(result)) == {1, 2}


@pytest.mark.parametrize("features", [np.zeros((32, 31, 1)), np.full((32, 32, 1), np.nan),
                                      np.zeros((32, 32, 0))])
def test_wand_rejects_misaligned_or_invalid_features(features):
    with pytest.raises(ValueError):
        smooth_tile_boundaries(np.zeros((32, 32, 3), np.uint8),
                               np.ones((32, 32), np.uint8), 16, feature_image=features)
