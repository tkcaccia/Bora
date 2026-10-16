import numpy as np
import pytest
from scipy.stats import kruskal

from experiments.gigatime_flash.top3_rgb import bh_adjust, markers_to_rgb, rank_markers


def test_bh_preserves_input_order_and_adjusts_the_whole_family():
    assert bh_adjust([.04, .001, .02, .9]) == pytest.approx([.05333333333, .004, .04, .9])


def test_ranking_matches_scipy_and_shifts_keep_spatial_structure():
    rng = np.random.default_rng(303)
    tiles = []
    for y in range(6):
        for x in range(6):
            group = 1 if x < 3 else 2
            tiles.append({"x": x*20, "y": y*20, "label": group,
                          "central_mean": [float(group*.3+rng.uniform(0, .02)),
                                           float(rng.uniform()), float(rng.uniform()), 0.]})
    result = rank_markers(tiles, ["strong", "noise1", "noise2", "DAPI"],
                          ["strong", "noise1", "noise2"], 20)
    top = result["ranking"][0]
    assert top["channel"] == "strong"
    samples = [[t["central_mean"][0] for t in tiles if t["label"] == g] for g in [1, 2]]
    assert top["nominal_p"] == pytest.approx(kruskal(*samples).pvalue)
    # There are only six distinct horizontal shifts, repeated vertically.
    # The original and reversed half-plane groups have identical H statistics.
    assert top["spatial_shift_tail_fraction"] == pytest.approx(1/3)
    assert result["spatial_diagnostic"]["shifts_including_identity"] == 36
    assert "DAPI" not in result["selected_channels"]
    assert result["fine_reference_mask_used_for_selection"] is False


def test_sparse_grid_is_rejected_instead_of_filling_missing_tiles():
    tiles = [{"x": (i % 6)*20, "y": (i//6)*20, "label": 1+i % 2,
              "central_mean": [.2, .3, .4]} for i in range(35)]
    with pytest.raises(ValueError, match="complete rectangular"):
        rank_markers(tiles, ["a", "b", "c"], ["a", "b", "c"], 20)


def test_constant_markers_do_not_produce_false_significance_or_nan_metadata():
    tiles = [{"x": (i % 4)*20, "y": (i//4)*20, "label": 1+i % 2,
              "central_mean": [.2, .3, .4]} for i in range(16)]
    result = rank_markers(tiles, ["a", "b", "c"], ["a", "b", "c"], 20)
    assert all(row["nominal_p"] == 1 for row in result["ranking"])
    assert all(row["spatial_shift_tail_fraction"] == 1 for row in result["ranking"])
    assert all(row["pearson"] is None for row in result["selected_tile_correlations"])


def test_rgb_preserves_frame_and_requested_channel_identity():
    values = np.zeros((12, 13, 4), np.float32)
    values[..., 0], values[..., 1], values[..., 2], values[..., 3] = .1, .4, .8, .2
    rgb, meta = markers_to_rgb(values, [2, 0, 1], np.ones((12, 13), bool), robust=False)
    assert rgb.shape == (12, 13, 3)
    assert rgb.dtype == np.uint8
    np.testing.assert_array_equal(rgb[0, 0], [204, 26, 102])
    assert meta["channel_indices_rgb"] == [2, 0, 1]
    assert meta["fine_reference_used"] is False


def test_rgb_scaling_uses_coarse_support_without_expanding_constant_noise():
    values = np.full((10, 10, 3), .2, np.float32)
    values[0] = 1
    support = np.ones((10, 10), bool)
    support[0] = False
    rgb, meta = markers_to_rgb(values, [0, 1, 2], support)
    assert meta["effective_range"] == pytest.approx([.01]*3)
    assert (rgb[1:] == 0).all()
    assert (rgb[0] == 255).all()
