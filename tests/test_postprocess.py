import numpy as np
import tifffile

from bora.postprocess import postprocess_streaming, smooth_and_fill


def test_cleanup_fills_small_hole_and_smooths_internal_staircase():
    labels = np.ones((128, 128), np.uint8)
    for y in range(128):
        labels[y, 64 + (4 if y % 12 < 6 else -4):] = 2
    labels[40:45, 30:35] = 0
    before = sum(np.count_nonzero(labels[y] != labels[y-1]) for y in range(1, 128))
    result = smooth_and_fill(labels, sigma=4, band_radius=12, max_hole_area=50)
    after = sum(np.count_nonzero(result[y] != result[y-1]) for y in range(1, 128))
    assert np.all(result[40:45, 30:35] == 1)
    assert after < before
    assert result[64, 20] == 1
    assert result[64, 110] == 2


def test_wide_symmetric_competition_rounds_an_internal_right_angle():
    yy, xx = np.mgrid[:256, :256]
    labels = np.where((xx < 128) | (yy < 128), 1, 2).astype(np.uint8)
    result = smooth_and_fill(labels, sigma=24, band_radius=80,
                             max_hole_area=0)
    assert result[135, 135] == 1  # The sharp concave corner is rounded.
    assert result[100, 100] == 1
    assert result[200, 200] == 2
    assert np.count_nonzero(result != labels) > 100


def test_streaming_cleanup_matches_full_frame_on_small_mask(tmp_path):
    labels = np.ones((128, 128), np.uint8)
    labels[:, 66:] = 2
    labels[35:39, 30:34] = 0
    source, output = tmp_path / "source.tif", tmp_path / "output.tif"
    tifffile.imwrite(source, labels)
    report = postprocess_streaming(source, output, block_size=64,
                                   sigma=4, band_radius=12,
                                   max_hole_area=100)
    assert np.array_equal(tifffile.imread(output), smooth_and_fill(
        labels, sigma=4, band_radius=12, max_hole_area=100))
    assert report["filled_background_pixels"] == 16


def test_cleanup_keeps_large_background_region_and_label_cores():
    labels = np.zeros((160, 160), np.uint8)
    labels[10:150, 10:80] = 1
    labels[10:150, 80:150] = 2
    labels[50:90, 35:75] = 0  # Larger than the hole threshold.
    result = smooth_and_fill(labels, max_hole_area=100)
    assert result[0, 0] == 0
    assert result[70, 50] == 0
    assert result[120, 30] == 1
    assert result[120, 130] == 2


def test_cleanup_absorbs_tiny_foreground_island_without_changing_large_regions():
    labels = np.ones((160, 160), np.uint8)
    labels[:, 80:] = 2
    labels[25:31, 25:31] = 2
    labels[90:125, 15:50] = 2
    result = smooth_and_fill(labels, sigma=0, band_radius=0,
                             max_hole_area=0, max_island_area=100)
    assert np.all(result[25:31, 25:31] == 1)
    assert np.all(result[90:125, 15:50] == 2)
    assert np.all(result[:, 100:] == 2)


def test_cleanup_preserves_island_touching_background():
    labels = np.ones((80, 80), np.uint8)
    labels[30:34, 30:34] = 2
    labels[29:30, 30:34] = 0
    result = smooth_and_fill(labels, sigma=0, band_radius=0,
                             max_hole_area=0, max_island_area=100)
    assert np.all(result[30:34, 30:34] == 2)


def test_streaming_island_cleanup_matches_full_frame(tmp_path):
    labels = np.ones((160, 160), np.uint8)
    labels[:, 80:] = 2
    labels[35:43, 35:43] = 2
    source, output = tmp_path / "source.tif", tmp_path / "output.tif"
    tifffile.imwrite(source, labels)
    postprocess_streaming(source, output, block_size=64, sigma=0, band_radius=0,
                          max_hole_area=0, max_island_area=100)
    assert np.array_equal(tifffile.imread(output), smooth_and_fill(
        labels, sigma=0, band_radius=0, max_hole_area=0, max_island_area=100))


def test_shape_filter_removes_round_island_and_preserves_thin_island():
    labels = np.ones((160, 160), np.uint8)
    labels[25:37, 25:37] = 2
    labels[90:94, 20:60] = 2
    result = smooth_and_fill(labels, sigma=0, band_radius=0,
                             max_hole_area=0, max_island_area=200,
                             min_circularity=0.7)
    assert np.all(result[25:37, 25:37] == 1)
    assert np.all(result[90:94, 20:60] == 2)
