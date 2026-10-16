import numpy as np
from scipy import ndimage as ndi

from bora.boundary import smooth_tile_boundaries


def test_color_guided_wand_rounds_tile_outline_and_improves_overlap():
    yy, xx = np.mgrid[:160, :160]
    truth = (xx - 80) ** 2 + (yy - 80) ** 2 <= 53 ** 2
    tile = np.zeros((160, 160), np.uint8)
    for y in range(0, 160, 20):
        for x in range(0, 160, 20):
            if truth[y:y+20, x:x+20].mean() >= 0.5:
                tile[y:y+20, x:x+20] = 1
    image = np.full((160, 160, 3), 245, np.uint8)
    image[truth] = (125, 66, 140)
    result = smooth_tile_boundaries(image, tile, 20)
    before = (tile.astype(bool) & truth).sum() / (tile.astype(bool) | truth).sum()
    after = (result.astype(bool) & truth).sum() / (result.astype(bool) | truth).sum()
    assert after > before
    assert result[80, 80] == 1
    assert result[0, 0] == 0


def test_local_competition_keeps_distinct_class_interiors():
    image = np.full((120, 120, 3), 235, np.uint8)
    labels = np.zeros((120, 120), np.uint8)
    labels[20:100, 20:60] = 1
    labels[20:100, 60:100] = 2
    image[20:100, 20:60] = (160, 80, 145)
    image[20:100, 60:100] = (170, 98, 135)
    result = smooth_tile_boundaries(image, labels, 40)
    assert result[60, 40] == 1
    assert result[60, 80] == 2
    assert set(np.unique(result)) == {0, 1, 2}


def test_image_guides_an_internal_class_border_off_the_tile_grid():
    yy, xx = np.mgrid[:192, :192]
    truth = np.where(xx < 96 + 13 * np.sin(yy / 20), 1, 2).astype(np.uint8)
    tile = np.where(xx < 96, 1, 2).astype(np.uint8)
    image = np.empty((192, 192, 3), np.uint8)
    image[truth == 1] = (160, 75, 150)
    image[truth == 2] = (220, 165, 125)
    result = smooth_tile_boundaries(image, tile, 48)
    assert np.mean(result == truth) > np.mean(tile == truth)
    assert np.count_nonzero(result != tile) > 100
    assert result[96, 24] == 1
    assert result[96, 168] == 2


def test_local_wand_moves_class_border_without_detached_color_islands():
    yy, xx = np.mgrid[:192, :192]
    truth = np.where(xx < 96 + 15 * np.sin(yy / 19), 1, 2).astype(np.uint8)
    tile = np.where(xx < 96, 1, 2).astype(np.uint8)
    image = np.empty((192, 192, 3), np.uint8)
    image[truth == 1] = (150, 75, 145)
    image[truth == 2] = (215, 155, 125)
    # Scattered misleading colors must not become new disconnected ROIs.
    image[40:150:9, 60:85:11] = (215, 155, 125)
    result = smooth_tile_boundaries(image, tile, 48)
    assert np.count_nonzero((result != tile) & (result > 0) & (tile > 0)) > 100
    assert np.mean(result == truth) > np.mean(tile == truth)
    for label in (1, 2):
        _, count = ndi.label(result == label)
        assert count == 1
