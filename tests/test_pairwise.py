import numpy as np

from bora.pairwise import PairwiseConfig, compete_pair


def test_pairwise_contest_moves_boundary_to_image_and_preserves_cores():
    yy, xx = np.mgrid[:192, :192]
    truth = np.where(xx < 96 + 14 * np.sin(yy / 20), 1, 2).astype(np.uint8)
    tiled = np.where(xx < 96, 1, 2).astype(np.uint8)
    image = np.empty((192, 192, 3), np.uint8)
    image[truth == 1] = (150, 70, 145)
    image[truth == 2] = (215, 150, 125)
    result = compete_pair(image, tiled,
                          PairwiseConfig(core_distance=22, search_radius=32))
    assert np.mean(result == truth) > np.mean(tiled == truth)
    assert result[90, 20] == 1 and result[90, 170] == 2
    assert np.count_nonzero(result != tiled) > 100


def test_pairwise_leaves_junctions_and_external_edges_unchanged():
    image = np.full((64, 64, 3), 100, np.uint8)
    labels = np.ones((64, 64), np.uint8)
    labels[:, 32:] = 2
    labels[10:15, 10:15] = 3
    assert np.array_equal(compete_pair(image, labels), labels)
    labels[10:15, 10:15] = 0
    assert np.array_equal(compete_pair(image, labels), labels)
