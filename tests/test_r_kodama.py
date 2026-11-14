import csv

import numpy as np
import pytest

from bora.r_feature_preflight import sha256
from bora.r_kodama import _admitted_tile_rows, _cluster_labels, _group_head
from bora.supervised import _scores


def test_strict_background_admission_and_padding(tmp_path):
    # A 224-square tile has 50,176 pixels. Strictly <10% background means
    # at least 45,159 supported pixels. Right padding is never tissue.
    counts = np.asarray([[45158, 45159], [50176, 1000]], np.uint32)
    eligible = np.ones((2, 2), bool)
    count_path = tmp_path / 'counts.npy'
    eligible_path = tmp_path / 'eligible.npy'
    np.save(count_path, counts)
    np.save(eligible_path, eligible)
    source = {'tissue_pixel_counts_path': str(count_path),
              'tissue_pixel_counts_sha256': sha256(count_path),
              'eligible_path': str(eligible_path),
              'eligible_sha256': sha256(eligible_path),
              'tile_grid_shape_yx': [2, 2], 'base_grid_shape_yx': [2, 2],
              'image_shape_yx': [448, 250], 'tissue_support': 'mask.tif'}
    # The second column exists only as 26 native pixels; its count cannot
    # exceed that width, even if it had all 224 image rows.
    with pytest.raises(ValueError, match='counts or geometry'):
        _admitted_tile_rows(source)
    counts = np.asarray([[45158, 5000], [50176, 1000]], np.uint32)
    np.save(count_path, counts)
    source['tissue_pixel_counts_sha256'] = sha256(count_path)
    rows = _admitted_tile_rows(source)
    assert rows.tolist() == [[1, 0]]
    counts[0, 0] = 45159
    np.save(count_path, counts)
    source['tissue_pixel_counts_sha256'] = sha256(count_path)
    assert _admitted_tile_rows(source).tolist() == [[0, 0], [1, 0]]


def test_joint_group_table_requires_exact_row_identity(tmp_path):
    table = tmp_path / 'groups.csv'
    with table.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['row_index', 'group'])
        writer.writerows([[0, 1], [1, 2], [2, 1]])
    assert _cluster_labels(table, 3).tolist() == [1, 2, 1]
    with table.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['row_index', 'group'])
        writer.writerows([[0, 1], [2, 2], [1, 1]])
    with pytest.raises(ValueError, match='rows or group IDs'):
        _cluster_labels(table, 3)


def test_anonymous_head_selects_full_channel_contrasts():
    rng = np.random.default_rng(20261114)
    features = rng.normal(0, .1, size=(12, 1280)).astype(np.float32)
    labels = np.repeat([1, 2, 3], 4)
    features[labels == 1, 100] += 3
    features[labels == 2, 700] += 3
    features[labels == 3, 1100] += 3
    head = _group_head(features, labels, selected_channels=8)
    assert head['classes'].tolist() == [1, 2, 3]
    assert {100, 700, 1100} <= set(head['selected'])
    assert head['coef'].shape == (3, 8)
    assert head['training_tile_counts'].tolist() == [4, 4, 4]


def test_binary_group_head_matches_nearest_centroid(tmp_path):
    rng = np.random.default_rng(42)
    features = rng.normal(size=(8, 1280)).astype(np.float32)
    labels = np.repeat([1, 2], 4)
    features[labels == 1, 100] += 4
    features[labels == 2, 900] += 4
    head = _group_head(features, labels, selected_channels=16)
    grid = features.reshape(2, 4, 1280).astype(np.float16)
    path = tmp_path / 'scores.npy'
    _scores(grid, head, path)
    projected = np.load(path)
    z = (grid.astype(np.float32)[..., head['selected']] - head['mean']) / head['std']
    direct = np.stack([np.sum((z - center) ** 2, axis=-1)
                       for center in head['centroids']])
    assert np.array_equal(np.argmin(direct, axis=0), np.argmax(projected, axis=0))
