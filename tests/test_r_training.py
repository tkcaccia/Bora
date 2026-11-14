import json

import numpy as np
import pytest

from bora.r_training import _annotation_grid, _sample_case


def test_class_raster_reads_exact_token_centres_and_requires_mapping(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import Affine

    labels = np.zeros((31, 31), np.uint8)
    labels[7, 7] = 11
    labels[7, 21] = 22
    labels[21, 7] = 11
    labels[21, 21] = 22
    labels[6, 6] = 99  # adjacent pixel cannot change the centre label
    target = tmp_path / "target.tif"
    with rasterio.open(target, "w", driver="GTiff", width=31, height=31,
                       count=1, dtype="uint8", transform=Affine.identity()) as dataset:
        dataset.write(labels, 1)
    assert np.array_equal(_annotation_grid(target, (31, 31),
                                          {"11": 1, "22": 2}), [[1, 2], [1, 2]])
    with pytest.raises(ValueError, match="unmapped"):
        _annotation_grid(target, (31, 31), {"11": 1})


def test_class_geojson_mapping_and_overlap_guard(tmp_path):
    geo = tmp_path / "classes.geojson"
    square_a = {"type": "Polygon", "coordinates": [[[0, 0], [14, 0],
                  [14, 28], [0, 28], [0, 0]]]}
    square_b = {"type": "Polygon", "coordinates": [[[14, 0], [28, 0],
                  [28, 28], [14, 28], [14, 0]]]}
    features = [{"type": "Feature", "properties": {"class": key},
                 "geometry": geometry}
                for key, geometry in (("tumor", square_a), ("stroma", square_b))]
    geo.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    assert np.array_equal(_annotation_grid(geo, (28, 28),
                                          {"tumor": 1, "stroma": 2}), [[1, 2], [1, 2]])
    features[1]["geometry"] = square_a
    geo.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    with pytest.raises(ValueError, match="overlap"):
        _annotation_grid(geo, (28, 28), {"tumor": 1, "stroma": 2})


def test_sample_case_rejects_nonfinite_features():
    grid = np.ones((2, 2, 1280), np.float16)
    grid[0, 0] = np.nan
    labels = np.array([[1, 1], [2, 2]], np.uint8)
    features, targets, counts = _sample_case({1: grid}, labels, 2,
                                             np.random.default_rng(3))
    assert features.shape == (3, 1280)
    assert sorted(targets.tolist()) == [1, 2, 2]
    assert counts["1"]["sampled_nonfinite_tokens"] == 1
