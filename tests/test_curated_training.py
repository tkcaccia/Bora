import json

import numpy as np

from bora.curated_training import (
    appearance_features, rasterize_geojson, regularize_prediction,
    segmentation_metrics, write_scaled_geojson, predict_curated_tiled,
)


def test_metrics_are_class_aware():
    truth = np.array([[0, 1, 1], [0, 2, 2]], dtype=np.uint8)
    prediction = np.array([[0, 1, 2], [0, 2, 2]], dtype=np.uint8)
    result = segmentation_metrics(truth, prediction)
    assert result["foreground_iou"] == 1.0
    assert result["macro_iou"] < 1.0
    assert set(result["per_class"]) == {"1", "2"}


def test_features_treat_cluster_ids_as_categories():
    rgb = np.full((5, 6, 3), 128, np.uint8)
    coarse = np.array([[0, 1, 2, 2, 1, 0]] * 5, np.uint8)
    values = appearance_features(rgb, coarse)
    assert values.shape[:2] == coarse.shape
    # RGB/context features plus one channel for each of labels 0, 1 and 2.
    assert values.shape[2] >= 3 + 3
    assert np.isfinite(values).all()


def test_scaled_geojson_uses_full_resolution_coordinates(tmp_path):
    # Deliberately omit class 1: exported IDs must not be renumbered by feature
    # order when a small class disappears at calibration resolution.
    prediction = np.array([[0, 2], [0, 2]], np.uint16)
    output = tmp_path / "result.geojson"
    count = write_scaled_geojson(prediction, output, 8,
                                 {2: {"classification": {"name": "tissue"},
                                      "value": 99}},
                                 min_area=1, simplify=0)
    data = json.loads(output.read_text())
    assert count == 1
    assert data["features"][0]["properties"]["value"] == 2
    coordinates = data["features"][0]["geometry"]["coordinates"]
    flat = np.asarray(coordinates[0])
    assert flat[:, 0].max() == 16
    assert flat[:, 1].max() == 16


def test_regularization_removes_tiny_island():
    source = np.zeros((9, 9), np.uint16)
    source[2:7, 2:7] = 1
    source[0, 0] = 2
    result = regularize_prediction(source, min_component=3, radius=0)
    assert result[0, 0] == 0
    assert np.all(result[2:7, 2:7] == 1)


def test_tiled_prediction_preserves_global_coordinates_and_seams():
    class CoordinateModel:
        def predict(self, values):
            # The final feature is normalized x position.
            return (values[:, -1] > 0).astype(np.uint16)

    rgb = np.full((15, 17, 3), 128, np.uint8)
    coarse = np.ones((15, 17), np.uint16)
    result, _ = predict_curated_tiled(CoordinateModel(), rgb, coarse,
                                      tile_size=5, halo=20)
    assert np.all(result[:, :9] == 0)
    assert np.all(result[:, 9:] == 1)
