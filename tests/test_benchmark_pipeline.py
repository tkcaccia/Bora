import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

from bora.benchmark_pipeline import (
    compare_masks, discover_samples, expand_tile_grid, prepare_coarse_mask,
)


def _write_tiff(path, values):
    values = np.asarray(values, dtype=np.uint8)
    with rasterio.open(path, "w", driver="GTiff", width=values.shape[1],
                       height=values.shape[0], count=1, dtype="uint8",
                       transform=Affine.identity()) as dst:
        dst.write(values, 1)


def test_discover_samples_normalizes_cellphenotyper_suffix(tmp_path):
    _write_tiff(tmp_path / "case.tif", np.zeros((2, 2)))
    (tmp_path / "case_standard_grown_mask_smooth_class.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": []}))
    samples = discover_samples(tmp_path)
    assert len(samples) == 1
    assert samples[0].name == "case"


def test_discover_samples_accepts_separate_image_directory(tmp_path):
    annotations = tmp_path / "annotations"
    images = tmp_path / "images"
    annotations.mkdir(); images.mkdir()
    _write_tiff(images / "case.btf", np.zeros((2, 2)))
    (annotations / "case_standard_grown_mask_smooth_class.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": []}))
    samples = discover_samples(annotations, image_dir=images)
    assert samples[0].image == images / "case.btf"


def test_expand_tile_grid_and_compare(tmp_path):
    image = tmp_path / "image.tif"
    grid = tmp_path / "grid.tif"
    expanded = tmp_path / "expanded.tif"
    truth = tmp_path / "truth.tif"
    _write_tiff(image, np.zeros((5, 7)))
    _write_tiff(grid, [[1, 2, 3], [4, 5, 6]])
    expected = np.array([
        [1, 1, 1, 2, 2, 2, 3],
        [1, 1, 1, 2, 2, 2, 3],
        [1, 1, 1, 2, 2, 2, 3],
        [4, 4, 4, 5, 5, 5, 6],
        [4, 4, 4, 5, 5, 5, 6],
    ], dtype=np.uint8)
    _write_tiff(truth, expected)
    expand_tile_grid(grid, image, expanded, tile_width=3, tile_height=3)
    with rasterio.open(expanded) as src:
        np.testing.assert_array_equal(src.read(1), expected)
    metrics = compare_masks(expanded, truth, block_size=2)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_iou"] == 1.0
    assert len(metrics["per_class"]) == 6


def test_prepare_coarse_mask_uses_full_resolution_grid_without_copy(tmp_path):
    image = tmp_path / "image.tif"
    grid = tmp_path / "grid.tif"
    output = tmp_path / "unused.tif"
    _write_tiff(image, np.zeros((5, 7)))
    _write_tiff(grid, np.ones((5, 7)))
    assert prepare_coarse_mask(grid, image, output, 3, 3) == grid
    assert not output.exists()


def test_compare_masks_reports_multiclass_error(tmp_path):
    truth = tmp_path / "truth.tif"
    pred = tmp_path / "pred.tif"
    _write_tiff(truth, [[0, 1], [2, 2]])
    _write_tiff(pred, [[0, 1], [1, 2]])
    metrics = compare_masks(pred, truth, block_size=1)
    assert metrics["accuracy"] == 0.75
    assert metrics["per_class"][0]["label"] == 1
    assert metrics["per_class"][0]["fp"] == 1
    assert metrics["per_class"][1]["label"] == 2
    assert metrics["per_class"][1]["fn"] == 1
