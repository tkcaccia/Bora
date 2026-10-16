"""Synthetic checks for internal tissue-interface scoring."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from score_multisize_borders import empty_internal_score, score


def _write(path: Path, values: np.ndarray):
    with rasterio.open(path, "w", driver="GTiff", height=values.shape[0],
                       width=values.shape[1], count=1, dtype="uint16") as dst:
        dst.write(values, 1)


def test_perfect_contour_has_best_multiscale_scores(tmp_path):
    ground = np.ones((128, 128), np.uint16)
    ground[:, 64:] = 2
    tile = np.ones_like(ground)
    tile[:, 48:] = 2
    for name, values in (("truth", ground), ("tile", tile), ("vector", ground)):
        _write(tmp_path / f"{name}.tif", values)
    result = score(tmp_path / "truth.tif", tmp_path / "tile.tif",
                   tmp_path / "vector.tif", radius=32, block_size=64)
    assert result["bora"]["macro_iou"] == 1
    assert result["bora"]["boundary_f1_8px"] == 1
    assert result["bora"]["hausdorff95_pixels"] == 0
    assert result["tile"]["macro_iou"] < 1
    assert result["changed_between_tissue_labels_pixels"] > 0


def test_one_class_slide_has_no_internal_border_score(tmp_path):
    labels = np.ones((64, 64), np.uint16)
    for name in ("truth", "tile", "vector"):
        _write(tmp_path / f"{name}.tif", labels)
    result = score(tmp_path / "truth.tif", tmp_path / "tile.tif",
                   tmp_path / "vector.tif", radius=16, block_size=32)
    assert result["bora"]["macro_iou"] is None
    assert result["bora"]["boundary_f1_8px"] is None
    shortcut = empty_internal_score()
    assert shortcut["bora"]["macro_iou"] is None
    assert shortcut["bora"]["false_tile_imprint_edge_pixels"] == 0


def test_spurious_island_outside_border_band_is_reported(tmp_path):
    truth = np.ones((256, 256), np.uint16)
    truth[:, 64:] = 2
    candidate = truth.copy()
    candidate[180:190, 180:190] = 1
    for name, values in (("truth", truth), ("tile", truth),
                         ("vector", candidate)):
        _write(tmp_path / f"{name}.tif", values)
    result = score(tmp_path / "truth.tif", tmp_path / "tile.tif",
                   tmp_path / "vector.tif", radius=32, block_size=128)
    assert result["bora"]["off_band_internal_edge_pixels"] > 0
