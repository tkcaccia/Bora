"""Boundary evaluation keeps interiors out of the acceptance signal."""
import importlib.util
from pathlib import Path

import numpy as np
import rasterio


spec = importlib.util.spec_from_file_location(
    "score_216_borders",
    Path(__file__).resolve().parents[1] / "scripts" / "score_216_borders.py")
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


def _write(path, data):
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[0],
                       width=data.shape[1], count=1, dtype=data.dtype) as out:
        out.write(data, 1)


def test_border_score_ignores_unchanged_interior(tmp_path):
    truth = np.zeros((48, 48), np.uint8)
    truth[8:40, 8:24] = 1
    truth[8:40, 24:40] = 2
    tile = truth.copy()
    candidate = truth.copy()
    candidate[8:40, 24:27] = 1  # Move only the internal class border.
    for name, data in (("truth", truth), ("tile", tile),
                       ("candidate", candidate)):
        _write(tmp_path / f"{name}.tif", data)
    result = scorer.score(tmp_path / "truth.tif", tmp_path / "tile.tif",
                          tmp_path / "candidate.tif", radius=3, block_size=16)
    assert result["internal"]["bora"]["macro_iou"] < 1
    assert result["internal"]["tile"]["macro_iou"] == 1
    assert result["internal"]["bora"]["pixels"] < truth.size
    assert result["external"]["bora"]["pixels"] > 0
    assert result["changed_between_tissue_labels_pixels"] == 96
    assert result["internal"]["bora"]["boundary_f1_8px"] == 1


def test_false_tile_imprint_detects_stale_internal_step(tmp_path):
    truth = np.ones((64, 64), np.uint8)
    truth[:, 36:] = 2
    tile = np.ones_like(truth)
    tile[:, 20:] = 2
    for name, data in (("truth", truth), ("tile", tile),
                       ("candidate", truth)):
        _write(tmp_path / f"{name}.tif", data)
    result = scorer.score(tmp_path / "truth.tif", tmp_path / "tile.tif",
                          tmp_path / "candidate.tif", radius=20,
                          block_size=32, internal_only=True)
    assert result["internal"]["tile"]["false_tile_imprint_fraction"] > 0
    assert result["internal"]["bora"]["false_tile_imprint_fraction"] == 0
