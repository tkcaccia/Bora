"""Preview backend checks use fabricated images and no expert reference."""

from pathlib import Path

import numpy as np
import pytest
import rasterio

from bora.r_prediction import _preview


def test_fast_preview_keeps_mask_grid_and_source_identity(tmp_path: Path):
    source = tmp_path / "synthetic-rgb.tif"
    mask = tmp_path / "synthetic-mask.tif"
    rgb = np.zeros((3, 256, 512), dtype=np.uint8)
    rgb[0] = np.arange(512, dtype=np.uint16)[None, :] % 256
    rgb[1] = np.arange(256, dtype=np.uint8)[:, None]
    rgb[2] = 180
    labels = np.zeros((64, 128), np.uint8)
    labels[8:50, 12:100] = 1
    with rasterio.open(source, "w", driver="GTiff", width=512, height=256,
                       count=3, dtype="uint8", tiled=True,
                       blockxsize=64, blockysize=64) as target:
        target.write(rgb)
    with rasterio.open(mask, "w", driver="GTiff", width=128, height=64,
                       count=1, dtype="uint8") as target:
        target.write(labels, 1)
    old = _preview(source, mask, [1], tmp_path / "old.png")
    fast = _preview(source, mask, [1], tmp_path / "fast.png",
                    backend="gdal-lanczos")
    assert old["shape_yx"] == fast["shape_yx"] == [256, 512]
    assert fast["source_thumbnail_backend"] == "gdal-lanczos"
    assert "source_thumbnail_backend" not in old
    assert (tmp_path / "old.png").exists() and (tmp_path / "fast.png").exists()
    with pytest.raises(ValueError, match="preview backend"):
        _preview(source, mask, [1], tmp_path / "bad.png", backend="unknown")
