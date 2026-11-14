import json

import numpy as np
import pytest

from bora.r_api import _case_plan


def test_case_plan_preserves_mask_and_full_glass_modes(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    pyvips = pytest.importorskip("pyvips")
    from PIL import Image

    image = tmp_path / "he.tif"
    Image.new("RGB", (450, 450), "white").save(image)
    tissue = tmp_path / "tissue.tif"
    pixels = np.zeros((450, 450), np.uint8)
    pixels[0, 0] = 1
    with rasterio.open(tissue, "w", driver="GTiff", width=450, height=450,
                       count=1, dtype="uint8") as dataset:
        dataset.write(pixels, 1)
    model = {"checkpoint_sha256": "model-hash", "config_sha256": "config-hash"}
    full = _case_plan(image, None, tmp_path / "full", tmp_path, model, [1, 2, 4], 3600)
    masked = _case_plan(image, tissue, tmp_path / "masked", tmp_path, model,
                        [1, 2, 4], 3600)
    assert full["source"]["selection"] == "all-original-image-tiles-including-glass"
    assert masked["source"]["selection"] == "intersects-supplied-tissue"
    assert full["source"]["support_positive_pixels"] is None
    assert full["source"]["selected_source_pixels"] == 450 * 450
    assert masked["source"]["support_positive_pixels"] == 1
    assert masked["source"]["selected_source_pixels"] == 1
    assert np.load(tmp_path / "full" / "eligible.npy").sum() == 9
    assert np.load(tmp_path / "masked" / "eligible.npy").sum() == 1
    assert len(full["worker_specs"]) == 3
    spec = json.loads((tmp_path / "masked" / "factor-4-WORKER_FROZEN.json").read_text())
    assert spec["padded_shape_yx"] == [896, 896]
    assert spec["base_grid_shape_yx"] == [4, 4]
