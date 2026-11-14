import json

import numpy as np
import pytest

from bora.r_training import _annotation_grid, _sample_case, train
from bora.r_feature_preflight import sha256


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


def test_fit_uses_frozen_factor_one_head_with_multiscale_cache(tmp_path):
    from PIL import Image

    image = tmp_path / "he.tif"
    Image.new("RGB", (28, 28), "white").save(image)
    labels = np.ones((28, 28), np.uint8)
    labels[:, 14:] = 2
    annotation = tmp_path / "classes.tif"
    Image.fromarray(labels).save(annotation)
    preparation = tmp_path / "features"
    preparation.mkdir()
    case_dir = preparation / "image-000"
    case_dir.mkdir()
    feature_rows = []
    for factor, shape in ((1, (2, 2, 1280)), (2, (1, 1, 1280))):
        folder = case_dir / f"factor-{factor}"
        folder.mkdir()
        values = np.zeros(shape, np.float16)
        if factor == 1:
            values[:, 1, :16] = 2
        else:
            values[:] = 1000
        feature_path = folder / "features.npy"
        np.save(feature_path, values)
        receipt_path = folder / "FEATURES_COMPLETE.json"
        receipt_path.write_text(json.dumps({"status": "COMPLETE", "factor": factor,
            "channels": 1280, "features_path": str(feature_path),
            "features_sha256": sha256(feature_path), "shape_yxc": list(shape)}))
        feature_rows.append({"case_dir": str(case_dir), "factor": factor,
                             "receipt_path": str(receipt_path),
                             "receipt_sha256": sha256(receipt_path)})
    (preparation / "PREPARATION_COMPLETE.json").write_text(json.dumps({
        "format": "bora-r-preparation-v1", "status": "COMPLETE",
        "cases": [{"case_dir": str(case_dir), "source": {
            "image": str(image), "image_sha256": sha256(image),
            "image_shape_yx": [28, 28]}}],
        "scales": feature_rows,
    }))
    result = train([image], [annotation], preparation, None, tmp_path / "model",
                   max_tokens_per_class=2, channels_per_class=4)
    assert result["factor_scales"] == [1]
    assert result["classes"] == [1, 2]
    with np.load(result["model_path"], allow_pickle=False) as fitted:
        assert fitted["selected"].max() < 1280
