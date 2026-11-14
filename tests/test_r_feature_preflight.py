import json

import numpy as np
import pytest

from bora.r_feature_preflight import _tile_counts, tissue_tile_counts


def test_exact_counts_include_padding_as_background():
    mask = np.ones((3, 3), dtype=np.uint8)
    assert _tile_counts(mask[:2], 3, 2).tolist() == [4, 2]
    assert _tile_counts(mask[2:], 3, 2).tolist() == [2, 1]
    assert tissue_tile_counts((3, 3), None, tile=2).tolist() == [[4, 2], [2, 1]]


def test_raster_support_excludes_glass(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    path = tmp_path / "support.tif"
    with rasterio.open(path, "w", driver="GTiff", width=3, height=3,
                       count=1, dtype="uint8") as dataset:
        array = np.zeros((3, 3), np.uint8)
        array[0, 0] = 1
        dataset.write(array, 1)
    assert tissue_tile_counts((3, 3), path, tile=2).tolist() == [[1, 0], [0, 0]]


def test_geojson_support_uses_image_pixel_edges(tmp_path):
    pytest.importorskip("rasterio")
    path = tmp_path / "support.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {}, "geometry": {
            "type": "Polygon", "coordinates": [[[0, 0], [2, 0], [2, 2],
                                                 [0, 2], [0, 0]]]
        }}]}))
    assert tissue_tile_counts((3, 3), path, tile=2).tolist() == [[4, 0], [0, 0]]
