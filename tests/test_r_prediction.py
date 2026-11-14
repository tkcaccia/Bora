import json

import numpy as np

from bora.r_feature_preflight import sha256
from bora.r_prediction import output_support, source_coordinate_geojson


def test_output_support_matches_source_centres_and_full_glass(tmp_path):
    from PIL import Image

    source = {"image_shape_yx": [28, 28], "tissue_support": None,
              "tissue_support_sha256": None}
    full_path = tmp_path / "full.dat"
    result = output_support(source, (15, 15), (2, 2), full_path)
    full = np.memmap(full_path, mode="r", dtype=np.uint8, shape=(15, 15))
    assert result["positive_pixels"] == 14 * 14
    assert full[:14, :14].all()
    assert not full[14].any() and not full[:, 14].any()

    raster = tmp_path / "tissue.tif"
    image = np.zeros((28, 28), np.uint8)
    image[4:20, 4:20] = 1
    Image.fromarray(image).save(raster)
    source["tissue_support"] = str(raster)
    source["tissue_support_sha256"] = sha256(raster)
    raster_path = tmp_path / "raster.dat"
    result = output_support(source, (14, 14), (2, 2), raster_path)
    sampled = np.memmap(raster_path, mode="r", dtype=np.uint8, shape=(14, 14))
    assert result["positive_pixels"] == 8 * 8
    assert np.array_equal(sampled, image[1::2, 1::2])


def test_output_geojson_uses_same_pixel_edge_geometry(tmp_path):
    geo = tmp_path / "tissue.geojson"
    polygon = {"type": "Polygon", "coordinates": [
        [[4, 4], [20, 4], [20, 20], [4, 20], [4, 4]]]}
    geo.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": polygon}]}))
    source = {"image_shape_yx": [28, 28], "tissue_support": str(geo),
              "tissue_support_sha256": sha256(geo)}
    path = tmp_path / "geo.dat"
    result = output_support(source, (14, 14), (2, 2), path)
    pixels = np.memmap(path, mode="r", dtype=np.uint8, shape=(14, 14))
    assert result["positive_pixels"] == 64
    assert pixels[2:10, 2:10].all()
    assert pixels.sum() == 64


def test_streamed_source_geojson_uses_calibrated_pixel_edges(tmp_path):
    old = tmp_path / "grid.geojson"
    new = tmp_path / "native.geojson"
    header = {"type": "FeatureCollection", "coordinate_space": "native_image_pixel_edges",
              "native_shape_yx": [28, 28], "native_mpp_xy": [4, 4]}
    feature = {"type": "Feature", "properties": {"area_pixels": 100},
               "geometry": {"type": "Polygon", "coordinates": [
                   [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]}}
    old.write_text(json.dumps({**header, "features": [feature]},
                              separators=(",", ":")) + "\n")
    receipt = source_coordinate_geojson(old, new, (224, 224), (.5, .5), (4, 4))
    transformed = json.loads(new.read_text())
    assert receipt["feature_count"] == 1
    assert transformed["coordinate_space"] == "calibrated_source_image_pixel_edges"
    assert transformed["output_grid_shape_yx"] == [28, 28]
    assert transformed["features"][0]["geometry"]["coordinates"][0][1] == [80, 0]
    assert transformed["features"][0]["properties"]["area_pixels"] == 6400
