"""Regression for reviewed tissue boundaries that cross fitted polygons."""

import hashlib
import json

import numpy as np
import pytest
from affine import Affine
from rasterio.features import rasterize
from shapely.affinity import translate
from shapely.geometry import box, mapping, shape

from bora.support_vector_clip import clip_vectors_to_reviewed_support


def test_fitted_vector_is_clipped_to_exact_reviewed_support(tmp_path):
    tissue = box(5, 5, 25, 25)
    geojson = tmp_path / "tissue.geojson"
    geojson.write_text(json.dumps({
        "type": "FeatureCollection",
        "coordinate_system": "calibrated_image_pixel_xy",
        "image_shape_yx": [32, 32],
        "mask_shape_yx": [32, 32],
        "features": [{"type": "Feature", "geometry": mapping(tissue),
                      "properties": {"class": "tissue"}}],
    }))
    expected = rasterize(
        [(translate(tissue, xoff=.499999, yoff=.499999), 1)],
        out_shape=(32, 32), transform=Affine.identity(), dtype=np.uint8)
    identity = {"quality_control": {
        "mode": "background-only", "tissue_geojson_path": str(geojson),
        "tissue_geojson_sha256": hashlib.sha256(geojson.read_bytes()).hexdigest(),
        "calibrated_frame_shape_yx": [32, 32], "mask_shape_yx": [32, 32],
    }}
    fitted = {"features": [{"type": "Feature", "geometry": mapping(box(2, 2, 21, 21)),
                            "properties": {"value": 1}}]}
    clipped, receipt = clip_vectors_to_reviewed_support(
        fitted, identity, expected, (.5, .5), minimum_area_um2=1)
    result = rasterize([(shape(clipped["features"][0]["geometry"]), 1)],
                       out_shape=(32, 32), transform=Affine.identity(), dtype=np.uint8)
    assert receipt["expanded_support_raster_exact"] is True
    assert receipt["classes_preserved"] == [1]
    assert np.count_nonzero((result != 0) & (expected == 0)) == 0

    identity["quality_control"]["tissue_geojson_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="GeoJSON changed"):
        clip_vectors_to_reviewed_support(fitted, identity, expected, (.5, .5), 1)


def test_noninteger_nearest_expansion_keeps_exact_reviewed_pixels(tmp_path):
    # A noninteger thumbnail resize makes the right boundary step at
    # different output pixels in different rows; no global half-pixel shift
    # of the source vector can reproduce this support.
    tissue = box(2, 2, 8, 8)
    geojson = tmp_path / "reviewed.geojson"
    geojson.write_text(json.dumps({
        "type": "FeatureCollection",
        "coordinate_system": "calibrated_image_pixel_xy",
        "image_shape_yx": [20, 20], "mask_shape_yx": [7, 7],
        "features": [{"type": "Feature", "geometry": mapping(tissue),
                      "properties": {"class": "tissue"}}],
    }))
    support = np.zeros((20, 20), np.uint8)
    support[2:13, 2:13] = 1
    support[8:13, 13] = 1
    identity = {"quality_control": {
        "mode": "background-only", "tissue_geojson_path": str(geojson),
        "tissue_geojson_sha256": hashlib.sha256(geojson.read_bytes()).hexdigest(),
        "calibrated_frame_shape_yx": [20, 20], "mask_shape_yx": [7, 7],
        "geojson_is_authoritative": True,
        "mapping": "nearest neighbor over complete padded calibrated image; crop to analysis frame",
    }}
    fitted = {"features": [{"type": "Feature",
                            "geometry": mapping(box(0, 0, 18, 18)),
                            "properties": {"value": 1}}]}
    result, receipt = clip_vectors_to_reviewed_support(
        fitted, identity, support, (.5, .5), minimum_area_um2=1)
    raster = rasterize([(shape(result["features"][0]["geometry"]), 1)],
                       out_shape=support.shape, transform=Affine.identity(),
                       dtype=np.uint8)
    assert receipt["support_vectorization"] == "exact_expanded_reviewed_raster"
    assert receipt["shift_xy_pixels"] is None
    assert np.array_equal(raster, support)
