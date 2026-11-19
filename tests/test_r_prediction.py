import json

import numpy as np

from bora.r_feature_preflight import sha256
from bora.r_prediction import (finalize_projected_mask, output_support,
                               source_coordinate_geojson)


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


def test_vector_export_preserves_a_small_zero_support_hole(tmp_path):
    from rasterio.features import rasterize
    from rasterio.transform import Affine

    support = np.ones((32, 32), np.uint8)
    support[16, 16] = 0
    projected = np.ones((32, 32), np.uint8)
    projected[:, 16:] = 2
    projected[support == 0] = 0
    support_path = tmp_path / "support.dat"
    projected_path = tmp_path / "projected.dat"
    support.tofile(support_path)
    projected.tofile(projected_path)
    out = tmp_path / "final"
    result = finalize_projected_mask(
        projected_path, support_path, support.shape, (4., 4.), [1, 2],
        "a" * 64, "b" * 64, out,
        minimum_area_um2=2000., contour_sigma_um=6.)
    final = np.fromfile(result["final_mask_path"], dtype=np.uint8).reshape(support.shape)
    exact = np.fromfile(out / "vectors-output-grid/exact/mask.uint8.dat",
                        dtype=np.uint8).reshape(support.shape)
    geo = json.loads((out / "vectors-output-grid/exact/mask.geojson").read_text())
    drawn = rasterize(((feature["geometry"], int(feature["properties"]["value"]))
                       for feature in geo["features"]),
                      out_shape=support.shape, transform=Affine.identity(),
                      fill=0, dtype=np.uint8)
    assert final[16, 16] == exact[16, 16] == drawn[16, 16] == 0
    assert np.array_equal(final, exact)
    assert np.array_equal(final, drawn)


def test_final_geometry_removes_diagonal_only_subthreshold_islands(tmp_path):
    from bora.supervised_wsi import area_repair

    support = np.ones((32, 32), np.uint8)
    projected = np.ones((32, 32), np.uint8)
    projected[2:10, 2:10] = 2
    projected[10:18, 10:18] = 2
    first, first_report = area_repair(projected, support, tmp_path / "eight.dat",
                                      (4., 4.), 2000., connectivity=8)
    final, final_report = area_repair(first, support, tmp_path / "four.dat",
                                      (4., 4.), 2000., connectivity=4)
    assert first_report["positive_parts"] == 0
    assert final_report["positive_parts"] == 2
    assert not np.any(final == 2)
