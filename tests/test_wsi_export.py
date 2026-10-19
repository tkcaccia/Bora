import json

import numpy as np
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import Polygon, shape

from bora.wsi_export import export, fit_window


def test_sparse_window_removes_tiny_parts_and_fills_holes_exactly():
    tile = np.zeros((256, 256), np.uint8)
    tile[20:60, 30:70] = 1
    tile[32:35, 42:45] = 0
    tile[200:203, 210:213] = 1
    report = {"native_mpp_xy": [.27, .27], "class_names": {"1": "tumor"}}
    features, final, audit = fit_window(tile, 1000, 2000, report)
    assert audit["sparse_lossless_contours"]
    assert audit["small_area_cleanup"]["isolated_regions_filled_with_background"] == 1
    assert audit["small_area_cleanup"]["filled_holes"] == 1
    assert np.all(final[32:35, 42:45] == 1)
    assert np.all(final[200:203, 210:213] == 0)
    rebuilt = rasterize(((feature["geometry"], feature["properties"]["value"])
                         for feature in features), out_shape=tile.shape,
                        transform=Affine(1, 0, 1000, 0, 1, 2000), dtype=np.uint8)
    np.testing.assert_array_equal(final, rebuilt)


def test_fragmented_window_uses_bounded_lossless_path():
    yy, xx = np.indices((128, 128))
    tile = np.where((xx // 4 + yy // 4) % 2 == 0, 1, 3).astype(np.uint8)
    report = {"native_mpp_xy": [.27, .27],
              "class_names": {"1": "tumor", "3": "stroma"}}
    _, final, audit = fit_window(tile, 0, 0, report)
    assert audit["fast_lossless_reason"] == "fragmented_contours"
    assert not np.any(final)  # Every four-connected 4×4 island has area 16.


def test_windowed_vector_export_matches_raster_and_removes_tiny_parts(tmp_path):
    shape_yx = (193, 281)
    source = np.zeros(shape_yx, np.uint8)
    source[12:180, 12:130] = 1
    source[12:180, 130:269] = 3
    source[30:33, 30:33] = 0  # enclosed nine-pixel hole
    source[185:188, 270:273] = 1  # isolated nine-pixel part
    report = {"native_shape_yx": list(shape_yx),
              "native_mpp_xy": [.27, .28],
              "class_names": {"1": "tumor", "3": "stroma"}}
    (tmp_path / "CLASSIFIED_COMPLETE.json").write_text(json.dumps(report))
    (tmp_path / "CLEANED_COMPLETE.json").write_text("{}")
    mm = np.memmap(tmp_path / "classified.cleaned.dat", mode="w+",
                   dtype=np.uint8, shape=shape_yx)
    mm[:] = source
    mm.flush()
    result = export(tmp_path, core=128, fit_core=64)
    assert result["windows"] == 6
    assert result["fit_core_pixels_for_new_windows"] == 64
    final = np.memmap(tmp_path / "classified.final.dat", mode="r",
                      dtype=np.uint8, shape=shape_yx)
    assert np.all(final[30:33, 30:33] == 1)
    assert np.all(final[185:188, 270:273] == 0)
    data = json.loads((tmp_path / "mask.geojson").read_text())
    rebuilt = rasterize(((feature["geometry"], feature["properties"]["value"])
                         for feature in data["features"]),
                        out_shape=shape_yx, dtype=np.uint8)
    np.testing.assert_array_equal(final, rebuilt)
    for feature in data["features"]:
        geometry = shape(feature["geometry"])
        parts = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
        assert all(part.area >= 20 for part in parts)
        assert all(Polygon(ring).area >= 20 for part in parts for ring in part.interiors)


def test_vector_export_uses_post_bora_background_mask_when_present(tmp_path):
    shape_yx=(96,96)
    report={"native_shape_yx":list(shape_yx),"native_mpp_xy":[.27,.27],
            "class_names":{"1":"tumor"}}
    (tmp_path/"CLASSIFIED_COMPLETE.json").write_text(json.dumps(report))
    (tmp_path/"CLEANED_COMPLETE.json").write_text("{}")
    (tmp_path/"BACKGROUND_COMPLETE.json").write_text("{}")
    cleaned=np.memmap(tmp_path/"classified.cleaned.dat",mode="w+",dtype=np.uint8,shape=shape_yx)
    cleaned[10:85,10:85]=1;cleaned.flush()
    background=np.memmap(tmp_path/"classified.background.dat",mode="w+",dtype=np.uint8,shape=shape_yx)
    background[:]=cleaned;background[35:60,35:60]=0;background.flush()
    audit=export(tmp_path,core=64,workers=2)
    final=np.memmap(tmp_path/"classified.final.dat",mode="r",dtype=np.uint8,shape=shape_yx)
    assert audit["vector_source_raster"]=="classified.background.dat"
    assert np.all(final[40:55,40:55]==0)
