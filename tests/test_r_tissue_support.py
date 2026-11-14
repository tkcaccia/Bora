"""A reviewed polygon must preserve exact tissue, void and source identity."""

import json

import numpy as np
from PIL import Image
import rasterio
from shapely.geometry import Polygon, mapping

from bora.r_feature_preflight import sha256
from bora.r_tissue_support import materialize_reviewed_tissue_support


def test_reviewed_tissue_geojson_roundtrips_a_hole_and_rejects_wrong_source(tmp_path):
    image = tmp_path / "calibrated-he.tif"
    Image.fromarray(np.full((448, 448, 3), (180, 86, 131), np.uint8)).save(image)
    outer = Polygon([(20, 20), (300, 20), (300, 300), (20, 300)],
                    holes=[[(100, 100), (200, 100), (200, 200), (100, 200)]])
    island = Polygon([(320, 320), (400, 320), (400, 400), (320, 400)])
    document = {"type": "FeatureCollection",
                "coordinate_space": "calibrated_image_pixel_edges",
                "image_shape_yx": [448, 448],
                "source_image_sha256": sha256(image),
                "review_status": "human_reviewed",
                "features": [{"type": "Feature", "geometry": mapping(p),
                              "properties": {}} for p in (outer, island)]}
    geojson = tmp_path / "reviewed.geojson"
    geojson.write_text(json.dumps(document) + "\n")
    receipt = materialize_reviewed_tissue_support(
        image, geojson, tmp_path / "output", minimum_free_gib=0)
    assert receipt["status"] == "VERIFIED"
    assert receipt["tissue_pixels"] == 280 * 280 - 100 * 100 + 80 * 80
    assert receipt["raster_vector_pixel_agreement"]
    with rasterio.open(receipt["tissue_mask"]) as source:
        mask = source.read(1)
    assert mask[50, 50] == mask[350, 350] == 255
    assert mask[150, 150] == mask[0, 0] == 0

    document["source_image_sha256"] = "0" * 64
    geojson.write_text(json.dumps(document) + "\n")
    try:
        materialize_reviewed_tissue_support(
            image, geojson, tmp_path / "wrong-source", minimum_free_gib=0)
    except ValueError as error:
        assert "not bound" in str(error)
    else:
        raise AssertionError("Wrong-image tissue geometry was admitted")
    assert not (tmp_path / "wrong-source").exists()
