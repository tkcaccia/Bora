"""Materialize a reviewed pixel-edge tissue polygon as reusable raster support.

This function does not infer a tissue boundary from the image. A human must
review the GeoJSON before benchmark use. The raster and reviewed vector are
bound to one calibrated H&E source and checked for exact pixel agreement.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import numpy as np

from .r_feature_preflight import inspect_image, sha256, tissue_tile_counts


def materialize_reviewed_tissue_support(
    image_path: str | Path, reviewed_geojson: str | Path, output_dir: str | Path,
    *, minimum_free_gib: float = 1.0, block_rows: int = 512,
) -> dict:
    """Write a ZSTD TIFF with the exact pixels selected by reviewed GeoJSON.

    GeoJSON coordinates are calibrated image pixel edges, with origin at the
    upper left. The document must contain ``coordinate_space``,
    ``image_shape_yx``, ``source_image_sha256``, and an explicit
    ``review_status`` so a polygon cannot be silently applied to a different
    image or mistaken for an accepted automatic candidate. Positive pixels
    are 255.
    """
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    from rasterio.windows import Window
    from shapely.geometry import shape

    image_path = Path(image_path).resolve()
    reviewed_geojson = Path(reviewed_geojson).resolve()
    output_dir = Path(output_dir).resolve()
    if not reviewed_geojson.is_file() or not image_path.is_file():
        raise FileNotFoundError("Calibrated image and reviewed GeoJSON are required")
    if output_dir.exists() or not output_dir.parent.is_dir():
        raise FileExistsError("Use a fresh output directory with an existing parent")
    if (not math.isfinite(minimum_free_gib) or minimum_free_gib < 0 or
            not isinstance(block_rows, int) or not 1 <= block_rows <= 4096):
        raise ValueError("Finite disk reserve and bounded block height required")
    height, width = inspect_image(image_path)
    source_hash = sha256(image_path)
    vector_hash = sha256(reviewed_geojson)
    document = json.loads(reviewed_geojson.read_text())
    if (document.get("type") != "FeatureCollection" or
            document.get("coordinate_space") != "calibrated_image_pixel_edges" or
            document.get("image_shape_yx") != [height, width] or
            document.get("source_image_sha256") != source_hash or
            document.get("review_status") != "human_reviewed"):
        raise ValueError("Reviewed geometry is not bound to this calibrated image")
    features = document.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError("Reviewed tissue GeoJSON must contain polygons")
    geometries = []
    for feature in features:
        geometry = shape(feature["geometry"])
        if (geometry.geom_type not in ("Polygon", "MultiPolygon") or
                geometry.is_empty or not geometry.is_valid):
            raise ValueError("Reviewed tissue geometry is empty or invalid")
        left, top, right, bottom = geometry.bounds
        if left < 0 or top < 0 or right > width or bottom > height:
            raise ValueError("Reviewed tissue geometry lies outside the image")
        geometries.append(feature["geometry"])
    projected = height * width + (1 << 30)
    free = shutil.disk_usage(output_dir.parent).free
    if free - projected < minimum_free_gib * (1 << 30):
        raise OSError("Tissue-mask materialization would breach disk reserve")
    output_dir.mkdir()
    target_path = output_dir / "tissue-support.tif"
    temporary = output_dir / "tissue-support.partial.tif"

    def expected_strip(top: int, rows: int) -> np.ndarray:
        return rasterize(((geometry, 255) for geometry in geometries),
                         out_shape=(rows, width), fill=0, dtype=np.uint8,
                         transform=Affine.translation(0, top), all_touched=False)

    with rasterio.open(temporary, "w", driver="GTiff", width=width,
                       height=height, count=1, dtype="uint8", tiled=True,
                       blockxsize=512, blockysize=512, compress="zstd",
                       BIGTIFF="IF_SAFER") as target:
        for top in range(0, height, block_rows):
            rows = min(block_rows, height - top)
            target.write(expected_strip(top, rows), 1,
                         window=Window(0, top, width, rows))
    os.replace(temporary, target_path)
    positive = 0
    with rasterio.open(target_path) as source:
        if (source.height, source.width, source.count, source.dtypes[0]) != \
                (height, width, 1, "uint8"):
            raise ValueError("Tissue mask raster schema differs")
        for top in range(0, height, block_rows):
            rows = min(block_rows, height - top)
            actual = source.read(1, window=Window(0, top, width, rows))
            expected = expected_strip(top, rows)
            if not np.array_equal(actual, expected):
                raise ValueError("Tissue mask and reviewed GeoJSON differ in pixels")
            positive += int(np.count_nonzero(actual))
    if positive == 0:
        raise ValueError("Reviewed geometry produced an empty tissue mask")
    raster_counts = tissue_tile_counts((height, width), target_path)
    vector_counts = tissue_tile_counts((height, width), reviewed_geojson)
    if not np.array_equal(raster_counts, vector_counts):
        raise ValueError("Tissue mask changes encoder tile admission counts")
    receipt = {
        "format": "bora-reviewed-tissue-support-v1", "status": "VERIFIED",
        "calibrated_image": str(image_path),
        "calibrated_image_sha256": source_hash,
        "reviewed_geojson": str(reviewed_geojson),
        "reviewed_geojson_sha256": vector_hash,
        "tissue_mask": str(target_path), "tissue_mask_sha256": sha256(target_path),
        "image_shape_yx": [height, width], "tissue_pixels": positive,
        "positive_value": 255, "raster_vector_pixel_agreement": True,
        "encoder_tile_counts_equal": True,
        "background_identification": "reviewed external GeoJSON; no automatic inference",
        "source_only": True, "expert_annotation_used": False,
    }
    receipt_path = output_dir / "SUPPORT_VERIFIED.json"
    partial_receipt = output_dir / "SUPPORT_VERIFIED.json.partial"
    partial_receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    os.replace(partial_receipt, receipt_path)
    return receipt
