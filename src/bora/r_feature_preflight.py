"""Bounded source-only admission for the R Virchow2 preparation interface.

The caller supplies tissue support explicitly for benchmark work. With no
support, every original image tile is eligible, including glass. Artificial
right/bottom padding never becomes a separate eligible tile.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


TILE = 224
SCALE_PAD = 4 * TILE


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tile_counts(mask: np.ndarray, width: int, tile: int = TILE) -> np.ndarray:
    """Count true mask pixels in one bounded image strip, padding as false."""
    if mask.ndim != 2 or mask.shape[1] != width or mask.shape[0] > tile:
        raise ValueError("Mask strip shape differs from the image")
    cols = math.ceil(width / tile)
    padded = np.zeros((tile, cols * tile), dtype=np.bool_)
    padded[: mask.shape[0], :width] = mask != 0
    return padded.reshape(tile, cols, tile).sum(axis=(0, 2), dtype=np.uint32)


def tissue_tile_counts(image_shape_yx: tuple[int, int], support: Path | None,
                       *, tile: int = TILE) -> np.ndarray:
    """Return exact tissue counts per tile without loading a whole WSI mask.

    Raster support uses band 1 and treats nonzero values as tissue. GeoJSON
    geometry must use image pixel-edge coordinates; it is rasterized in bounded
    strips with the same pixel-center inclusion rule used by rasterio.
    """
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    from rasterio.windows import Window

    height, width = map(int, image_shape_yx)
    if height <= 0 or width <= 0 or tile <= 0:
        raise ValueError("Image and tile dimensions must be positive")
    rows, cols = math.ceil(height / tile), math.ceil(width / tile)
    counts = np.zeros((rows, cols), dtype=np.uint32)
    if support is None:
        for row in range(rows):
            valid_h = min(tile, height - row * tile)
            for col in range(cols):
                counts[row, col] = valid_h * min(tile, width - col * tile)
        return counts

    support = Path(support)
    if not support.is_file():
        raise FileNotFoundError(support)
    if support.suffix.lower() in (".json", ".geojson"):
        source = json.loads(support.read_text())
        if source.get("type") != "FeatureCollection":
            raise ValueError("Tissue GeoJSON must be a FeatureCollection")
        shapes = [feature["geometry"] for feature in source.get("features", [])
                  if feature.get("geometry") is not None]
        if not shapes:
            raise ValueError("Tissue GeoJSON has no geometry")
        for row in range(rows):
            y0 = row * tile
            strip_h = min(tile, height - y0)
            strip = rasterize(((geometry, 1) for geometry in shapes),
                              out_shape=(strip_h, width), fill=0,
                              transform=Affine.translation(0, y0), dtype=np.uint8)
            counts[row] = _tile_counts(strip, width, tile)
    else:
        with rasterio.open(support) as dataset:
            if (dataset.width, dataset.height) != (width, height) or dataset.count < 1:
                raise ValueError("Tissue raster must have the exact image dimensions")
            for row in range(rows):
                y0 = row * tile
                strip = dataset.read(1, window=Window(0, y0, width,
                                                    min(tile, height - y0)))
                counts[row] = _tile_counts(strip, width, tile)
    return counts


def inspect_image(image: Path) -> tuple[int, int]:
    import pyvips

    image = Path(image)
    if not image.is_file():
        raise FileNotFoundError(image)
    source = pyvips.Image.new_from_file(str(image), access="sequential")
    if source.bands != 3 or source.format != "uchar":
        raise ValueError("Calibrated H&E image must be uint8 RGB")
    return source.height, source.width


def inspect_input(image: Path, support: Path | None,
                  *, tile: int = TILE) -> dict:
    image = Path(image).resolve()
    support = Path(support).resolve() if support is not None else None
    height, width = inspect_image(image)
    counts = tissue_tile_counts((height, width), support, tile=tile)
    eligible = counts > 0
    return {
        "format": "bora-r-feature-preflight-v1",
        "image": str(image), "image_sha256": sha256(image),
        "tissue_support": str(support) if support else None,
        "tissue_support_sha256": sha256(support) if support else None,
        "selection": "intersects-supplied-tissue" if support else "all-original-image-tiles-including-glass",
        "image_shape_yx": [height, width],
        "tile_size_yx": [tile, tile],
        "tile_grid_shape_yx": list(counts.shape),
        "eligible_tiles": int(eligible.sum()),
        "selected_source_pixels": int(counts.sum()),
        "support_positive_pixels": int(counts.sum()) if support else None,
        "padded_shape_yx": [math.ceil(height / SCALE_PAD) * SCALE_PAD,
                            math.ceil(width / SCALE_PAD) * SCALE_PAD],
        "counts": counts,
        "eligible": eligible,
    }
