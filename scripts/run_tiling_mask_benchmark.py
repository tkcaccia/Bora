#!/usr/bin/env python3
"""Generate 56/90/216-pixel C++ tile masks for Bora's four-tissue benchmark."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import rasterio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executable", type=Path,
                        default=Path("/home/chiamaka/.local/bin/tiling-mask-cpp"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    images = sorted(args.input.glob("*.btf"))
    if len(images) != 4:
        raise SystemExit(f"Expected four .btf images, found {len(images)}")
    for image in images:
        annotation = image.with_suffix(".geojson")
        if not annotation.is_file():
            raise SystemExit(f"Missing annotation for {image.name}")
        destination = args.output / image.stem
        destination.mkdir(parents=True, exist_ok=True)
        command = [str(args.executable), str(image), str(annotation),
                   "--tile-size", "56", "--tile-size", "90",
                   "--tile-size", "216", "--workers", str(args.workers),
                   "--raster-workers", str(args.workers),
                   "--border-error-percent", "0", "--output", str(destination)]
        with (destination / "tiling-mask.log").open("w") as log:
            log.write("$ " + " ".join(command) + "\n\n")
            log.flush()
            subprocess.run(command, check=True, stdout=log,
                           stderr=subprocess.STDOUT)
        with rasterio.open(image) as original:
            image_shape = (original.height, original.width)
        for size in (56, 90, 216):
            grid_path = destination / f"mask_tile_grid_{size}x{size}_cpp.tif"
            with rasterio.open(grid_path) as grid:
                if (grid.height, grid.width) != image_shape:
                    raise RuntimeError(f"{grid_path}: unexpected shape")
        metadata = {"image": str(image), "annotation": str(annotation),
                    "tile_sizes": [56, 90, 216], "border_error_percent": 0,
                    "image_shape": image_shape}
        (destination / "bora_tiling_manifest.json").write_text(
            json.dumps(metadata, indent=2))
        print(f"{image.stem}: completed {image_shape}", flush=True)


if __name__ == "__main__":
    main()
