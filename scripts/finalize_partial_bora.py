#!/usr/bin/env python3
"""Finalize a flat streaming Bora mask after a post-refinement interruption."""
import argparse
import json
from pathlib import Path
from time import perf_counter

from bora.multicpu_geojson import convert
from bora.pyramid import pyramidize_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mask", type=Path)
    parser.add_argument("geojson", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    started = perf_counter()
    pyramidize_mask(args.mask, "LZW", 8)
    pyramid_seconds = perf_counter() - started
    polygon_started = perf_counter()
    polygon = convert(args.mask, args.geojson, page=0, min_area=500.0,
                      simplify=2.0, workers=4)
    report = {"recovered_after_refinement": True,
              "pyramid_seconds": pyramid_seconds,
              "polygon_seconds": perf_counter() - polygon_started,
              "geojson_conversion": polygon}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
