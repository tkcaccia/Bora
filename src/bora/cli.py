import argparse, json
import numpy as np
import tifffile
from pathlib import Path
from .backends import make_backend
from .geojson import write_geojson
from .io import read_image, read_mask, write_ome_mask
from .refine import RefineConfig, refine_labels
from .streaming import refine_streaming, write_geojson_streaming
from .pyramid import pyramidize_mask


def parser():
    root = argparse.ArgumentParser(prog="bora")
    sub = root.add_subparsers(dest="command", required=True)
    p = sub.add_parser("refine", help="refine a coarse label mask")
    p.add_argument("image"); p.add_argument("mask")
    p.add_argument("--output", required=True); p.add_argument("--geojson", required=True); p.add_argument("--report")
    p.add_argument("--backend", choices=("watershed","medsam","pathsegmentor"), default="watershed")
    p.add_argument("--checkpoint"); p.add_argument("--repo-dir"); p.add_argument("--device", default="cuda")
    p.add_argument("--core-erosion", type=int, default=16); p.add_argument("--outer-dilation", type=int, default=24)
    p.add_argument("--tile-size", type=int, default=2048); p.add_argument("--tile-overlap", type=int, default=256)
    p.add_argument("--min-area", type=int, default=32); p.add_argument("--smooth-radius", type=int, default=2)
    p.add_argument("--geojson-simplify", type=float, default=0)
    p.add_argument("--stream", action=argparse.BooleanOptionalAction, default=None,
                   help="stream blocks (automatic above 100 million pixels)")
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--pyramid", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pyramid-compression", choices=("LZW","UNCOMPRESSED"), default="LZW")
    p.add_argument("--pyramid-workers", type=int, default=8)
    return root


def main(argv=None):
    a = parser().parse_args(argv)
    cfg = RefineConfig(a.core_erosion, a.outer_dilation, a.tile_size, a.tile_overlap, a.min_area, a.smooth_radius)
    backend = make_backend(a.backend, a.checkpoint, a.device, a.repo_dir)
    with tifffile.TiffFile(a.mask) as tif:
        shape = tif.series[0].shape
    stream = a.stream if a.stream is not None else int(np.prod(shape[-2:])) > 100_000_000
    if stream:
        report = refine_streaming(a.image, a.mask, a.output, backend, cfg, a.block_size)
        report["geojson_feature_count"] = write_geojson_streaming(
            a.geojson, a.output, max(2048,a.block_size), a.geojson_simplify, a.min_area)
    else:
        image, labels = read_image(a.image), read_mask(a.mask)
        refined, report = refine_labels(image, labels, backend, cfg)
        write_ome_mask(a.output, refined); write_geojson(a.geojson, refined, a.geojson_simplify, a.min_area)
    if a.pyramid:
        pyramidize_mask(a.output, a.pyramid_compression, a.pyramid_workers)
    report["pyramidal"] = bool(a.pyramid)
    report.update({"image":a.image,"mask":a.mask,"output":a.output,"geojson":a.geojson,"backend":a.backend})
    report_path = Path(a.report) if a.report else Path(a.output).with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
