import argparse, json
from pathlib import Path
from .backends import make_backend
from .geojson import write_geojson
from .io import read_image, read_mask, write_ome_mask
from .refine import RefineConfig, refine_labels


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
    return root


def main(argv=None):
    a = parser().parse_args(argv)
    image, labels = read_image(a.image), read_mask(a.mask)
    cfg = RefineConfig(a.core_erosion, a.outer_dilation, a.tile_size, a.tile_overlap, a.min_area, a.smooth_radius)
    refined, report = refine_labels(image, labels, make_backend(a.backend, a.checkpoint, a.device, a.repo_dir), cfg)
    write_ome_mask(a.output, refined); write_geojson(a.geojson, refined, a.geojson_simplify, a.min_area)
    report.update({"image":a.image,"mask":a.mask,"output":a.output,"geojson":a.geojson,"backend":a.backend})
    report_path = Path(a.report) if a.report else Path(a.output).with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
