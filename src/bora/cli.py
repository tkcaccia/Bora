import argparse, json
import numpy as np
import tifffile
from pathlib import Path
from .backends import make_backend
from .io import read_image, read_mask, write_ome_mask
from .refine import RefineConfig, refine_labels
from .streaming import refine_streaming
from .pyramid import pyramidize_mask
from .wand import run_annealed_wand
from .cellphenotyper import mask_to_geojson


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
    p.add_argument("--geojson-page", type=int, default=-1)
    p.add_argument("--geojson-max-page-side", type=int, default=2048)
    p.add_argument("--geojson-min-area", type=float, default=500)
    p.add_argument("--geojson-smooth-buffer", type=float, default=10.0)
    p.add_argument("--geojson-smooth-passes", type=int, default=3)
    p.add_argument("--geojson-simplify", type=float, default=6.0)
    p.add_argument("--geojson-group-prefix", default="group_")
    p.add_argument("--geojson-polygon-backend",
                   choices=("auto","opencv","skimage","rasterio"), default="auto")
    p.add_argument("--geojson-dissolve-by-value", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--geojson-fill-holes", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--geojson-preserve-topology", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--stream", action=argparse.BooleanOptionalAction, default=None,
                   help="stream blocks (automatic above 100 million pixels)")
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--pyramid", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pyramid-compression", choices=("LZW","UNCOMPRESSED"), default="LZW")
    p.add_argument("--pyramid-workers", type=int, default=8)
    p.add_argument("--annealed-wand", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wand-boundary-radius", type=int, default=64)
    p.add_argument("--wand-downsample", type=int, default=4)
    p.add_argument("--wand-iterations", type=int, default=16)
    p.add_argument("--wand-initial-temperature", type=float, default=2.0)
    p.add_argument("--wand-final-temperature", type=float, default=0.05)
    p.add_argument("--wand-data-weight", type=float, default=1.0)
    p.add_argument("--wand-smoothness-weight", type=float, default=0.3)
    p.add_argument("--wand-edge-beta", type=float, default=0.7)
    p.add_argument("--wand-connectivity", type=int, choices=(4,8), default=8)
    p.add_argument("--watershed-resolution", choices=("native","accelerated"), default="native")
    p.add_argument("--watershed-max-side", type=int, default=256,
                   help="analysis size used only in accelerated watershed mode")
    return root


def main(argv=None):
    a = parser().parse_args(argv)
    cfg = RefineConfig(a.core_erosion, a.outer_dilation, a.tile_size, a.tile_overlap, a.min_area, a.smooth_radius)
    watershed_max_side = 0 if a.watershed_resolution == "native" else a.watershed_max_side
    watershed = make_backend("watershed", watershed_max_side=watershed_max_side)
    selected = make_backend(a.backend, a.checkpoint, a.device, a.repo_dir, watershed_max_side)
    backends = [watershed] if a.backend == "watershed" else [watershed, selected]
    wand = ({"boundary_radius":a.wand_boundary_radius,"iterations":a.wand_iterations,
             "initial_temperature":a.wand_initial_temperature,"final_temperature":a.wand_final_temperature,
             "data_weight":a.wand_data_weight,"smoothness_weight":a.wand_smoothness_weight,
             "edge_beta":a.wand_edge_beta,"connectivity":a.wand_connectivity}
            if a.annealed_wand else None)
    with tifffile.TiffFile(a.mask) as tif:
        shape = tif.series[0].shape
    stream = a.stream if a.stream is not None else int(np.prod(shape[-2:])) > 100_000_000
    if stream:
        report = refine_streaming(a.image, a.mask, a.output, backends, cfg, a.block_size,
                                  wand, a.wand_downsample)
    else:
        image, labels = read_image(a.image), read_mask(a.mask)
        if wand:
            labels, wand_report = run_annealed_wand(
                image, labels, labels > 0, downsample=a.wand_downsample, **wand)
        else:
            wand_report = {"enabled":False,"applied":False,"changed_pixels":0}
        refined = labels
        stage_reports = []
        for stage_backend in backends:
            refined, stage_report = refine_labels(image, refined, stage_backend, cfg)
            stage_report["backend"] = type(stage_backend).__name__
            stage_reports.append(stage_report)
        report = {"stages":stage_reports}
        report["annealed_wand"] = wand_report
        write_ome_mask(a.output, refined)
    if a.pyramid:
        pyramidize_mask(a.output, a.pyramid_compression, a.pyramid_workers)
    report["geojson_feature_count"] = mask_to_geojson(
        a.output, a.geojson, page=a.geojson_page,
        max_page_side=a.geojson_max_page_side, min_area=a.geojson_min_area,
        smooth_buffer=a.geojson_smooth_buffer, smooth_passes=a.geojson_smooth_passes,
        simplify=a.geojson_simplify, group_prefix=a.geojson_group_prefix,
        dissolve_by_value=a.geojson_dissolve_by_value,
        fill_holes=a.geojson_fill_holes,
        preserve_topology=a.geojson_preserve_topology,
        polygon_backend=a.geojson_polygon_backend)
    report["pyramidal"] = bool(a.pyramid)
    report["watershed_resolution"] = a.watershed_resolution
    report["refinement_chain"] = ((["annealed_wand"] if wand else []) +
                                  [type(item).__name__ for item in backends])
    report["wand_downsample"] = a.wand_downsample if wand else None
    report["geojson_converter"] = "CellPhenotyper/bin/mask_to_geojson.py@fd2dd40"
    report.update({"image":a.image,"mask":a.mask,"output":a.output,"geojson":a.geojson,"backend":a.backend})
    report_path = Path(a.report) if a.report else Path(a.output).with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
