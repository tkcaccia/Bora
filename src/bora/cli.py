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
    p.add_argument("--class-metadata", help="tiling-mask metrics_cpp.json for named wsiTools annotations")
    p.add_argument("--backend", choices=("watershed","adaptive-wand","medsam","pathsegmentor","pathsegmentator"), default="watershed")
    p.add_argument("--source-tile-size", type=int,
                   help="original square tile width, required for adaptive-wand")
    p.add_argument("--external-core-fraction", type=float, default=0.37)
    p.add_argument("--adaptive-external-core", action="store_true")
    p.add_argument("--internal-core-fraction", type=float, default=0.28)
    p.add_argument("--grabcut-iterations", type=int, default=1)
    p.add_argument("--watershed-compactness", type=float, default=0.1)
    p.add_argument("--median-size", type=int, default=5)
    p.add_argument("--repair-block-seams", action="store_true")
    p.add_argument("--min-seam-run", type=int, default=24)
    p.add_argument("--seam-roi-tiles", type=int, default=2)
    p.add_argument("--repair-external-only", action="store_true")
    p.add_argument("--protect-internal-band", type=int, default=0)
    p.add_argument("--internal-search-fraction", type=float, default=0.28)
    p.add_argument("--local-color-fraction", type=float, default=0.70)
    p.add_argument("--internal-spatial-weight", type=float, default=1.5)
    p.add_argument("--internal-median-size", type=int, default=9)
    p.add_argument("--checkpoint"); p.add_argument("--repo-dir"); p.add_argument("--device", default="cuda")
    p.add_argument("--label-map", help="JSON object mapping label IDs to PathSegmentor text prompts")
    p.add_argument("--pathsegmentor-config", help="override the official inference YAML")
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
    p.add_argument("--geojson-backend", choices=("multicpu","cuda","cellphenotyper"), default="multicpu")
    p.add_argument("--geojson-cuda-device", type=int, default=0)
    p.add_argument("--geojson-cuda-fallback", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--geojson-profile", choices=("accurate","balanced","fast","custom"), default="accurate")
    p.add_argument("--geojson-workers", type=int, default=4)
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
    b = sub.add_parser("benchmark", help="run tiling-mask, Bora, and compare with GeoJSON truth")
    b.add_argument("input_dir"); b.add_argument("--output", required=True)
    b.add_argument("--image-dir", help="optional directory containing the matching TIFF images")
    b.add_argument("--tile-size", default="224", help="source tile size: N or WIDTHxHEIGHT")
    b.add_argument("--tiling-mask-command", default="tiling-mask")
    b.add_argument("--tiling-mask-backend", choices=("python", "cpp"), default="python")
    b.add_argument("--coordinates", choices=("pixel", "world"), default="pixel")
    b.add_argument("--class-property"); b.add_argument("--workers", type=int, default=4)
    b.add_argument("--backend", choices=("watershed","adaptive-wand","medsam","pathsegmentor","pathsegmentator"), default="watershed")
    b.add_argument("--checkpoint"); b.add_argument("--repo-dir"); b.add_argument("--device", default="cuda")
    b.add_argument("--block-size", type=int, default=2048)
    b.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    b.add_argument("--bora-arg", action="append", default=[], help="extra argument passed to bora refine; repeatable")
    e = sub.add_parser("evidence", help="whole-field refinement of a fixed converted RGB image")
    e.add_argument("image"); e.add_argument("mask")
    e.add_argument("--tissue-mask", required=True, help="independent aligned tissue support; nonzero is tissue")
    e.add_argument("--output", required=True); e.add_argument("--report")
    e.add_argument("--method", choices=("simplex", "linear"), default="simplex")
    e.add_argument("--palette", help="JSON array of conversion RGB colours indexed by label ID minus one")
    e.add_argument("--mpp-x", type=float, required=True); e.add_argument("--mpp-y", type=float, required=True)
    e.add_argument("--source-tile-size", type=int, default=90)
    return root


def main(argv=None):
    a = parser().parse_args(argv)
    if a.command == "evidence":
        from PIL import Image
        from .evidence import refine_evidence
        rgb = np.array(Image.open(a.image).convert("RGB"))
        coarse = read_mask(a.mask); tissue = read_mask(a.tissue_mask) > 0
        palette = json.loads(Path(a.palette).read_text()) if a.palette else None
        labels, report = refine_evidence(rgb, coarse, tissue, [a.mpp_x, a.mpp_y],
            method=a.method, palette=palette, tile_size=a.source_tile_size)
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(a.output, labels, ome=True, compression="deflate", photometric="minisblack",
            metadata={"axes": "YX", "PhysicalSizeX": a.mpp_x, "PhysicalSizeY": a.mpp_y,
                      "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm"})
        if a.report:
            Path(a.report).parent.mkdir(parents=True, exist_ok=True)
            Path(a.report).write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2)); return
    if a.command == "benchmark":
        from .benchmark_pipeline import run_pipeline
        try:
            summary = run_pipeline(
                a.input_dir, a.output, image_dir=a.image_dir, tile_size=a.tile_size,
                tiling_mask_command=a.tiling_mask_command,
                tiling_mask_backend=a.tiling_mask_backend, coordinates=a.coordinates,
                class_property=a.class_property, workers=a.workers, backend=a.backend,
                checkpoint=a.checkpoint, repo_dir=a.repo_dir, device=a.device,
                block_size=a.block_size, resume=a.resume, extra_bora_args=a.bora_arg)
        except (OSError, RuntimeError, ValueError) as exc:
            raise SystemExit(f"bora benchmark: error: {exc}") from exc
        print(json.dumps(summary, indent=2))
        return
    if a.backend == "adaptive-wand":
        if not a.source_tile_size or a.source_tile_size <= 0:
            raise SystemExit("--source-tile-size must be positive for adaptive-wand")
        from .boundary import BoundaryConfig, smooth_boundaries_streaming
        boundary_config = BoundaryConfig(
            external_core_fraction=a.external_core_fraction,
            adaptive_external_core=a.adaptive_external_core,
            internal_core_fraction=a.internal_core_fraction,
            grabcut_iterations=a.grabcut_iterations,
            watershed_compactness=a.watershed_compactness,
            median_size=a.median_size,
            repair_block_seams=a.repair_block_seams,
            min_seam_run=a.min_seam_run,
            seam_roi_tiles=a.seam_roi_tiles,
            repair_external_only=a.repair_external_only,
            protect_internal_band=a.protect_internal_band,
            internal_search_fraction=a.internal_search_fraction,
            local_color_fraction=a.local_color_fraction,
            internal_spatial_weight=a.internal_spatial_weight,
            internal_median_size=a.internal_median_size)
        report = smooth_boundaries_streaming(
            Path(a.image), Path(a.mask), Path(a.output),
            a.source_tile_size, a.block_size, boundary_config)
        backends, wand = [], None
    else:
        cfg = RefineConfig(a.core_erosion, a.outer_dilation, a.tile_size,
                           a.tile_overlap, a.min_area, a.smooth_radius)
        watershed_max_side = 0 if a.watershed_resolution == "native" else a.watershed_max_side
        watershed = make_backend("watershed", watershed_max_side=watershed_max_side)
        selected = make_backend(a.backend, a.checkpoint, a.device, a.repo_dir, watershed_max_side,
                                a.label_map, a.pathsegmentor_config)
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
    if a.geojson_backend in ("multicpu", "cuda"):
        if a.geojson_backend == "cuda":
            from .cuda_geojson import convert
        else:
            from .multicpu_geojson import convert
        profiles = {
            "accurate": {"page":0, "min_area":500.0, "simplify":2.0},
            "balanced": {"page":0, "min_area":500.0, "simplify":4.0},
            "fast": {"page":3, "min_area":10000.0, "simplify":16.0},
            "custom": {"page":a.geojson_page, "min_area":a.geojson_min_area,
                       "simplify":a.geojson_simplify},
        }
        settings = profiles[a.geojson_profile]
        converter_options = dict(
            mask_path=a.output, output_path=a.geojson,
            max_page_side=a.geojson_max_page_side,
            workers=a.geojson_workers, group_prefix=a.geojson_group_prefix,
            **settings)
        if a.geojson_backend == "cuda":
            converter_options.update(device=a.geojson_cuda_device,
                                     fallback=a.geojson_cuda_fallback)
        geojson_report = convert(**converter_options)
        report["geojson_feature_count"] = geojson_report["features"]
        report["geojson_conversion"] = geojson_report
    else:
        report["geojson_feature_count"] = mask_to_geojson(
            a.output, a.geojson, page=a.geojson_page,
            max_page_side=a.geojson_max_page_side, min_area=a.geojson_min_area,
            smooth_buffer=a.geojson_smooth_buffer, smooth_passes=a.geojson_smooth_passes,
            simplify=a.geojson_simplify, group_prefix=a.geojson_group_prefix,
            dissolve_by_value=a.geojson_dissolve_by_value,
            fill_holes=a.geojson_fill_holes,
            preserve_topology=a.geojson_preserve_topology,
            polygon_backend=a.geojson_polygon_backend)
    if a.class_metadata:
        from .geojson_compat import enrich_wsitools_geojson
        with tifffile.TiffFile(a.output) as tif:
            height, width = tif.series[0].shape[-2:]
        report["annotation_metadata"] = enrich_wsitools_geojson(
            Path(a.geojson), Path(a.class_metadata), width, height)
    report["pyramidal"] = bool(a.pyramid)
    report["watershed_resolution"] = a.watershed_resolution
    report["refinement_chain"] = (["adaptive_boundary_wand"] if a.backend == "adaptive-wand" else
                                  ((["annealed_wand"] if wand else []) +
                                   [type(item).__name__ for item in backends]))
    report["wand_downsample"] = a.wand_downsample if wand else None
    report["geojson_converter"] = a.geojson_backend
    report["geojson_profile"] = a.geojson_profile
    report.update({"image":a.image,"mask":a.mask,"output":a.output,"geojson":a.geojson,"backend":a.backend})
    report_path = Path(a.report) if a.report else Path(a.output).with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
