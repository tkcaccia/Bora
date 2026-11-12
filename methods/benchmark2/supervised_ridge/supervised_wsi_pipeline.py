#!/usr/bin/env python3
"""Finalize a public supervised Virchow2 whole-slide mask and native GeoJSON.

All decisions are made on the complete calibrated slide. Crops are display-only.
The frozen 397-image training head and training-only class-presence rule are
immutable inputs. This CPU route does not run Virchow2, KODAMA, or any GPU work.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import cv2
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.windows import Window
from shapely.geometry import Polygon, mapping, shape
from shapely.ops import transform

ROOT = Path("/mnt/sata_ssd/ChatGPT/Bora")
DATA = Path("/mnt/sata_ssd/ChatGPT/Bora-Data")
SITE = Path("/tmp/bora-analysis-bw106-v2")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from bora.contour_polish import smooth_global
from bora.wsi_global_vector import publish_global_wsi_vectors

OUT = DATA / "benchmark2-supervised-ridge-2026-11-12-v5"
SCOPE = ROOT / "migration/benchmark2-full-wsi-comparators-2026-11-10-v2/SCOPE.json"
CAMPAIGN = ROOT / "migration/benchmark2-unified-wsi-panels-2026-11-11-v17/CAMPAIGN_VERIFIED.json"
MODEL = ROOT / "migration/benchmark1-eight-class-head-search-2026-11-12-v1/FINAL_MODEL_FROZEN.json"
ONTOLOGY = ROOT / "migration/benchmark-label-reduction-2026-11-11-v1/ONTOLOGY_FROZEN.json"
PRESENCE = ROOT / "migration/benchmark2-supervised-ridge-2026-11-12-v2/CLASS_PRESENCE_FROZEN.json"
SCORES = DATA / "benchmark2-supervised-ridge-2026-11-12-v3/wsi-scores"
GEO = DATA / "current-wsi-links-2026-11-06"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path: Path, value: dict) -> None:
    partial = path.with_name(path.name + ".partial")
    with partial.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def inputs(slug: str) -> dict:
    campaign = json.loads(CAMPAIGN.read_text())
    allowed = {row["slug"] for row in campaign["pages"]}
    if slug not in allowed or "pm016" in slug.lower():
        raise ValueError("Only the thirteen public campaign slides are admitted")
    source = [r for r in json.loads(SCOPE.read_text())["slides"] if r["slug"] == slug]
    if len(source) != 1:
        raise ValueError("Calibrated image scope differs")
    source = source[0]
    source_path = Path(source["path"])
    if sha(source_path) != source["sha256"]:
        raise ValueError("Calibrated H&E SHA256 differs")
    score_dir = SCORES / slug
    score_receipt = score_dir / "RIDGE_SCORE_GRID_VERIFIED.json"
    score_info = json.loads(score_receipt.read_text())
    score_path = score_dir / "smoothed-scores.npy"
    if (score_info["status"] != "VERIFIED_EIGHT_CLASS_RIDGE_CPU_SCORE_GRID" or
            sha(score_path) != score_info["smoothed_scores_sha256"] or
            score_info["model_receipt_sha256"] != sha(MODEL) or
            score_info["ontology_sha256"] != sha(ONTOLOGY)):
        raise ValueError("Frozen full-slide supervised scores differ")
    scores = np.load(score_path, mmap_mode="r", allow_pickle=False)
    if scores.shape[0] != 8 or scores.dtype != np.float32:
        raise ValueError("Expected eight float32 score channels")
    geo_path = GEO / f"{slug}__Bora.geojson"
    geoj = json.loads(geo_path.read_text())
    if geoj["coordinate_space"] != "original_WSI_native_pixel_edges":
        raise ValueError("Bora support is not in original native coordinates")
    if geoj["native_origin_xy"] != [0, 0]:
        raise ValueError("Nonzero native origin requires a new reviewed transform")
    working_shape = geoj["full_slide_input_identity"]["shape_yx"]
    if working_shape[0] > source["height"] or working_shape[1] > source["width"]:
        raise ValueError("Reviewed geometry exceeds calibrated source")
    sx, sy = geoj["working_pixel_to_native_scale_xy"]
    if min(sx, sy) <= 0:
        raise ValueError("Invalid calibrated-to-native transform")
    working = [transform(lambda x, y, z=None: (x / sx, y / sy),
                         shape(feature["geometry"])) for feature in geoj["features"]]
    if not working or any(not g.is_valid for g in working):
        raise ValueError("Invalid reviewed Bora support geometry")
    presence = json.loads(PRESENCE.read_text())
    if (presence["status"] != "TRAINING_ONLY_RIDGE_CLASS_PRESENCE_RULE_FROZEN" or
            presence["development_masks_opened"] != 0 or
            presence["whole_slide_expert_masks_opened"] != 0):
        raise ValueError("Training-only presence rule differs")
    return dict(source=source, geoj=geoj, geo_path=geo_path, working=working,
                scores=scores, score_path=score_path, score_receipt=score_receipt,
                threshold=float(presence["minimum_predicted_tissue_fraction"]))


def reviewed_support(geometries: list, shape_yx: tuple[int, int], out: Path) -> np.memmap:
    """Rasterize Bora's final geometry once on the entire calibrated frame."""
    if out.exists():
        raise FileExistsError(out)
    support = np.memmap(out, mode="w+", dtype=np.uint8, shape=shape_yx)
    support[:] = 0
    rasterize(((mapping(g), 1) for g in geometries), out=support,
              transform=Affine.identity(), fill=0, dtype=np.uint8, all_touched=False)
    support.flush()
    if not np.any(support):
        raise ValueError("No reviewed tissue support")
    return support


def select_classes(scores: np.ndarray, geometries: list,
                   minimum_fraction: float) -> tuple[list[int], dict]:
    """Use complete-slide tissue score-grid fractions, never a selected crop."""
    height, width = scores.shape[1:]
    support = rasterize(((mapping(g), 1) for g in geometries),
                        out_shape=(height, width),
                        transform=Affine.scale(14, 14), fill=0, dtype=np.uint8) > 0
    if not support.any():
        raise ValueError("Reviewed support misses score grid")
    if not np.isfinite(scores[:, support]).all():
        raise ValueError("Nonfinite supervised score inside tissue")
    winner = np.argmax(scores, axis=0)[support]
    counts = np.bincount(winner, minlength=8)
    fractions = counts / counts.sum()
    kept = np.flatnonzero(fractions >= minimum_fraction)
    if len(kept) == 0:
        kept = np.array([int(np.argmax(counts))])
    return [int(k + 1) for k in kept], {
        "score_grid_tissue_tokens": int(counts.sum()),
        "winner_counts": counts.tolist(),
        "winner_fractions": fractions.tolist(),
        "retained_class_ids": (kept + 1).tolist(),
        "excluded_class_ids": [i for i in range(1, 9) if i not in kept + 1],
        "minimum_fraction": minimum_fraction,
    }


def project_mask(scores: np.ndarray, support: np.memmap,
                 class_ids: list[int], out: Path, *, block: int = 768) -> np.memmap:
    """Bicubic continuous scores, one global decision and exact Bora support."""
    if out.exists():
        raise FileExistsError(out)
    height, width = support.shape
    mask = np.memmap(out, mode="w+", dtype=np.uint8, shape=(height, width))
    sources = [np.asarray(scores[cid-1], np.float32) for cid in class_ids]
    for y in range(0, height, block):
        y1 = min(height, y + block)
        for x in range(0, width, block):
            x1 = min(width, x + block)
            if not np.any(support[y:y1, x:x1]):
                mask[y:y1, x:x1] = 0
                continue
            xx = (np.arange(x, x1, dtype=np.float32) + .5) / 14 - .5
            yy = (np.arange(y, y1, dtype=np.float32) + .5) / 14 - .5
            map_x, map_y = np.meshgrid(xx, yy)
            best = np.full((y1-y, x1-x), -np.inf, np.float32)
            labels = np.zeros(best.shape, np.uint8)
            for cid, source in zip(class_ids, sources):
                response = cv2.remap(source, map_x, map_y, cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
                take = response > best
                labels[take] = cid
                best[take] = response[take]
            labels[np.asarray(support[y:y1, x:x1]) == 0] = 0
            mask[y:y1, x:x1] = labels
        mask.flush()
        print("PROJECT_ROWS", y1, "/", height, flush=True)
    return mask


def area_repair(source: np.memmap, support: np.memmap, out: Path, mpp_xy: tuple[float, float],
                minimum_um2: float, *, max_passes: int = 8) -> tuple[np.memmap, dict]:
    """Repair whole-slide 8-connected positive islands and enclosed empty holes."""
    if out.exists():
        raise FileExistsError(out)
    if min(mpp_xy) <= 0 or minimum_um2 <= 0:
        raise ValueError("Positive physical area required")
    height, width = source.shape
    if support.shape != source.shape:
        raise ValueError("Cleanup support shape differs")
    result = np.memmap(out, mode="w+", dtype=np.uint8, shape=source.shape)
    for y in range(0, height, 512):
        result[y:y+512] = source[y:y+512]
    result.flush()
    threshold = math.ceil(minimum_um2 / (mpp_xy[0] * mpp_xy[1]) - 1e-9)
    counts = {"positive_parts": 0, "enclosed_holes": 0, "pixels_reassigned": 0,
              "isolated_to_background": 0}
    kernel = np.ones((3, 3), np.uint8)
    active = [int(i) for i in np.unique(result) if i > 0]
    for iteration in range(max_passes):
        changed = 0
        for value in active + [0]:
            binary = np.asarray(result == value, np.uint8)
            n, components, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            del binary
            small = np.flatnonzero((stats[:, cv2.CC_STAT_AREA] < threshold) &
                                   (np.arange(n) > 0))
            for component in small:
                x, y, w, h, area = map(int, stats[component])
                if value == 0 and (x == 0 or y == 0 or x+w == width or y+h == height):
                    continue
                xa, ya = max(0, x-1), max(0, y-1)
                xb, yb = min(width, x+w+1), min(height, y+h+1)
                local_cc = components[ya:yb, xa:xb]
                at = local_cc == component
                if value == 0 and not np.all(support[ya:yb, xa:xb][at]):
                    continue
                ring = (cv2.dilate(at.astype(np.uint8), kernel) > 0) & ~at
                view = result[ya:yb, xa:xb]
                neighbors = np.asarray(view[ring], np.uint8)
                neighbors = neighbors[neighbors != value]
                # A small positive island joins adjacent tissue in preference
                # to glass. Exterior support is still protected by the final
                # source support and geometry agreement checks.
                positive = neighbors[neighbors > 0]
                if len(positive):
                    neighbors = positive
                if len(neighbors) == 0:
                    if value == 0:
                        continue
                    winner = 0
                    counts["isolated_to_background"] += 1
                else:
                    winner = int(np.bincount(neighbors, minlength=9).argmax())
                view[at] = winner
                changed += 1
                counts["pixels_reassigned"] += area
                counts["enclosed_holes" if value == 0 else "positive_parts"] += 1
            del components, stats
            gc.collect()
        result.flush()
        print("AREA_PASS", minimum_um2, iteration+1, changed, flush=True)
        if changed == 0:
            break
    else:
        raise ValueError("Global small-region repair did not converge")
    counts.update(minimum_area_um2=minimum_um2, minimum_pixels=threshold,
                  passes=iteration+1, connectivity=8, strictly_less_than=True)
    return result, counts


def bind_assembly(slug: str, final: Path, context: dict,
                  class_ids: list[int], out: Path) -> Path:
    assembly = out / "assembly"
    assembly.mkdir()
    target = assembly / "semantic-mask.uint8.dat"
    os.link(final, target)
    binding = {"model_receipt_sha256": sha(MODEL),
               "class_presence_sha256": sha(PRESENCE),
               "score_receipt_sha256": sha(context["score_receipt"]),
               "bora_support_geojson_sha256": sha(context["geo_path"]),
               "source_code_sha256": sha(Path(__file__))}
    bind_sha = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    source = context["source"]
    atomic(assembly / "ASSEMBLY_COMPLETE.json", {
        "format": "bora-wsi-label-assembly-v1", "core_partition_complete": True,
        "independently_fitted_windows": False, "native_shape_yx": [source["height"], source["width"]],
        "native_mpp_xy": context["geoj"]["working_mpp_xy"],
        "class_ids": class_ids, "semantic_mask_path": target.name,
        "semantic_mask_sha256": sha(target), "source_image_sha256": source["sha256"],
        "source_frame_id": slug + ":calibrated_0p5um", "global_model_binding_sha256": bind_sha,
        "binding": binding,
    })
    return assembly


def native_geojson(exact_path: Path, context: dict, out: Path) -> dict:
    geoj = json.loads(exact_path.read_text())
    sx, sy = context["geoj"]["working_pixel_to_native_scale_xy"]
    native_mpp = context["geoj"]["native_mpp_xy"]
    for feature in geoj["features"]:
        geometry = transform(lambda x, y, z=None: (x*sx, y*sy),
                             shape(feature["geometry"]))
        if not geometry.is_valid:
            raise ValueError("Invalid native supervised polygon")
        feature["geometry"] = mapping(geometry)
        feature["properties"]["area_pixels"] = float(geometry.area)
        feature["properties"]["area_um2"] = float(geometry.area * np.prod(native_mpp))
    geoj["coordinate_space"] = "original_WSI_native_pixel_edges"
    geoj["native_shape_yx"] = context["geoj"]["native_shape_yx"]
    geoj["native_mpp_xy"] = native_mpp
    geoj["working_pixel_to_native_scale_xy"] = [sx, sy]
    geoj["processing_scope"] = "entire calibrated slide; no diagnostic crop fitting"
    geoj["supervised_model"] = "training-selected eight-class Virchow2 ridge head"
    atomic(out, geoj)
    minimum = min(p.area * np.prod(native_mpp)
                  for feature in geoj["features"]
                  for p in ([shape(feature["geometry"])] if feature["geometry"]["type"] == "Polygon"
                            else shape(feature["geometry"]).geoms))
    if minimum < 2000 - 1e-4:
        raise ValueError("Native GeoJSON contains a sub-2000 um2 polygon")
    return {"feature_count": len(geoj["features"]), "minimum_polygon_um2": float(minimum)}


def write_mask_tiff(mask_path: Path, shape_yx: tuple[int, int], path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    mask = np.memmap(mask_path, mode="r", dtype=np.uint8, shape=shape_yx)
    temporary = path.with_suffix(".partial.tif")
    with rasterio.open(temporary, "w", driver="GTiff", width=shape_yx[1], height=shape_yx[0],
                       count=1, dtype="uint8", tiled=True, blockxsize=512,
                       blockysize=512, compress="DEFLATE", BIGTIFF="IF_SAFER",
                       transform=Affine.identity()) as dest:
        for y in range(0, shape_yx[0], 512):
            for x in range(0, shape_yx[1], 512):
                h, w = min(512, shape_yx[0]-y), min(512, shape_yx[1]-x)
                dest.write(mask[y:y+h, x:x+w], 1, window=Window(x, y, w, h))
    os.replace(temporary, path)
    with rasterio.open(path) as src:
        for y in range(0, shape_yx[0], 512):
            for x in range(0, shape_yx[1], 512):
                h, w = min(512, shape_yx[0]-y), min(512, shape_yx[1]-x)
                if not np.array_equal(src.read(1, window=Window(x, y, w, h)), mask[y:y+h, x:x+w]):
                    raise ValueError("Full-resolution TIFF pixel round-trip differs")


def audit_final(mask_path: Path, support_path: Path, native_path: Path,
                context: dict) -> dict:
    """Independently check physical islands, support, and native vector identity."""
    source = context["source"]
    frame = (source["height"], source["width"])
    mpp = context["geoj"]["working_mpp_xy"]
    threshold = math.ceil(2000 / (mpp[0] * mpp[1]) - 1e-9)
    mask = np.memmap(mask_path, mode="r", dtype=np.uint8, shape=frame)
    support = np.memmap(support_path, mode="r", dtype=np.uint8, shape=frame)
    if np.any((mask > 0) & (support == 0)):
        raise ValueError("Final supervised mask extends outside reviewed Bora support")
    counts = []
    for value in [int(v) for v in np.unique(mask) if v > 0] + [0]:
        binary = np.asarray(mask == value, np.uint8)
        n, components, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        del binary
        ids = np.arange(1, n)
        if value == 0:
            exterior = np.unique(np.concatenate((components[0], components[-1],
                                                   components[:, 0], components[:, -1])))
            ids = np.setdiff1d(ids, exterior)
        sizes = stats[ids, cv2.CC_STAT_AREA]
        bad = int(np.count_nonzero(sizes < threshold))
        counts.append({"class_id": value, "components": len(ids),
                       "minimum_pixels": int(sizes.min()) if len(sizes) else None,
                       "below_2000um2": bad})
        del components, stats
        gc.collect()
        if bad:
            raise ValueError("Final exact vector raster has sub-2000 um2 component")
    native = json.loads(native_path.read_text())
    sx, sy = context["geoj"]["working_pixel_to_native_scale_xy"]
    pairs = []
    for feature in native["features"]:
        geometry = transform(lambda x, y, z=None: (x/sx, y/sy),
                             shape(feature["geometry"]))
        pairs.append((mapping(geometry), int(feature["properties"]["value"])))
    roundtrip = np.memmap(native_path.with_suffix(".roundtrip.partial.dat"), mode="w+",
                          dtype=np.uint8, shape=frame)
    rasterize(pairs, out=roundtrip, transform=Affine.identity(),
              fill=0, dtype=np.uint8, all_touched=False)
    different = 0
    for y in range(0, frame[0], 512):
        different += int(np.count_nonzero(roundtrip[y:y+512] != mask[y:y+512]))
    del roundtrip
    native_path.with_suffix(".roundtrip.partial.dat").unlink()
    if different:
        raise ValueError(f"Native GeoJSON round-trip differs by {different} calibrated pixels")
    return {"status": "INDEPENDENT_FULL_SLIDE_GEOMETRY_AUDIT_PASSED",
            "physical_minimum_um2": 2000, "minimum_pixels": threshold,
            "positive_and_hole_components": counts,
            "outside_reviewed_support_pixels": 0,
            "native_geojson_roundtrip_different_pixels": different}


def main(slug: str) -> None:
    start = time.monotonic()
    context = inputs(slug)
    slide = OUT / slug
    if slide.exists():
        raise FileExistsError("Preserve existing slide output; recover in a new version: " + str(slide))
    slide.mkdir(parents=True)
    state = slide / "STATE.json"
    try:
        source = context["source"]
        shape_yx = (source["height"], source["width"])
        mpp = tuple(context["geoj"]["working_mpp_xy"])
        atomic(state, {"status": "RUNNING", "stage": "reviewed_support"})
        support = reviewed_support(context["working"], shape_yx, slide / "reviewed-support.uint8.dat")
        classes, class_report = select_classes(context["scores"], context["working"], context["threshold"])
        atomic(slide / "CLASS_SELECTION.json", class_report)
        atomic(state, {"status": "RUNNING", "stage": "global_projection", "class_ids": classes})
        raw = project_mask(context["scores"], support, classes, slide / "raw-mask.uint8.dat")
        pre, pre_report = area_repair(raw, support, slide / "precontour-400um2.uint8.dat", mpp, 400)
        atomic(state, {"status": "RUNNING", "stage": "six_um_contours"})
        smooth_path = slide / "contour-6um.npy"
        contour_report = smooth_global(pre, support, mpp, smooth_path, sigma_um=6, block_size=1024)
        smoothed = np.load(smooth_path, mmap_mode="r")
        atomic(state, {"status": "RUNNING", "stage": "global_2000um2_cleanup"})
        final, final_report = area_repair(smoothed, support, slide / "final-clean-2000um2.uint8.dat", mpp, 2000)
        del raw, pre, smoothed, final, support
        gc.collect()
        atomic(state, {"status": "RUNNING", "stage": "global_vector"})
        assembly = bind_assembly(slug, slide / "final-clean-2000um2.uint8.dat", context, classes, slide)
        names = {int(k): v["name"].replace("_", " ") for k, v in json.loads(ONTOLOGY.read_text())["classes"].items()}
        vector = publish_global_wsi_vectors(
            assembly, slide / "global-vectors", names, minimum_area_um2=2000,
            smoothing_tolerance_um=0, maximum_cleanup_change=.02,
            max_input_regions=250_000, max_coordinates=10_000_000,
            max_boundary_edges=10_000_000, disk_budget_bytes=12 * 1024**3)
        exact = slide / "global-vectors/exact/mask.uint8.dat"
        native = slide / "supervised-bora.native.geojson"
        native_report = native_geojson(slide / "global-vectors/exact/mask.geojson", context, native)
        tiff = slide / "supervised-bora.mask-0p5um.tif"
        write_mask_tiff(exact, shape_yx, tiff)
        final_audit = audit_final(exact, slide / "reviewed-support.uint8.dat", native, context)
        result = {"status": "SUPERVISED_BORA_FULL_SLIDE_VECTOR_VERIFIED", "slug": slug,
                  "source_image_sha256": source["sha256"], "score_sha256": sha(context["score_path"]),
                  "source_bora_geojson_sha256": sha(context["geo_path"]),
                  "training_model_receipt_sha256": sha(MODEL), "training_presence_sha256": sha(PRESENCE),
                  "ontology_sha256": sha(ONTOLOGY), "class_ids": classes,
                  "class_selection": class_report, "precontour_cleanup": pre_report,
                  "contour_smoothing": contour_report, "final_cleanup": final_report,
                  "vector_receipt_sha256": sha(slide / "global-vectors/GLOBAL_VECTOR_COMPLETE.json"),
                  "vector_summary": {"input_regions": vector["input_regions"],
                                     "exact_component_count": vector["exact_component_count"],
                                     "cleanup_changed_pixels": vector["cleanup_changed_pixels"],
                                     "global_contours": vector["global_contours"]},
                  "native_vector": native_report, "calibrated_shape_yx": list(shape_yx),
                  "independent_geometry_audit": final_audit,
                  "calibrated_mpp_xy": list(mpp), "native_mpp_xy": context["geoj"]["native_mpp_xy"],
                  "output_sha256": {"mask_tiff": sha(tiff), "native_geojson": sha(native),
                                    "exact_mask": sha(exact)},
                  "source_code_sha256": sha(Path(__file__)), "gpu_used": False,
                  "whole_slide_expert_accuracy": None, "pm016_included": False,
                  "elapsed_seconds": time.monotonic()-start}
        atomic(slide / "FULL_SLIDE_VERIFIED.json", result)
        atomic(state, {"status": "COMPLETE", "receipt_sha256": sha(slide / "FULL_SLIDE_VERIFIED.json")})
        print("FULL_SLIDE_VERIFIED", slug, classes, round(result["elapsed_seconds"], 1), flush=True)
    except BaseException as exc:
        atomic(state, {"status": "FAILED_PRESERVE_PARTIAL", "error": repr(exc),
                       "elapsed_seconds": time.monotonic()-start})
        raise


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: supervised_wsi_pipeline.py PUBLIC_SLUG")
    main(sys.argv[1])
