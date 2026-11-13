"""Full-slide supervised mask, cleanup and vector route.

Ported from the checksum-frozen Chiamaka ridge finalizer. It accepts an
explicit manifest instead of project-specific absolute paths or public slugs.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

import cv2
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.windows import Window
from shapely.geometry import mapping, shape
from shapely.ops import transform

from .contour_polish import smooth_global
from .wsi_global_vector import publish_global_wsi_vectors
from .supervised import digest, _input


def sha(path: Path) -> str:
    return digest(path)


def atomic(path: Path, value: dict) -> None:
    if path.exists() and path.name != "STATE.json":
        raise FileExistsError(path)
    temp = path.with_name(path.name + ".partial")
    if temp.exists():
        raise FileExistsError(temp)
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _context(manifest: Path, spec: dict, score_path: Path) -> dict:
    wsi = spec["wsi"]
    source_path = _input(manifest, wsi, "source_image")
    geo_path = _input(manifest, wsi, "support_geojson")
    geo = json.loads(geo_path.read_text())
    if geo.get("coordinate_space") != "original_WSI_native_pixel_edges" or geo.get("native_origin_xy") != [0, 0]:
        raise ValueError("Expected reviewed Bora native-coordinate tissue support")
    sx, sy = map(float, geo["working_pixel_to_native_scale_xy"])
    if min(sx, sy) <= 0:
        raise ValueError("Invalid working-to-native scaling")
    height, width = map(int, wsi["calibrated_shape_yx"])
    if min(height, width) <= 0:
        raise ValueError("Invalid calibrated shape")
    working_shape = geo["full_slide_input_identity"]["shape_yx"]
    if working_shape[0] > height or working_shape[1] > width:
        raise ValueError("Reviewed support exceeds calibrated frame")
    with rasterio.open(source_path) as src:
        if (src.height, src.width) != (height, width):
            raise ValueError("Calibrated image header differs")
    working = [transform(lambda x, y, z=None: (x/sx, y/sy),
                         shape(feature["geometry"])) for feature in geo["features"]]
    if not working or any(not g.is_valid for g in working):
        raise ValueError("Invalid reviewed tissue geometry")
    scores = np.load(score_path, mmap_mode="r", allow_pickle=False)
    if scores.ndim != 3 or scores.shape[0] != 8 or scores.dtype != np.float32:
        raise ValueError("Expected eight float32 score grids")
    if scores.shape[1:] != (math.ceil(height/14), math.ceil(width/14)):
        raise ValueError("Score grid and calibrated image differ")
    presence_path = _input(manifest, wsi, "class_presence_receipt")
    presence = json.loads(presence_path.read_text())
    threshold = float(wsi["minimum_predicted_tissue_fraction"])
    if threshold != .02:
        raise ValueError("The frozen full-slide class-presence floor is 0.02")
    if (presence.get("status") != "TRAINING_ONLY_RIDGE_CLASS_PRESENCE_RULE_FROZEN" or
            float(presence.get("minimum_predicted_tissue_fraction", -1)) != threshold or
            presence.get("development_masks_opened") != 0 or
            presence.get("whole_slide_expert_masks_opened") != 0):
        raise ValueError("The training-only class-presence receipt differs")
    mpp = geo["working_mpp_xy"]
    if len(mpp) != 2 or any(not math.isfinite(v) or v <= 0 for v in mpp):
        raise ValueError("Invalid physical spacing")
    ontology_path = _input(manifest, wsi, "ontology")
    ontology = json.loads(ontology_path.read_text())
    names = {int(k): v["name"].replace("_", " ") for k, v in ontology["classes"].items()}
    if set(names) != set(range(1, 9)) or not all(names.values()):
        raise ValueError("Eight tissue-class names are required")
    return {"source": {"height": height, "width": width, "sha256": sha(source_path)},
            "geoj": geo, "geo_path": geo_path, "working": working,
            "scores": scores, "score_path": score_path, "threshold": threshold,
            "names": names, "model_path": _input(manifest, spec, "model"),
            "presence_sha256": sha(presence_path), "ontology_sha256": sha(ontology_path),
            "source_id": wsi["source_frame_id"]}


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
    binding = {"model_sha256": sha(context["model_path"]),
               "class_presence_sha256": context["presence_sha256"],
               "score_sha256": sha(context["score_path"]),
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
        "source_frame_id": context["source_id"], "global_model_binding_sha256": bind_sha,
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


def run_wsi(manifest: Path, spec: dict, score_path: Path, output: Path) -> dict:
    """Finalize one complete score grid; refuse crops and in-place retries."""
    start = time.monotonic()
    context = _context(Path(manifest), spec, Path(score_path))
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    state = output / "STATE.json"
    try:
        source = context["source"]
        shape_yx = (source["height"], source["width"])
        mpp = tuple(context["geoj"]["working_mpp_xy"])
        atomic(state, {"status": "RUNNING", "stage": "reviewed_support"})
        support = reviewed_support(context["working"], shape_yx,
                                   output / "reviewed-support.uint8.dat")
        classes, class_report = select_classes(context["scores"], context["working"], context["threshold"])
        if len(classes) < 2:
            raise ValueError("Global vector requires at least two evidence-supported classes")
        atomic(output / "CLASS_SELECTION.json", class_report)
        atomic(state, {"status": "RUNNING", "stage": "global_projection", "class_ids": classes})
        raw = project_mask(context["scores"], support, classes, output / "raw-mask.uint8.dat")
        pre, pre_report = area_repair(raw, support, output / "precontour-400um2.uint8.dat", mpp, 400)
        atomic(state, {"status": "RUNNING", "stage": "six_um_contours"})
        smooth_path = output / "contour-6um.npy"
        contour_report = smooth_global(pre, support, mpp, smooth_path, sigma_um=6, block_size=1024)
        smoothed = np.load(smooth_path, mmap_mode="r")
        atomic(state, {"status": "RUNNING", "stage": "global_2000um2_cleanup"})
        final, final_report = area_repair(smoothed, support,
                                          output / "prevector-2000um2.uint8.dat", mpp, 2000)
        del raw, pre, smoothed, final, support
        gc.collect()
        atomic(state, {"status": "RUNNING", "stage": "global_vector"})
        assembly = bind_assembly(context["source_id"], output / "prevector-2000um2.uint8.dat",
                                 context, classes, output)
        vector = publish_global_wsi_vectors(
            assembly, output / "global-vectors", context["names"],
            minimum_area_um2=2000, smoothing_tolerance_um=0,
            maximum_cleanup_change=.02, max_input_regions=250_000,
            max_coordinates=10_000_000, max_boundary_edges=10_000_000,
            disk_budget_bytes=12 * 1024**3)
        exact = output / "global-vectors/exact/mask.uint8.dat"
        old = np.memmap(exact, mode="r", dtype=np.uint8, shape=shape_yx)
        candidate = output / "postvector-candidate.uint8.dat"
        support = np.memmap(output / "reviewed-support.uint8.dat", mode="r", dtype=np.uint8,
                            shape=shape_yx)
        cleaned, post_report = area_repair(old, support, candidate, mpp, 2000)
        changed = old_positive = 0
        for y in range(0, shape_yx[0], 512):
            different = old[y:y+512] != cleaned[y:y+512]
            changed += int(different.sum())
            old_positive += int(np.count_nonzero(np.asarray(old[y:y+512])[different]))
        if old_positive:
            raise ValueError("Global vector has a subthreshold positive raster")
        if changed > .001 * int(np.count_nonzero(old)):
            raise ValueError("Vector gap repair would change too much tissue")
        working_path = output / "supervised-bora.working.geojson"
        final_mask = output / "supervised-bora.final-2000um2.uint8.dat"
        if changed:
            from .supervised_vector_gaps import patched_working_geojson, audit_working_raster
            atomic(state, {"status": "RUNNING", "stage": "unoccupied_vector_gap_repair",
                           "gap_pixels": changed})
            patch = patched_working_geojson(output / "global-vectors/exact/mask.geojson",
                                            old, cleaned, working_path, mpp)
            audit_working_raster(working_path, cleaned, output / "working-roundtrip.partial.dat")
            os.replace(candidate, final_mask)
        else:
            candidate.unlink()
            os.link(exact, final_mask)
            os.link(output / "global-vectors/exact/mask.geojson", working_path)
            patch = {"patches": 0, "source": "exact global vector unchanged"}
        del old, cleaned, support
        gc.collect()
        native = output / "supervised-bora.native.geojson"
        native_report = native_geojson(working_path, context, native)
        tiff = output / "supervised-bora.mask-0p5um.tif"
        write_mask_tiff(final_mask, shape_yx, tiff)
        geometry_audit = audit_final(final_mask, output / "reviewed-support.uint8.dat", native, context)
        result = {"status": "SUPERVISED_BORA_FULL_SLIDE_VECTOR_VERIFIED",
                  "format": "bora-supervised-wsi-v1", "source_frame_id": context["source_id"],
                  "source_image_sha256": source["sha256"], "score_sha256": sha(score_path),
                  "source_bora_geojson_sha256": sha(context["geo_path"]),
                  "training_model_sha256": sha(context["model_path"]),
                  "training_presence_sha256": context["presence_sha256"],
                  "ontology_sha256": context["ontology_sha256"], "class_ids": classes,
                  "class_selection": class_report, "precontour_cleanup": pre_report,
                  "contour_smoothing": contour_report, "prevector_cleanup": final_report,
                  "postvector_cleanup": post_report, "postvector_gap_pixels": changed,
                  "vector_patch": patch,
                  "vector_receipt_sha256": sha(output / "global-vectors/GLOBAL_VECTOR_COMPLETE.json"),
                  "vector_summary": {"input_regions": vector["input_regions"],
                                     "exact_component_count": vector["exact_component_count"],
                                     "cleanup_changed_pixels": vector["cleanup_changed_pixels"],
                                     "global_contours": vector["global_contours"]},
                  "native_vector": native_report, "calibrated_shape_yx": list(shape_yx),
                  "independent_geometry_audit": geometry_audit,
                  "calibrated_mpp_xy": list(mpp), "native_mpp_xy": context["geoj"]["native_mpp_xy"],
                  "output_sha256": {"mask_tiff": sha(tiff), "native_geojson": sha(native),
                                    "exact_mask": sha(final_mask),
                                    "working_geojson": sha(working_path)},
                  "source_code_sha256": sha(Path(__file__)), "gpu_used": False,
                  "whole_slide_expert_accuracy": None, "elapsed_seconds": time.monotonic()-start}
        atomic(output / "FULL_SLIDE_VERIFIED.json", result)
        atomic(state, {"status": "COMPLETE", "receipt_sha256": sha(output / "FULL_SLIDE_VERIFIED.json")})
        return result
    except BaseException as exc:
        atomic(state, {"status": "FAILED_PRESERVE_PARTIAL", "error": repr(exc),
                       "elapsed_seconds": time.monotonic()-start})
        raise


def verify_wsi(folder: Path) -> dict:
    folder = Path(folder)
    receipt_path = folder / "FULL_SLIDE_VERIFIED.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("format") != "bora-supervised-wsi-v1" or receipt.get("status") != "SUPERVISED_BORA_FULL_SLIDE_VECTOR_VERIFIED":
        raise ValueError("Invalid full-slide receipt")
    for name, relative in (("mask_tiff", "supervised-bora.mask-0p5um.tif"),
                           ("native_geojson", "supervised-bora.native.geojson"),
                           ("exact_mask", "supervised-bora.final-2000um2.uint8.dat"),
                           ("working_geojson", "supervised-bora.working.geojson")):
        if sha(folder / relative) != receipt["output_sha256"][name]:
            raise ValueError(f"Changed full-slide output: {name}")
    if sha(folder / "global-vectors/GLOBAL_VECTOR_COMPLETE.json") != receipt["vector_receipt_sha256"]:
        raise ValueError("Changed full-slide vector receipt")
    with rasterio.open(folder / "supervised-bora.mask-0p5um.tif") as src:
        if (src.height, src.width) != tuple(receipt["calibrated_shape_yx"]):
            raise ValueError("Full-slide TIFF shape differs")
        mask = np.memmap(folder / "supervised-bora.final-2000um2.uint8.dat", mode="r",
                         dtype=np.uint8, shape=(src.height, src.width))
        for y in range(0, src.height, 512):
            for x in range(0, src.width, 512):
                h, w = min(512, src.height-y), min(512, src.width-x)
                if not np.array_equal(src.read(1, window=Window(x, y, w, h)), mask[y:y+h, x:x+w]):
                    raise ValueError("Full-slide TIFF pixels differ from exact mask")
        del mask
    native = json.loads((folder / "supervised-bora.native.geojson").read_text())
    # The native vector stores native spacing; the calibrated spacing is bound
    # separately by the completed full-slide receipt. Reconstruct verification
    # context without changing the frozen vector's public coordinate schema.
    native["working_mpp_xy"] = receipt["calibrated_mpp_xy"]
    context = {"source": {"height": receipt["calibrated_shape_yx"][0],
                           "width": receipt["calibrated_shape_yx"][1]}, "geoj": native}
    audit = audit_final(folder / "supervised-bora.final-2000um2.uint8.dat",
                        folder / "reviewed-support.uint8.dat",
                        folder / "supervised-bora.native.geojson", context)
    if audit != receipt["independent_geometry_audit"]:
        raise ValueError("Full-slide geometry audit differs")
    return {"status": "VERIFIED", "class_ids": receipt["class_ids"],
            "receipt_sha256": sha(receipt_path)}

