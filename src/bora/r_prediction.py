"""Source-bound supervised score projection for the developing R interface.

Score arithmetic reuses the frozen Python supervised implementation. The
spatial projection is bounded in output blocks and accepts an explicit output
pixel spacing; no whole-slide RGB array is copied into R.
"""
from __future__ import annotations

import json
import hashlib
import math
import mmap
import os
from pathlib import Path
import colorsys
import resource

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from .r_feature_preflight import sha256
from .r_training import CHANNELS, FORMAT as TRAINING_FORMAT, _verified_scales
from .supervised import _scores


def load_model(predictor_dir: Path) -> dict:
    """Verify a new R predictor and return numeric arrays with source hashes."""
    root = Path(predictor_dir).resolve()
    receipt_path = root / "TRAINING_COMPLETE.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("format") != TRAINING_FORMAT or receipt.get("status") != "COMPLETE":
        raise ValueError("A complete Bora R training receipt is required")
    model_path = Path(receipt["model_path"]).resolve()
    if model_path.parent != root or sha256(model_path) != receipt["model_sha256"]:
        raise ValueError("Predictor path or SHA256 differs")
    with np.load(model_path, allow_pickle=False) as data:
        if str(data["format"].item()) != TRAINING_FORMAT or \
           str(data["model"].item()) != "classwise-ridge":
            raise ValueError("Unsupported Bora R predictor format")
        model = {name: data[name].copy() for name in
                 ("selected", "mean", "std", "coef", "intercept", "classes", "factors")}
        model["preparation_sha256"] = str(data["preparation_sha256"].item())
        model["training_inputs_sha256"] = str(data["training_inputs_sha256"].item())
        model["binary_margin_threshold"] = (float(data["binary_margin_threshold"].item())
            if "binary_margin_threshold" in data.files else 0.0)
    selected, classes = model["selected"], model["classes"]
    nclass = len(classes)
    if (selected.ndim != 1 or len(selected) < 1 or len(selected) > CHANNELS or
            selected.min() < 0 or selected.max() >= CHANNELS or
            len(np.unique(selected)) != len(selected) or
            classes.ndim != 1 or nclass < 2 or nclass > 32 or
            classes.min() < 1 or classes.max() > 254 or
            not np.all(np.diff(classes) > 0) or
            not np.array_equal(model["factors"], [1]) or
            model["mean"].shape != selected.shape or
            model["std"].shape != selected.shape or
            model["coef"].shape != (1 if nclass == 2 else nclass, len(selected)) or
            model["intercept"].shape != (1 if nclass == 2 else nclass,) or
            not all(np.isfinite(model[name]).all() for name in
                    ("mean", "std", "coef", "intercept")) or
            np.any(model["std"] <= 0)):
        raise ValueError("Bora R predictor arrays differ from their contract")
    if not math.isfinite(model["binary_margin_threshold"]) or \
       (nclass != 2 and model["binary_margin_threshold"] != 0):
        raise ValueError("Binary margin threshold differs from predictor contract")
    if "binary_threshold_calibration" in receipt and nclass == 2 and \
       receipt["binary_threshold_calibration"] is not None and \
       receipt["binary_threshold_calibration"]["threshold"] != model["binary_margin_threshold"]:
        raise ValueError("Predictor threshold differs from training receipt")
    if model["training_inputs_sha256"] != sha256(root / "TRAINING_INPUTS.json"):
        raise ValueError("Predictor training inputs changed")
    if model["preparation_sha256"] != receipt["preparation_sha256"]:
        raise ValueError("Predictor preparation identity differs")
    if receipt["classes"] != list(map(int, classes)) or \
       receipt["selected_concatenated_channels"] != selected.tolist():
        raise ValueError("Predictor and training receipt differ")
    model["model_path"] = str(model_path)
    model["model_sha256"] = receipt["model_sha256"]
    model["receipt_path"] = str(receipt_path)
    model["receipt_sha256"] = sha256(receipt_path)
    return model


def score_case(preparation: dict, case: dict, model: dict, output: Path,
               *, backend: str = "cpu") -> dict:
    """Project verified factor-1 tokens, preserving the historical math."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    grids = _verified_scales(preparation, case)
    grid = grids[1]
    path = output / "raw-scores.npy"
    if backend == "cuda":
        from .r_compute import scores_cuda
        counts = scores_cuda(grid, model, path)
    elif backend == "cpu":
        counts = _scores(grid, model, path, roi_order=False)
    else:
        raise ValueError("backend must be 'cpu' or 'cuda'")
    raw = np.load(path, mmap_mode="r", allow_pickle=False)
    if raw.dtype != np.float32 or raw.shape[0] != len(model["classes"]):
        raise ValueError("Projected score grid differs from trained classes")
    smooth_path = output / "smoothed-scores.npy"
    smooth = np.lib.format.open_memmap(smooth_path, mode="w+", dtype=np.float32,
                                       shape=raw.shape)
    for index in range(raw.shape[0]):
        smooth[index] = gaussian_filter(np.asarray(raw[index]), sigma=.75,
                                        mode="nearest")
        smooth.flush()
    del smooth
    return {"raw_scores_path": str(path), "raw_scores_sha256": sha256(path),
            "smoothed_scores_path": str(smooth_path),
            "smoothed_scores_sha256": sha256(smooth_path),
            "score_shape_class_yx": list(raw.shape),
            "finite_token_count": counts["finite_tokens"],
            "nearest_imputed_token_count": counts["imputed_tokens"]}


def output_shape(image_shape_yx: tuple[int, int],
                 source_mpp_xy: tuple[float, float],
                 output_mpp_xy: tuple[float, float]) -> tuple[int, int]:
    if len(image_shape_yx) != 2 or min(image_shape_yx) < 1 or \
       len(source_mpp_xy) != 2 or len(output_mpp_xy) != 2 or \
       not all(math.isfinite(float(v)) and v > 0
               for v in (*source_mpp_xy, *output_mpp_xy)):
        raise ValueError("Positive image dimensions and measured pixel spacing required")
    return (math.ceil(image_shape_yx[0] * source_mpp_xy[1] / output_mpp_xy[1]),
            math.ceil(image_shape_yx[1] * source_mpp_xy[0] / output_mpp_xy[0]))


def output_support(source: dict, shape_yx: tuple[int, int],
                   ratio_xy: tuple[float, float], output: Path) -> dict:
    """Sample supplied tissue at output pixel centres in bounded strips.

    Without a supplied support mask, all original image pixels, including
    glass, remain eligible. Output pixels centred in artificial padding are
    zero. GeoJSON coordinates must be aligned source-image pixel edges.
    """
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    from rasterio.windows import Window

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    image_h, image_w = map(int, source["image_shape_yx"])
    out_h, out_w = shape_yx
    if min(image_h, image_w, out_h, out_w) < 1 or \
       len(ratio_xy) != 2 or not all(math.isfinite(v) and v > 0 for v in ratio_xy):
        raise ValueError("Invalid source/output support geometry")
    xs_float = (np.arange(out_w, dtype=np.float64) + .5) * ratio_xy[0]
    valid_x = xs_float < image_w
    xs = np.minimum(xs_float.astype(np.int64), image_w - 1)
    mask = np.memmap(output, mode="w+", dtype=np.uint8, shape=shape_yx)
    mask[:] = 0
    path = source.get("tissue_support")
    if path is None:
        for y in range(out_h):
            valid_y = (y + .5) * ratio_xy[1] < image_h
            if valid_y:
                mask[y, valid_x] = 1
    else:
        path = Path(path)
        if sha256(path) != source["tissue_support_sha256"]:
            raise ValueError("Prepared tissue support SHA256 differs")
        if path.suffix.lower() in (".json", ".geojson"):
            data = json.loads(path.read_text())
            if data.get("type") != "FeatureCollection":
                raise ValueError("Tissue GeoJSON is not a FeatureCollection")
            geometries = [item["geometry"] for item in data.get("features", [])
                          if item.get("geometry") is not None]
            if not geometries:
                raise ValueError("Tissue GeoJSON has no geometry")
            rasterize(((geometry, 1) for geometry in geometries), out=mask,
                      transform=Affine.scale(*ratio_xy), fill=0,
                      all_touched=False)
            mask[:, ~valid_x] = 0
            for y in range(out_h):
                if (y + .5) * ratio_xy[1] >= image_h:
                    mask[y] = 0
        else:
            with rasterio.open(path) as dataset:
                if (dataset.height, dataset.width) != (image_h, image_w) or dataset.count < 1:
                    raise ValueError("Prepared tissue raster dimensions differ")
                bytes_per_source_row = image_w * np.dtype(dataset.dtypes[0]).itemsize
                rows_per_read = max(1, min(64, int((64 << 20) /
                    max(1, bytes_per_source_row * max(1, math.ceil(ratio_xy[1]))))))
                for y0 in range(0, out_h, rows_per_read):
                    y1 = min(out_h, y0 + rows_per_read)
                    yy_float = (np.arange(y0, y1, dtype=np.float64) + .5) * ratio_xy[1]
                    valid_y = yy_float < image_h
                    if not valid_y.any():
                        continue
                    yy = yy_float[valid_y].astype(np.int64)
                    first, last = int(yy.min()), int(yy.max()) + 1
                    pixels = dataset.read(1, window=Window(0, first, image_w, last - first))
                    rows = np.flatnonzero(valid_y) + y0
                    mask[rows[:, None], np.flatnonzero(valid_x)[None, :]] = \
                        (pixels[(yy - first)[:, None], xs[valid_x][None, :]] != 0)
    mask.flush()
    pixels = int(np.count_nonzero(mask))
    return {"support_path": str(output), "support_sha256": sha256(output),
            "shape_yx": list(shape_yx), "positive_pixels": pixels,
            "source_support_path": str(path) if path else None,
            "source_support_sha256": source.get("tissue_support_sha256"),
            "source_to_output_ratio_xy": list(ratio_xy),
            "sampling": "source pixel at output pixel centre; padding excluded"}


def project_mask(scores: np.ndarray, classes: np.ndarray,
                 shape_yx: tuple[int, int], ratio_xy: tuple[float, float],
                 output: Path, *, support: np.ndarray | None = None,
                 block: int = 512,
                 binary_margin_threshold: float | None = None) -> dict:
    """Bicubic class decision in bounded output blocks, with optional support."""
    if output.exists():
        raise FileExistsError(output)
    if (scores.ndim != 3 or scores.dtype != np.float32 or
            scores.shape[0] != len(classes) or not 2 <= len(classes) <= 32 or
            not np.all(np.diff(classes) > 0) or classes.min() < 1 or
            classes.max() > 254 or len(shape_yx) != 2 or min(shape_yx) < 1 or
            len(ratio_xy) != 2 or not all(math.isfinite(v) and v > 0 for v in ratio_xy) or
            block < 64 or
            (binary_margin_threshold is not None and
             (len(classes) != 2 or not math.isfinite(binary_margin_threshold)))):
        raise ValueError("Invalid registered score projection")
    if not np.isfinite(scores).all():
        raise ValueError("Projected class scores contain nonfinite values")
    if support is not None and support.shape != shape_yx:
        raise ValueError("Output support shape differs")
    h, w = shape_yx
    mask = np.memmap(output, mode="w+", dtype=np.uint8, shape=(h, w))
    mapped_classes = [np.asarray(scores[i], np.float32) for i in range(len(classes))]
    total_supported = 0
    for y in range(0, h, block):
        y1 = min(h, y + block)
        yy = (np.arange(y, y1, dtype=np.float32) + .5) * ratio_xy[1] / 14 - .5
        for x in range(0, w, block):
            x1 = min(w, x + block)
            target_support = None if support is None else np.asarray(support[y:y1, x:x1]) > 0
            if target_support is not None and not target_support.any():
                mask[y:y1, x:x1] = 0
                continue
            xx = (np.arange(x, x1, dtype=np.float32) + .5) * ratio_xy[0] / 14 - .5
            map_x, map_y = np.meshgrid(xx, yy)
            if binary_margin_threshold is not None:
                first = cv2.remap(mapped_classes[0], map_x, map_y, cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
                second = cv2.remap(mapped_classes[1], map_x, map_y, cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)
                chosen = np.where(second - first > binary_margin_threshold,
                                  classes[1], classes[0]).astype(np.uint8)
            else:
                best = np.full((y1-y, x1-x), -np.inf, np.float32)
                chosen = np.zeros(best.shape, np.uint8)
                for cls, source in zip(classes, mapped_classes):
                    response = cv2.remap(source, map_x, map_y, cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_REPLICATE)
                    take = response > best
                    chosen[take] = cls
                    best[take] = response[take]
            if target_support is not None:
                chosen[~target_support] = 0
                total_supported += int(target_support.sum())
            else:
                total_supported += chosen.size
            mask[y:y1, x:x1] = chosen
        mask.flush()
    return {"mask_path": str(output), "mask_sha256": sha256(output),
            "shape_yx": [h, w], "positive_pixels": total_supported,
            "support_applied": support is not None,
            "score_interpolation": "OpenCV INTER_CUBIC with replicated border",
            "source_to_output_ratio_xy": list(ratio_xy),
            "binary_margin_threshold": binary_margin_threshold}


def finalize_projected_mask(shared_mask_path: Path, support_path: Path,
                            shape_yx: tuple[int, int], mpp_xy: tuple[float, float],
                            classes: list[int], source_image_sha256: str,
                            model_sha256: str, output: Path,
                            *, minimum_area_um2: float = 2000.0,
                            contour_sigma_um: float = 6.0,
                            class_names: dict[str, str] | None = None) -> dict:
    """Reuse frozen global repair, smoothing and vector guards on one grid.

    The vector publisher emits geometry in the *output grid* pixel coordinates.
    A separate native-coordinate conversion is required before public release.
    """
    from .contour_polish import smooth_global
    from .supervised_wsi import area_repair
    from .wsi_global_vector import publish_global_wsi_vectors

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if (len(shape_yx) != 2 or min(shape_yx) < 1 or
            len(mpp_xy) != 2 or not all(math.isfinite(v) and v > 0 for v in mpp_xy) or
            not 2 <= len(classes) <= 32 or len(set(classes)) != len(classes) or
            any(not 1 <= cls <= 254 for cls in classes)):
        raise ValueError("Invalid finalization geometry or class set")
    if class_names is None:
        class_names = {str(cls): f"class_{cls}" for cls in classes}
    if set(class_names) != set(map(str, classes)) or \
       any(not isinstance(value, str) or not value.strip()
           for value in class_names.values()):
        raise ValueError("Every final class requires one nonempty name")
    expected_bytes = shape_yx[0] * shape_yx[1]
    if Path(shared_mask_path).stat().st_size != expected_bytes or \
       Path(support_path).stat().st_size != expected_bytes:
        raise ValueError("Projected mask or support byte count differs")
    output.mkdir(parents=True)
    source = np.memmap(shared_mask_path, mode="r", dtype=np.uint8, shape=shape_yx)
    support = np.memmap(support_path, mode="r", dtype=np.uint8, shape=shape_yx)
    if np.any((source > 0) & (support == 0)) or \
       not np.isin(source, [0, *classes]).all():
        raise ValueError("Projected mask violates fixed support or class IDs")
    try:
        first_path = output / "area-repaired.dat"
        first, first_report = area_repair(source, support, first_path, mpp_xy,
                                         minimum_area_um2)
        smooth_path = output / "contour-smoothed.npy"
        smooth_report = smooth_global(first, support, mpp_xy, smooth_path,
                                      sigma_um=contour_sigma_um)
        smoothed = np.load(smooth_path, mmap_mode="r", allow_pickle=False)
        final_path = output / "final-mask.uint8.dat"
        final, second_report = area_repair(smoothed, support, final_path,
                                           mpp_xy, minimum_area_um2,
                                           connectivity=4)
        if np.any((final > 0) & (support == 0)):
            raise ValueError("Final cleanup expanded beyond supplied tissue")
        assembly = output / "vector-input"
        assembly.mkdir()
        semantic = assembly / "semantic-mask.uint8.dat"
        os.link(final_path, semantic)
        binding = {"source_image_sha256": source_image_sha256,
                   "model_sha256": model_sha256,
                   "support_sha256": sha256(Path(support_path)),
                   "projected_mask_sha256": sha256(Path(shared_mask_path)),
                   "output_mpp_xy": list(mpp_xy)}
        binding_sha = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
        assembly_receipt = {"format": "bora-wsi-label-assembly-v1",
                            "core_partition_complete": True,
                            "independently_fitted_windows": False,
                            "native_shape_yx": list(shape_yx),
                            "native_mpp_xy": list(mpp_xy),
                            "class_ids": sorted(map(int, classes)),
                            "semantic_mask_path": "semantic-mask.uint8.dat",
                            "semantic_mask_sha256": sha256(semantic),
                            "source_image_sha256": source_image_sha256,
                            "global_model_binding_sha256": binding_sha,
                            "source_frame_id": "bora-r-output-grid"}
        (assembly / "ASSEMBLY_COMPLETE.json").write_text(
            json.dumps(assembly_receipt, indent=2) + "\n")
        vector = publish_global_wsi_vectors(
            assembly, output / "vectors-output-grid",
            class_names,
            # Raster cleanup has already enforced the physical area floor.
            # The shared vector publisher also fills holes below its own
            # floor; applying that floor again can turn a reviewed zero
            # support pixel into tissue. A subpixel vector floor preserves
            # every cleaned raster pixel, including source-support holes.
            minimum_area_um2=min(minimum_area_um2,
                                 .5 * mpp_xy[0] * mpp_xy[1]),
            smoothing_tolerance_um=3.0,
            minimum_review_part_area_um2=minimum_area_um2)
        exact_vector_mask = output / "vectors-output-grid/exact/mask.uint8.dat"
        if sha256(exact_vector_mask) != sha256(final_path):
            raise ValueError("Final vector raster differs from the cleaned tissue mask")
        result = {"format": "bora-r-output-grid-finalization-v1",
                  "status": "COMPLETE", "coordinate_space": "output_grid_pixel_edges",
                  "not_native_coordinate_release": True,
                  "source_image_sha256": source_image_sha256,
                  "model_sha256": model_sha256,
                  "input_mask_sha256": sha256(Path(shared_mask_path)),
                  "support_sha256": sha256(Path(support_path)),
                  "output_mpp_xy": list(mpp_xy),
                  "minimum_area_um2": minimum_area_um2,
                  "vector_pixel_preserving_area_um2": min(
                      minimum_area_um2, .5 * mpp_xy[0] * mpp_xy[1]),
                  "contour_sigma_um": contour_sigma_um,
                  "first_area_repair": first_report,
                  "contour_smoothing": smooth_report,
                  "second_area_repair": second_report,
                  "final_mask_path": str(final_path),
                  "final_mask_sha256": sha256(final_path),
                  "vector_receipt_path": str(output / "vectors-output-grid/GLOBAL_VECTOR_COMPLETE.json"),
                  "vector_receipt_sha256": sha256(
                      output / "vectors-output-grid/GLOBAL_VECTOR_COMPLETE.json"),
                  "exact_vector_sha256": vector["output_sha256"]["exact/mask.geojson"]}
        (output / "FINALIZATION_COMPLETE.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    except BaseException as error:
        (output / "FINALIZATION_FAILED.json").write_text(json.dumps({
            "format": "bora-r-output-grid-finalization-failure-v1",
            "status": "FAILED_PRESERVE_PARTIAL", "error": repr(error)}) + "\n")
        raise


def source_coordinate_geojson(output_grid_geojson: Path, destination: Path,
                              source_shape_yx: tuple[int, int],
                              source_mpp_xy: tuple[float, float],
                              output_mpp_xy: tuple[float, float],
                              *, max_feature_bytes: int = 256 << 20) -> dict:
    """Stream globally fitted vector features into calibrated-image coordinates.

    The source is the verified global output-grid GeoJSON, not a set of
    independently fitted windows. Only one feature object is decoded at a time.
    """
    if destination.exists():
        raise FileExistsError(destination)
    if (len(source_shape_yx) != 2 or min(source_shape_yx) < 1 or
            len(source_mpp_xy) != 2 or len(output_mpp_xy) != 2 or
            not all(math.isfinite(float(v)) and v > 0
                    for v in (*source_mpp_xy, *output_mpp_xy)) or
            max_feature_bytes < 1024):
        raise ValueError("Measured source/output pixel geometry is required")
    ratio_x = output_mpp_xy[0] / source_mpp_xy[0]
    ratio_y = output_mpp_xy[1] / source_mpp_xy[1]
    marker = b',"features":['

    def transformed(node):
        if isinstance(node, (list, tuple)) and len(node) >= 2 and \
           isinstance(node[0], (int, float)):
            return [float(node[0]) * ratio_x, float(node[1]) * ratio_y]
        return [transformed(child) for child in node]

    with Path(output_grid_geojson).open("rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            index = mapped.find(marker)
            if index < 0 or index > (1 << 20):
                raise ValueError("Global vector collection header differs")
            header = json.loads(mapped[:index].decode() + "}")
            if header.get("type") != "FeatureCollection" or \
               header.get("coordinate_space") != "native_image_pixel_edges":
                raise ValueError("Global vector coordinate contract differs")
            output_shape = header["native_shape_yx"]
            header.update({"coordinate_space": "calibrated_source_image_pixel_edges",
                           "native_shape_yx": list(source_shape_yx),
                           "native_mpp_xy": list(source_mpp_xy),
                           "output_grid_shape_yx": output_shape,
                           "output_grid_mpp_xy": list(output_mpp_xy),
                           "output_grid_geojson_sha256": sha256(Path(output_grid_geojson)),
                           "source_pixels_per_output_pixel_xy": [ratio_x, ratio_y]})
            cursor = index + len(marker)
            temporary = destination.with_name(destination.name + ".partial")
            count = 0
            with temporary.open("x") as target:
                target.write(json.dumps(header, separators=(",", ":"), allow_nan=False)[:-1])
                target.write(',"features":[')
                while mapped[cursor] != ord("]"):
                    if mapped[cursor] != ord("{"):
                        raise ValueError("Expected one GeoJSON feature object")
                    begin = cursor
                    depth, quoted, escaped = 0, False, False
                    while cursor < len(mapped):
                        byte = mapped[cursor]
                        cursor += 1
                        if quoted:
                            if escaped:
                                escaped = False
                            elif byte == ord("\\"):
                                escaped = True
                            elif byte == ord('"'):
                                quoted = False
                        elif byte == ord('"'):
                            quoted = True
                        elif byte == ord("{"):
                            depth += 1
                        elif byte == ord("}"):
                            depth -= 1
                            if depth == 0:
                                break
                        if cursor - begin > max_feature_bytes:
                            raise ValueError("One vector feature exceeds memory bound")
                    if depth != 0:
                        raise ValueError("Incomplete global vector feature")
                    feature = json.loads(mapped[begin:cursor])
                    if feature.get("type") != "Feature" or \
                       feature.get("geometry", {}).get("type") not in ("Polygon", "MultiPolygon"):
                        raise ValueError("Global vector feature is not polygonal")
                    feature["geometry"]["coordinates"] = transformed(
                        feature["geometry"]["coordinates"])
                    properties = feature.setdefault("properties", {})
                    properties["area_output_grid_pixels"] = properties.get("area_pixels")
                    if properties.get("area_pixels") is not None:
                        properties["area_pixels"] *= ratio_x * ratio_y
                    if count:
                        target.write(",")
                    json.dump(feature, target, separators=(",", ":"), allow_nan=False)
                    count += 1
                    if mapped[cursor] == ord(","):
                        cursor += 1
                    elif mapped[cursor] != ord("]"):
                        raise ValueError("Global vector feature delimiter differs")
                if mapped[cursor:cursor + 2] != b"]}":
                    raise ValueError("Global vector collection terminator differs")
                target.write("]}\n")
                target.flush()
                os.fsync(target.fileno())
    os.replace(temporary, destination)
    return {"source_geojson_path": str(destination),
            "source_geojson_sha256": sha256(destination),
            "output_grid_geojson_sha256": sha256(Path(output_grid_geojson)),
            "feature_count": count,
            "source_pixels_per_output_pixel_xy": [ratio_x, ratio_y],
            "source_shape_yx": list(source_shape_yx)}


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_mask_tiff(array: np.ndarray, path: Path, *, equals: int | None = None) -> dict:
    import rasterio
    from rasterio.windows import Window

    if path.exists():
        raise FileExistsError(path)
    height, width = array.shape
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height,
                       count=1, dtype="uint8", tiled=True, blockxsize=512,
                       blockysize=512, compress="zstd", BIGTIFF="IF_SAFER") as target:
        for y in range(0, height, 512):
            for x in range(0, width, 512):
                block = np.asarray(array[y:y + 512, x:x + 512], np.uint8)
                if equals is not None:
                    block = np.where(block == equals, 255, 0).astype(np.uint8)
                target.write(block, 1, window=Window(x, y, block.shape[1], block.shape[0]))
    return {"path": str(path), "sha256": sha256(path),
            "shape_yx": [height, width], "class_id": equals}


def _preview(image_path: Path, final_tiff: Path, classes: list[int], path: Path,
             *, source_rgb_cache: Path | None = None,
             backend: str = "vips-lanczos3") -> dict:
    import rasterio
    from rasterio.enums import Resampling
    from PIL import Image

    if backend not in ("vips-lanczos3", "gdal-lanczos"):
        raise ValueError("Unknown preview backend")
    if source_rgb_cache is not None and source_rgb_cache.exists():
        rgb = np.load(source_rgb_cache, allow_pickle=False)
        if (rgb.ndim != 3 or rgb.shape[2] != 3 or max(rgb.shape[:2]) > 1024
                or rgb.dtype != np.uint8):
            raise ValueError("Cached preview is not calibrated uint8 RGB")
    else:
        if backend == "vips-lanczos3":
            import pyvips
            source = pyvips.Image.new_from_file(str(image_path), access="sequential")
            scale = min(1, 1024 / max(source.width, source.height))
            small = source.resize(scale, kernel="lanczos3")
            if small.bands != 3 or small.format != "uchar":
                raise ValueError("Preview source is not calibrated uint8 RGB")
            rgb = np.ndarray(buffer=small.write_to_memory(), dtype=np.uint8,
                             shape=(small.height, small.width, 3)).copy()
        else:
            with rasterio.open(image_path) as source:
                if source.count != 3 or any(dtype != "uint8" for dtype in source.dtypes):
                    raise ValueError("Preview source is not calibrated uint8 RGB")
                scale = min(1, 1024 / max(source.width, source.height))
                height = max(1, round(source.height * scale))
                width = max(1, round(source.width * scale))
                bands = source.read((1, 2, 3), out_shape=(3, height, width),
                                    resampling=Resampling.lanczos)
                rgb = np.ascontiguousarray(np.moveaxis(bands, 0, -1))
        if source_rgb_cache is not None:
            temporary = source_rgb_cache.with_name(source_rgb_cache.name + ".partial")
            with temporary.open("xb") as stream:
                np.save(stream, rgb, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, source_rgb_cache)
    with rasterio.open(final_tiff) as raster:
        labels = raster.read(1, out_shape=rgb.shape[:2],
                             resampling=Resampling.nearest)
    overlay = rgb.astype(np.float32)
    for cls in classes:
        color = np.asarray(colorsys.hsv_to_rgb((cls * .6180339887) % 1, .75, 1)) * 255
        inside = labels == cls
        overlay[inside] = .45 * overlay[inside] + .55 * color
    Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).save(path)
    result = {"path": str(path), "sha256": sha256(path),
            "shape_yx": list(rgb.shape[:2]),
            "format": "RGB overlay; colors identify class IDs"}
    if backend != "vips-lanczos3":
        result["source_thumbnail_backend"] = backend
    return result


def _resource_admission(preparation: dict, model: dict,
                        source_mpp_xy: tuple[float, float],
                        output_mpp_xy: tuple[float, float], output: Path,
                        minimum_free_gib: float, max_new_gib: float | None) -> dict:
    if not math.isfinite(minimum_free_gib) or minimum_free_gib < 0 or \
       (max_new_gib is not None and (not math.isfinite(max_new_gib) or max_new_gib <= 0)):
        raise ValueError("Finite disk reserve and optional campaign cap required")
    estimates = []
    for case in preparation["cases"]:
        h, w = output_shape(tuple(case["source"]["image_shape_yx"]),
                            source_mpp_xy, output_mpp_xy)
        token_h, token_w = (case["source"]["padded_shape_yx"][0] // 14,
                            case["source"]["padded_shape_yx"][1] // 14)
        pixels = h * w
        score_bytes = len(model["classes"]) * token_h * token_w * 4 * 2
        # Native, shared support, repair/smoothing, exact vectors' raster,
        # TIFFs/BW and temporary polygonization; reserve extra for metadata.
        projected_bytes = pixels * (16 + len(model["classes"])) + score_bytes + (1 << 30)
        estimates.append({"image": case["source"]["image"],
                          "output_shape_yx": [h, w], "output_pixels": pixels,
                          "projected_peak_new_bytes": projected_bytes})
    projected = sum(item["projected_peak_new_bytes"] for item in estimates)
    vfs = os.statvfs(output.parent)
    free = vfs.f_bavail * vfs.f_frsize
    if free - projected < minimum_free_gib * 1024**3:
        raise OSError("Prediction would breach free-space reserve")
    if max_new_gib is not None and projected > max_new_gib * 1024**3:
        raise OSError("Prediction would breach declared campaign peak cap")
    with Path("/proc/meminfo").open() as stream:
        memory_kib = next(int(line.split()[1]) for line in stream
                          if line.startswith("MemAvailable:"))
    peak_case_pixels = max(item["output_pixels"] for item in estimates)
    estimated_host_bytes = peak_case_pixels * 24 + max(
        item["projected_peak_new_bytes"] - item["output_pixels"] *
        (16 + len(model["classes"])) - (1 << 30) for item in estimates)
    if estimated_host_bytes > memory_kib * 1024 * .7:
        raise MemoryError("Prediction projected host memory exceeds admission bound")
    return {"free_bytes_before": free, "projected_peak_new_bytes": projected,
            "minimum_free_gib": minimum_free_gib, "max_new_gib": max_new_gib,
            "mem_available_kib_before": memory_kib,
            "estimated_peak_host_bytes": estimated_host_bytes,
            "cases": estimates}


def predict_supervised(preparation_dir: Path, predictor_dir: Path, output_dir: Path,
                       *, source_mpp_xy: tuple[float, float] = (.5, .5),
                       output_mpp_xy: tuple[float, float] = (4.0, 4.0),
                       minimum_area_um2: float = 2000.0,
                       contour_sigma_um: float = 6.0,
                       minimum_free_gib: float = 1.0,
                       max_new_gib: float | None = None,
                       preview_backend: str = "vips-lanczos3",
                       class_names: dict[str, str] | None = None,
                       backend: str = "cpu", n_cores: int = 1) -> dict:
    """Complete the supervised route with native and common-support outputs."""
    from .r_compute import validate_execution
    backend, n_cores = validate_execution(backend, n_cores)
    root = Path(preparation_dir).resolve()
    manifest_path = root / "PREPARATION_COMPLETE.json"
    preparation = json.loads(manifest_path.read_text())
    if preparation.get("format") != "bora-r-preparation-v1" or \
       preparation.get("status") != "COMPLETE" or not preparation.get("cases"):
        raise ValueError("Complete Bora R preparation required")
    model = load_model(predictor_dir)
    if class_names is not None and (
            not isinstance(class_names, dict) or
            set(class_names) != set(map(str, model["classes"])) or
            any(not isinstance(value, str) or not value.strip()
                for value in class_names.values())):
        raise ValueError("class_names must name exactly the fitted class IDs")
    if not all(math.isfinite(v) and v > 0 for v in (*source_mpp_xy, *output_mpp_xy)) or \
       minimum_area_um2 <= 0 or contour_sigma_um <= 0:
        raise ValueError("Measured spacing and positive cleanup limits required")
    if preview_backend not in ("vips-lanczos3", "gdal-lanczos"):
        raise ValueError("Unknown preview backend")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    admission = _resource_admission(preparation, model, source_mpp_xy,
                                     output_mpp_xy, output,
                                     minimum_free_gib, max_new_gib)
    output.mkdir(parents=True)
    _atomic_json(output / "RESOURCE_ADMISSION.json", admission)
    cases = []
    try:
        for index, case in enumerate(preparation["cases"]):
            source = case["source"]
            image_path = Path(source["image"])
            if sha256(image_path) != source["image_sha256"]:
                raise ValueError("Prepared H&E image SHA256 differs")
            folder = output / f"image-{index:03d}"
            folder.mkdir()
            score = score_case(preparation, case, model, folder / "scores",
                               backend=backend)
            shape_yx = output_shape(tuple(source["image_shape_yx"]),
                                    source_mpp_xy, output_mpp_xy)
            ratio = (output_mpp_xy[0] / source_mpp_xy[0],
                     output_mpp_xy[1] / source_mpp_xy[1])
            support_receipt = output_support(source, shape_yx, ratio,
                                              folder / "output-support.uint8.dat")
            support = np.memmap(support_receipt["support_path"], mode="r",
                                dtype=np.uint8, shape=shape_yx)
            smooth = np.load(score["smoothed_scores_path"], mmap_mode="r")
            native = project_mask(smooth, model["classes"], shape_yx, ratio,
                                  folder / "native-mask.uint8.dat",
                                  binary_margin_threshold=model["binary_margin_threshold"])
            clipped = project_mask(smooth, model["classes"], shape_yx, ratio,
                                   folder / "support-clipped-mask.uint8.dat",
                                   support=support,
                                   binary_margin_threshold=model["binary_margin_threshold"])
            final = finalize_projected_mask(
                Path(clipped["mask_path"]), Path(support_receipt["support_path"]),
                shape_yx, output_mpp_xy, list(map(int, model["classes"])),
                source["image_sha256"], model["model_sha256"], folder / "cleanup",
                minimum_area_um2=minimum_area_um2,
                contour_sigma_um=contour_sigma_um,
                class_names=class_names)
            # Benchmark 2's guarded fitted contours are the downloadable
            # vector view. The exact vector remains beside the scored raster.
            review_geojson = folder / "cleanup/vectors-output-grid/review/mask.geojson"
            source_vector = source_coordinate_geojson(
                review_geojson, folder / "class-mask.geojson",
                tuple(source["image_shape_yx"]), source_mpp_xy, output_mpp_xy)
            final_mask = np.memmap(final["final_mask_path"], mode="r",
                                   dtype=np.uint8, shape=shape_yx)
            masks = {"native": _write_mask_tiff(
                np.memmap(native["mask_path"], mode="r", dtype=np.uint8,
                          shape=shape_yx), folder / "native-mask.tif"),
                     "final": _write_mask_tiff(final_mask, folder / "class-mask.tif")}
            binary = {}
            for cls in model["classes"]:
                binary[str(int(cls))] = _write_mask_tiff(
                    final_mask, folder / f"bw-class-{int(cls):03d}.tif",
                    equals=int(cls))
            preview = _preview(image_path, Path(masks["final"]["path"]),
                               list(map(int, model["classes"])),
                               folder / "preview.png", backend=preview_backend)
            record = {"format": "bora-r-prediction-case-v1", "status": "COMPLETE",
                      "image": str(image_path),
                      "image_sha256": source["image_sha256"],
                      "preparation_manifest_sha256": sha256(manifest_path),
                      "predictor_sha256": model["model_sha256"],
                      "backend": backend, "n_cores": n_cores,
                      "source_mpp_xy": list(source_mpp_xy),
                      "output_mpp_xy": list(output_mpp_xy),
                      "classes": list(map(int, model["classes"])),
                      "scores": score, "support": support_receipt,
                      "native_projection": native,
                      "support_clipped_projection": clipped,
                      "finalization": {"receipt_path": str(folder / "cleanup/FINALIZATION_COMPLETE.json"),
                                       "receipt_sha256": sha256(
                                           folder / "cleanup/FINALIZATION_COMPLETE.json")},
                      "source_vector": source_vector,
                      "masks": masks, "binary_class_images": binary,
                      "preview": preview}
            receipt_path = folder / "CASE_COMPLETE.json"
            _atomic_json(receipt_path, record)
            cases.append({"receipt_path": str(receipt_path),
                          "receipt_sha256": sha256(receipt_path)})
        result = {"format": "bora-r-prediction-v1", "status": "COMPLETE",
                  "route": "supervised-classwise-ridge",
                  "preparation_manifest_sha256": sha256(manifest_path),
                  "predictor_sha256": model["model_sha256"],
                  "resource_admission_sha256": sha256(output / "RESOURCE_ADMISSION.json"),
                  "backend": backend, "n_cores": n_cores,
                  "cases": cases,
                  "peak_host_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "gpu_used": backend == "cuda"}
        _atomic_json(output / "PREDICTION_COMPLETE.json", result)
        return result
    except BaseException as error:
        _atomic_json(output / "PREDICTION_FAILED.json",
                     {"format": "bora-r-prediction-failure-v1", "status": "FAILED",
                      "error": repr(error), "completed_cases": cases})
        raise
