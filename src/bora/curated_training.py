"""Supervised calibration of Bora against curated WSI GeoJSON annotations.

This module deliberately keeps calibration separate from inference: targets are
used only by ``fit_curated_model`` and never by ``predict_curated``.  Models are
trained on image appearance plus the imperfect cluster mask, then Bora's usual
boundary/topology stages can be applied to their predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import json

import numpy as np
from scipy import ndimage as ndi
from rasterio.features import rasterize
from rasterio.features import shapes as raster_shapes
from rasterio.transform import Affine
from shapely.geometry import mapping, shape as geometry_shape
from shapely.ops import unary_union
from shapely import coverage_union_all


@dataclass(frozen=True)
class CuratedConfig:
    downsample: int = 16
    samples_per_class: int = 30000
    max_iter: int = 120
    max_leaf_nodes: int = 31
    learning_rate: float = 0.1
    random_seed: int = 17


def _spatial_shape(image_path):
    import pyvips
    image = pyvips.Image.new_from_file(str(image_path), access="sequential")
    return image.height, image.width


def read_rgb_downsampled(image_path, downsample):
    """Read a WSI at a bounded calibration resolution with libvips."""
    import pyvips
    image = pyvips.Image.new_from_file(str(image_path), access="sequential")
    if image.bands == 1 and image.get_typeof("n-pages") and image.get("n-pages") in (3, 4):
        pages = [pyvips.Image.new_from_file(str(image_path), page=i, access="sequential") for i in range(3)]
        image = pages[0].bandjoin(pages[1:])
    if image.bands > 3:
        image = image.extract_band(0, n=3)
    # ``shrink`` is an area/box reduction in libvips and avoids aliasing the
    # histology texture before feature extraction.
    image = image.shrink(float(downsample), float(downsample))
    dtype = {"uchar": np.uint8, "ushort": np.uint16, "float": np.float32}[image.format]
    return np.ndarray(buffer=image.write_to_memory(), dtype=dtype,
                      shape=(image.height, image.width, image.bands)).copy()[..., :3]


def rasterize_geojson(path, shape, downsample, *, target=False):
    data = json.loads(Path(path).read_text())
    items = []
    names = {}
    for index, feature in enumerate(data.get("features", []), 1):
        properties = feature.get("properties") or {}
        value = index if target else int(properties.get("value", index))
        classification = properties.get("classification")
        name = classification.get("name") if isinstance(classification, dict) else classification
        names[value] = name or f"class_{value}"
        items.append((feature["geometry"], value))
    labels = rasterize(items, out_shape=shape,
                       transform=Affine(downsample, 0, 0, 0, downsample, 0),
                       fill=0, dtype="uint16")
    return labels, names


def appearance_features(rgb, coarse, *, coordinate_origin=(0, 0),
                        full_shape=None, max_label=None):
    """Compact stain/texture/context features suitable for CPU WSI inference."""
    values = rgb.astype(np.float32) / (65535.0 if rgb.dtype == np.uint16 else 255.0)
    channels = [values]
    for sigma in (1.5, 5.0, 15.0):
        channels.append(np.stack([ndi.gaussian_filter(values[..., c], sigma) for c in range(3)], -1))
    gray = values.mean(axis=2)
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    channels.append(np.stack([gray, np.hypot(gx, gy)], -1))
    # Cluster IDs are categorical, not ordinal.  A one-hot representation lets
    # the model learn that (for example) clusters 1 and 9 may be similar without
    # inventing a numeric distance between their identifiers.
    max_label = max(1, int(coarse.max())) if max_label is None else int(max_label)
    channels.append(np.stack([(coarse == value).astype(np.float32)
                              for value in range(max_label + 1)], axis=-1))
    foreground = coarse > 0
    signed_distance = (ndi.distance_transform_edt(foreground) -
                       ndi.distance_transform_edt(~foreground))
    channels.append((np.clip(signed_distance, -64, 64) / 64.0)[..., None])
    # Coarse slide position provides weak anatomical context; it is deliberately
    # low dimensional and is validated on spatially held-out blocks.
    full_h, full_w = coarse.shape if full_shape is None else tuple(map(int, full_shape))
    origin_y, origin_x = map(int, coordinate_origin)
    yy = (2.0 * (np.arange(coarse.shape[0], dtype=np.float32) + origin_y) /
          max(1, full_h - 1) - 1.0)[:, None]
    xx = (2.0 * (np.arange(coarse.shape[1], dtype=np.float32) + origin_x) /
          max(1, full_w - 1) - 1.0)[None, :]
    channels.append(np.stack([np.broadcast_to(yy, coarse.shape),
                              np.broadcast_to(xx, coarse.shape)], axis=-1))
    return np.concatenate(channels, axis=2).astype(np.float32)


def stratified_indices(labels, samples_per_class, seed=17, eligible=None):
    rng = np.random.default_rng(seed)
    flat = labels.ravel()
    allowed = np.ones(flat.shape, bool) if eligible is None else np.asarray(eligible, bool).ravel()
    chosen = []
    for value in np.unique(flat):
        indices = np.flatnonzero((flat == value) & allowed)
        if len(indices) > samples_per_class:
            indices = rng.choice(indices, samples_per_class, replace=False)
        chosen.append(indices)
    return np.concatenate(chosen)


def fit_curated_model(features, target, config=CuratedConfig(), eligible=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    started = perf_counter()
    indices = stratified_indices(target, config.samples_per_class,
                                 config.random_seed, eligible)
    x = features.reshape(-1, features.shape[-1])[indices]
    y = target.ravel()[indices]
    counts = np.bincount(y)
    weights = np.array([len(y) / (max(1, len(counts)) * max(1, counts[v])) for v in y])
    model = HistGradientBoostingClassifier(
        learning_rate=config.learning_rate, max_iter=config.max_iter,
        max_leaf_nodes=config.max_leaf_nodes, l2_regularization=1.0,
        random_state=config.random_seed)
    model.fit(x, y, sample_weight=weights)
    return model, {"training_samples": int(len(indices)),
                   "runtime_seconds": perf_counter() - started}


def predict_curated(model, features, chunk_pixels=1_000_000):
    flat = features.reshape(-1, features.shape[-1])
    output = np.empty(len(flat), np.uint16)
    started = perf_counter()
    for begin in range(0, len(flat), int(chunk_pixels)):
        output[begin:begin + chunk_pixels] = model.predict(flat[begin:begin + chunk_pixels])
    return output.reshape(features.shape[:2]), perf_counter() - started


def predict_curated_tiled(model, rgb, coarse, *, tile_size=512, halo=96,
                          probability_sigma=0.0):
    """Bounded-memory inference with seam-free multiscale feature halos."""
    height, width = coarse.shape
    output = np.empty((height, width), np.uint16)
    max_label = max(1, int(coarse.max()))
    started = perf_counter()
    for y in range(0, height, int(tile_size)):
        for x in range(0, width, int(tile_size)):
            y1, x1 = min(height, y + int(tile_size)), min(width, x + int(tile_size))
            y0h, x0h = max(0, y - int(halo)), max(0, x - int(halo))
            y1h, x1h = min(height, y1 + int(halo)), min(width, x1 + int(halo))
            features = appearance_features(
                rgb[y0h:y1h, x0h:x1h], coarse[y0h:y1h, x0h:x1h],
                coordinate_origin=(y0h, x0h), full_shape=(height, width),
                max_label=max_label)
            flat = features.reshape(-1, features.shape[-1])
            if probability_sigma:
                probabilities = model.predict_proba(flat).reshape(
                    features.shape[0], features.shape[1], -1)
                probabilities = np.stack([
                    ndi.gaussian_filter(probabilities[..., index],
                                        float(probability_sigma))
                    for index in range(probabilities.shape[-1])], axis=-1)
                prediction = model.classes_[probabilities.argmax(axis=-1)]
            else:
                prediction = model.predict(flat).reshape(features.shape[:2])
            output[y:y1, x:x1] = prediction[
                y-y0h:y1-y0h, x-x0h:x1-x0h]
    return output, perf_counter() - started


def segmentation_metrics(truth, prediction):
    truth = np.asarray(truth); prediction = np.asarray(prediction)
    classes = sorted(set(np.unique(truth).tolist()) - {0})
    per_class = {}
    for value in classes:
        a, b = truth == value, prediction == value
        intersection = int(np.count_nonzero(a & b))
        union = int(np.count_nonzero(a | b))
        denom = int(a.sum() + b.sum())
        per_class[str(value)] = {"iou": intersection / union if union else 1.0,
                                 "dice": 2 * intersection / denom if denom else 1.0}
    a, b = truth > 0, prediction > 0
    intersection = int(np.count_nonzero(a & b)); union = int(np.count_nonzero(a | b))
    return {"foreground_iou": intersection / union if union else 1.0,
            "macro_iou": float(np.mean([x["iou"] for x in per_class.values()])),
            "macro_dice": float(np.mean([x["dice"] for x in per_class.values()])),
            "per_class": per_class}


def regularize_prediction(prediction, min_component=8, radius=1):
    """Remove sampling noise while retaining a mutually exclusive label map."""
    source = np.asarray(prediction)
    output = source.copy()
    if radius:
        output = ndi.median_filter(output, size=2 * int(radius) + 1)
    for value in np.unique(output):
        if value == 0:
            continue
        components, count = ndi.label(output == value)
        sizes = np.bincount(components.ravel())
        remove = np.flatnonzero(sizes < int(min_component))
        if len(remove):
            output[np.isin(components, remove)] = 0
    return output.astype(np.uint16, copy=False)


def write_scaled_geojson(prediction, output_path, downsample, properties,
                         min_area=500.0, simplify=4.0):
    """Polygonize calibrated labels in full-resolution pixel coordinates."""
    raster = np.asarray(prediction)
    grouped = {}
    transform = Affine(float(downsample), 0, 0, 0, float(downsample), 0)
    for encoded, raw_value in raster_shapes(
            raster, mask=raster > 0, transform=transform, connectivity=8):
        value = int(raw_value)
        candidate = geometry_shape(encoded)
        if candidate.area >= float(min_area):
            grouped.setdefault(value, []).append(candidate)
    features = []
    for value, parts in sorted(grouped.items()):
        # Raster components form a non-overlapping planar coverage.  The
        # coverage union is linear-time for this case and avoids the extremely
        # expensive general overlay used by ``unary_union`` on many islands.
        try:
            geometry = coverage_union_all(parts)
            if not geometry.is_valid:
                geometry = unary_union(parts)
        except Exception:
            geometry = unary_union(parts)
        if simplify:
            geometry = geometry.simplify(float(simplify), preserve_topology=True)
        feature_properties = dict(properties.get(value, {"value": value,
                                                          "classification": f"class_{value}"}))
        # Curated files may omit ``value`` or carry legacy values unrelated to
        # feature order.  The calibrated raster class is authoritative; always
        # encode it so missing/empty classes cannot shift subsequent IDs when
        # the GeoJSON is imported again.
        feature_properties["value"] = int(value)
        features.append({"type": "Feature", "properties": feature_properties,
                         "geometry": mapping(geometry)})
    result = {"type": "FeatureCollection", "features": features,
              "bora": {"calibration_downsample": int(downsample)}}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(result), encoding="utf-8")
    return len(features)


def geojson_properties(path):
    data = json.loads(Path(path).read_text())
    return {index: dict(feature.get("properties") or {})
            for index, feature in enumerate(data.get("features", []), 1)}


def write_full_pyramidal_mask(prediction, output_path, full_shape, downsample):
    """Upscale categorical labels exactly and write a tiled pyramidal TIFF."""
    import pyvips
    source = np.ascontiguousarray(prediction.astype(np.uint16, copy=False))
    image = pyvips.Image.new_from_memory(source.data, source.shape[1], source.shape[0],
                                         1, "ushort")
    full_h, full_w = map(int, full_shape)
    image = image.resize(full_w / image.width, vscale=full_h / image.height,
                         kernel="nearest")
    image = image.crop(0, 0, full_w, full_h)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.tiffsave(str(output_path), tile=True, tile_width=512, tile_height=512,
                   pyramid=True, subifd=True, compression="lzw", bigtiff=True,
                   properties=True)
