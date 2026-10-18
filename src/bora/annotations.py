"""Training-only tissue annotations for whole-group Neighbor context labels.

Reference masks never enter ``load_case`` or the inference pipeline.  A training
manifest may use this module after ``bora prepare`` has frozen the anonymous
groups and their Virchow2 descriptors.
"""
import json
from pathlib import Path

import numpy as np
import tifffile
from rasterio.features import rasterize
from shapely.geometry import shape


def _class_map(value, names):
    if not isinstance(value, dict) or not value:
        raise ValueError("annotation_values must map source mask values to model class IDs")
    mapping = {}
    for raw, target in value.items():
        if not isinstance(raw, str) or not raw:
            raise ValueError("Annotation value keys must be nonempty strings")
        if isinstance(target, bool) or not isinstance(target, int) or not 1 <= target <= 255:
            raise ValueError("Annotation targets must be integer model class IDs in 1..255")
        index = target
        if index not in names:
            raise ValueError("Annotation target is absent from manifest classes: " + str(index))
        mapping[raw] = index
    return mapping


def _property(properties, field):
    value = properties
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError("GeoJSON feature lacks annotation property: " + field)
        value = value[part]
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        raise ValueError("GeoJSON annotation property must be a scalar class value")
    return str(value)


def read_annotation(path, shape_yx, mapping, *, field="value", ignore_values=("0",),
                    coordinate_space=None):
    """Return native-grid model class IDs, with zero reserved for unannotated pixels."""
    path = Path(path)
    ignored = set(map(str, ignore_values)) - set(mapping)
    if path.suffix.lower() in (".tif", ".tiff"):
        source = tifffile.imread(path)
        if (source.ndim != 2 or source.shape != tuple(shape_yx)
                or source.dtype.kind not in "iu"):
            raise ValueError("Annotation TIFF must be an aligned two-dimensional integer native mask")
        output = np.zeros(shape_yx, np.uint8)
        for raw in np.unique(source):
            key = str(int(raw))
            if key in mapping:
                output[source == raw] = mapping[key]
            elif key not in ignored:
                raise ValueError("Unmapped annotation value: " + key)
        return output
    if path.suffix.lower() not in (".json", ".geojson"):
        raise ValueError("Annotation must be an integer TIFF or GeoJSON FeatureCollection")
    if coordinate_space != "native_image_pixel_edges":
        raise ValueError("Declare annotation_coordinate_space=native_image_pixel_edges for GeoJSON")
    data = json.loads(path.read_text())
    if data.get("type") != "FeatureCollection" or not isinstance(data.get("features"), list):
        raise ValueError("Expected a GeoJSON FeatureCollection")
    if (data.get("coordinate_space") not in (None, coordinate_space)
            or data.get("native_shape_yx") not in (None, list(shape_yx))):
        raise ValueError("GeoJSON coordinate space or native shape differs from prepared groups")
    by_class = {}
    for feature in data["features"]:
        raw = _property(feature.get("properties", {}), field)
        if raw in ignored:
            continue
        if raw not in mapping:
            raise ValueError("Unmapped GeoJSON annotation value: " + raw)
        geometry = shape(feature["geometry"])
        if (geometry.geom_type not in ("Polygon", "MultiPolygon") or not geometry.is_valid
                or geometry.is_empty or geometry.area <= 0):
            raise ValueError("GeoJSON training features must be valid positive-area polygons")
        xmin, ymin, xmax, ymax = geometry.bounds
        if xmin < -1e-6 or ymin < -1e-6 or xmax > shape_yx[1] + 1e-6 or ymax > shape_yx[0] + 1e-6:
            raise ValueError("GeoJSON annotation lies outside the native image")
        by_class.setdefault(mapping[raw], []).append(geometry.__geo_interface__)
    output = np.zeros(shape_yx, np.uint8)
    for target, geometries in by_class.items():
        layer = rasterize(((g, 1) for g in geometries), out_shape=shape_yx,
                          fill=0, dtype=np.uint8)
        if np.any((output > 0) & (output != target) & (layer > 0)):
            raise ValueError("GeoJSON tissue classes overlap on native pixels")
        output[layer > 0] = target
    return output


def group_targets(groups, group_ids, annotation, *, min_coverage=.5, min_purity=.8):
    """Assign one class to each sufficiently covered, sufficiently pure group."""
    groups, annotation = np.asarray(groups), np.asarray(annotation)
    ids = np.asarray(group_ids)
    if (groups.ndim != 2 or groups.shape != annotation.shape or
            not np.array_equal(np.unique(groups[groups > 0]), np.sort(ids))):
        raise ValueError("Prepared group raster and descriptor IDs must agree")
    if not (0 < min_coverage <= 1 and 0 < min_purity <= 1):
        raise ValueError("min_coverage and min_purity must be in (0, 1]")
    retained, labels, excluded = [], [], []
    for group in ids:
        values = annotation[groups == group]
        valid = values[values > 0]
        coverage = len(valid) / len(values)
        counts = np.bincount(valid, minlength=256) if len(valid) else np.zeros(256, int)
        winner = int(np.argmax(counts[1:]) + 1) if len(valid) else 0
        purity = int(counts[winner]) / len(valid) if len(valid) else 0.
        item = dict(group_id=int(group), coverage=float(coverage), purity=float(purity))
        if coverage >= min_coverage and purity >= min_purity:
            retained.append(int(group)); labels.append(winner)
        else:
            item["reason"] = "insufficient_annotation" if coverage < min_coverage else "mixed_tissue"
            excluded.append(item)
    return np.asarray(retained, np.int64), np.asarray(labels, np.uint8), excluded
