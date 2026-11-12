"""Bind fitted WSI polygons to the exact reviewed tissue raster."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from affine import Affine
from rasterio.features import rasterize, shapes
from shapely.affinity import translate
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from .mask_cleanup import remove_small_vector_parts


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_support_match(tissue, support, *, block_rows=512) -> bool:
    height, width = support.shape
    for top in range(0, height, block_rows):
        rows = min(block_rows, height - top)
        raster = rasterize([(tissue, 1)], out_shape=(rows, width),
                           transform=Affine(1, 0, 0, 0, 1, top),
                           fill=0, dtype=np.uint8)
        if not np.array_equal(raster != 0, np.asarray(support[top:top + rows]) != 0):
            return False
    return True


def _polygonize_exact_expanded_support(support):
    """Recover pixel-edge vectors when noninteger NN scaling defeats a shift.

    ``stream_support`` has already bound the reviewed GeoJSON to its cached
    low-resolution mask and expanded that mask by nearest neighbor. This
    polygonizes that *same* expanded raster without adding or dropping any
    supported pixel. The full-frame raster equality below is the acceptance
    gate; no approximate contour simplification is used here.
    """
    pixels = np.asarray(support, dtype=np.uint8)
    polygons = []
    for geometry, value in shapes(pixels, mask=pixels, connectivity=4,
                                  transform=Affine.identity()):
        if int(value) != 1:
            continue
        polygon = shape(geometry)
        if polygon.is_empty or not polygon.is_valid:
            raise ValueError('Expanded reviewed support produced an invalid polygon')
        polygons.append(polygon)
        if len(polygons) > 100000:
            raise ValueError('Expanded reviewed support has excessive components')
    if not polygons:
        raise ValueError('Expanded reviewed support is empty')
    tissue = unary_union(polygons)
    if tissue.is_empty or not tissue.is_valid or not _exact_support_match(tissue, support):
        raise ValueError('Expanded reviewed support vector failed exact raster equality')
    return tissue, len(polygons)


def clip_vectors_to_reviewed_support(data, identity, support, mpp, minimum_area_um2):
    """Intersect every class with a vector that exactly reproduces support.

    The accepted GeoJSON is drawn in calibrated pixel-edge coordinates. The
    nearest-neighbor expansion of its small mask uses a half-pixel center
    convention. Test the finite candidate offsets against *every* working
    pixel before clipping. A slight tie nudge resolves integer-edge ties.
    """
    qc = identity['quality_control']
    if qc['mode'] != 'background-only':
        raise ValueError('Reviewed tissue support is required for vector clipping')
    path = Path(qc['tissue_geojson_path'])
    if _sha(path) != qc['tissue_geojson_sha256']:
        raise ValueError('Reviewed tissue GeoJSON changed')
    document = json.loads(path.read_text())
    if (document.get('type') != 'FeatureCollection' or
            document.get('coordinate_system') != 'calibrated_image_pixel_xy' or
            document.get('image_shape_yx') != qc['calibrated_frame_shape_yx'] or
            document.get('mask_shape_yx') != qc['mask_shape_yx']):
        raise ValueError('Reviewed tissue GeoJSON frame differs')
    polygons = [shape(feature['geometry']) for feature in document['features']]
    if not polygons or any(g.is_empty or not g.is_valid for g in polygons):
        raise ValueError('Reviewed tissue vector is empty or invalid')
    source = unary_union(polygons)
    matched = None
    for shift_x, shift_y in ((.499999, .499999), (.5, .5),
                             (.499999, .5), (.5, .499999)):
        candidate = translate(source, xoff=shift_x, yoff=shift_y)
        if _exact_support_match(candidate, support):
            matched = (candidate, shift_x, shift_y)
            break
    if matched is None:
        # Noninteger thumbnail-to-calibrated resizing can move a boundary by
        # an entire nearest-neighbor source pixel. A global half-pixel shift
        # cannot express that. The binder has already verified the original
        # GeoJSON against the reviewed thumbnail mask; derive an exact vector
        # from the expanded mask instead of changing its tissue support.
        if (qc.get('geojson_is_authoritative') is not True or
                qc.get('mapping') != 'nearest neighbor over complete padded calibrated image; crop to analysis frame'):
            raise ValueError('Reviewed GeoJSON cannot reproduce the exact expanded tissue raster')
        tissue, expanded_components = _polygonize_exact_expanded_support(support)
        shift_x = shift_y = None
        vectorization = 'exact_expanded_reviewed_raster'
    else:
        tissue, shift_x, shift_y = matched
        expanded_components = None
        vectorization = 'translated_reviewed_geojson'
    original_groups = {int(f['properties']['value']) for f in data['features']}
    for feature in data['features']:
        clipped = shape(feature['geometry']).intersection(tissue)
        if clipped.is_empty or not clipped.is_valid:
            raise ValueError('Clipping removed or invalidated an original group')
        feature['geometry'] = mapping(clipped)
    data = remove_small_vector_parts(data, minimum_area_um2 / float(np.prod(mpp)))
    if {int(f['properties']['value']) for f in data['features']} != original_groups:
        raise ValueError('Small-part cleanup removed an original group')
    outside_area = 0.0
    for feature in data['features']:
        polygon = shape(feature['geometry'])
        if not polygon.is_valid:
            raise ValueError('A clipped class polygon is invalid')
        outside_area += polygon.difference(tissue).area
    if outside_area > 1e-7:
        raise ValueError('A clipped class extends beyond reviewed tissue geometry')
    receipt = {
        'status': 'COMPLETE', 'reviewed_geojson_sha256': qc['tissue_geojson_sha256'],
        'expanded_support_raster_exact': True,
        'shift_xy_pixels': None if shift_x is None else [shift_x, shift_y],
        'support_vectorization': vectorization,
        'expanded_support_components': expanded_components,
        'classes_preserved': sorted(original_groups),
        'small_parts_removed': data['small_polygon_part_cleanup']['removed_parts'],
        'outside_reviewed_geometry_area_pixels': outside_area,
        'minimum_area_um2': minimum_area_um2,
    }
    return data, receipt
