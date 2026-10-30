"""Bind and stream a calibrated exterior-background mask for whole-slide Bora."""
from pathlib import Path
import json
import numpy as np
from .wsi_frozen import bound_file
from .selected_wsi import atomic


def bind_support(spec, base, image_sha, frame_shape, mpp_xy):
    import pyvips
    if not isinstance(spec, dict) or set(spec) != {'format', 'mode', 'mask', 'summary', 'tissue_geojson'}:
        raise ValueError('Tissue support requires bound GeoJSON, mask, and summary')
    if spec['format'] != 'bora-background-only-support-v1' or spec['mode'] != 'background-only':
        raise ValueError('Wrong background-only support policy')
    mask_path, mask_hash = bound_file(base, spec['mask'], 'background-only mask')
    summary_path, summary_hash = bound_file(base, spec['summary'], 'background-only summary')
    geojson_path, geojson_hash = bound_file(base, spec['tissue_geojson'], 'tissue-area GeoJSON')
    summary = json.loads(summary_path.read_text())
    if summary['source_image_sha256'] != image_sha or summary['tissue_mask_sha256'] != mask_hash:
        raise ValueError('Background mask belongs to another calibrated image')
    if summary['artifact_model_used'] or summary['policy'] != 'external glass and large internal white voids excluded; small enclosed adipose spaces retained':
        raise ValueError('Background mask policy differs')
    if summary['tissue_geojson']['sha256'] != geojson_hash or not summary['tissue_geojson']['roundtrip_exact']:
        raise ValueError('Tissue GeoJSON differs from reviewed support')
    image = pyvips.Image.new_from_file(str(mask_path), access='random')
    expected = summary['segmentation_thumbnail_shape_yx']
    if image.bands != 1 or image.format != 'uchar' or [image.height, image.width] != expected:
        raise ValueError('Background mask shape or dtype differs')
    document = json.loads(geojson_path.read_text())
    if (document.get('type') != 'FeatureCollection' or
            document.get('coordinate_system') != 'calibrated_image_pixel_xy' or
            document.get('image_shape_yx') != list(frame_shape) or
            document.get('mask_shape_yx') != expected or
            document.get('source_image_sha256') != image_sha):
        raise ValueError('Tissue GeoJSON belongs to another calibrated frame')
    from affine import Affine
    from rasterio.features import rasterize
    import tifffile
    scale_x = frame_shape[1] / expected[1]
    scale_y = frame_shape[0] / expected[0]
    geometry = ((item['geometry'], 1) for item in document['features'])
    raster = rasterize(geometry, out_shape=tuple(expected),
                       transform=Affine.scale(scale_x, scale_y),
                       fill=0, all_touched=False, dtype=np.uint8)
    if not np.array_equal(raster != 0, tifffile.imread(mask_path) != 0):
        raise ValueError('Tissue GeoJSON and cached raster select different pixels')
    histogram = np.frombuffer(image.hist_find().write_to_memory(), dtype=np.uint32)
    if np.any(histogram[1:255]):
        raise ValueError('Background mask must be binary 0/255')
    if any(abs(float(x) - float(y)) > .01 for x, y in zip(mpp_xy, [.5, .5])):
        raise ValueError('Use the calibrated half-micron image')
    return dict(mode='background-only', tissue_geojson_path=str(geojson_path),
                tissue_geojson_sha256=geojson_hash, geojson_is_authoritative=True,
                mask_path=str(mask_path), mask_sha256=mask_hash,
                summary_sha256=summary_hash, mask_shape_yx=expected,
                calibrated_frame_shape_yx=list(frame_shape),
                source='reviewed tissue-area GeoJSON; large voids excluded and small adipose lacunae retained',
                mapping='nearest neighbor over complete padded calibrated image; crop to analysis frame',
                artifact_predictions_used=False)


def stream_support(identity, shape_yx, out):
    import pyvips
    h, w = shape_yx
    fh, fw = identity['calibrated_frame_shape_yx']
    image = pyvips.Image.new_from_file(identity['mask_path'], access='random')
    image = image.resize(fw / image.width, vscale=fh / image.height, kernel='nearest')
    if (image.height, image.width) != (fh, fw):
        raise ValueError('Background support resampling frame differs')
    support = np.lib.format.open_memmap(Path(out) / 'support.npy', mode='w+', dtype=np.uint8, shape=(h, w))
    for y in range(0, h, 512):
        rows = min(512, h-y)
        pixels = np.frombuffer(image.crop(0, y, w, rows).write_to_memory(), np.uint8).reshape(rows, w)
        support[y:y+rows] = pixels > 0
    support.flush()
    atomic(Path(out) / 'BACKGROUND_SUPPORT_USED.json', dict(identity, status='COMPLETE',
        working_shape_yx=[h, w], no_inference_rerun=True, expert_annotation_used=False))
    return support
