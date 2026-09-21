"""Blockwise WSI refinement and polygonization with bounded memory."""
from pathlib import Path
from time import perf_counter
import json
import numpy as np
import tifffile
from shapely.geometry import Polygon, box, mapping
from skimage import measure
from .io import label_dtype
from .refine import refine_labels


def _vips_crop(image, x, y, w, h):
    crop = image.crop(x, y, w, h)
    dtype = {"uchar":np.uint8,"ushort":np.uint16,"float":np.float32}[crop.format]
    return np.ndarray(buffer=crop.write_to_memory(), dtype=dtype,
                      shape=(crop.height, crop.width, crop.bands)).copy()


def refine_streaming(image_path, mask_path, output_path, backend, config, block_size=1024):
    import pyvips
    source = tifffile.memmap(mask_path)
    if source.ndim != 2:
        raise ValueError(f"Streaming mask must be a flat 2-D TIFF; got {source.shape}")
    # Boundary-only access is sparse and non-monotonic because halo windows
    # overlap; random access prevents repeated decoding from the image origin.
    image = pyvips.Image.new_from_file(str(image_path), access="random")
    if (image.height, image.width) != source.shape:
        raise ValueError(f"Image/mask mismatch: {(image.height,image.width)} vs {source.shape}")
    dtype = label_dtype(int(source.max()))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    dest = tifffile.memmap(output_path, shape=source.shape, dtype=dtype,
                           photometric="minisblack", metadata={"axes":"YX"}, bigtiff=True)
    halo = max(config.core_erosion, config.outer_dilation, config.smooth_radius) + 4
    total = refined = copied = 0
    t0 = perf_counter()
    for y in range(0, source.shape[0], block_size):
        for x in range(0, source.shape[1], block_size):
            h, w = min(block_size, source.shape[0]-y), min(block_size, source.shape[1]-x)
            y0, x0 = max(0,y-halo), max(0,x-halo)
            y1, x1 = min(source.shape[0],y+h+halo), min(source.shape[1],x+w+halo)
            mask = np.asarray(source[y0:y1,x0:x1])
            center = mask[y-y0:y-y0+h,x-x0:x-x0+w]
            total += 1
            if np.unique(mask).size == 1:
                dest[y:y+h,x:x+w] = center
                copied += 1
                continue
            rgb = _vips_crop(image, x0, y0, x1-x0, y1-y0)
            local, _ = refine_labels(rgb, mask, backend, config)
            dest[y:y+h,x:x+w] = local[y-y0:y-y0+h,x-x0:x-x0+w].astype(dtype, copy=False)
            refined += 1
        dest.flush()
        print(f"[Bora] row {min(y+block_size,source.shape[0])}/{source.shape[0]} blocks={total} refined={refined}", flush=True)
    dest.flush()
    return {"mode":"streaming", "shape":list(source.shape), "blocks":total,
            "refined_blocks":refined, "copied_interior_blocks":copied,
            "runtime_seconds":perf_counter()-t0}


def write_geojson_streaming(path, mask_path, block_size=2048, simplify=1.0, min_area=32):
    labels = tifffile.memmap(mask_path)
    features = []
    for y in range(0, labels.shape[0], block_size):
        for x in range(0, labels.shape[1], block_size):
            h, w = min(block_size,labels.shape[0]-y), min(block_size,labels.shape[1]-x)
            y0,x0,y1,x1=max(0,y-1),max(0,x-1),min(labels.shape[0],y+h+1),min(labels.shape[1],x+w+1)
            tile=np.asarray(labels[y0:y1,x0:x1]); clip=box(x,y,x+w,y+h)
            for raw_id in np.unique(tile):
                lid=int(raw_id)
                if lid <= 0: continue
                for contour in measure.find_contours(tile==raw_id,.5,fully_connected="high"):
                    if len(contour)<4: continue
                    poly=Polygon([(float(c+x0),float(r+y0)) for r,c in contour])
                    if not poly.is_valid: poly=poly.buffer(0)
                    poly=poly.intersection(clip)
                    if simplify: poly=poly.simplify(simplify,preserve_topology=True)
                    if poly.is_empty or poly.area<min_area: continue
                    features.append({"type":"Feature","properties":{"label":lid,"area_px":round(poly.area,2)},"geometry":mapping(poly)})
    Path(path).write_text(json.dumps({"type":"FeatureCollection","features":features}))
    return len(features)
