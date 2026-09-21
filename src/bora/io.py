from pathlib import Path
import numpy as np
import tifffile


def read_mask(path):
    with tifffile.TiffFile(path) as tif:
        arr = np.squeeze(tif.series[0].levels[0].asarray())
    if arr.ndim != 2:
        raise ValueError(f"Mask must be one 2-D label plane; got {arr.shape}")
    if np.min(arr, initial=0) < 0 or not np.all(arr == np.floor(arr)):
        raise ValueError("Mask labels must be non-negative integers")
    return arr


def read_image(path):
    with tifffile.TiffFile(path) as tif:
        arr = np.squeeze(tif.series[0].levels[0].asarray())
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, -1)
    elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Image must resolve to YX, YXC, or CYX; got {arr.shape}")
    return np.repeat(arr, 3, -1) if arr.shape[-1] == 1 else arr[..., :3]


def label_dtype(max_label):
    for dtype in (np.uint8, np.uint16, np.uint32):
        if max_label <= np.iinfo(dtype).max:
            return np.dtype(dtype)
    raise ValueError("Label IDs above uint32 are unsupported")


def write_ome_mask(path, labels):
    out = labels.astype(label_dtype(int(labels.max(initial=0))), copy=False)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, out, ome=True, metadata={"axes": "YX", "Name": "Bora refined labels"},
                     compression="deflate", photometric="minisblack", bigtiff=out.nbytes >= 2**32)
