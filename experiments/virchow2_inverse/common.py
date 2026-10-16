"""Native-frame data and audit helpers for the Virchow2 inversion study."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

PATCH = 14
WINDOW = 224
MODEL_REPOSITORY = "paige-ai/Virchow2"
MODEL_REVISION = "3158645804b69e3f3bc4439d4116edddf0840a72"
MODEL_SHA256 = "8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b"
SELECTORS = ["pairwise_fisher", "class_l1", "balanced_trees", "balanced_mrmr"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def pad_image(image):
    h, w = image.shape[:2]
    return np.pad(image, ((0, (-h) % PATCH), (0, (-w) % PATCH), (0, 0)), mode="reflect")


def starts(length):
    if length < WINDOW or length % PATCH:
        raise ValueError("Expected a patch-aligned canvas at least 224 pixels wide")
    last = length-WINDOW
    return sorted(set([*range(0, last+1, WINDOW//2), last]))


def pixel_patches(padded):
    h, w = padded.shape[:2]
    if h % PATCH or w % PATCH:
        raise ValueError("Canvas must be divisible by the token patch size")
    return padded.reshape(h//PATCH, PATCH, w//PATCH, PATCH, 3).transpose(0,2,1,3,4).reshape(-1,PATCH*PATCH*3)


def assemble_patches(patches, grid_shape, native_shape):
    gh, gw = grid_shape
    result = np.asarray(patches).reshape(gh,gw,PATCH,PATCH,3).transpose(0,2,1,3,4).reshape(gh*PATCH,gw*PATCH,3)
    return result[:native_shape[0], :native_shape[1]]


def reconstruction_folds(grid_shape, block_tokens=8, n_folds=4):
    """Fixed 112-pixel spatial blocks; every output belongs to one held-out fold."""
    yy, xx = np.indices(grid_shape)
    return ((yy//block_tokens + 2*(xx//block_tokens)) % n_folds).ravel()


def coarse_tile_observations(features, coarse, tile_size=90):
    """One central-half dense-token mean per coarse tile, with native coordinates."""
    h, w = coarse.shape
    gh, gw = features.shape[:2]
    cy, cx = (np.arange(gh)+.5)*PATCH, (np.arange(gw)+.5)*PATCH
    values, labels, groups, records = [], [], [], []
    margin = max(1,tile_size//4)
    for y in range(0,h,tile_size):
        for x in range(0,w,tile_size):
            y1,x1=min(y+tile_size,h),min(x+tile_size,w)
            if min(y1-y,x1-x)<.8*tile_size:continue
            ids,counts=np.unique(coarse[y:y1,x:x1],return_counts=True)
            label=int(ids[np.argmax(counts)])
            if label==0 or counts.max()/counts.sum()<.95:continue
            yi=np.flatnonzero((cy>=y+margin)&(cy<y1-margin))
            xi=np.flatnonzero((cx>=x+margin)&(cx<x1-margin))
            if not len(yi) or not len(xi):continue
            value=features[np.ix_(yi,xi)].mean((0,1),dtype=np.float64)
            values.append(value);labels.append(label)
            # 3x3 coarse tiles per spatial group, not random pixel folds.
            groups.append((y//tile_size//3)*100+x//tile_size//3)
            records.append({"x":x,"y":y,"label":label,"tokens":int(len(yi)*len(xi))})
    if not values:raise ValueError("No eligible coarse tile summaries")
    return np.asarray(values),np.asarray(labels),np.asarray(groups),records
