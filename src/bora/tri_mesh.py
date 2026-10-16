"""Triangular-mesh tissue interface refinement using the OpenMP C++ kernel."""
from __future__ import annotations

import ctypes
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import numpy as np
import rasterio
import tifffile
from rasterio.windows import Window


LIBRARY = Path(__file__).resolve().parents[2] / "cpp" / "libbora_tri_mesh.so"
_KERNEL = None
_SOURCE = None
_IMAGE = None


@dataclass(frozen=True)
class TriMeshConfig:
    mesh_spacing: int = 64
    support_radius: int = 80
    band_radius: int = 216
    color_weight: float = 0.0
    kernel_threads: int = 2
    post_smooth_radius: int = 0


def scale_for_tile_size(config: TriMeshConfig, tile_size: int) -> TriMeshConfig:
    """Use smaller interface geometry when the coarse input tiles are smaller."""
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    factor = tile_size / 216
    mesh = max(16, round(config.mesh_spacing*factor/8)*8)
    support = max(24, round(config.support_radius*factor/8)*8)
    band = max(48, round(config.band_radius*factor))
    post = (max(2, round(config.post_smooth_radius*factor))
            if config.post_smooth_radius else 0)
    return TriMeshConfig(mesh, support, band,
                         config.color_weight, config.kernel_threads, post)


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        if not LIBRARY.is_file():
            raise FileNotFoundError(f"Compile the C++ tri-mesh kernel first: {LIBRARY}")
        function = ctypes.CDLL(str(LIBRARY)).bora_tri_mesh_u16
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                             ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             ctypes.c_int, ctypes.c_int, ctypes.c_float,
                             ctypes.c_int, ctypes.c_int]
        function.restype = ctypes.c_int64
        _KERNEL = function
    return _KERNEL


def refine_tri_mesh(labels: np.ndarray, image: np.ndarray | None = None,
                    config: TriMeshConfig = TriMeshConfig()) -> tuple[np.ndarray, int]:
    labels = np.ascontiguousarray(labels, dtype=np.uint16)
    if labels.ndim != 2:
        raise ValueError("Expected a two-dimensional label mask")
    if image is not None:
        image = np.ascontiguousarray(image[..., :3], dtype=np.uint8)
        if image.shape != (*labels.shape, 3):
            raise ValueError("RGB image and label mask must have equal dimensions")
    result = np.empty_like(labels)
    changed = _kernel()(labels.ctypes.data,
                        image.ctypes.data if image is not None else None,
                        result.ctypes.data, labels.shape[1], labels.shape[0],
                        config.mesh_spacing, config.support_radius,
                        config.band_radius, config.color_weight,
                        config.kernel_threads, config.post_smooth_radius)
    if changed < 0:
        raise ValueError(f"Tri-mesh kernel rejected input or settings: {changed}")
    return result, changed


def _worker_init(mask_path: str, image_path: str | None):
    global _SOURCE, _IMAGE
    _SOURCE = rasterio.open(mask_path)
    if image_path:
        import pyvips
        image = pyvips.Image.new_from_file(image_path, access="random")
        if image.bands == 1 and image.get_typeof("n-pages") and image.get("n-pages") in (3, 4):
            pages = [pyvips.Image.new_from_file(image_path, page=i,
                                                access="random") for i in range(3)]
            image = pages[0].bandjoin(pages[1:])
        if (image.height, image.width) != (_SOURCE.height, _SOURCE.width):
            raise ValueError("Image and mask dimensions differ")
        _IMAGE = image


def _process(task: tuple[int, int, int, int, int, TriMeshConfig]):
    x, y, width, height, block_size, config = task
    halo = config.band_radius + config.support_radius + config.mesh_spacing + 8
    x0 = max(0, (max(0, x-halo)//config.mesh_spacing)*config.mesh_spacing)
    y0 = max(0, (max(0, y-halo)//config.mesh_spacing)*config.mesh_spacing)
    x1 = min(_SOURCE.width, x+width+halo)
    y1 = min(_SOURCE.height, y+height+halo)
    labels = _SOURCE.read(1, window=Window(x0, y0, x1-x0, y1-y0))
    center = labels[y-y0:y-y0+height, x-x0:x-x0+width]
    if len(np.unique(labels)) < 3 and 0 in labels:
        # A single foreground label plus background has no internal interface.
        return x, y, center.copy(), 0
    rgb = None
    if config.color_weight:
        from .streaming import _vips_crop
        rgb = _vips_crop(_IMAGE, x0, y0, x1-x0, y1-y0)
    refined, _ = refine_tri_mesh(labels, rgb, config)
    chosen = refined[y-y0:y-y0+height, x-x0:x-x0+width].copy()
    return x, y, chosen, int(np.count_nonzero(chosen != center))


def refine_tri_mesh_streaming(mask_path: Path, output_path: Path,
                              image_path: Path | None = None, *,
                              config: TriMeshConfig = TriMeshConfig(),
                              block_size: int = 2048, workers: int = 2) -> dict:
    """Refine a WSI in bounded blocks; C++ also parallelizes each block."""
    if config.color_weight and not image_path:
        raise ValueError("Color guidance requires an original image")
    started = perf_counter()
    with rasterio.open(mask_path) as source:
        height, width = source.height, source.width
        dtype = np.dtype(source.dtypes[0])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    destination = tifffile.memmap(output_path, shape=(height, width), dtype=dtype,
                                  photometric="minisblack", metadata={"axes": "YX"},
                                  bigtiff=True)
    tasks = ((x, y, min(block_size, width-x), min(block_size, height-y),
              block_size, config)
             for y in range(0, height, block_size)
             for x in range(0, width, block_size))
    changed = blocks = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"),
                             initializer=_worker_init,
                             initargs=(str(mask_path), str(image_path) if image_path else None)) as pool:
        pending = {}
        for _ in range(workers*2):
            try:
                job = next(tasks)
            except StopIteration:
                break
            pending[pool.submit(_process, job)] = job
        while pending:
            future = next(as_completed(pending))
            pending.pop(future)
            x, y, part, count = future.result()
            destination[y:y+part.shape[0], x:x+part.shape[1]] = part
            changed += count
            blocks += 1
            try:
                job = next(tasks)
                pending[pool.submit(_process, job)] = job
            except StopIteration:
                pass
            if blocks % 16 == 0:
                destination.flush()
                print(f"[Bora tri-mesh] blocks {blocks}, changed {changed}", flush=True)
    destination.flush()
    return {"shape": [height, width], "changed_pixels": changed,
            "blocks": blocks, "workers": workers,
            "parameters": asdict(config),
            "runtime_seconds": perf_counter()-started}
