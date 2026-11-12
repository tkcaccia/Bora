"""Bora-owned selection from a reusable, label-free 256-channel WSI cache.

The retained channels are selected before Bora without using cluster labels.
Bora then fits its usual 64-channel selector to the current cluster labels.
Original Virchow2 channel IDs and exact compact-cache identities are preserved.
"""

from pathlib import Path
import json

import numpy as np

from .conversion import select_dimensions
from .selected_wsi import atomic, sha
from .wsi_frozen import bound_file


FORMAT = "bora-wsi-response-input-v3-reduced256"


def is_reduced_input(manifest):
    return json.loads(Path(manifest).read_text()).get("format") == FORMAT


def read_reduced_input(manifest):
    manifest = Path(manifest).resolve()
    spec = json.loads(manifest.read_text())
    expected = {
        "format", "source_frame_id", "image", "shape_yx", "group_grid",
        "features", "channel_indices", "compact_receipt",
    }
    if spec.get("format") != FORMAT or set(spec) != expected:
        raise ValueError("Use a complete reduced-256 Bora input manifest")
    if not isinstance(spec["source_frame_id"], str) or not spec["source_frame_id"].strip():
        raise ValueError("Source frame identity required")
    image, image_sha = bound_file(manifest.parent, spec["image"], "image")
    group_path, group_sha = bound_file(manifest.parent, spec["group_grid"], "group_grid")
    indices_path, indices_sha = bound_file(
        manifest.parent, spec["channel_indices"], "channel_indices"
    )
    compact_path, compact_sha = bound_file(
        manifest.parent, spec["compact_receipt"], "compact_receipt"
    )
    compact = json.loads(Path(compact_path).read_text())
    if (
        compact.get("status") != "COMPLETE"
        or compact.get("retained_channels") != 256
        or compact.get("original_channels") != 1280
        or compact.get("image_sha256") != image_sha
        or compact.get("channel_indices_sha256") != indices_sha
        or not isinstance(compact.get("outputs"), dict)
    ):
        raise ValueError("Compact cache provenance differs from this image")
    channel_indices = np.load(indices_path, allow_pickle=False)
    if (
        channel_indices.shape != (256,)
        or channel_indices.dtype.kind not in "iu"
        or channel_indices.min() < 0
        or channel_indices.max() >= 1280
        or len(np.unique(channel_indices)) != 256
    ):
        raise ValueError("Retained original channel indices are invalid")
    grid = np.load(group_path, allow_pickle=False)
    if (
        grid.ndim != 2
        or grid.dtype.kind not in "iu"
        or grid.min() < 0
        or grid.max() > 255
        or any(side % 4 for side in grid.shape)
    ):
        raise ValueError("A padded categorical group grid is required")
    eligible = grid > 0
    eligibility_path = Path(compact.get("eligibility", ""))
    tile_qc_path = Path(compact.get("tile_qc", ""))
    if (
        not eligibility_path.is_file()
        or sha(eligibility_path) != compact.get("eligibility_sha256")
        or not tile_qc_path.is_file()
        or sha(tile_qc_path) != compact.get("tile_qc_sha256")
    ):
        raise ValueError("Exact reviewed tile-QC provenance is missing or changed")
    reviewed_eligible = np.load(eligibility_path, allow_pickle=False)
    if reviewed_eligible.dtype != np.dtype("bool") or not np.array_equal(reviewed_eligible, eligible):
        raise ValueError("Current groups differ from the compact cache's strict admitted tiles")
    tile_qc = json.loads(tile_qc_path.read_text())
    policy = tile_qc.get("clustering_policy", {})
    if (
        tile_qc.get("status") != "COMPLETE"
        or policy.get("comparison") != "<"
        or policy.get("background_percent") != 10
        or not policy.get("exactly_ten_percent_excluded")
        or tile_qc.get("input_identity", {}).get("calibrated_image_sha256") != image_sha
    ):
        raise ValueError("The compact cache is not bound to strict reviewed tile QC")
    labels = grid[eligible]
    classes = np.unique(labels)
    if len(classes) < 1 or not np.array_equal(classes, np.arange(1, len(classes) + 1)):
        raise ValueError("At least one contiguous anonymous group required")
    shape = spec["shape_yx"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(n) is not int or n < 1 for n in shape)
        or np.any(np.array(shape) > np.array(grid.shape) * 224)
    ):
        raise ValueError("Calibrated frame and group grid differ")
    if set(spec["features"]) != {"1", "2", "4"}:
        raise ValueError("All three compact spatial scales are required")
    arrays = {}
    hashes = {}
    for factor in (1, 2, 4):
        path, digest = bound_file(
            manifest.parent, spec["features"][str(factor)], f"factor_{factor}"
        )
        output = compact["outputs"].get(str(factor), {})
        if Path(output.get("compact", "")).resolve() != Path(path).resolve() or output.get("compact_sha256") != digest:
            raise ValueError(f"Factor-{factor} is not the verified compact cache")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_shape = tuple(np.array(grid.shape) * 16 // factor) + (256,)
        if array.shape != expected_shape or array.dtype != np.dtype("float16"):
            raise ValueError(f"Factor-{factor} compact feature shape or precision differs")
        arrays[factor] = array
        hashes[str(factor)] = digest
    identity = {
        "reduced_input_manifest_sha256": sha(manifest),
        "compact_receipt_sha256": compact_sha,
        "source_image_sha256": image_sha,
        "group_grid_sha256": group_sha,
        "channel_indices_sha256": indices_sha,
        "reduced_features_sha256": hashes,
    }
    return spec, arrays, eligible, labels, channel_indices, identity


def prepare_reduced_selection(manifest, output):
    spec, arrays, eligible, labels, indices, identity = read_reduced_input(manifest)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh reduced-cache selection folder")
    values = np.empty((int(eligible.sum()), 256), np.float32)
    for i, (y, x) in enumerate(np.argwhere(eligible)):
        block = arrays[1][y * 16:(y + 1) * 16, x * 16:(x + 1) * 16]
        if not np.isfinite(block).all():
            raise ValueError("A fitting tile lacks retained spatial tokens")
        values[i] = block.mean((0, 1), dtype=np.float64)
    if len(np.unique(labels)) == 1:
        center = np.median(values, axis=0)
        mad = np.median(np.abs(values - center), axis=0)
        positions = np.argsort(-mad, kind="stable")[:64].astype(int)
        basis = "within-community median absolute deviation over 256 retained tile-mean channels"
    else:
        positions = select_dimensions(values, labels).astype(int)
        basis = "all 256 label-free retained channels and current anonymous tile clusters"
    original_ids = indices[positions].astype(int)
    output.mkdir(parents=True)
    files = {}
    for factor, array in arrays.items():
        path = output / f"bora-selected-factor-{factor}.npy"
        selected = np.lib.format.open_memmap(
            path, mode="w+", dtype=array.dtype, shape=array.shape[:2] + (64,)
        )
        for y in range(0, array.shape[0], 16):
            selected[y:y + 16] = np.take(array[y:y + 16], positions, axis=2)
        selected.flush()
        del selected
        files[str(factor)] = {"path": str(path.resolve()), "sha256": sha(path)}
    source_manifest = Path(manifest).resolve()
    image_path, image_sha = bound_file(source_manifest.parent, spec["image"], "image")
    group_path, group_sha = bound_file(source_manifest.parent, spec["group_grid"], "group_grid")
    selected_spec = {
        "format": "bora-wsi-response-input-v1",
        "source_frame_id": spec["source_frame_id"],
        "image": {"path": str(image_path), "sha256": image_sha},
        "shape_yx": spec["shape_yx"],
        "group_grid": {"path": str(group_path), "sha256": group_sha},
        "features": files,
        "selected_dimensions": original_ids.tolist(),
    }
    target = output / "bora-selected-input.json"
    atomic(target, selected_spec)
    atomic(
        output / "BORA_REDUCED_SELECTION_COMPLETE.json",
        {
            **identity,
            "status": "COMPLETE",
            "source_encoder_channels": 1280,
            "reduced_cache_channels": 256,
            "retained_channels": 64,
            "selected_original_dimensions": original_ids.tolist(),
            "selected_compact_positions": positions.tolist(),
            "selection_owner": "Bora",
            "selection_basis": basis,
            "selected_input_manifest_sha256": sha(target),
        },
    )
    return target, identity
