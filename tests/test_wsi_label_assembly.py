"""Compact native label assembly with exact seam and full-frame checks."""

import hashlib
import json

import numpy as np
import pytest

from bora.kodama_bora import assemble_wsi_label_cores
from bora.wsi_global_vector import publish_global_wsi_vectors


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(tmp_path):
    y, x = np.indices((12, 12))
    support = np.ones((12, 12), bool)
    support[0, 0] = False
    labels = np.where(x < 4, 1, np.where(x < 8, 2, 3)).astype(np.uint8)
    labels[~support] = 0
    support_path = tmp_path / "support.npy"
    np.save(support_path, support)
    image_path = tmp_path / "image.bin"
    image_path.write_bytes(b"registered synthetic native image")
    image_hash, support_hash = _hash(image_path), _hash(support_path)
    model_hash = "a" * 64
    windows = []
    for y0 in (0, 6):
        for x0 in (0, 6):
            oy, ox = max(0, y0 - 1), max(0, x0 - 1)
            ey, ex = min(12, y0 + 7), min(12, x0 + 7)
            path = tmp_path / f"labels_{y0}_{x0}.npy"
            np.save(path, labels[oy:ey, ox:ex])
            windows.append({"labels_path": str(path),
                "labels_sha256": _hash(path), "origin_yx": [oy, ox],
                "core_yxhw": [y0, x0, 6, 6],
                "global_model_binding_sha256": model_hash,
                "source_frame_id": "case/native",
                "source_image_sha256": image_hash,
                "tissue_support_sha256": support_hash,
                "independently_fitted": False})
    kwargs = dict(tissue_support_path=support_path,
        tissue_support_sha256=support_hash, native_he_path=image_path,
        source_image_sha256=image_hash,
        global_model_binding_sha256=model_hash,
        source_frame_id="case/native", native_mpp_xy=[1., 1.])
    return labels, windows, kwargs


def test_categorical_cores_assemble_exact_and_vectorize_globally(tmp_path):
    expected, windows, kwargs = _case(tmp_path)
    output = tmp_path / "assembly"
    receipt = assemble_wsi_label_cores(windows, [12, 12], [1, 2, 3], output,
        **kwargs)
    assert receipt["format"] == "bora-wsi-label-assembly-v1"
    assert receipt["compared_halo_pixels"] > 0
    assert receipt["halo_mismatch_pixels"] == 0
    assert np.array_equal(np.fromfile(output / "semantic-mask.uint8.dat",
        np.uint8).reshape(12, 12), expected)
    vector = publish_global_wsi_vectors(output, tmp_path / "vectors",
        {1: "Group 1", 2: "Group 2", 3: "Group 3"},
        minimum_area_um2=1., smoothing_tolerance_um=0.)
    assert vector["exact_component_count"] == 3
    assert vector["source_semantic_mask_sha256"] == receipt["semantic_mask_sha256"]
    document = json.loads((tmp_path / "vectors" / "exact" / "mask.geojson").read_text())
    assert {item["properties"]["value"] for item in document["features"]} == {1, 2, 3}


def test_categorical_assembler_rejects_halo_disagreement_and_gap(tmp_path):
    _, windows, kwargs = _case(tmp_path)
    changed = np.load(windows[1]["labels_path"])
    changed[2, 0] = 3 if changed[2, 0] != 3 else 2
    np.save(windows[1]["labels_path"], changed)
    windows[1]["labels_sha256"] = _hash(tmp_path / "labels_0_6.npy")
    with pytest.raises(ValueError, match="halos disagree"):
        assemble_wsi_label_cores(windows, [12, 12], [1, 2, 3],
            tmp_path / "bad-halo", **kwargs)
    _, fresh, kwargs = _case(tmp_path)
    fresh[1]["core_yxhw"] = [0, 7, 6, 5]
    with pytest.raises(ValueError, match="gap"):
        assemble_wsi_label_cores(fresh, [12, 12], [1, 2, 3],
            tmp_path / "bad-gap", **kwargs)
