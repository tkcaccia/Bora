"""Fabricated inputs only; no dataset images or encoder weights are read."""
import json

import numpy as np
import pytest

from bora.r_api import prepare
from bora.r_feature_preflight import sha256
from bora.r_training import _verified_scales


def _fixture(tmp_path):
    pytest.importorskip("pyvips")
    rasterio = pytest.importorskip("rasterio")
    from PIL import Image

    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"fabricated checkpoint")
    (model / "config.json").write_text('{"fabricated":true}')
    images = []
    for index in range(2):
        path = tmp_path / f"case-{index}.tif"
        Image.new("RGB", (224, 224), (100 + index, 60, 120)).save(path)
        images.append(path)
    mask = tmp_path / "support.tif"
    with rasterio.open(mask, "w", driver="GTiff", width=224, height=224,
                       count=1, dtype="uint8") as dataset:
        dataset.write(np.ones((224, 224), np.uint8), 1)
    entries = []
    for index, image in enumerate(images):
        scales = {}
        for factor in (1, 2, 4):
            path = tmp_path / f"features-{index}-{factor}.npy"
            grid = np.full((64 // factor, 64 // factor, 1280),
                           index + factor, dtype=np.float16)
            np.save(path, grid, allow_pickle=False)
            receipt = tmp_path / f"source-{index}-{factor}.json"
            registration = {
                "axis_order": "yxc", "origin_source_px_yx": [0, 0],
                "token_stride_source_px_yx": [14 * factor, 14 * factor],
                "padding": "right-bottom-to-multiple-896"}
            receipt.write_text(json.dumps({
                "format": "bora-precomputed-scale-receipt-v1", "status": "VERIFIED",
                "image_sha256": sha256(image),
                "model_checkpoint_sha256": sha256(model / "model.safetensors"),
                "model_config_sha256": sha256(model / "config.json"),
                "factor": factor, "features_sha256": sha256(path),
                "shape_yxc": list(grid.shape), "registration": registration,
                "encoder_selection": "full-field", "encoder_tissue_sha256": None}))
            scales[str(factor)] = {
                "path": str(path), "sha256": sha256(path),
                "source_receipt_path": str(receipt),
                "source_receipt_sha256": sha256(receipt),
                "registration": registration}
        entries.append({"image_path": str(image), "image_sha256": sha256(image),
                        "encoder_selection": "full-field", "scales": scales})
    spec = {"format": "bora-precomputed-features-v1",
            "model_checkpoint_sha256": sha256(model / "model.safetensors"),
            "model_config_sha256": sha256(model / "config.json"),
            "images": entries}
    manifest = tmp_path / "precomputed.json"
    manifest.write_text(json.dumps(spec))
    return images, mask, model, manifest, spec


def test_adopt_two_images_without_encoder_and_verify_training_grids(tmp_path):
    images, mask, model, manifest, _ = _fixture(tmp_path)
    output = tmp_path / "adopted"
    result = prepare(images, [mask, mask], output, model, [1, 2, 4],
                     precomputed_manifest=str(manifest))
    assert result["status"] == "COMPLETE"
    assert result["cache_adoption_not_new_extraction"] is True
    assert len(result["cases"]) == 2 and len(result["scales"]) == 6
    assert all(case["worker_specs"] == [] for case in result["cases"])
    assert [int(_verified_scales(result, case)[1][0, 0, 0])
            for case in result["cases"]] == [1, 2]
    for case in result["cases"]:
        assert case["source"]["historical_encoder_selection"] == "full-field"
        assert case["source"]["selection"] == "intersects-supplied-tissue"
    assert not list(output.rglob("*WORKER_FROZEN*"))


def test_adoption_rejects_hash_and_registration_before_output(tmp_path):
    images, mask, model, manifest, spec = _fixture(tmp_path)
    spec["images"][0]["scales"]["1"]["registration"]["origin_source_px_yx"] = [1, 0]
    manifest.write_text(json.dumps(spec))
    output = tmp_path / "bad-registration"
    with pytest.raises(ValueError, match="registration"):
        prepare(images, [mask, mask], output, model, [1, 2, 4],
                precomputed_manifest=str(manifest))
    assert not output.exists()
    spec["images"][0]["scales"]["1"]["registration"]["origin_source_px_yx"] = [0, 0]
    spec["images"][0]["scales"]["1"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="array or provenance"):
        prepare(images, [mask, mask], output, model, [1, 2, 4],
                precomputed_manifest=str(manifest))
    assert not output.exists()


def test_adoption_rejects_masked_encoder_support_mismatch(tmp_path):
    images, mask, model, manifest, spec = _fixture(tmp_path)
    spec["images"][0]["encoder_selection"] = "supplied-tissue"
    spec["images"][0]["encoder_tissue_sha256"] = "0" * 64
    manifest.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="support differs"):
        prepare(images, [mask, mask], tmp_path / "no-output", model, [1, 2, 4],
                precomputed_manifest=str(manifest))
    assert not (tmp_path / "no-output").exists()


def test_adoption_rejects_unbound_source_receipt(tmp_path):
    images, mask, model, manifest, spec = _fixture(tmp_path)
    source_path = tmp_path / "source-0-1.json"
    proof = json.loads(source_path.read_text())
    proof["image_sha256"] = "0" * 64
    source_path.write_text(json.dumps(proof))
    spec["images"][0]["scales"]["1"]["source_receipt_sha256"] = sha256(source_path)
    manifest.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="not source-bound"):
        prepare(images, [mask, mask], tmp_path / "no-output", model, [1, 2, 4],
                precomputed_manifest=str(manifest))
    assert not (tmp_path / "no-output").exists()
