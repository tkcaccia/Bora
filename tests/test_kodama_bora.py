"""End-to-end frozen numeric head and cross-seam scoring contract."""

import hashlib
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from bora.cli import parser
from bora.kodama_bora import (
    fit_three_group_model, publish_three_group_contours, score_three_group_model,
    score_three_group_tokens,
)


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fit_score_roundtrip_binds_projection_and_frame(tmp_path):
    rng = np.random.default_rng(17)
    samples = tmp_path / "samples.npz"
    features = np.concatenate([
        rng.normal(loc=i, scale=.08, size=(8, 32)).astype(np.float32)
        for i in (0, 1, 2)])
    labels = np.repeat(np.asarray([1, 2, 3], np.uint8), 8)
    np.savez_compressed(samples, features=features, labels=labels)
    projection = "a" * 64
    fit_manifest = tmp_path / "fit.json"
    fit_manifest.write_text(json.dumps({
        "format": "bora-kodama-three-group-fit-v1",
        "source_frame_id": "train/registered-token-grid",
        "source_image_sha256": "d" * 64,
        "training_annotation_sha256": "b" * 64,
        "pca_projection_sha256": projection, "class_ids": [1, 2, 3],
        "samples": {"path": samples.name, "sha256": _hash(samples)}}))
    model = tmp_path / "model"
    fitted = fit_three_group_model(fit_manifest, model)
    assert fitted["class_counts"] == [8, 8, 8]
    assert fitted["model_sha256"] == _hash(model / "numeric_head.npz")

    field = tmp_path / "field.npy"
    support = tmp_path / "support.npy"
    lattice = np.tile(features[8].reshape(1, 1, 32), (70, 83, 1))
    lattice[:, :28] = features[0]
    lattice[:, 55:] = features[16]
    np.save(field, lattice)
    tissue = np.ones((70, 83), bool)
    tissue[:, 40] = False
    np.save(support, tissue)
    score_manifest = tmp_path / "score.json"
    source = {"format": "bora-kodama-three-group-score-v1",
        "registration_method": "same_global_token_grid",
        "source_frame_id": "train/registered-token-grid",
        "source_image_sha256": "d" * 64,
        "pca_projection_sha256": projection,
        "field": {"path": field.name, "sha256": _hash(field)},
        "support": {"path": support.name, "sha256": _hash(support)}}
    score_manifest.write_text(json.dumps(source))
    output = tmp_path / "score"
    scored = score_three_group_model(score_manifest, model, output,
        sigma_tokens=1.375, prior_exponent=.5, tile_size=32)
    result = np.load(output / "three_group_probability.npy", allow_pickle=False)
    assert result.shape == (70, 83, 3)
    assert np.allclose(result.sum(axis=-1), 1, atol=1e-6)
    assert scored["source_frame_id"] == source["source_frame_id"]
    assert scored["probability_sha256"] == _hash(output / "three_group_probability.npy")
    with np.load(model / "numeric_head.npz", allow_pickle=False) as frozen:
        head = {key: frozen[key] for key in (
            "scaler_mean", "scaler_scale", "logistic_coef",
            "logistic_intercept", "class_ids")}
        prior = frozen["training_prior"]
    whole = score_three_group_tokens(lattice, tissue, head, prior,
        sigma_tokens=1.375, prior_exponent=.5)
    assert np.max(np.abs(result - whole)) < 1e-6
    assert result[35, 27, 0] < result[35, 2, 0]

    narrow = tmp_path / "score-small-sigma"
    score_three_group_model(score_manifest, model, narrow,
        sigma_tokens=.125, prior_exponent=.5, tile_size=32)
    assert np.max(np.abs(np.load(narrow / "three_group_probability.npy") -
        score_three_group_tokens(lattice, tissue, head, prior,
            sigma_tokens=.125, prior_exponent=.5))) < 1e-6

    bad = dict(source, pca_projection_sha256="c" * 64)
    score_manifest.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="PCA projection differs"):
        score_three_group_model(score_manifest, model, tmp_path / "bad",
            sigma_tokens=1.0, prior_exponent=.5)
    score_manifest.write_text(json.dumps(dict(source, source_image_sha256="e" * 64)))
    with pytest.raises(ValueError, match="image"):
        score_three_group_model(score_manifest, model, tmp_path / "wrong-slide",
            sigma_tokens=1.0, prior_exponent=.5)


def test_kodama_cli_has_separate_fit_score_and_contour_steps():
    assert parser().parse_args(["kodama-fit", "fit.json", "--output", "model"]).command == "kodama-fit"
    assert parser().parse_args(["kodama-score", "score.json", "--model", "model",
        "--output", "score", "--sigma-tokens", "2", "--prior-exponent", ".7"]).command == "kodama-score"
    assert parser().parse_args(["kodama-contours", "assembly", "he.tif", "names.json",
        "--output", "contours"]).command == "kodama-contours"
    assert parser().parse_args(["kodama-native", "native.json",
        "--output", "native"]).command == "kodama-native"


def test_complete_native_three_group_mask_becomes_diagonal_tissue_annotations(tmp_path):
    """A large synthetic stair edge exercises global, not windowed, contours."""
    h = w = 2048
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    support = (y > 80 + x // 4) & (y < 1900)
    mask = np.where(support, np.where(x < 680, 1,
        np.where(x < 1360, 2, 3)), 0).astype(np.uint8)
    assembly = tmp_path / "assembly"
    assembly.mkdir()
    source_mask = assembly / "semantic-mask.uint8.dat"
    source_mask.write_bytes(mask.tobytes())
    he = tmp_path / "he.tif"
    with rasterio.open(he, "w", driver="GTiff", height=h, width=w,
            count=3, dtype="uint8", transform=Affine.translation(2, 3)) as image:
        for channel in (1, 2, 3):
            image.write(np.full((h, w), 160, np.uint8), channel)
    (assembly / "ASSEMBLY_COMPLETE.json").write_text(json.dumps({
        "format": "bora-wsi-probability-assembly-v1",
        "core_partition_complete": True,
        "independently_fitted_windows": False,
        "native_shape_yx": [h, w], "native_mpp_xy": [1., 1.],
        "class_ids": [1, 2, 3], "source_frame_id": "test/native",
        "source_image_sha256": _hash(he),
        "global_model_binding_sha256": "a" * 64,
        "semantic_mask_sha256": _hash(source_mask),
        "semantic_mask_path": source_mask.name}))
    result = publish_three_group_contours(assembly, he, tmp_path / "contours",
        {1: "KODAMA Group 1", 2: "KODAMA Group 2", 3: "KODAMA Group 3"},
        minimum_area_um2=1000, minimum_free_bytes=0,
        deadline_seconds=600)
    assert result["status"] == "COMPLETE"
    assert result["component_count"] == 3
    assert result["selected_outer_tolerance_um"] > 0
    assert (tmp_path / "contours" / "COMPLETE.json").is_file()
    assert _hash(tmp_path / "contours" / "tissue_annotations" /
        "BORA_TISSUE_ANNOTATIONS.geojson") == result["predicted_tissue_geojson_sha256"]
    tissue = json.loads((tmp_path / "contours" / "tissue_annotations" /
        "BORA_TISSUE_ANNOTATIONS.geojson").read_text())
    assert len(tissue["features"]) == 3
    assert {f["properties"]["value"] for f in tissue["features"]} == {1, 2, 3}
    smoothing = json.loads((tmp_path / "contours" / "tissue_annotations" /
        "SMOOTHING_COMPLETE.json").read_text())
    before = smoothing["selected"]["segment_stats_before"]
    after = smoothing["selected"]["segment_stats_after"]
    assert after["diagonal_segments"] > before["diagonal_segments"]
    assert after["axis_aligned_segments"] < before["axis_aligned_segments"] / 2
    assert not list((tmp_path / "contours").rglob("*PM016*"))
