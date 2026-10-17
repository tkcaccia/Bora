"""Calibration, one-class inference and frozen-cache integrity, without a GPU."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import tifffile

from experiments.bora_extension.features import (
    sampling, choose_dimensions, freeze_case, verify_case)
from experiments.virchow2_inverse.common import digest, pad_image, starts


def test_sampling_unknown_never_invents_micrometres():
    row = dict(working_mpp_xy=None, fov_um_xy=None, working_shape_yx=[301, 419])
    for shape in ([301, 419], [150, 210], [75, 105]):
        result = sampling(row, shape)
        assert result["mpp_xy"] is None
        assert result["window_fov_um_xy"] is None
        assert not result["calibrated_model_sampling"]
    with pytest.raises(ValueError, match="contradictory"):
        sampling(dict(row, fov_um_xy=[209.5, 150.5]), [301, 419])


def test_sampling_uses_actual_resized_axes_not_nominal_scale():
    row = dict(working_mpp_xy=[.501, .499], working_shape_yx=[301, 419])
    result = sampling(row, [150, 210])
    assert result["mpp_xy"] == [pytest.approx(.501*419/210), pytest.approx(.499*301/150)]
    assert result["window_fov_um_xy"][0] == pytest.approx(224*.501*419/210)


def test_single_class_cache_is_explicitly_nondiscriminative():
    features = np.random.default_rng(91).normal(size=(16, 16, 1280)).astype(np.float16)
    selected, info = choose_dimensions(features, np.ones((180, 180), np.uint8))
    assert len(selected) == len(set(selected)) == 64
    assert info["observed_classes"] == [1]
    assert info["single_coarse_class"]
    assert not info["selection"]["discriminative_selection_possible"]
    assert info["conversion_degeneracy"] == "constant observed-class RGB"


def test_multiclass_selection_matches_unchanged_context():
    from experiments.virchow2_multi.methods import Context
    features = np.random.default_rng(92).normal(size=(26, 26, 1280)).astype(np.float16)
    coarse = np.ones((360, 360), np.uint8)
    coarse[:, 180:] = 2
    image = np.full((360, 360, 3), 130, np.uint8)
    selected, info = choose_dimensions(features, coarse)
    context = Context(image, coarse, features, [[220, 45, 65], [55, 165, 75]])
    assert np.array_equal(selected, context.selected)
    assert not info["single_coarse_class"]


def test_full_field_encoder_does_not_read_reference_and_resume_checks_hashes(tmp_path, monkeypatch):
    import experiments.bora_extension.features as module
    monkeypatch.setattr(module.shutil, "disk_usage", lambda p: SimpleNamespace(free=25*1024**3))
    src, out = tmp_path/"source", tmp_path/"output"
    case = src/"cases"/"uncalibrated_one_class"
    case.mkdir(parents=True)
    out.mkdir()
    image = np.random.default_rng(93).integers(0, 255, (180, 180, 3), dtype=np.uint8)
    Image.fromarray(image).save(case/"original.png")
    tifffile.imwrite(case/"coarse.tif", np.ones((180, 180), np.uint8))
    (case/"reference.native.tif").write_bytes(b"not a readable TIFF; predictor must not open")
    sample = dict(case=case.name, dataset="example", working_mpp_xy=None,
        fov_um_xy=None, working_shape_yx=[180, 180],
        original_sha256=digest(case/"original.png"), coarse_sha256=digest(case/"coarse.tif"))
    calls = []

    def encode(model, rgb, selected=None):
        calls.append(rgb.copy())
        padded = pad_image(rgb)
        channels = 1280 if selected is None else len(selected)
        values = np.random.default_rng(94).normal(size=(*np.array(padded.shape[:2])//14, channels)).astype(np.float16)
        return values, dict(window_coordinates_yx=[(y, x) for y in starts(padded.shape[0]) for x in starts(padded.shape[1])],
            patch_px=14, padded_shape_yx=list(padded.shape[:2]))

    record = freeze_case(src, out, sample, None, encode)
    assert len(calls) == 3
    assert calls[0].shape == (224, 224, 3)
    assert np.array_equal(calls[0][:180, :180], image)  # No annotation/tissue gate or resampling.
    assert record["all_original_tokens_covered"]
    assert not record["fine_reference_read"] and not record["bora_run"]
    assert record["sampling"]["mpp_xy"] is None
    assert verify_case(out/"cases"/case.name, sample) == record
    freeze_case(src, out, sample, None, encode)
    assert len(calls) == 3  # Verified frozen resume does not encode again.
    (out/"cases"/case.name/"features/base.npy").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="cache changed"):
        verify_case(out/"cases"/case.name, sample)
