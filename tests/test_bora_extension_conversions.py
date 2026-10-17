import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import tifffile

from experiments.bora_extension.conversions import convert_maps, copy_verified, json_safe
from experiments.bora_extension.features import choose_dimensions
from experiments.virchow2_inverse.common import digest


def test_single_eligible_class_is_constant_without_fitting_missing_classes(monkeypatch):
    import experiments.bora_extension.conversions as module
    def forbidden(*args, **kwargs):
        raise AssertionError("Single-class branch must not fit a classifier")
    monkeypatch.setattr(module, "Context", forbidden)
    image = np.full((180, 190, 3), 130, np.uint8)
    coarse = np.full(image.shape[:2], 2, np.uint8)
    coarse[:, -10:] = 1  # Too narrow to form an eligible central training bag.
    f = np.random.default_rng(72).normal(size=(16, 16, 1280)).astype(np.float16)
    _, selection = choose_dimensions(f, coarse)
    outputs, plain, classes = convert_maps(image, coarse, f, [[220, 45, 65], [55, 165, 75]],
        np.ones(coarse.shape, bool), selection, {})
    assert classes.tolist() == [2] and set(np.unique(coarse)) == {1, 2}
    assert set(outputs) == {"09", "10", "15", "19", "ensemble"}
    for rgb, detail in outputs.values():
        assert np.all(rgb == [55, 165, 75])
        assert detail["degenerate_single_eligible_class"]
    assert plain.shape == (180, 190, 1) and np.all(plain == 1)


def test_multiclass_conversions_are_byte_exact_with_original_formulas():
    from experiments.virchow2_multi.methods import Context, align_variant, run_method
    from experiments.virchow2_ensemble.run import mean_evidence
    rng = np.random.default_rng(73)
    image = rng.integers(60, 245, (360, 360, 3), dtype=np.uint8)
    coarse = np.ones((360, 360), np.uint8)
    coarse[:, 180:] = 2
    support = np.ones(coarse.shape, bool)
    f = rng.normal(size=(26, 26, 1280)).astype(np.float16)
    _, selection = choose_dimensions(f, coarse)
    variants = {n: (rng.normal(size=(16, 16, 64)).astype(np.float32), {"shape_yx": shape})
        for n, shape in (("scale_1", [180, 180]), ("scale_2", [90, 90]))}
    palette = [[220, 45, 65], [55, 165, 75]]
    outputs, _, classes = convert_maps(image, coarse, f, palette, support, selection, variants)
    c = Context(image, coarse, f, palette, analysis_support=support)
    aligned = {n: align_variant(v, m["shape_yx"], c) for n, (v, m) in variants.items()}
    evidence = {}
    for method in (9, 10, 15, 19):
        rgb, _, _, _, p = run_method(method, c, aligned, return_probabilities=True)
        assert np.array_equal(outputs[f"{method:02d}"][0], rgb)
        if method in (9, 19):
            evidence[method] = p
    expected = c.probability_rgb(mean_evidence([evidence[9], evidence[19]], [classes, classes]))
    assert np.array_equal(outputs["ensemble"][0], expected)
    assert outputs["15"][1]["object_gaussian_sigma_px"] == [12, 40, 24]
    assert "object_scales_um" not in outputs["15"][1]


def test_byte_transfer_preserves_source_and_rejects_changed_destination(tmp_path):
    source, dest = tmp_path/"source", tmp_path/"dst"
    source.write_bytes(b"original reference bytes, never decoded")
    copy_verified(source, dest, digest(source))
    assert source.read_bytes() == dest.read_bytes()
    dest.write_bytes(b"changed")
    with pytest.raises(ValueError, match="assembled input changed"):
        copy_verified(source, dest, digest(source))


def test_undefined_diagnostic_correlations_are_null_not_fake_zero():
    data = dict(correlations=[[float("nan"), .2], [float("inf"), -.7]])
    assert json_safe(data) == {"correlations": [[None, .2], [None, -.7]]}
    json.dumps(json_safe(data), allow_nan=False)
