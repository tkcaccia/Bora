"""CPU-only cached-feature tests; no encoder, GPU, native fastPLS or downloads."""
import csv
import json
from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pytest
import tifffile
from PIL import Image
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from appearance_fixtures import (assert_frozen_rgb_quantization,
                                legacy_rgb9plus19, spatial_case)
from bora.appearance_state import (linear_state, linear_values, probabilities,
                                  save_state, scale_values, scaler_state)
from bora.appearance_bridge import run_request, main
from bora.conversion import convert_rgb
from bora.discriminative import DiscriminativeAppearance, select_pairwise
from bora.rgb_recipe import RGB9Plus19Recipe

ENCODER = dict(revision="synthetic-test-only", sha256="a" * 64)


def row_case(mode="shared"):
    rng = np.random.default_rng(93)
    x = rng.normal(size=(36, 1280))
    y = np.asarray(["1", "2", "3"] * 12)
    for j in range(3):
        x[y == str(j + 1), j * 20:(j + 1) * 20] += 3
    kwargs = dict(mode=mode, feature_kind="virchow2-spatial",
        feature_columns=[f"f{i:04d}" for i in range(1280)], encoder=ENCODER,
        label_kind="semantic" if mode == "shared" else "anonymous",
        classes={"1": "tumor", "2": "stroma", "3": "immune"}, splits=["train"] * len(x))
    return x, y, kwargs


@pytest.mark.parametrize("classes", [2, 3])
def test_rgb_recipe_matches_full_legacy_formula_and_roundtrips(classes, tmp_path, monkeypatch):
    f, coarse, palette, variants = spatial_case(classes)
    legacy_rgb, legacy_probability, selected = legacy_rgb9plus19(f, coarse, palette, variants)
    recipe = RGB9Plus19Recipe.fit(f, coarse, palette, variants)
    rgb, _ = recipe.transform(f, variants)
    expected = recipe.predict_proba(f, variants)
    # Compare the complete legacy formula with models fitted on this runtime.
    # Two float32 epsilon units accommodate BLAS/layout arithmetic order; this
    # is an absolute probability bound, never a relative or image-wide average.
    np.testing.assert_array_equal(recipe.state["selected_dimensions"], selected)
    np.testing.assert_allclose(expected, legacy_probability, rtol=0, atol=2e-7)
    np.testing.assert_allclose(rgb, legacy_rgb, rtol=0, atol=1)
    # Frozen macOS reference: isolated half-integer quantization crossings are
    # permitted across BLAS/CPU platforms, with BOTH magnitude and count limits.
    with np.load(Path(__file__).parent / "data" / f"rgb9plus19_legacy_{classes}.npz") as old:
        np.testing.assert_array_equal(recipe.state["selected_dimensions"], old["selected"])
        assert_frozen_rgb_quantization(rgb, old["rgb"])
    path = tmp_path / "local.json"
    recipe.save(path)
    def forbidden(*args, **kwargs):
        raise AssertionError("Loading/transforming must not refit")
    for model in (StandardScaler, LogisticRegression, Ridge):
        monkeypatch.setattr(model, "fit", forbidden)
    loaded = RGB9Plus19Recipe.load(path)
    np.testing.assert_array_equal(loaded.predict_proba(f, variants), expected)
    np.testing.assert_array_equal(loaded.transform(f, variants)[0], rgb)
    alternate = list(reversed(palette))
    assert np.any(loaded.transform(f, variants, palette=alternate)[0] != rgb)
    np.testing.assert_array_equal(loaded.predict_proba(f, variants), expected)
    with pytest.raises(FileExistsError):
        loaded.save(path)


@pytest.mark.parametrize("delta,count,accepted", [(1, 3, True), (1, 4, False), (2, 1, False)])
def test_frozen_rgb_quantization_limits_reject_color_or_spatial_drift(delta, count, accepted):
    reference = np.full((100, 1000, 3), 127, np.uint8)
    changed = reference.copy()
    changed.reshape(-1)[:count] += delta
    if accepted:
        assert_frozen_rgb_quantization(changed, reference)
    else:
        with pytest.raises(AssertionError):
            assert_frozen_rgb_quantization(changed, reference)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_replay_numeric_primitives_match_sklearn_predictions(dtype):
    rng = np.random.default_rng(48)
    x = rng.normal(size=(40, 8)).astype(dtype)
    scaler = StandardScaler().fit(x)
    np.testing.assert_array_equal(scale_values(scaler_state(scaler), x), scaler.transform(x))
    ridge = Ridge(alpha=100).fit(x, rng.normal(size=(40, 5)).astype(dtype))
    tolerance = 4 * np.finfo(dtype).eps
    np.testing.assert_allclose(linear_values(linear_state(ridge), x), ridge.predict(x),
                               rtol=tolerance, atol=tolerance)
    for classes in (2, 3):
        clf = LogisticRegression(C=.3, class_weight="balanced", max_iter=2000,
                                 random_state=430).fit(x, np.arange(len(x)) % classes)
        np.testing.assert_allclose(probabilities(linear_state(clf, logistic=True), x),
                                   clf.predict_proba(x), rtol=0, atol=1e-14)


def test_local_recipe_rejects_wrong_field_and_malformed_state(tmp_path):
    f, c, p, v = spatial_case()
    recipe = RGB9Plus19Recipe.fit(f, c, p, v)
    changed = f.copy(); changed[0, 0, 0] += .1
    with pytest.raises(ValueError, match="different token field"):
        recipe.predict_proba(changed, v)
    broken = json.loads(json.dumps(recipe.state)); broken["ridge"][0]["coef"] = [[0]]
    path = tmp_path / "invalid.json"; save_state(path, broken)
    with pytest.raises(ValueError, match="dimensions"):
        RGB9Plus19Recipe.load(path)
    path.write_text('{"payload":{},"sha256":"wrong"}')
    with pytest.raises(ValueError, match="checksum"):
        RGB9Plus19Recipe.load(path)


def test_local_recipe_single_class_and_fallback_are_explicit():
    f, c, p, v = spatial_case()
    c[:] = 1
    with pytest.raises(ValueError, match="two groups"):
        RGB9Plus19Recipe.fit(f, c, p, v)
    _, report = convert_rgb(f, c, p, v, allow_local_fallback=True)
    assert report["conversion"] == "coarse-palette fallback"
    assert not report["full_RGB9_plus_19_applied"]


def test_pipeline_reuses_recipe_and_rejects_changed_coarse_mask(monkeypatch):
    from bora.pipeline import Case, prepare
    f, c, palette, variants = spatial_case()
    recipe = RGB9Plus19Recipe.fit(f, c, palette, variants)
    rng = np.random.default_rng(4)
    he = rng.integers(50, 220, (*c.shape, 3), dtype=np.uint8)
    tissue = np.ones(c.shape, bool)
    case = Case(he, c, tissue, f, np.array([.5, .5]), np.array([.5, .5]), tissue,
                variants=variants, rgb_recipe=recipe)
    monkeypatch.setattr(RGB9Plus19Recipe, "fit", lambda *a, **k: pytest.fail("Pipeline must not refit"))
    data = prepare(case)
    assert data["details"]["conversion"]["fit_repeated"] is False
    assert data["descriptors"].shape == (2, 2560)
    case.coarse = 3 - c
    with pytest.raises(ValueError, match="different coarse mask"):
        prepare(case)


def test_calibrated_case_manifest_loads_recipe_without_native_fit(tmp_path, monkeypatch):
    from bora.pipeline import ENCODER_REVISION, ENCODER_SHA256, load_case, prepare
    f, c, palette, variants = spatial_case()
    RGB9Plus19Recipe.fit(f, c, palette, variants).save(tmp_path / "local.json")
    np.save(tmp_path / "base.npy", f)
    tifffile.imwrite(tmp_path / "coarse.tif", c)
    tifffile.imwrite(tmp_path / "tissue.tif", np.ones(c.shape, np.uint8))
    Image.fromarray(np.full((*c.shape, 3), 150, np.uint8)).save(tmp_path / "he.png")
    spec = dict(image="he.png", coarse="coarse.tif", tissue="tissue.tif", tokens="base.npy",
        mpp_xy=[.5, .5], rgb_recipe="local.json", encoder_revision=ENCODER_REVISION,
        encoder_sha256=ENCODER_SHA256, variants={})
    for name, factor in (("scale_1", 2), ("scale_2", 4)):
        np.save(tmp_path / f"{name}.npy", variants[name]["values"])
        spec["variants"][name] = dict(features=f"{name}.npy",
            shape_yx=variants[name]["shape_yx"], mpp_xy=[.5 * factor] * 2)
    path = tmp_path / "case.json"; path.write_text(json.dumps(spec))
    monkeypatch.setattr(RGB9Plus19Recipe, "fit", lambda *a, **k: pytest.fail("Must not refit cached recipe"))
    case = load_case(path)
    assert "rgb_recipe" in case.provenance["input_sha256"]
    assert prepare(case)["details"]["conversion"]["fit_repeated"] is False
    spec["evidence_rgb"] = "he.png"; path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="not both"):
        load_case(path)


def test_shared_fit_is_deterministic_frozen_transfer_and_palette_independent(tmp_path, monkeypatch):
    x, y, kwargs = row_case()
    a = DiscriminativeAppearance.fit(x, y, **kwargs)
    b = DiscriminativeAppearance.fit(x, y, **kwargs)
    assert a.state == b.state
    contract = {k: kwargs[k] for k in ("feature_kind", "feature_columns", "encoder")}
    query = x[:5] + .17
    expected = a.predict_proba(query, **contract)
    path = tmp_path / "shared.json"; a.save(path)
    def forbidden(*args, **kwargs):
        raise AssertionError("Frozen transfer must not fit")
    monkeypatch.setattr(LogisticRegression, "fit", forbidden)
    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    loaded = DiscriminativeAppearance.load(path)
    np.testing.assert_array_equal(expected, loaded.predict_proba(query, **contract))
    np.testing.assert_allclose(expected.sum(1), 1)
    np.testing.assert_array_equal(loaded.rgb(expected, {k: [0, 0, 0] for k in a.state["classes"]}), 0)
    np.testing.assert_array_equal(expected, loaded.predict_proba(query, **contract))
    altered = dict(contract, feature_columns=list(reversed(contract["feature_columns"])))
    with pytest.raises(ValueError, match="feature order"):
        loaded.predict_proba(query, **altered)
    with pytest.raises(ValueError, match="pooled/spatial"):
        loaded.predict_proba(query[:, :-1], **contract)


@pytest.mark.parametrize("bad", ["query_fit", "anonymous_shared", "nan", "small", "one_class", "wrong_width", "unnamed", "constant"])
def test_invalid_transfer_contract_is_rejected(bad):
    x, y, kwargs = row_case()
    if bad == "query_fit": kwargs["splits"][0] = "test"
    if bad == "anonymous_shared": kwargs["label_kind"] = "anonymous"
    if bad == "nan": x[0, 0] = np.nan
    if bad == "small": x, y, kwargs["splits"] = x[:9], y[:9], ["train"] * 9
    if bad == "one_class": y[:] = "1"
    if bad == "wrong_width": x = x[:, :-1]
    if bad == "unnamed": kwargs["classes"].pop("3")
    if bad == "constant": x[:] = 1
    with pytest.raises(ValueError):
        DiscriminativeAppearance.fit(x, y, **kwargs)


def test_shared_selector_is_the_local_pairwise_rule():
    from bora.conversion import select_dimensions
    x, y, _ = row_case()
    np.testing.assert_array_equal(select_pairwise(x, y), select_dimensions(x, y))


def test_csv_bridge_excludes_query_labels_and_replays_without_refit(tmp_path):
    x, y, kwargs = row_case()
    table = tmp_path / "rows.csv"
    columns = kwargs["feature_columns"]
    with table.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "x", "y", "label", "split", *columns])
        for i, row in enumerate(x):
            writer.writerow([f"tile-{i}", i * 14 + 7, 7,
                y[i] if i < 30 else "poison_target_label", "train" if i < 30 else "test", *row])
    spec = dict(format="bora-appearance-request-v1", mode="shared", feature_table="rows.csv",
        feature_columns=columns, feature_kind="virchow2-spatial", encoder=ENCODER,
        label_column="label", split_column="split", id_column="id", x_column="x", y_column="y",
        coordinate_space="native_pixel_centres", label_kind="semantic", classes=kwargs["classes"],
        output_prefix="fitted")
    request = tmp_path / "request.json"; request.write_text(json.dumps(spec))
    report = run_request(request)
    assert report["fit_rows"] == 30 and report["rows"] == 36
    assert "probability:poison_target_label" not in report["probability_columns"]
    spec.update(recipe_path="fitted.recipe.json", output_prefix="replayed")
    request.write_text(json.dumps(spec))
    replay = run_request(request)
    assert replay["loaded_without_refit"] and replay["fit_rows"] == 0
    assert (tmp_path / "fitted.scores.csv").read_bytes() == (tmp_path / "replayed.scores.csv").read_bytes()
    with pytest.raises(FileExistsError): run_request(request)
    spec["output_prefix"] = "bad-coordinate"; spec["coordinate_space"] = "screen"
    request.write_text(json.dumps(spec))
    assert main([str(request)]) == 2
    assert not (tmp_path / "bad-coordinate.recipe.json").exists()


def test_pooled_contract_stays_distinct_and_inline_bridge_is_cpu_only(tmp_path):
    x, y, kwargs = row_case("local")
    pooled = np.concatenate([x, x * .2], axis=1)
    columns = [f"f{i:04d}" for i in range(2560)]
    spec = dict(format="bora-appearance-request-v1", mode="local", features=pooled.tolist(),
        labels=y.tolist(), splits=["reference"] * len(y), feature_columns=columns,
        feature_kind="virchow2-pooled", label_kind="anonymous", encoder=ENCODER,
        classes=kwargs["classes"], output_prefix="pooled")
    path = tmp_path / "request.json"; path.write_text(json.dumps(spec))
    script = (
        "import sys; from bora.appearance_bridge import main; "
        "code=main(sys.argv[1:]); assert not any('fastpls' in k or k=='torch' for k in sys.modules); "
        "raise SystemExit(code)"
    )
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / "src"))
    run = subprocess.run([sys.executable, "-c", script, str(path)], env=env,
                         capture_output=True, text=True, timeout=45)
    assert run.returncode == 0, run.stderr
    report = json.loads(run.stdout)
    assert report["feature_kind"] == "virchow2-pooled"
    assert report["label_kind"] == "anonymous" and not report["full_RGB9_plus_19"]
    recipe = DiscriminativeAppearance.load(tmp_path / "pooled.recipe.json")
    with pytest.raises(ValueError, match="feature kind"):
        recipe.predict_proba(x, feature_kind="virchow2-spatial",
            feature_columns=kwargs["feature_columns"], encoder=ENCODER)
