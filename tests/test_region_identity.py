import copy
import json

import numpy as np
import pytest
from rasterio.features import shapes

from bora.classifier import NeighborContextClassifier
from bora.groups import boundary_context, compact_groups, group_means
from bora.pipeline import digest
from bora.region_identity import classify_regions, describe_region_polygons, geometry_digest


def fixture_data():
    rng = np.random.default_rng(41)
    mask = np.zeros((57, 93), np.uint8)
    mask[2:55, 3:50] = 2
    mask[5:45, 50:90] = 10
    mask[34:50, 20:60] = 30
    mask[8:11, 5:10] = 30  # A disconnected piece must share group 30's prediction.
    features = [dict(type="Feature", properties=dict(value=int(value), name=f"Group {value}"),
        geometry=geometry) for geometry, value in shapes(mask) if value]
    tokens = rng.normal(size=(3, 4, 1280)).astype(np.float32)
    return mask, features, tokens


@pytest.mark.parametrize("stripe_rows", [1, 7, 32, 57, 100])
def test_streamed_polygons_match_original_descriptors_across_stripe_seams(stripe_rows):
    mask, features, tokens = fixture_data()
    compact, expected_ids = compact_groups(mask)
    means, mass = group_means(tokens, compact, (29, 47))
    expected = boundary_context(compact, means, (.25, .27))
    ids, actual, actual_mass, graph = describe_region_polygons(tokens, features,
        mask.shape, (29, 47), (.25, .27), stripe_rows=stripe_rows)
    np.testing.assert_array_equal(ids, expected_ids)
    np.testing.assert_array_equal(actual_mass, mass)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(graph, graph.T)


def test_padding_not_stretched_and_isolated_group_has_zero_contrast():
    mask = np.zeros((15, 15), np.uint8)
    mask[14, 14] = 1
    features = [dict(properties=dict(value=1), geometry=g) for g, v in shapes(mask) if v]
    tokens = np.zeros((2, 2, 1280), np.float32)
    tokens[1, 1] = 7
    _, actual, mass, _ = describe_region_polygons(tokens, features, mask.shape,
        mask.shape, (.5, .5), stripe_rows=1)
    np.testing.assert_array_equal(actual[0, :1280], 7)
    np.testing.assert_array_equal(actual[0, 1280:], 0)
    assert mass.tolist() == [1]


def test_small_budget_reduces_stripes_without_changing_descriptors():
    mask, features, tokens = fixture_data()
    expected = describe_region_polygons(tokens, features, mask.shape, (29, 47), (.25, .27))
    actual = describe_region_polygons(tokens, features, mask.shape, (29, 47), (.25, .27),
        stripe_rows=100, maximum_working_bytes=400000)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def test_overlapping_same_group_pieces_do_not_hide_a_conflicting_group():
    mask, features, tokens = fixture_data()
    duplicate = copy.deepcopy(features[0])
    other = copy.deepcopy(duplicate)
    other["properties"]["value"] = 99
    # Input painter order could hide 99 between two copies of the same group.
    features.extend([other, duplicate])
    with pytest.raises(ValueError, match="overlap"):
        describe_region_polygons(tokens, features, mask.shape, (29, 47), (.25, .27))


@pytest.mark.parametrize("invalid", ["overlap", "outside", "nan", "channels", "padding", "allocation", "no_pixel"])
def test_invalid_polygon_or_feature_input_fails(invalid):
    mask, features, tokens = fixture_data()
    opts = {}
    if invalid == "overlap":
        extra = copy.deepcopy(features[0]); extra["properties"]["value"] = 99
        features.append(extra)
    if invalid == "outside":
        features[0]["geometry"]["coordinates"][0][0] = [-1, -1]
    if invalid == "nan": tokens[0, 0, 0] = np.nan
    if invalid == "channels": tokens = tokens[:, :, :-1]
    if invalid == "padding": tokens = tokens[:-1]
    if invalid == "allocation": opts["maximum_working_bytes"] = 1
    if invalid == "no_pixel":
        features.append(dict(properties=dict(value=99), geometry=dict(type="Polygon",
            coordinates=[[[0, 0], [.1, 0], [.1, .1], [0, .1], [0, 0]]])))
    with pytest.raises(ValueError):
        describe_region_polygons(tokens, features, mask.shape, (29, 47), (.25, .27), **opts)


def files(tmp_path):
    mask, features, tokens = fixture_data()
    doc = dict(type="FeatureCollection", features=features,
        native_shape_yx=list(mask.shape), source_image_sha256="a" * 64)
    contours = tmp_path / "contours.geojson"
    contours.write_text(json.dumps(doc))
    token_path = tmp_path / "tokens.npy"
    np.save(token_path, tokens)
    reg = dict(format="bora-region-registration-v1", source_frame_id="synthetic/native",
        native_shape_yx=list(mask.shape), working_shape_yx=[29, 47],
        native_mpp_xy=[.25, .27], pixel_aligned=True, registration_method="native_same_grid",
        image_sha256="a" * 64, contours_sha256=digest(contours), tokens_sha256=digest(token_path))
    registration = tmp_path / "registration.json"
    registration.write_text(json.dumps(reg))
    rng = np.random.default_rng(12)
    x = rng.normal(size=(40, 2560)); y = np.tile([1, 3], 20)
    model_path = tmp_path / "frozen.npz"
    NeighborContextClassifier().fit(x, y, {1: "tissue A", 3: "tissue B"}).save(model_path)
    return contours, registration, token_path, model_path, json.loads(contours.read_text()), reg


def test_frozen_native_head_changes_only_properties_and_publishes_receipt(tmp_path):
    contours, registration, tokens, model, original, _ = files(tmp_path)
    input_hashes = [digest(p) for p in (contours, registration, tokens, model)]
    output = tmp_path / "named"
    report = classify_regions(contours, registration, tokens, model, output, stripe_rows=7)
    result = json.loads((output / "tissue-annotations.geojson").read_text())
    assert result["features"] != original["features"]
    assert [f["geometry"] for f in result["features"]] == [f["geometry"] for f in original["features"]]
    assert report["geometry_sha256"] == geometry_digest(original["features"])
    assert report["geometry_unchanged"] and report["complete"]
    assert report["group_count"] == 3 and report["feature_count"] > 3
    assert report["biological_accuracy_qualified"] is False
    assert report["responses_are_calibrated_probabilities"] is False
    assert all(f["properties"]["name"] in ("tissue A", "tissue B") for f in result["features"])
    # Two separated pieces of the same anonymous group retain one identity.
    source_30 = [f for f in result["features"] if f["properties"]["source_group_id"] == 30]
    assert len(source_30) == 2
    assert len({f["properties"]["predicted_tissue_id"] for f in source_30}) == 1
    assert [digest(p) for p in (contours, registration, tokens, model)] == input_hashes
    with np.load(output / "descriptors.npz", allow_pickle=False) as data:
        loaded = NeighborContextClassifier.load(model)
        np.testing.assert_array_equal(data["predicted_class"], loaded.predict(data["x"]))
    for name, sha in report["output_sha256"].items():
        assert digest(output / name) == sha
    with pytest.raises(FileExistsError):
        classify_regions(contours, registration, tokens, model, output)


@pytest.mark.parametrize("change", ["hash", "registration", "frame", "dimension", "spacing"])
def test_registered_input_changes_fail_without_publishing(tmp_path, change):
    contours, registration, tokens, model, doc, reg = files(tmp_path)
    if change == "hash": reg["tokens_sha256"] = "0" * 64
    if change == "registration": reg["pixel_aligned"] = False
    if change == "frame": reg["image_sha256"] = "b" * 64
    if change == "dimension": reg["native_shape_yx"] = [58, 93]
    if change == "spacing":
        doc["native_mpp_xy"] = [.5, .5]
        contours.write_text(json.dumps(doc))
        reg["contours_sha256"] = digest(contours)
    registration.write_text(json.dumps(reg))
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        classify_regions(contours, registration, tokens, model, output)
    assert not output.exists()
