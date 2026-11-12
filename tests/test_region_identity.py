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


def test_cropped_native_bounds_map_groups_to_the_correct_spatial_tokens():
    mask = np.zeros((30, 28), np.uint8)
    mask[3:8, 4:9] = 1
    mask[18:23, 19:24] = 2
    features = [dict(properties=dict(value=int(v)), geometry=g)
                for g, v in shapes(mask) if v]
    tokens = np.zeros((2, 2, 1280), np.float32)
    tokens[0, 0] = 1
    tokens[1, 1] = 9
    ids, descriptors, mass, _ = describe_region_polygons(
        tokens, features, mask.shape, (28, 28), (.25, .25),
        working_content_shape_yx=(20, 20), native_content_bounds_xyxy=(4, 3, 24, 23))
    np.testing.assert_array_equal(ids, [1, 2])
    np.testing.assert_array_equal(mass, [25, 25])
    np.testing.assert_array_equal(descriptors[:, 0], [1, 9])
    # The old full-frame scale gives the second group a different descriptor.
    _, wrong, _, _ = describe_region_polygons(
        tokens, features, mask.shape, (28, 28), (.25, .25),
        working_content_shape_yx=(20, 20))
    assert wrong[1, 0] != 9
    with pytest.raises(ValueError, match="content bounds"):
        describe_region_polygons(tokens, features, mask.shape, (28, 28), (.25, .25),
            working_content_shape_yx=(20, 20), native_content_bounds_xyxy=(4, 3, 23, 23))


def test_uncovered_glass_tokens_can_remain_nan_in_saved_full_channel_cache():
    mask = np.zeros((15, 15), np.uint8)
    mask[14, 14] = 1
    features = [dict(properties=dict(value=1), geometry=g) for g, v in shapes(mask) if v]
    tokens = np.full((2, 2, 1280), np.nan, np.float16)
    tokens[1, 1] = 7
    _, descriptors, mass, _ = describe_region_polygons(
        tokens, features, mask.shape, mask.shape, (.5, .5), stripe_rows=1)
    assert mass.tolist() == [1]
    np.testing.assert_array_equal(descriptors[0, :1280], 7)
    np.testing.assert_array_equal(descriptors[0, 1280:], 0)


def test_small_budget_reduces_stripes_without_changing_descriptors():
    mask, features, tokens = fixture_data()
    expected = describe_region_polygons(tokens, features, mask.shape, (29, 47), (.25, .27))
    actual = describe_region_polygons(tokens, features, mask.shape, (29, 47), (.25, .27),
        stripe_rows=100, maximum_working_bytes=400000)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def test_subpixel_roundoff_at_native_edge_is_accepted_but_real_overflow_is_not():
    mask = np.ones((15, 15), np.uint8)
    features = [dict(properties=dict(value=1), geometry=dict(type='Polygon',
        coordinates=[[[0, 0], [15, 0], [15, 15], [0, 15], [0, 0]]]))]
    tokens = np.ones((2, 2, 1280), np.float32)
    exact = describe_region_polygons(tokens, features, mask.shape, mask.shape, (.5, .5))
    rounded = copy.deepcopy(features)
    rounded[0]['geometry']['coordinates'][0][1] = [15 + 5e-10, 0]
    rounded[0]['geometry']['coordinates'][0][2] = [15 + 5e-10, 15]
    actual = describe_region_polygons(tokens, rounded, mask.shape, mask.shape, (.5, .5))
    for left, right in zip(actual, exact):
        np.testing.assert_array_equal(left, right)
    rounded[0]['geometry']['coordinates'][0][1] = [15.001, 0]
    rounded[0]['geometry']['coordinates'][0][2] = [15.001, 15]
    with pytest.raises(ValueError, match='outside the native image frame'):
        describe_region_polygons(tokens, rounded, mask.shape, mask.shape, (.5, .5))


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


def test_calibrated_resize_registration_uses_unpadded_content_mapping(tmp_path):
    contours, registration, tokens, model, doc, reg = files(tmp_path)
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(dict(status="COMPLETE", source_sha256="a" * 64,
        source_original_shape_yx=[57, 93], source_mpp_xy=[.25, .27],
        target_shape_yxc=[29, 47, 3], true_working_size_wh=[20, 15],
        target_sha256="b" * 64)))
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps(dict(status="COMPLETE", stage="virchow2_spatial",
        identity=dict(image_sha256="b" * 64, source_sha256="a" * 64,
                      encoder=dict(channels=1280)),
        outputs={"factor-1": dict(path="tokens.npy", bytes=tokens.stat().st_size,
                                  sha256=digest(tokens))})))
    reg.update(format="bora-region-registration-v2",
        registration_method="calibrated_resize_pad", working_content_shape_yx=[15, 20],
        calibration_receipt=str(calibration), calibration_sha256=digest(calibration),
        spatial_cache_manifest=str(cache), spatial_cache_manifest_sha256=digest(cache),
        data_store=str(tmp_path))
    registration.write_text(json.dumps(reg))
    output = tmp_path / "classified"
    report = classify_regions(contours, registration, tokens, model, output, stripe_rows=7)
    assert report["working_content_shape_yx"] == [15, 20]
    data = np.load(tokens, allow_pickle=False)
    _, expected, _, _ = describe_region_polygons(data, doc["features"], (57, 93),
        (29, 47), (.25, .27), working_content_shape_yx=(15, 20))
    with np.load(output / "descriptors.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["x"], expected)


@pytest.mark.parametrize("shape_key", ["source_native_shape_yxc", "source_original_shape_yx"])
def test_two_step_registration_binds_native_resize_pad_and_full_channel_cache(tmp_path, shape_key):
    contours, registration, tokens, model, doc, reg = files(tmp_path)
    native = tmp_path / "native.json"
    first = dict(status="COMPLETE", source_sha256="a" * 64,
        source_mpp_xy=[.25, .27],
        native_analysis_bounds_xyxy=[0, 0, 93, 57],
        target_shape_yxc=[15, 20, 3], target_sha256="c" * 64)
    first[shape_key] = [57, 93, 3] if shape_key.endswith("yxc") else [57, 93]
    native.write_text(json.dumps(first))
    pad = tmp_path / "pad.json"
    second = dict(status="COMPLETE", source_sha256="c" * 64,
        source_shape_yxc=[15, 20, 3], target_shape_yxc=[29, 47, 3],
        true_working_size_wh=[20, 15], target_sha256="b" * 64)
    pad.write_text(json.dumps(second))
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps(dict(status="COMPLETE", stage="virchow2_spatial",
        identity=dict(image_sha256="b" * 64, source_sha256="c" * 64,
                      encoder=dict(channels=1280)),
        outputs={"factor-1": dict(path="tokens.npy", bytes=tokens.stat().st_size,
                                  sha256=digest(tokens))})))
    reg.update(format="bora-region-registration-v3",
        registration_method="calibrated_resize_chain", working_content_shape_yx=[15, 20],
        native_content_bounds_xyxy=[0, 0, 93, 57],
        native_identity_receipt=str(native), native_identity_sha256=digest(native),
        native_identity_kind="calibration_source",
        calibration_receipts=[dict(path=str(p), sha256=digest(p)) for p in (native, pad)],
        spatial_cache_manifest=str(cache), spatial_cache_manifest_sha256=digest(cache),
        data_store=str(tmp_path))
    doc["full_slide_input_identity"] = dict(source_frame_id="synthetic/native",
        analysis_bounds_native_xyxy=[0, 0, 93, 57],
        input_sha256=dict(image="b" * 64))
    contours.write_text(json.dumps(doc))
    reg["contours_sha256"] = digest(contours)
    registration.write_text(json.dumps(reg))
    output = tmp_path / "classified-v3"
    report = classify_regions(contours, registration, tokens, model, output)
    assert report["registration_format"] == "bora-region-registration-v3"
    assert report["native_content_bounds_xyxy"] == [0, 0, 93, 57]
    assert set(report["input_sha256"]) == {"contours", "registration", "tokens", "model",
                                              "native_identity", "calibration_0", "calibration_1",
                                              "spatial_cache_manifest"}
    second["source_sha256"] = "d" * 64
    pad.write_text(json.dumps(second))
    reg["calibration_receipts"][1]["sha256"] = digest(pad)
    registration.write_text(json.dumps(reg))
    with pytest.raises(ValueError, match="chain"):
        classify_regions(contours, registration, tokens, model, tmp_path / "broken-chain")


def test_frozen_native_head_changes_only_properties_and_publishes_receipt(tmp_path):
    contours, registration, tokens, model, original, reg = files(tmp_path)
    # Published Bora WSI polygons use a text classification, not a mapping.
    for feature in original["features"]:
        feature["properties"]["classification"] = "anonymous_group"
    contours.write_text(json.dumps(original))
    reg["contours_sha256"] = digest(contours)
    registration.write_text(json.dumps(reg))
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
    for source, named in zip(original["features"], result["features"]):
        for field in ("value", "name", "classification"):
            assert named["properties"][field] == source["properties"][field]
        assert named["properties"]["predicted_tissue"] in ("tissue A", "tissue B")
    assert all(f["properties"]["source_group_classification"] == "anonymous_group"
               for f in result["features"])
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
