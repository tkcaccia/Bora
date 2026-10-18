import numpy as np
import pytest
from bora.groups import boundary_context, compact_groups, group_means
from bora.classifier import NeighborContextClassifier
from bora.classifier import FASTPLS_CORE_REVISION, FASTPLS_REVISION
from bora._vendor.fastpls import PLS


def test_context_weights_use_physical_shared_boundary_lengths():
    regions = np.array([[1, 2], [1, 3]], np.int32)
    means = np.repeat(np.array([[1.], [2.], [4.]], np.float32), 1280, axis=1)
    x = boundary_context(regions, means, [1., 2.])
    np.testing.assert_array_equal(x[:, :1280], means)
    np.testing.assert_allclose(x[:, 1280], [-2, 0, 8/3], rtol=1e-6)


def test_isolated_group_has_zero_context_and_sparse_ids_are_safe():
    mask = np.array([[1000000, 0, 2], [0, 0, 2]], np.int64)
    regions, ids = compact_groups(mask)
    assert ids.tolist() == [2, 1000000]
    means = np.ones((2, 1280), np.float32)
    assert np.all(boundary_context(regions, means, [.5, .5])[:, 1280:] == 0)


def test_disconnected_pieces_share_one_area_weighted_token_mean():
    tokens = np.zeros((2, 2, 1280), np.float32)
    tokens[1, 1] = 10
    regions = np.zeros((28, 28), np.int32)
    regions[0, 0] = 1
    regions[20, 20:23] = 1
    means, mass = group_means(tokens, regions, (28, 28))
    assert mass.tolist() == [4]
    np.testing.assert_allclose(means, 7.5)


def test_token_padding_is_not_stretched_to_the_crop():
    tokens = np.zeros((2, 2, 1280), np.float32)
    tokens[1, 1] = 7
    regions = np.zeros((15, 15), np.int32); regions[14, 14] = 1
    means, _ = group_means(tokens, regions, (15, 15))
    assert np.all(means == 7)


def test_native_neighbor_context_artifact_roundtrip_is_identical(tmp_path):
    rng = np.random.default_rng(19)
    x = rng.normal(size=(40, 2560)); y = np.tile([1, 3], 20)
    model = NeighborContextClassifier().fit(x, y, {1: "tumor", 3: "stroma"})
    expected = model.predict(x[:7])
    path = tmp_path / "model.npz"
    meta = model.save(path)
    loaded = NeighborContextClassifier.load(path)
    np.testing.assert_array_equal(loaded.predict(x[:7]), expected)
    assert loaded.actual_components_ == 1 and meta["native_head_count"] == 1
    assert meta["fastpls_revision"] == FASTPLS_REVISION
    assert meta["fastpls_core_revision"] == FASTPLS_CORE_REVISION
    with pytest.raises(FileExistsError):
        model.save(path)


def test_corrected_fastpls_lda_handles_wide_24_component_fit():
    # Regression case from the upstream SIMPLS score-moment correction.
    rng = np.random.default_rng(19)
    x = np.asfortranarray(rng.normal(size=(40, 2560)))
    y = np.tile([1, 3], 20)
    model = PLS(n_components=24, method="simpls", classifier="lda",
                scaling="autoscaling", backend="cpu", seed=314159).fit(x, y)
    assert model.n_components_ == 24
    assert np.isfinite(model.predict_scores(x)).all()
    np.testing.assert_array_equal(model.predict(x), y)


@pytest.mark.parametrize("mutate", ["nan", "columns", "single_class", "float_targets", "missing_name"])
def test_invalid_training_contract_is_rejected(mutate):
    x = np.ones((8, 2560)); y = np.tile([1, 2], 4); names = {1: "a", 2: "b"}
    if mutate == "nan": x[0, 0] = np.nan
    if mutate == "columns": x = x[:, :-1]
    if mutate == "single_class": y[:] = 1
    if mutate == "float_targets": y = y.astype(float)
    if mutate == "missing_name": names = {1: "a"}
    with pytest.raises(ValueError):
        NeighborContextClassifier().fit(x, y, names)
