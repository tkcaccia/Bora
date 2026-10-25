import numpy as np
import pytest

from bora.contextual_token_head import (
    contextualize_pca, fit_numeric_head, predict_numeric_head,
)


def test_global_context_crosses_patch_seam_and_respects_support():
    field = np.zeros((5, 33, 3), np.float32)
    field[:, :16, 0] = 1
    field[:, 16:, 1] = 1
    support = np.ones((5, 33), bool)
    support[:, 32] = False
    result = contextualize_pca(field, support, 1)
    assert 0 < result[2, 15, 1] < 1
    assert 0 < result[2, 16, 0] < 1
    assert np.array_equal(result[:, 32], field[:, 32])


def test_numeric_head_is_deterministic_and_normalized():
    samples = np.asarray([[0, 0], [0, .1], [1, 0], [1, .1],
                          [0, 1], [.1, 1]], np.float32)
    labels = np.asarray([1, 1, 2, 2, 3, 3], np.uint8)
    head = fit_numeric_head(samples, labels)
    field = samples.reshape(2, 3, 2)
    p = predict_numeric_head(field, head,
        training_prior=np.asarray([.6, .2, .2], np.float32),
        prior_exponent=.5)
    assert p.shape == (2, 3, 3)
    assert np.allclose(p.sum(-1), 1, atol=1e-6)
    assert np.array_equal(p, predict_numeric_head(field, head,
        training_prior=np.asarray([.6, .2, .2], np.float32),
        prior_exponent=.5))


def test_invalid_frame_or_head_fails_closed():
    field = np.zeros((3, 4, 2), np.float32)
    with pytest.raises(ValueError):
        contextualize_pca(field, np.ones((4, 3), bool), 1)
    with pytest.raises(ValueError):
        predict_numeric_head(field, {}, prior_exponent=0)
    with pytest.raises(ValueError):
        fit_numeric_head(field.reshape(-1, 2),
            np.tile(np.array([1, 2], np.uint8), 6))
