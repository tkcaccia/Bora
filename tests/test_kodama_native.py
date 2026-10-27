"""Checks for the absolute-coordinate WSI handoff used by selected Bora."""

import numpy as np

from bora.kodama_native import project_token_window


def test_native_projection_does_not_depend_on_window_partition():
    generator = np.random.default_rng(17)
    scores = generator.random((8, 9, 3), dtype=np.float32)
    scores /= scores.sum(axis=-1, keepdims=True)
    whole = project_token_window(scores, (0, 0), (224, 252), (2.0, 2.0))
    left = project_token_window(scores, (0, 0), (224, 87), (2.0, 2.0))
    right = project_token_window(scores, (0, 87), (224, 165), (2.0, 2.0))
    assert np.array_equal(whole, np.concatenate((left, right), axis=1))


def test_native_projection_preserves_absolute_origin():
    scores = np.zeros((4, 4, 3), np.float32)
    scores[..., 0] = 1
    scores[1, 2] = (0, 1, 0)
    full = project_token_window(scores, (0, 0), (112, 112), (2.0, 2.0))
    crop = project_token_window(scores, (23, 51), (42, 37), (2.0, 2.0))
    assert np.array_equal(full[23:65, 51:88], crop)
