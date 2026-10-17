import numpy as np
import pytest
from experiments.bora_cycles.methods import predict
from experiments.bora_cycles.registry import CYCLE_ONE


@pytest.mark.parametrize("key", [k for k, v in CYCLE_ONE.items() if v.implemented])
def test_candidates_keep_ids_and_do_not_freeze_coarse_interiors(key):
    h, w = 40, 48
    yy, xx = np.indices((h, w))
    truth_pattern = xx > 19+yy//5
    palette = np.array([[200, 40, 30], [25, 45, 210]], np.uint8)
    rgb = palette[truth_pattern.astype(int)]
    coarse = np.where(xx >= 24, 2, 1).astype(np.uint8)
    coarse[:8, :8] = 0
    support = np.ones((h, w), bool)
    originals = [a.copy() for a in (rgb, coarse, support)]
    labels, details = predict(key, rgb, coarse, support, [1., 1.], palette, he=rgb)
    assert labels.shape == tuple(details["analysis_shape_yx"])
    assert np.isfinite(labels).all()
    assert set(np.unique(labels)) <= {1, 2}
    assert details["fine_reference_used"] is False
    assert details["hard_anchors"] is False
    assert details["fixed_margin"] is False
    for original, current in zip(originals, (rgb, coarse, support)):
        np.testing.assert_array_equal(current, original)


def test_unknown_candidate_cannot_be_run_as_a_completed_algorithm():
    with pytest.raises(ValueError, match="not implemented"):
        predict("C99", None, None, None, None, None)


def test_roster_has_twenty_explicitly_distinct_mechanism_tags():
    assert len(CYCLE_ONE) == 20
    assert len({value.mechanism for value in CYCLE_ONE.values()}) == 20
    assert all(value.hypothesis and value.risk for value in CYCLE_ONE.values())
    assert all(value.implemented for value in CYCLE_ONE.values())


def test_white_candidate_preserves_enclosed_lumina_and_reports_exclusions():
    coarse = np.ones((24, 24), np.uint8)
    rgb = np.full((24, 24, 3), [190, 40, 40], np.uint8)
    he = rgb.copy()
    he[:3] = 255
    he[10:13, 10:13] = 255
    labels, detail = predict("C20", rgb, coarse, np.ones_like(coarse, bool),
                             [1., 1.], [[190, 40, 40]], he=he)
    assert np.all(labels[:3] == 0)
    assert np.all(labels[10:13, 10:13] == 1)
    assert detail["proposed_glass_pixels"] == 72


def test_adaptive_colour_labels_match_unquantized_active_set():
    from bora.evidence import Evidence
    from experiments.bora_cycles.methods import adaptive_colors, correct_coarse
    from experiments.bora_cycles.native import simplex_qp
    rng = np.random.default_rng(123)
    palette = rng.integers(20, 230, size=(5, 3), dtype=np.uint8)
    fractions = rng.dirichlet(np.ones(5), size=(24, 25))
    rgb = np.rint(fractions@palette).astype(np.uint8)
    coarse = np.tile(np.arange(25)//5+1, (24, 1)).astype(np.uint8)
    c = Evidence(rgb, coarse, np.ones_like(coarse, bool), [1., 1.])
    actual, details = adaptive_colors(c, palette)
    exact, _ = simplex_qp(c.rgb, c.lattice(), palette/255)
    exact = correct_coarse(exact, c)
    np.testing.assert_array_equal(actual.argmax(-1), exact.argmax(-1))
    assert sum(r["accepted_colours"] for r in details["refinement"]) == details["unique_rgb_colours"]


def test_probability_bounds_are_exercised_for_an_ambiguous_five_class_palette():
    from bora.evidence import Evidence
    from experiments.bora_cycles.methods import interval_palette
    rng = np.random.default_rng(62)
    palette = rng.integers(25, 225, size=(5, 3), dtype=np.uint8)
    fractions = rng.dirichlet(np.full(5, 5.), size=(15, 20))
    rgb = np.rint(fractions@palette).astype(np.uint8)
    coarse = np.tile(np.arange(20)//4+1, (15, 1)).astype(np.uint8)
    c = Evidence(rgb, coarse, np.ones_like(coarse, bool), [1., 1.])
    q, detail = interval_palette(c, palette)
    assert detail["bounded_colour_cells"] > 0
    assert 0 <= detail["mean_probability_interval_width"] <= 1
    assert np.isfinite(q).all()
    np.testing.assert_allclose(q.sum(-1), 1, atol=2e-6)
