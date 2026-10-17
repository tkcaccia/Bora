import numpy as np
import pytest
from experiments.bora_cycles.metrics import RegionEvaluator, discovery_score
from experiments.bora_cycles.diagnostics import white_regions, novelty, rare_colors, mask_diagnostics


def test_small_regions_absent_from_coarse_are_not_omitted():
    truth = np.ones((30, 30), np.uint8)
    truth[2:4, 2:4] = 7
    truth[0, :] = 0
    prediction = np.ones_like(truth)
    prediction[8:10, 8:10] = 0
    score = RegionEvaluator(truth, [2., 3.]).score(prediction)
    assert set(score["per_class"]) == {"1", "7"}
    assert score["per_class"]["7"]["iou"] == 0
    assert score["regions"]["small_le_400_um2"]["completely_missed"] == 1
    assert score["missed_regions"][0]["area_um2"] == 24
    assert score["annotated_unassigned_pixels"] == 4
    assert score["valid_pixels"] == 870


def test_empty_size_strata_are_null():
    truth = np.ones((3, 3), np.uint8)
    score = RegionEvaluator(truth, [1., 1.]).score(truth)
    assert score["regions"]["medium_400_2500_um2"]["recall_at_50pct"] is None
    assert score["macro_iou"] == 1


def test_unannotated_tissue_does_not_change_accuracy():
    truth = np.zeros((10, 10), np.uint8)
    truth[2:5, 2:5] = 1
    a = np.ones_like(truth)
    b = a.copy()
    b[truth == 0] = 8
    evaluator = RegionEvaluator(truth, [1., 1.])
    assert evaluator.score(a)["macro_iou"] == evaluator.score(b)["macro_iou"] == 1


def test_white_lumen_is_distinct_from_border_white():
    he = np.full((20, 20, 3), 120, np.uint8)
    he[0:3, :, :] = 255
    he[8:12, 8:12] = 255
    white, exterior, _ = white_regions(he)
    assert white[9, 9] and not exterior[9, 9]
    assert exterior[1, 1]


def test_novel_colour_has_no_biological_class_assigned():
    rgb = np.full((20, 20, 3), [200, 30, 30], np.uint8)
    coarse = np.ones((20, 20), np.uint8)
    coarse[12:, :] = 0
    rgb[14:18, 4:8] = [20, 20, 230]
    _, regions, detail = novelty(rgb, coarse, np.ones_like(coarse, bool), [1., 1.])
    assert regions[15, 5] > 0
    assert detail["fine_reference_used"] is False
    assert "unresolved" in detail["identity"]
    assert discovery_score(regions, coarse)["semantic_identity"].startswith("unknown")


def test_rarity_detects_minority_colours_even_when_all_pixels_are_in_training_domain():
    rgb = np.full((64, 64, 3), [200, 30, 30], np.uint8)
    rgb[30:32, 30:32] = [20, 40, 230]
    support = np.ones((64, 64), bool)
    coarse = np.ones((64, 64), np.uint8)
    _, unseen, _ = novelty(rgb, coarse, support, [1., 1.])
    _, rare, detail = rare_colors(rgb, support, [1., 1.])
    assert unseen[30, 30] == 0
    assert rare[30, 30] > 0
    assert detail["coarse_labels_used"] is False
    assert detail["proposed_pixels"] == 4


def test_cpp_active_set_matches_independent_scipy_solver():
    from scipy.optimize import minimize
    from experiments.bora_cycles.native import simplex_qp
    rng = np.random.default_rng(91)
    for k in (2, 5, 9):
        colors = rng.uniform(size=(k, 3))
        rgb = rng.uniform(size=(8, 3)).astype(np.float32)
        prior = rng.dirichlet(np.ones(k), size=8).astype(np.float32)
        actual, detail = simplex_qp(rgb, prior, colors, threads=1)
        for i in range(len(rgb)):
            fun = lambda p: .5*np.sum((p@colors-rgb[i])**2)+.006*np.sum((p-prior[i])**2)
            jac = lambda p: (p@colors-rgb[i])@colors.T+.012*(p-prior[i])
            result = minimize(fun, prior[i], jac=jac, bounds=[(0, 1)]*k,
                              constraints=[dict(type="eq", fun=lambda p:p.sum()-1,
                                                jac=lambda p:np.ones(k))],
                              method="SLSQP", options=dict(ftol=1e-12, maxiter=300))
            assert result.success
            np.testing.assert_allclose(actual[i], result.x, atol=2e-5)
        assert detail["max_kkt_residual"] < 1e-8
        np.testing.assert_allclose(actual.sum(-1), 1, atol=2e-7)
        assert actual.min() >= 0


def test_cpp_linear_kernel_parity_across_threads():
    from experiments.bora_cycles.native import logits
    rng = np.random.default_rng(8)
    x = rng.normal(size=(31, 47, 10)).astype(np.float32)
    coef = rng.normal(size=(7, 10))
    intercept = rng.normal(size=7)
    expected = x@coef.T+intercept
    actual = logits(x, coef, intercept, threads=1)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
    np.testing.assert_array_equal(actual, logits(x, coef, intercept, threads=2))


def test_cpp_semimarkov_matches_exhaustive_sequence_optimization():
    import itertools
    from experiments.bora_cycles.native import semimarkov
    rng = np.random.default_rng(103)
    unary = rng.uniform(0, 1, size=(4, 7, 2)).astype(np.float32)
    penalty, limit = .6, 3
    actual = semimarkov(unary, penalty=penalty, duration_limit=limit, threads=1)
    np.testing.assert_array_equal(
        actual, semimarkov(unary, penalty=penalty, duration_limit=limit, threads=2))
    for row in range(len(unary)):
        def cost(labels):
            value = sum(unary[row, i, label] for i, label in enumerate(labels))
            for _, run in itertools.groupby(labels):
                length = len(list(run))
                value += penalty*(1+.1*np.log1p(min(length, limit)))
                value += penalty*.1/(1+limit)*max(0, length-limit)
            return value
        choices = list(itertools.product(range(2), repeat=unary.shape[1]))
        expected = min(choices, key=cost)
        np.testing.assert_array_equal(actual[row], expected)


def test_cpp_spatial_certificate_matches_dense_class_decisions():
    from experiments.bora_cycles.native import adaptive_linear, logits
    rng = np.random.default_rng(618)
    rgb = rng.uniform(size=(117, 139, 3)).astype(np.float32)
    rgb[:64, :64] = [.8, .1, .2]
    coefficient = rng.normal(size=(6, 3))
    intercept = rng.normal(size=6)
    expected = logits(rgb, coefficient, intercept, threads=1).argmax(-1)
    actual, details = adaptive_linear(rgb, coefficient, intercept, threads=2)
    np.testing.assert_array_equal(actual, expected)
    assert details["visited_blocks"] > 0
    assert details["certified_blocks"] > 0
    assert details["densely_classified_pixels"] < rgb.shape[0]*rgb.shape[1]


def test_compiled_metric_matches_every_numpy_metric_field_exactly():
    rng = np.random.default_rng(950)
    truth = rng.choice([0, 1, 7, 21], size=(127, 179)).astype(np.uint8)
    prediction = rng.choice([0, 1, 7, 21, 255], size=truth.shape).astype(np.uint8)
    evaluator = RegionEvaluator(truth, [2., .25])
    expected = evaluator.score(prediction, backend="numpy")
    actual = evaluator.score(prediction, backend="cpp")
    assert actual == expected


@pytest.mark.parametrize("spacing", [[1., 1.], [.25, .25], [2., .5]])
def test_compiled_white_queries_equal_full_per_class_ellipse_dilation(spacing):
    rng = np.random.default_rng(612)
    labels = rng.choice([0, 1, 7, 255], size=(47, 53)).astype(np.uint8)
    support = rng.uniform(size=labels.shape) > .1
    white = rng.uniform(size=labels.shape) > .7
    exterior = white & (rng.uniform(size=labels.shape) > .7)
    expected = mask_diagnostics(labels, support, white, exterior, spacing)
    actual = mask_diagnostics(labels, support, white, exterior, spacing, backend="cpp")
    assert actual == expected
