"""Independent likelihood, optimization and physical-policy checks for cycle 4."""
from itertools import product

import numpy as np
import pytest
from scipy.optimize import check_grad, minimize, minimize_scalar
from scipy.special import softmax

from bora.evidence import Evidence
from experiments.bora_cycles import native
from experiments.bora_cycles.cycle_four import (
    clearance_glass, count_objective, density_basins, fit_lad, legendre_features,
    mass_projection, maxent_objective, modal_colours, ordinal_palette,
    poisson_inverse, tyler_scatter)
from experiments.bora_cycles.methods import predict
from experiments.bora_cycles.registry import (
    CHILDREN_FOUR, CYCLE_FOUR, CYCLE_FOUR_SURVIVORS)
from experiments.bora_cycles.review import validate_children


@pytest.mark.parametrize("key", [c["id"] for c in CHILDREN_FOUR])
def test_children_keep_input_bytes_and_original_semantic_ids(key):
    yy, xx = np.indices((40, 48))
    palette = np.array([[200, 40, 30], [25, 45, 210]], np.uint8)
    rgb = palette[(xx > 18+yy//5).astype(int)]
    coarse = np.where(xx >= 24, 2, 1).astype(np.uint8)
    coarse[:8, :8] = 0
    support = np.ones_like(coarse, bool)
    before = [a.copy() for a in (rgb, coarse, support)]
    labels, detail = predict(key, rgb, coarse, support, [1., 1.], palette, he=rgb)
    assert set(np.unique(labels)) <= {1, 2}
    assert labels.shape == coarse.shape
    assert detail["parent"] == CYCLE_FOUR[key].parent
    assert detail["fine_reference_used"] is False
    assert detail["hard_anchors"] is False
    assert detail["fixed_margin"] is False
    for a, b in zip(before, (rgb, coarse, support)):
        np.testing.assert_array_equal(a, b)


def test_cycle_four_has_exact_signed_parent_lineage_and_no_old_rejections():
    assert len(CYCLE_FOUR) == 20
    assert validate_children(dict(survivors=list(CYCLE_FOUR_SURVIVORS)), CHILDREN_FOUR)
    assert set(CYCLE_FOUR)-{c["id"] for c in CHILDREN_FOUR} == {"F01", "C17", "C01", "D08", "C15"}
    assert len({c.mechanism for c in CYCLE_FOUR.values()}) == 20


def test_categorical_count_kernel_matches_exhaustive_categorical_outcomes():
    rng = np.random.default_rng(8)
    probability = rng.dirichlet(np.ones(3), size=(3, 4))
    labels = np.array([0, 1, 2])
    actual, derivative = native.categorical_winner(probability, labels, threads=1)
    expected = []
    for row, own in zip(probability, labels):
        tails = np.zeros(3)
        for outcome in product(range(3), repeat=4):
            mass = np.prod(row[np.arange(4), outcome])
            counts = np.bincount(outcome, minlength=3)
            for rival in range(3):
                if rival != own and counts[own]-counts[rival] >= int(rival < own):
                    tails[rival] += mass
        expected.append(sum(np.log(tails[g]) for g in range(3) if g != own))
    np.testing.assert_allclose(actual, expected, atol=2e-14)
    other, other_derivative = native.categorical_winner(probability, labels, threads=2)
    np.testing.assert_array_equal(actual, other)
    np.testing.assert_array_equal(derivative, other_derivative)


@pytest.mark.parametrize("exact", [False, True])
def test_tile_count_objective_analytic_gradient(exact):
    rng = np.random.default_rng(68)
    bags = rng.normal(size=(6, 5, 4))
    labels = np.array([0, 1, 2, 0, 1, 2])
    initial = rng.normal(size=(4, 3))*.1
    weight = np.full(6, 1/6)
    theta = rng.normal(size=12)*.2
    f = lambda x: count_objective(x, bags, labels, weight, initial, design_effect=4.2, exact=exact)
    assert check_grad(lambda x: f(x)[0], lambda x: f(x)[1], theta) < 2e-6


def test_lad_fit_recovers_affine_surface_with_gross_response_outliers():
    rng = np.random.default_rng(6)
    design = np.c_[np.ones(60), rng.uniform(-1, 1, (60, 2))]
    beta = np.array([.5, .1, -.2])
    value = design@beta
    value[:6] += 100
    actual, report = fit_lad(design, value, np.ones(60), slope_penalty=0.)
    np.testing.assert_allclose(actual, beta, atol=1e-9)
    assert report["success"]
    least_squares = np.linalg.lstsq(design, value, rcond=None)[0]
    assert np.linalg.norm(least_squares-beta) > 1


def test_clearance_reconstruction_removes_broad_void_not_thin_slit_and_is_physical():
    he = np.full((90, 120, 3), [180, 100, 140], np.uint8)
    he[10:13, 10:100] = 255  # long and thin: opposite of inertia-based F03
    he[35:55, 20:80] = 255
    yy, xx = np.indices(he.shape[:2])
    lumen = (yy-72)**2+(xx-98)**2 <= 6**2
    he[lumen] = 255
    excluded, _ = clearance_glass(he, [1., 1.])
    assert not excluded[10:13, 10:100].any()
    assert excluded[35:55, 20:80].all()
    assert excluded[lumen].all()  # explicitly verify the intended risk
    rotated, _ = clearance_glass(np.rot90(he), [1., 1.])
    np.testing.assert_array_equal(rotated, np.rot90(excluded))
    small_physical, _ = clearance_glass(he, [.1, .1])
    assert not small_physical.any()


def test_poisson_inverse_matches_independent_one_dimensional_convex_fit():
    palette = np.array([[210, 30, 40], [35, 85, 205]], np.uint8)
    rgb = np.full((8, 8, 3), [100, 65, 130], np.uint8)
    coarse = np.tile([1, 1, 1, 1, 2, 2, 2, 2], (8, 1)).astype(np.uint8)
    c = Evidence(rgb, coarse, np.ones_like(coarse, bool), [1., 1.])
    c.weight[:] = 0.  # remove additive coarse correction, give uniform density prior
    q, detail = poisson_inverse(c, palette, None)
    colors, observed = palette/255, rgb[0, 0]/255
    def objective(fraction):
        p = np.array([fraction, 1-fraction])
        intensity = p@colors
        return np.sum(intensity-observed*np.log(intensity))+.006*np.sum((p-.5)**2)
    fit = minimize_scalar(objective, bounds=(0, 1), method="bounded", options=dict(xatol=1e-13))
    assert abs(float(q[0, 0, 0])-fit.x) < 5e-4
    assert detail["projected_gradient_residual"] < .002


def test_ordinal_observation_ignores_amplitude_when_prior_is_uniform():
    palette = np.array([[210, 40, 90], [40, 80, 210]], np.uint8)
    coarse = np.tile([1, 1, 1, 1, 2, 2, 2, 2], (8, 1)).astype(np.uint8)
    outputs = []
    for value in ([200, 30, 70], [110, 90, 100]):
        c = Evidence(np.full((8, 8, 3), value, np.uint8), coarse, np.ones_like(coarse, bool), [1., 1.])
        c.weight[:] = 0.
        outputs.append(ordinal_palette(c, palette, None)[0])
    np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-7)


def test_ordinal_newton_matches_independent_convex_minimum():
    palette = np.array([[210, 40, 90], [40, 80, 210], [30, 215, 70]], np.uint8)
    rgb = np.full((8, 9, 3), [120, 80, 100], np.uint8)
    coarse = np.tile(np.repeat([1, 2, 3], 3), (8, 1)).astype(np.uint8)
    c = Evidence(rgb, coarse, np.ones_like(coarse, bool), [1., 1.])
    c.weight[:] = 0.
    q, detail = ordinal_palette(c, palette, None)
    colors = palette/255
    a = np.stack([colors[:, i]-colors[:, j] for i, j in [(0, 1), (0, 2), (1, 2)]], -1)/.05
    sign = np.array([1., 1., -1.])
    objective = lambda v: np.logaddexp(0, -sign*(v@a)).sum()+.006*np.sum((v-1/3)**2)
    fit = minimize(objective, np.full(3, 1/3), method="SLSQP", bounds=[(0, 1)]*3,
                   constraints=dict(type="eq", fun=lambda v: v.sum()-1),
                   options=dict(ftol=1e-13, maxiter=500))
    assert fit.success
    np.testing.assert_allclose(q[0, 0], fit.x, atol=2e-5)
    assert detail["converged"]
    assert np.max(np.diff(detail["objective_history"])) < 1e-7


def test_mass_projection_matches_independent_constrained_kl_minimum():
    q0 = np.array([[.8, .1, .1], [.1, .6, .3], [.3, .3, .4], [.4, .4, .2]])
    target = np.array([.7, 1.5, 1.8])
    q, report = mass_projection(q0, target)
    np.testing.assert_allclose(q.sum(0), target, atol=4e-7)
    np.testing.assert_allclose(q.sum(1), 1, atol=1e-12)
    def value(x):
        return float(np.sum(x.reshape(4, 3)*np.log(x.reshape(4, 3)/q0)))
    constraint = lambda x: np.r_[x.reshape(4, 3).sum(1)-1, x.reshape(4, 3).sum(0)[:2]-target[:2]]
    fit = minimize(value, np.full(12, 1/3), method="SLSQP", bounds=[(1e-10, 1)]*12,
                   constraints=dict(type="eq", fun=constraint), options=dict(ftol=1e-12, maxiter=400))
    assert fit.success
    assert abs(value(q.ravel())-fit.fun) < 1e-7
    assert report["converged"]


def test_angular_scatter_is_individually_radial_invariant_and_rotation_equivariant():
    rng = np.random.default_rng(90)
    x = rng.normal(size=(250, 3))@np.diag([3, 1, .4])
    shape, detail = tyler_scatter(x)
    radial, _ = tyler_scatter(x*np.exp(rng.normal(size=(250, 1))*3))
    np.testing.assert_allclose(shape, radial, atol=1e-10)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    transformed, _ = tyler_scatter(x@rotation)
    np.testing.assert_allclose(transformed, rotation.T@shape@rotation, atol=1e-7)
    assert np.linalg.eigvalsh(shape).min() > 0
    assert abs(np.trace(shape)-3) < 1e-12
    assert detail["fixed_point_change"] < 1e-6


def test_colour_density_flow_ends_at_local_maxima_without_cycles():
    rng = np.random.default_rng(4)
    hist = rng.random((8, 7, 6))
    roots = density_basins(hist)
    flat = hist.ravel()
    assert np.all(flat[roots.ravel()] >= flat)
    for root in np.unique(roots):
        coord = np.asarray(np.unravel_index(root, hist.shape))
        for delta in product((-1, 0, 1), repeat=3):
            at = np.clip(coord+delta, 0, np.asarray(hist.shape)-1)
            assert hist[tuple(at)] <= flat[root]


def test_unsupervised_colour_modes_reveal_unknown_regions_without_semantic_invention():
    rgb = np.zeros((40, 60, 3), np.uint8)
    rgb[:, :20], rgb[:, 20:40], rgb[:, 40:] = [220, 30, 30], [30, 210, 30], [30, 30, 220]
    coarse = np.tile(np.r_[np.ones(20), np.zeros(20), np.full(20, 2)], (40, 1)).astype(np.uint8)
    c = Evidence(rgb, coarse, np.ones_like(coarse, bool), [1., 1.])
    q, detail = modal_colours(c, None, None)
    assert detail["unsupported_pixels"] > 0
    assert detail["unsupported_spatial_components"] > 0
    assert detail["semantic_new_classes_assigned"] is False
    np.testing.assert_allclose(q[:, 25:35], .5, atol=1e-12)
    assert len(detail["unsupported_colour_modes"]) > 0


def test_maximum_entropy_gradient_and_discrete_uniform_normalizer():
    rng = np.random.default_rng(30)
    basis = legendre_features(rng.uniform(0, 1, (120, 3)))
    moment = basis.mean(0)
    theta = rng.normal(size=26)*.2
    f = lambda v: maxent_objective(v, basis, moment)
    assert check_grad(lambda v: f(v)[0], lambda v: f(v)[1], theta) < 2e-6
    value, gradient = f(np.zeros(26))
    assert abs(value) < 1e-14
    np.testing.assert_allclose(gradient, 0, atol=1e-14)
    fit = minimize(f, theta, jac=True, method="BFGS", options=dict(gtol=1e-9))
    assert np.max(np.abs(fit.x)) < 1e-6
