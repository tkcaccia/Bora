"""Independent mathematics, native kernels and semantic invariants for cycle6."""
import json

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import check_grad, minimize
from scipy.stats import norm

from experiments.bora_cycles import native
from experiments.bora_cycles.cycle_five import field_laplacian
from experiments.bora_cycles.cycle_six import (
    additive_basis, calibration_objective, contaminated_objective, cubature_risk,
    fit_rigid_palette, glass_committor, guide_laplacian, hinge_objective,
    joint_normal_event, joint_normal_objective, neural_objective,
    positive_scatter, posterior_dominance, project_dominant,
    shared_field_system, simplex_quadrature)
from experiments.bora_cycles.methods import predict
from experiments.bora_cycles.registry import (
    CHILDREN_SIX, CYCLE_SIX, CYCLE_SIX_SURVIVORS, candidates)
from experiments.bora_cycles.review import validate_children


def example():
    yy, xx = np.indices((96, 270))
    palette = np.array([[200, 40, 30], [25, 45, 210], [30, 185, 65]], np.uint8)
    pattern = np.clip(((xx+12*np.sin(yy/11))/90).astype(int), 0, 2)
    rgb = palette[pattern]
    coarse = (xx//90+1).astype(np.uint8); coarse[:12, :12] = 0
    support = np.ones_like(coarse, bool)
    return rgb, coarse, support, palette


@pytest.mark.parametrize('key', [c['id'] for c in CHILDREN_SIX])
def test_children_execute_with_all_bag_classes_and_preserve_fixed_semantics(key):
    rgb, coarse, support, palette = example()
    original = [a.copy() for a in (rgb, coarse, support)]
    labels, report = predict(key, rgb, coarse, support, [1., 1.], palette, he=rgb)
    assert labels.shape == coarse.shape
    assert set(np.unique(labels)) <= {1, 2, 3}
    assert report['parent'] == CYCLE_SIX[key].parent
    assert not report.get('missing_known_bag_class', False)
    assert not report['fine_reference_used'] and not report['hard_anchors'] and not report['fixed_margin']
    json.dumps(report, allow_nan=False)
    for before, after in zip(original, (rgb, coarse, support)):
        np.testing.assert_array_equal(before, after)


def test_five_signed_parents_and_fifteen_new_mechanisms():
    assert len(CYCLE_SIX) == 20
    assert CYCLE_SIX_SURVIVORS == ('H02', 'C17', 'H10', 'H04', 'C01')
    assert validate_children(dict(survivors=list(CYCLE_SIX_SURVIVORS)), CHILDREN_SIX)
    old = {v.mechanism for cycle in range(1, 6) for v in candidates(cycle).values()}
    new = {c['intervention'] for c in CHILDREN_SIX}
    assert len(new) == 15 and not (old & new)


def test_guide_graph_is_physical_symmetric_and_has_constant_nullspace():
    spacing = np.array([.7, 1.3])
    guide = np.zeros((4, 6, 3))
    lap = guide_laplacian(guide, spacing)
    np.testing.assert_allclose(lap.toarray(), 1.02*field_laplacian((4, 6), spacing).toarray())
    np.testing.assert_allclose(lap.toarray(), lap.T.toarray())
    np.testing.assert_allclose(lap@np.ones(24), 0, atol=1e-12)
    guide[:, 3:] = 1
    edge = guide_laplacian(guide, spacing)
    assert abs(edge[2, 3]) < .021*abs(lap[2, 3])
    assert np.linalg.eigvalsh(edge.toarray()+.015*np.eye(24)).min() > .0149


def test_shared_field_identifies_constant_class_data_without_false_drift():
    rng = np.random.default_rng(73)
    mass = rng.uniform(.1, 1, (3, 20))
    means = np.array([[.8, .1, .2], [.1, .2, .8], [.2, .8, .3]])
    sums = mass[..., None]*means[:, None, :]
    field, fitted, report = shared_field_system(mass, sums, field_laplacian((4, 5), [1., 1.]), means)
    np.testing.assert_allclose(field, 0, atol=1e-11)
    np.testing.assert_allclose(fitted, means, atol=1e-11)
    assert report['linear_system_max_residual'] < 1e-11


def test_glass_access_distinguishes_an_open_corridor_from_a_sealed_lumen():
    he = np.full((25, 36, 3), [180, 100, 140], np.uint8)
    he[10:15, :18] = 255
    he[8:17, 25:30] = 255
    probability, report = glass_committor(he, np.ones((25, 36), bool), [1., 1.])
    assert probability[12, 10] > .99
    assert probability[12, 27] < .01
    assert probability.min() >= -1e-9 and probability.max() <= 1+1e-9
    assert report['linear_system_max_residual'] < 1e-10
    he[:, :18] = [180, 100, 140]
    probability, report = glass_committor(he, np.ones((25, 36), bool), [1., 1.])
    assert not probability.any() and report['no_two_sided_seed_evidence']


def test_palette_dominance_quadrature_respects_class_symmetry_including_tied_nodes():
    sample, weight = simplex_quadrature(3)
    colors = np.full((3, 3), .4)
    q = posterior_dominance(np.array([[.2, .5, .6]]), np.full((1, 3), 1/3), colors, sample, weight)
    np.testing.assert_allclose(q, 1/3, atol=1e-12)


def test_binary_palette_integral_against_independent_adaptive_quadrature():
    colors = np.array([[.8, .1, .2], [.1, .3, .8]])
    x = np.array([[.65, .2, .3]]); prior = np.array([[.6, .4]])
    variance = .05
    sample, weight = simplex_quadrature(2)
    actual = posterior_dominance(x, prior, colors, sample, weight, variance=variance)[0, 0]
    def density(value):
        q = np.array([value, 1-value])
        return np.exp(-.5*(np.sum((q@colors-x[0])**2)+.012*np.sum((q-prior[0])**2))/variance)
    expected = quad(density, .5, 1, epsabs=1e-12)[0]/quad(density, 0, 1, epsabs=1e-12)[0]
    assert abs(actual-expected) < .01


@pytest.mark.parametrize('kind', ['calibration', 'hinge', 'cubature', 'joint', 'contamination', 'neural'])
def test_analytic_gradients_against_independent_finite_differences(kind):
    rng = np.random.default_rng(81)
    bags = rng.normal(size=(5, 7, 4)); bags[..., -1] = 1
    labels = np.arange(5)%3; weight = np.full(5, .2)
    initial = rng.normal(size=(4, 3))*.1
    if kind == 'calibration':
        theta = initial.ravel()+rng.normal(size=12)*.05
        objective = lambda a: calibration_objective(a, bags.mean(1), labels, weight, initial)
    elif kind == 'hinge':
        theta = rng.normal(size=12)*.2
        objective = lambda a: hinge_objective(a, bags.mean(1), labels, weight, 3)
    elif kind == 'cubature':
        theta = rng.normal(size=12)*.2
        objective = lambda a: cubature_risk(a, bags, labels, weight, np.full(7, 1/7), 3)
    elif kind == 'joint':
        theta = initial.ravel()
        objective = lambda a: joint_normal_objective(a, bags, labels, weight, initial)
    elif kind == 'contamination':
        theta = np.r_[initial.ravel(), -2.]
        objective = lambda a: contaminated_objective(a, bags, labels, weight, initial)
    else:
        hidden = rng.normal(size=(4, 5))*.2
        theta = np.r_[initial.ravel(), hidden.ravel(), rng.normal(size=15)*.1]
        objective = lambda a: neural_objective(a, bags, labels, weight, initial, hidden)
    error = check_grad(lambda a: objective(a)[0], lambda a: objective(a)[1], theta)
    assert error < 3e-5, (kind, error)


def test_integrated_hinge_matches_scalar_gaussian_quadrature_and_is_convex():
    x = np.array([[.2, -.1, .4, 1.]])
    coefficient = np.array([[.1, .2, -.1, 0], [-.2, .3, .5, -.4]])
    delta = coefficient[1]-coefficient[0]
    mean = 1+x[0]@delta
    sd = np.sqrt(np.sum(delta[:3]**2)/25+1e-12)
    expected = quad(lambda z: max(0, mean+sd*z)**2*norm.pdf(z), -12, 12,
                    epsabs=1e-11)[0]+.0025*np.sum(coefficient**2)
    objective = lambda a: hinge_objective(a.ravel(), x, np.array([0]), np.array([1.]), 2)[0]
    assert abs(objective(coefficient)-expected) < 1e-9
    other = coefficient[::-1]+.4
    assert objective((coefficient+other)/2) <= (objective(coefficient)+objective(other))/2+1e-12


def test_two_class_joint_count_integral_matches_closed_normal_difference():
    p = np.array([[[.65, .35], [.4, .6], [.55, .45]]])
    mean = p.sum(1)[0]; variance = (p*(1-p)).sum(1)[0]
    actual = np.exp(joint_normal_event(p, np.array([0]), nodes=61)[0][0])
    expected = norm.cdf((mean[0]-mean[1])/np.sqrt(variance.sum()))
    assert abs(actual-expected) < 1e-8


def test_cpp_gaussian_scores_against_numpy_and_exact_thread_parity():
    rng = np.random.default_rng(13)
    x = rng.normal(size=(29, 17, 3)).astype(np.float32)
    means = rng.normal(size=(7, 3)); matrix = rng.normal(size=(7, 3, 3))
    precision = matrix@matrix.transpose(0, 2, 1)+np.eye(3)
    intercept = rng.normal(size=7)
    delta = x.astype(float)[..., None, :]-means
    expected = -.5*np.einsum('...gi,gij,...gj->...g', delta, precision, delta)+intercept
    one = native.gaussian_logits(x, means, precision, intercept, threads=1)
    two = native.gaussian_logits(x, means, precision, intercept, threads=2)
    np.testing.assert_allclose(one, expected, atol=1e-5, rtol=1e-7)
    np.testing.assert_array_equal(one, two)


def test_star_projection_against_independent_constrained_optimizer():
    rng = np.random.default_rng(19)
    values = rng.normal(size=(12, 5)); labels = np.arange(12)%5
    actual = project_dominant(values, labels)
    for value, own, result in zip(values, labels, actual):
        contrast = np.eye(5)[own]-np.eye(5)[np.arange(5) != own]
        fit = minimize(lambda q: (.5*np.sum((q-value)**2), q-value), np.full(5, .2),
            jac=True, method='SLSQP', bounds=[(0, 1)]*5,
            constraints=[dict(type='eq', fun=lambda q: q.sum()-1, jac=lambda q: np.ones(5)),
                         dict(type='ineq', fun=lambda q: contrast@q, jac=lambda q: contrast)],
            options=dict(ftol=1e-12, maxiter=200))
        assert fit.success
        np.testing.assert_allclose(result, fit.x, atol=1e-7)


def test_cpp_dominance_quadratic_matches_independent_qp_and_thread_parity():
    rng = np.random.default_rng(39)
    colors = rng.uniform(.1, .9, (4, 3)); x = rng.uniform(0, 1, (8, 3)).astype(np.float32)
    prior = rng.dirichlet(np.ones(4), 8).astype(np.float32); labels = np.arange(8)%4
    one, report = native.dominance_qp(x, prior, colors, labels, tolerance=1e-9, iterations=3000, threads=1)
    two, _ = native.dominance_qp(x, prior, colors, labels, tolerance=1e-9, iterations=3000, threads=2)
    np.testing.assert_array_equal(one, two)
    assert report['max_projected_gradient_residual'] < 1e-6
    for at, result in enumerate(one):
        def objective(q):
            delta = q@colors-x[at]
            return .5*delta@delta+.006*np.sum((q-prior[at])**2), delta@colors.T+.012*(q-prior[at])
        contrast = np.eye(4)[labels[at]]-np.eye(4)[np.arange(4) != labels[at]]
        fit = minimize(objective, np.full(4, .25), jac=True, method='SLSQP', bounds=[(0, 1)]*4,
            constraints=[dict(type='eq', fun=lambda q: q.sum()-1, jac=lambda q: np.ones(4)),
                         dict(type='ineq', fun=lambda q: contrast@q, jac=lambda q: contrast)],
            options=dict(ftol=1e-13, maxiter=300))
        assert fit.success
        np.testing.assert_allclose(result, fit.x, atol=3e-5)
        assert result[labels[at]] >= result.max()-1e-7


def test_rigid_palette_preserves_geometry_and_decreases_anchored_objective():
    canonical = np.array([[.8, .1, .2], [.1, .3, .8], [.2, .8, .3]])
    angle = .1
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    x = np.repeat(canonical@rotation+np.array([.02, -.01, .01]), 12, axis=0)
    prior = np.repeat(np.eye(3), 12, axis=0)
    fitted, report = fit_rigid_palette(x, prior, np.ones(len(x)), canonical)
    before = np.linalg.norm(canonical[:, None]-canonical[None], axis=-1)
    after = np.linalg.norm(fitted[:, None]-fitted[None], axis=-1)
    np.testing.assert_allclose(before, after, atol=1e-12)
    assert abs(report['rotation_determinant']-1) < 1e-12
    assert report['orthogonality_residual'] < 1e-12
    assert max(np.diff(report['objective_path']), default=0) < 1e-8


def test_expected_palette_risk_trace_matches_independent_monte_carlo():
    rng = np.random.default_rng(91)
    colors = rng.uniform(0, 1, (3, 3)); matrix = rng.normal(size=(3, 3))*.015
    covariance = matrix@matrix.T
    q = np.array([.2, .3, .5]); x = np.array([.6, .2, .4])
    noise = rng.normal(size=(100000, 3, 3))
    draws = colors+np.einsum('ij,njc->nic', np.linalg.cholesky(covariance), noise)
    empirical = np.mean(np.sum((np.einsum('i,nic->nc', q, draws)-x)**2, -1))
    expected = np.sum((q@colors-x)**2)+3*q@covariance@q
    assert abs(empirical-expected) < 2e-4


def test_positive_scatter_and_additive_basis_have_the_stated_geometry():
    projected = positive_scatter(np.diag([-.2, .1, .8]))
    np.testing.assert_allclose(projected, np.diag([0, .1, .8]))
    x = np.array([[.2, .3, .4], [.5, .1, -.2]])
    basis = additive_basis(x, np.zeros((3, 4)))
    assert basis.shape == (2, 16)
    np.testing.assert_allclose(basis[:, :4], np.c_[np.ones(2), x])


def test_glass_child_changes_only_parent_labels_to_zero():
    rgb, coarse, support, palette = example()
    he = rgb.copy(); he[35:50, :70] = 255
    parent, _ = predict('H02', rgb, coarse, support, [1., 1.], palette, he=he)
    child, report = predict('J03', rgb, coarse, support, [1., 1.], palette, he=he)
    assert np.all((child == parent) | (child == 0))
    assert report['proposed_glass_pixels'] > 0
