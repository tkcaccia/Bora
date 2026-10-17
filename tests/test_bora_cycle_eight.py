"""Independent model equations, derivative checks and boundary invariants."""
import json
from itertools import product

import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import check_grad, minimize

from experiments.bora_cycles.cycle_eight import (
    bounded_system, cvar_objective, hellinger_inverse, hellinger_objective,
    heteroscedastic_hinge, huber_radial, inverse_diagonal, kernel_scores,
    majorized_inverse, max_hinge_objective, mean_log_objective,
    nonlocal_laplacian, residual_design, robust_system, wide_path_glass)
from experiments.bora_cycles.cycle_five import field_laplacian
from experiments.bora_cycles.cycle_seven import gaussian_design
from experiments.bora_cycles.methods import predict
from experiments.bora_cycles.registry import CHILDREN_EIGHT, CYCLE_EIGHT, CYCLE_EIGHT_SURVIVORS, candidates
from experiments.bora_cycles.review import validate_children


@pytest.mark.parametrize('key', [v['id'] for v in CHILDREN_EIGHT])
def test_every_child_uses_all_tissue_without_mutating_fixed_input(key):
    yy, xx = np.indices((96, 270))
    palette = np.array([[200, 40, 30], [25, 45, 210], [30, 185, 65]], np.uint8)
    pattern = np.clip(((xx+12*np.sin(yy/11))/90).astype(int), 0, 2)
    rgb = palette[pattern]
    coarse = (xx//90+1).astype(np.uint8)
    coarse[:12, :12] = 0
    support = np.ones(coarse.shape, bool)
    before = [a.copy() for a in (rgb, coarse, support)]
    labels, report = predict(key, rgb, coarse, support, [1., 1.], palette, he=rgb)
    assert labels.shape == coarse.shape and set(np.unique(labels)) == {1, 2, 3}
    assert np.all(labels[:12, :12] > 0)
    assert report['parent'] == CYCLE_EIGHT[key].parent
    assert not report.get('missing_known_bag_class', False)
    assert not report['fine_reference_used'] and not report['hard_anchors'] and not report['fixed_margin']
    json.dumps(report, allow_nan=False)
    for a, b in zip(before, (rgb, coarse, support)):
        np.testing.assert_array_equal(a, b)


def test_lineage_is_signed_five_and_fifteen_previously_unused_mechanisms():
    assert CYCLE_EIGHT_SURVIVORS == ('J01', 'K05', 'C17', 'J07', 'K13')
    assert len(CYCLE_EIGHT) == 20
    assert validate_children(dict(survivors=list(CYCLE_EIGHT_SURVIVORS)), CHILDREN_EIGHT)
    old = {v.mechanism for cycle in range(1, 8) for v in candidates(cycle).values()}
    assert not old.intersection(v['intervention'] for v in CHILDREN_EIGHT)


def test_nonlocal_graph_is_positive_symmetric_and_constant_reproducing():
    rng = np.random.default_rng(150)
    lap = nonlocal_laplacian(rng.random((4, 5, 3))*.2, [1., 1.]).toarray()
    np.testing.assert_allclose(lap, lap.T, atol=1e-12)
    np.testing.assert_allclose(lap@np.ones(20), 0, atol=1e-12)
    assert np.linalg.eigvalsh(lap).min() > -1e-12
    assert np.count_nonzero(lap < 0) > 2*(4*4+3*5)  # More than local grid edges.


def test_uncertainty_diagonal_is_exact_for_independent_nodes():
    diagonal = np.arange(1, 12, dtype=float)
    values, report = inverse_diagonal(sparse.diags(diagonal))
    np.testing.assert_allclose(values, 1/diagonal, atol=1e-14)
    assert report['negative_diagonal_estimates_clipped'] == 0


def test_wide_route_does_not_delete_a_large_cavity_through_a_thin_neck():
    he = np.full((48, 48, 3), [130, 30, 100], np.uint8)
    he[:, :8] = 255
    he[10:30, 18:36] = 255
    he[19, 8:18] = 255
    excluded, _ = wide_path_glass(he, [1., 1.])
    assert excluded[20, 2] and not excluded[20, 25]
    he[16:24, 8:18] = 255
    wider, _ = wide_path_glass(he, [1., 1.])
    assert wider[20, 25]
    assert not wider[40, 25]


def test_bounded_field_matches_independent_constrained_quadratic():
    matrix = sparse.csr_matrix([[3., -1., .2], [-1., 4., -1.], [.2, -1., 2.]])
    rhs = np.array([[-1., 3., 2.], [2., .1, -1.], [5., 1., 2.]])
    actual, report = bounded_system(matrix, rhs)
    objective = lambda flat: .5*np.sum(flat.reshape(3, 3)*(matrix@flat.reshape(3, 3)))-np.sum(rhs*flat.reshape(3, 3))
    ref = minimize(objective, np.full(9, .5), method='SLSQP', bounds=[(0, 1)]*9,
        options=dict(ftol=1e-12, maxiter=200))
    assert ref.success and abs(objective(actual.ravel())-ref.fun) < 1e-8
    assert report['projected_gradient_residual'] < 1e-5
    assert actual.min() >= 0 and actual.max() <= 1


def test_robust_field_matches_independent_radial_objective():
    rng = np.random.default_rng(151)
    base = field_laplacian((2, 3), [1., 1.], length_um=1.)
    lap = base.T@base
    mass = rng.uniform(.5, 1, 6)
    target = rng.normal(size=(6, 3))
    target[0] += 20
    mean = np.zeros(3)
    actual, report = robust_system(lap, mass, target, mean, iterations=60)
    def objective(flat):
        u = flat.reshape(6, 3)
        return np.sum(mass*huber_radial(np.linalg.norm(u-target, axis=1)))+.5*np.sum(u*(lap@u))+.0075*np.sum(u*u)
    ref = minimize(objective, np.zeros(18), method='BFGS', options=dict(gtol=1e-7))
    assert abs(objective(actual.ravel())-ref.fun) < 1e-7
    assert max(np.diff(report['objective_path'])) < 1e-8


def test_hellinger_gradient_and_simplex_optimum_against_independent_solver():
    rng = np.random.default_rng(152)
    colors, x, prior = rng.random((4, 3)), rng.random((3, 3)), rng.dirichlet(np.ones(4), 3)
    f = lambda flat: hellinger_objective(flat.reshape(3, 4), x, prior, colors)
    assert check_grad(lambda a:f(a)[0], lambda a:f(a)[1].ravel(), prior.ravel()) < 2e-6
    actual, report = hellinger_inverse(x, prior, colors, iterations=3000)
    ref = minimize(lambda a:f(a)[0], prior.ravel(), jac=lambda a:f(a)[1].ravel(),
        method='SLSQP', bounds=[(0, 1)]*12,
        constraints=[dict(type='eq', fun=lambda a:a.reshape(3, 4).sum(1)-1)],
        options=dict(ftol=1e-12, maxiter=500))
    assert ref.success and abs(f(actual.ravel())[0]-ref.fun) < 1e-7
    assert max(np.diff(report['objective_path'])) < 1e-10


@pytest.mark.parametrize('mode', ['student', 'sparse'])
def test_majorized_palette_has_feasible_monotone_original_energy(mode):
    rng = np.random.default_rng(153)
    colors, x, prior = rng.random((5, 3)), rng.random((13, 3)), rng.dirichlet(np.ones(5), 13)
    q, report = majorized_inverse(x, prior, colors, mode)
    np.testing.assert_allclose(q.sum(1), 1, atol=1e-6)
    assert q.min() >= 0 and max(np.diff(report['objective_path'])) < 1e-6
    assert report['max_kkt_residual'] < 5e-8 and report['nonconvex_local_solution']


def test_entropy_penalty_is_concave_and_its_tangent_is_a_global_majorizer():
    rng = np.random.default_rng(155)
    old, new = rng.dirichlet(np.ones(5), size=(2, 11))
    entropy = lambda q: -.002*np.sum((q+1e-5)*np.log(q+1e-5), -1)
    gradient = -.002*(np.log(old+1e-5)+1)
    assert np.all(entropy(new) <= entropy(old)+np.sum(gradient*(new-old), -1)+1e-12)
    # Unlike the superseded quadratic, this can defeat the0.012 ridge near
    # a simplex boundary, so the original objective is genuinely nonconvex.
    assert .012-.002/(1e-6+1e-5) < 0


@pytest.mark.parametrize('kind', ['joint', 'heteroscedastic', 'cvar', 'mean-log'])
def test_risk_gradients_against_independent_finite_differences(kind):
    rng = np.random.default_rng(154)
    x = np.c_[rng.normal(size=(6, 3)), np.ones(6)]
    labels, weight = np.arange(6)%3, np.full(6, 1/6)
    theta = rng.normal(size=12)*.15
    design, quadrature = gaussian_design(x)
    if kind == 'joint':
        objective = lambda a:max_hinge_objective(a, design, labels, weight, quadrature, 3)
    elif kind == 'heteroscedastic':
        a = rng.normal(size=(6, 3, 3))
        covariance = a@a.swapaxes(1, 2)/100.
        objective = lambda a:heteroscedastic_hinge(a, x, labels, weight, covariance, 3)
    elif kind == 'cvar':
        theta = np.r_[theta, .7]
        objective = lambda a:cvar_objective(a, design, labels, weight, 3)
    else:
        objective = lambda a:mean_log_objective(a, design, labels, weight, 3)
    assert check_grad(lambda a:objective(a)[0], lambda a:objective(a)[1], theta) < 3e-6


def test_block_noise_preserves_within_row_correlation():
    bags = np.ones((1, 25, 4))
    rows = np.repeat(np.array([-2., -1., 0., 1., 2.]), 5)
    bags[0, :, :3] = rows[:, None]
    x = np.array([[0., 0., 0., 1.]])
    ordinary = residual_design(bags, np.eye(3), x, 'cvar')[0, :, 0]
    blocked = residual_design(bags, np.eye(3), x, 'block')[0, :, 0]
    np.testing.assert_allclose(blocked.mean(), 0, atol=1e-14)
    assert np.mean(blocked**2) > 2*np.mean(ordinary**2)


def test_kernel_noise_integral_matches_gaussian_hermite_reference():
    means = np.array([[.2, -.1, .6], [.8, .4, -.3]])
    query = np.array([[0., .2, .1], [1., -.2, .4]])
    labels = np.array([0, 1])
    actual = kernel_scores(query, means, labels, 2)
    point, weight = np.polynomial.hermite.hermgauss(6)
    offsets = np.array(list(product(point, repeat=3)))*np.sqrt(.08)
    qw = np.array([a*b*c for a,b,c in product(weight, repeat=3)])/np.pi**1.5
    expected = np.empty((2, 2))
    for g in range(2):
        norm = np.sum(qw*np.exp(-np.sum((np.sqrt(2)*offsets)**2, -1)/6))
        for i in range(2):
            cross = np.sum(qw*np.exp(-np.sum((query[i]-means[g]-offsets)**2, -1)/6))
            expected[i, g] = 2*cross-norm
    np.testing.assert_allclose(actual, expected, atol=1e-9)
