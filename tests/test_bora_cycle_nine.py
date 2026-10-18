"""Independent equations and geometry checks for cycle-nine model changes."""
import json
import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import check_grad, minimize
from scipy.sparse.linalg import spsolve

from experiments.bora_cycles.cycle_nine import (
    binary_cut_glass, cantelli_scores, interval_objective,
    moment_constrained_system, necessity_scores, projected_inverse, ray_glass,
    repulsion_objective, robust_palette_objective, sampling_variance,
    series_laplacian, stain_access, thin_plate)
from experiments.bora_cycles.methods import predict
from experiments.bora_cycles.registry import CHILDREN_NINE, CYCLE_NINE_SURVIVORS, candidates
from experiments.bora_cycles.review import validate_children


@pytest.mark.parametrize('key', [v['id'] for v in CHILDREN_NINE])
def test_full_tissue_prediction_preserves_inputs_and_known_class_identity(key):
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
    assert report['parent'] == candidates(9)[key].parent
    assert not report['fine_reference_used'] and not report['hard_anchors'] and not report['fixed_margin']
    json.dumps(report, allow_nan=False)
    for a, b in zip(before, (rgb, coarse, support)):
        np.testing.assert_array_equal(a, b)


def test_prospective_lineage_and_unique_model_interventions():
    assert CYCLE_NINE_SURVIVORS == ('J01', 'L02', 'K05', 'L03', 'C17')
    assert len(candidates(9)) == 20
    assert validate_children(dict(survivors=list(CYCLE_NINE_SURVIVORS)), CHILDREN_NINE)
    old = {v.mechanism for cycle in range(1, 9) for v in candidates(cycle).values()}
    assert not old.intersection(v['intervention'] for v in CHILDREN_NINE)


def test_schur_field_equals_independent_augmented_kkt_solve():
    matrix = sparse.csr_matrix([[3., -1., 0.], [-1., 4., -.5], [0., -.5, 2.]])
    mass = np.array([.2, .7, .5]); rhs = np.arange(9).reshape(3, 3)/5
    mean = np.array([.3, .4, .6])
    actual, report = moment_constrained_system(matrix, rhs, mass, mean)
    kkt = sparse.bmat([[matrix, mass[:, None]], [mass[None], None]], format='csc')
    expected = spsolve(kkt, np.vstack([rhs, mass.sum()*mean]))[:3]
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert report['equality_max_residual'] < 1e-12 and report['kkt_max_residual'] < 1e-12


def test_positive_sampling_variance_has_exact_independent_node_limit():
    diagonal = np.arange(1., 12.)
    mass = np.linspace(0, 1, 11); rhs = np.ones((11, 3))
    fit, variance, report = sampling_variance(sparse.diags(diagonal), mass, rhs)
    np.testing.assert_allclose(variance, .015*mass/diagonal**2, atol=1e-15)
    np.testing.assert_allclose(fit, rhs/diagonal[:, None], atol=1e-15)
    assert report['variance_minimum'] == 0 and report['linear_system_max_residual'] < 1e-14


def test_cantelli_correctly_ranks_exact_and_uncertain_distances():
    distance = np.array([[1., 4., 8.], [4., 1., 8.]])
    scores = cantelli_scores(distance, np.zeros_like(distance))
    np.testing.assert_array_equal(scores.argmax(-1), [0, 1])
    variance = np.array([[100., 0., 0.]])
    scores = cantelli_scores(np.array([[1., 4., 8.]]), variance)
    assert scores.argmax(-1)[0] == 1


def test_free_thin_plate_annihilates_all_affine_planes_not_curvature():
    yy, xx = np.indices((5, 7)); lap = thin_plate((5, 7), [1.3, .7], length=1.)
    for plane in (np.ones((5, 7)), xx, yy, 3*xx-2*yy+1):
        np.testing.assert_allclose(lap@plane.ravel(), 0, atol=1e-12)
    assert (xx.ravel()**2)@(lap@(xx.ravel()**2)) > 0
    assert np.linalg.eigvalsh(lap.toarray()).min() > -1e-12


def test_series_graph_is_symmetric_positive_and_detects_subgrid_barrier():
    he = np.full((16, 24, 3), 245, np.uint8)
    smooth = series_laplacian(he, (2, 3), np.array([8., 8.]), [1., 1.]).toarray()
    he[:, 7:9] = [100, 30, 90]
    barrier = series_laplacian(he, (2, 3), np.array([8., 8.]), [1., 1.]).toarray()
    assert abs(barrier[0, 1]) < abs(smooth[0, 1])
    np.testing.assert_allclose(barrier, barrier.T, atol=1e-14)
    np.testing.assert_allclose(barrier@np.ones(6), 0, atol=1e-14)
    assert np.linalg.eigvalsh(barrier).min() > -1e-12


def test_repulsive_energy_analytic_gradient_and_class_permutation():
    rng = np.random.default_rng(913)
    shape = (6, 3, 3)
    lap = sparse.diags(np.arange(1., 7.))
    mass = rng.uniform(.1, 1, (6, 3)); sums = rng.normal(size=shape); means = rng.normal(size=(3, 3))
    flat = rng.normal(size=np.prod(shape))
    f = lambda a: repulsion_objective(a, shape, lap, mass, sums, means)
    assert check_grad(lambda a:f(a)[0], lambda a:f(a)[1], flat) < 2e-5
    perm = [2, 0, 1]
    other = repulsion_objective(flat.reshape(shape)[:, perm].ravel(), shape, lap, mass[:, perm], sums[:, perm], means[perm])
    assert abs(other[0]-f(flat)[0]) < 1e-10


def test_ray_access_preserves_enclosed_cavity_and_curved_cleft():
    white = np.zeros((31, 31), bool)
    white[:, :3] = True
    white[14, :15] = True
    white[14:24, 14] = True
    white[20:25, 22:27] = True
    visible = ray_glass(white)
    assert visible[14, 12] and not visible[23, 14] and not visible[22, 24]
    assert np.all(~visible | white)


def test_stain_access_accepts_free_glass_and_preserves_stained_pixels():
    he = np.full((20, 24, 3), [130, 40, 90], np.uint8)
    he[:, :6] = 255
    he[8:12, 6:16] = 255
    he[14:17, 16:20] = 255
    excluded, report = stain_access(he, [1., 1.])
    assert excluded[10, 15] and not excluded[15, 18] and not excluded[19, 20]
    assert report['exterior_seed_pixels'] > 0


def test_cut_matches_bruteforce_binary_energy_and_terminal_constraints():
    # Certificate alone is not enough: independently enumerate every free node.
    he = np.array([[[255]*3, [235]*3, [130, 40, 90]],
                   [[255]*3, [250]*3, [130, 40, 90]],
                   [[255]*3, [235]*3, [130, 40, 90]]], np.uint8)
    mask, report = binary_cut_glass(he, [1., 1.])
    assert mask[:, 0].all() and not mask[:, 2].any()
    assert report['cut_energy_integer'] == report['maxflow_integer']
    # Only the central white node is not on the border; reconstruct both costs.
    x = he.astype(float)/255
    sat = np.ptp(x, -1)/x.max(-1)
    logit = (x.min(-1)-.96)/.015-sat/.02
    bg = np.maximum(1, np.rint(1000*np.logaddexp(0, -logit)))
    fg = np.maximum(1, np.rint(1000*np.logaddexp(0, logit)))
    costs = []
    for middle in (False, True):
        trial = mask.copy(); trial[1, 1] = middle
        # Border-white and stained terminal unaries are hard constraints, zero.
        unary = bg[1, 1] if middle else fg[1, 1]
        boundary = 2000*(np.count_nonzero(trial[1:] != trial[:-1])+np.count_nonzero(trial[:, 1:] != trial[:, :-1]))
        costs.append(unary+boundary)
    assert report['cut_energy_integer'] == min(costs)


@pytest.mark.parametrize('kind', ['interval', 'robust'])
def test_palette_gradient_and_optimum_against_independent_solver(kind):
    rng = np.random.default_rng(914)
    colors, x, prior = rng.random((4, 3)), rng.random((3, 3)), rng.dirichlet(np.ones(4), 3)
    f = lambda q: (interval_objective(q, x, prior, colors) if kind == 'interval'
                    else robust_palette_objective(q, x, prior, colors))
    assert check_grad(lambda a:f(a.reshape(3, 4))[0], lambda a:f(a.reshape(3, 4))[1].ravel(), prior.ravel()) < 2e-6
    actual, report = projected_inverse(f, prior, iterations=2000)
    ref = minimize(lambda a:f(a.reshape(3, 4))[0], prior.ravel(), jac=lambda a:f(a.reshape(3, 4))[1].ravel(),
        method='SLSQP', bounds=[(0, 1)]*12,
        constraints=[dict(type='eq', fun=lambda a:a.reshape(3, 4).sum(1)-1)],
        options=dict(ftol=1e-12, maxiter=500))
    assert ref.success and abs(f(actual)[0]-ref.fun) < 1e-7
    assert max(np.diff(report['objective_path'])) < 1e-10


def test_counterfactual_necessity_does_not_identify_redundant_vertex():
    colors = np.array([[0., 0., 0.], [1., 0., 0.], [.5, 0., 0.]])
    x = np.array([[.5, 0., 0.], [1., 0., 0.]])
    prior = np.full((2, 3), 1/3)
    scores, report = necessity_scores(x, prior, colors)
    assert scores[0, 2] < 1e-7
    assert scores[1].argmax() == 1 and report['max_kkt_residual'] < 1e-6
