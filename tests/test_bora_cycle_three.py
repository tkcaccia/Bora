"""Independent geometry, likelihood-gradient and inference checks for cycle 3."""
import numpy as np
import pytest
from scipy import ndimage as ndi
from scipy.optimize import check_grad, minimize
from scipy.spatial import ConvexHull

from bora.evidence import Evidence
from experiments.bora_cycles import native
from experiments.bora_cycles.cycle_three import (
    blur_inverse, cleft_components, copula_log_density, energy_potential,
    margin_objective, mean_logit_objective, prototype_system, support_depth,
    tessellated_weights, witness_objective)
from experiments.bora_cycles.methods import copula, predict
from experiments.bora_cycles.registry import CHILDREN_THREE, CYCLE_THREE, CYCLE_THREE_SURVIVORS
from experiments.bora_cycles.review import validate_children


@pytest.mark.parametrize("key", [v["id"] for v in CHILDREN_THREE])
def test_all_children_preserve_inputs_and_class_ids_with_unknown_tissue(key):
    yy, xx = np.indices((48, 56))
    palette = np.array([[200, 40, 30], [25, 45, 210]], np.uint8)
    rgb = palette[(xx > 20+yy//6).astype(int)]
    coarse = np.where(xx > 28, 2, 1).astype(np.uint8)
    coarse[:8, :8] = 0
    support = np.ones_like(coarse, bool)
    before = [v.copy() for v in (rgb, coarse, support)]
    result, detail = predict(key, rgb, coarse, support, [1., 1.], palette, he=rgb)
    assert result.shape == (48, 56)
    assert set(np.unique(result)) <= {1, 2}
    assert detail["fine_reference_used"] is False
    assert detail["parent"] == CYCLE_THREE[key].parent
    for a, b in zip(before, (rgb, coarse, support)):
        np.testing.assert_array_equal(a, b)


def test_cycle_three_exact_lineage_and_no_rejected_candidate_survives():
    assert len(CYCLE_THREE) == 20
    assert validate_children(dict(survivors=list(CYCLE_THREE_SURVIVORS)), CHILDREN_THREE)
    assert set(CYCLE_THREE)-{c["id"] for c in CHILDREN_THREE} == {"D10", "C17", "C01", "D08", "C15"}


def test_local_tile_exclusion_matches_independent_weighted_normal_equations():
    rng = np.random.default_rng(95)
    coarse = np.ones((65, 70), np.uint8)
    rgb = rng.integers(40, 220, (65, 70, 3), dtype=np.uint8)
    c = Evidence(rgb, coarse, np.ones_like(coarse, bool), [1., 1.], tile_size=15)
    matrix, rhs = prototype_system(c, 0, exclude_tile=True)
    yy, xx = np.indices(c.shape)
    y, x = 32, 36
    valid = ((abs(xx-x) <= 48) & (abs(yy-y) <= 48)
             & ~((xx//15 == x//15) & (yy//15 == y//15)))
    design = np.stack([np.ones(valid.sum()), (xx[valid]-x)/48, (yy[valid]-y)/48], -1)
    weight = c.weight[valid]/97**2
    expected = (design.T*weight)@design+.015*np.eye(3)
    expected_rhs = (design.T*weight)@c.rgb[valid]
    np.testing.assert_allclose(matrix[y, x], expected, atol=1e-12)
    np.testing.assert_allclose(rhs[y, x], expected_rhs, atol=1e-12)
    changed = rgb.copy()
    changed[30:45, 30:45] = 255-changed[30:45, 30:45]
    c2 = Evidence(changed, coarse, np.ones_like(coarse, bool), [1., 1.], tile_size=15)
    _, rhs2 = prototype_system(c2, 0, exclude_tile=True)
    np.testing.assert_allclose(rhs[y, x], rhs2[y, x], atol=1e-12)


def test_physical_white_shape_distinguishes_slit_from_round_lumen_and_rotates():
    he = np.full((100, 120, 3), [175, 110, 150], np.uint8)
    he[10:13, 10:65] = 255
    yy, xx = np.indices(he.shape[:2])
    circle = (yy-65)**2+(xx-65)**2 <= 15**2
    he[circle] = 255
    glass, detail = cleft_components(he, np.array([1., 1.]))
    assert glass[10:13, 10:65].all()
    assert not glass[circle].any()
    rotated, _ = cleft_components(np.rot90(he), np.array([1., 1.]))
    np.testing.assert_array_equal(np.rot90(glass), rotated)
    assert detail["cleft_components"] == 1
    # The same pixel slit at 0.1 um/px is too short in physical units.
    tiny, _ = cleft_components(he, np.array([.1, .1]))
    assert not tiny.any()


def test_forward_blur_inverse_matches_independent_constrained_minimum():
    palette = np.array([[210, 25, 35], [30, 40, 200]], np.uint8)
    coarse = np.tile([1, 1, 1, 1, 2, 2, 2, 2], (8, 1)).astype(np.uint8)
    truth = np.eye(2)[coarse-1]
    observed = ndi.gaussian_filter(truth@(palette/255), (2, 2, 0), mode="reflect")
    rgb = np.rint(observed*255).astype(np.uint8)
    c = Evidence(rgb, coarse, np.ones_like(coarse, bool), [1., 1.])
    q, detail = blur_inverse(c, palette, None)
    prior = c.lattice().astype(float)
    def objective(p):
        v = np.stack([p.reshape(c.shape), 1-p.reshape(c.shape)], -1)
        error = ndi.gaussian_filter(v@(palette/255), (2, 2, 0), mode="reflect")-c.rgb
        return .5*np.sum(error**2)+.006*np.sum((v-prior)**2)
    fit = minimize(objective, prior[..., 0].ravel(), method="L-BFGS-B",
                   bounds=[(0, 1)]*64, options=dict(ftol=1e-13, maxiter=800))
    assert fit.success
    assert objective(q[..., 0].ravel())-fit.fun < 2e-5
    assert detail["projected_gradient_residual"] < .003


def test_palette_mesh_reproduces_vertices_and_affine_tetrahedron_mixture():
    vertices = np.array([[.1, .1, .1], [.8, .1, .1], [.1, .8, .1], [.1, .1, .8]])
    fraction = np.array([[.1, .2, .3, .4]])
    q, detail = tessellated_weights(np.r_[vertices, fraction@vertices], vertices)
    np.testing.assert_allclose(q[:4], np.eye(4), atol=2e-7)
    np.testing.assert_allclose(q[4], fraction[0], atol=2e-7)
    assert detail["intrinsic_rank"] == 3
    q1, _ = tessellated_weights(np.array([[.3, 0, 0], [.8, 0, 0]]), np.array([[0, 0, 0], [1, 0, 0]]))
    np.testing.assert_allclose(q1, [[.7, .3], [.2, .8]], atol=1e-7)


def test_mean_logit_exact_gradient_hessian_and_independent_convex_fit():
    rng = np.random.default_rng(86)
    x, y = rng.normal(size=(15, 4)), rng.integers(0, 3, 15)
    weight, theta = np.full(15, 1/15), rng.normal(size=(4, 3))*.1
    def objective(v):
        return mean_logit_objective(v.reshape(4, 3), x, y, weight)
    assert check_grad(lambda v: objective(v)[0], lambda v: objective(v)[1], theta.ravel()) < 2e-7
    _, gradient, hessian = mean_logit_objective(theta, x, y, weight, hessian=True)
    direction = rng.normal(size=12)
    eps = 1e-5
    numerical = (objective(theta.ravel()+eps*direction)[1]-objective(theta.ravel()-eps*direction)[1])/(2*eps)
    np.testing.assert_allclose(hessian@direction, numerical, atol=1e-8)
    assert np.linalg.eigvalsh(hessian).min() >= .00499999
    fit = minimize(objective, theta.ravel(), jac=True, method="BFGS", options=dict(gtol=1e-8))
    assert np.max(np.abs(objective(fit.x)[1])) < 1e-7


@pytest.mark.parametrize("kind", ["witness", "margin"])
def test_tile_objective_gradients(kind):
    rng = np.random.default_rng(60)
    k, y = 3, np.array([0, 1, 2, 1, 0, 2])
    weight, theta = np.full(6, 1/6), rng.normal(size=12)*.2
    if kind == "witness":
        x = rng.normal(size=(6, 25, 4))
        f = lambda v: witness_objective(v, x, y, weight, k)
        # Bag pixel order carries no information in the witness model.
        np.testing.assert_allclose(f(theta)[0], witness_objective(theta, x[:, ::-1], y, weight, k)[0])
    else:
        x = rng.normal(size=(6, 4))
        f = lambda v: margin_objective(v, x, y, weight, k)
    assert check_grad(lambda v: f(v)[0], lambda v: f(v)[1], theta) < 2e-6
    if kind == "margin":
        # Every rival score ties at initialization. The original maximum-rival
        # pilot objective failed line search at this exact point.
        assert check_grad(lambda v: f(v)[0], lambda v: f(v)[1], np.zeros_like(theta)) < 2e-6


def test_fft_distributional_energy_matches_brute_force_distances():
    rng = np.random.default_rng(44)
    hist = rng.random((4, 4, 4))
    hist /= hist.sum()
    grid = np.stack(np.meshgrid(*[np.arange(4)/4]*3, indexing="ij"), -1).reshape(-1, 3)
    distances = np.linalg.norm(grid[:, None]-grid[None], axis=-1)
    potential, self_distance = energy_potential(hist)
    expected = distances@hist.ravel()
    np.testing.assert_allclose(potential.ravel(), expected, atol=1e-12)
    assert abs(self_distance-.5*(hist.ravel()@expected)) < 1e-12


def test_cpp_support_depth_matches_all_facet_numpy_and_thread_count():
    rng = np.random.default_rng(60)
    points, x = rng.normal(size=(55, 3)), rng.normal(size=(90, 3)).astype(np.float32)
    planes = ConvexHull(points).equations
    expected = -(x@planes[:, :3].T+planes[:, 3]).max(-1)
    a, b = native.hull_depth(x, planes, threads=1), native.hull_depth(x, planes, threads=2)
    np.testing.assert_allclose(a, expected, atol=2e-7)
    np.testing.assert_array_equal(a, b)
    depth, _ = support_depth(np.array([[.5, 0, 0], [.5, 1, 0], [2, 0, 0]]), np.array([[0, 0, 0], [1, 0, 0]]))
    np.testing.assert_allclose(depth, [.5, -.5, -1], atol=1e-7)


def test_typicality_retains_the_parent_copula_density_before_calibration():
    rng = np.random.default_rng(9)
    rgb = rng.integers(20, 235, (35, 40, 3), dtype=np.uint8)
    coarse = np.tile(np.r_[np.ones(20), np.full(20, 2)], (35, 1)).astype(np.uint8)
    c = Evidence(rgb, coarse, np.ones_like(coarse, bool), [1., 1.])
    from scipy.special import softmax
    actual = softmax(copula_log_density(c, c.weight), -1)
    expected, _ = copula(c)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
