"""Independent mathematical checks for cycle-two model interventions."""
import numpy as np
import pytest
from scipy.optimize import minimize, minimize_scalar

from bora.evidence import Evidence
from experiments.bora_cycles import native
from experiments.bora_cycles.cycle_two import poisson_winner, glass_competition
from experiments.bora_cycles.methods import predict
from experiments.bora_cycles.registry import CHILDREN_TWO, CYCLE_TWO, CYCLE_TWO_SURVIVORS
from experiments.bora_cycles.review import validate_children


@pytest.mark.parametrize("k", [2, 5, 9])
def test_general_quadratic_matches_independent_constrained_optimizer(k):
    rng = np.random.default_rng(423+k)
    factor = rng.normal(size=(k, k))
    h = factor@factor.T+.04*np.eye(k)
    b = rng.normal(size=(3, k))
    prior = rng.dirichlet(np.ones(k), size=3)
    actual, detail = native.quadratic_simplex(h, b, prior)
    for i in range(3):
        fit = minimize(lambda q: .5*q@h@q-b[i]@q, prior[i],
                       jac=lambda q: h@q-b[i], method="SLSQP",
                       bounds=[(0, 1)]*k,
                       constraints=[dict(type="eq", fun=lambda q: q.sum()-1,
                                         jac=lambda q: np.ones(k))],
                       options=dict(ftol=1e-12, maxiter=500))
        assert fit.success
        np.testing.assert_allclose(actual[i], fit.x, atol=2e-6)
    assert detail["max_kkt_residual"] < 1e-7


def test_pair_constraint_is_global_best_edge_including_singletons():
    rng = np.random.default_rng(56)
    colors = rng.uniform(.05, .9, (6, 3))
    x = rng.uniform(0, 1, (4, 3))
    pi = rng.dirichlet(np.ones(6), size=4)
    actual, _ = native.pair_palette(x, pi, colors)
    for row, q in enumerate(actual):
        assert (q > 1e-7).sum() <= 2
        def energy(v):
            return .5*np.sum((v@colors-x[row])**2)+.006*np.sum((v-pi[row])**2)
        best = np.inf
        for a in range(6):
            for b in range(a, 6):
                def face(t):
                    v = np.zeros(6)
                    v[a] += t
                    v[b] += 1-t
                    return energy(v)
                fit = minimize_scalar(face, bounds=(0, 1), method="bounded")
                best = min(best, fit.fun, face(0), face(1))
        assert abs(energy(q)-best) < 1e-7


def test_relative_entropy_inverse_matches_independent_objective():
    rng = np.random.default_rng(67)
    colors = rng.uniform(.1, .9, (5, 3))
    x = rng.uniform(.2, .8, (3, 3))
    pi = rng.dirichlet(np.ones(5)*3, size=3)
    actual, detail = native.entropy_palette(x, pi, colors)
    for row in range(3):
        def energy(q):
            return .5*np.sum((q@colors-x[row])**2)+.012*np.sum(q*np.log(q/pi[row]))
        def gradient(q):
            return (q@colors-x[row])@colors.T+.012*(np.log(q/pi[row])+1)
        fit = minimize(energy, pi[row], jac=gradient, method="SLSQP",
                       bounds=[(1e-12, 1)]*5,
                       constraints=[dict(type="eq", fun=lambda q: q.sum()-1,
                                         jac=lambda q: np.ones(5))],
                       options=dict(ftol=1e-12, maxiter=500))
        assert fit.success
        assert abs(energy(actual[row])-fit.fun) < 1e-8
    assert detail["max_stationarity_residual"] < 1e-4


def test_poisson_winners_partition_the_count_space_and_gradient_is_correct():
    means = np.array([[2., 4., 1.2], [6., 2.5, 3.]])
    probabilities = []
    for label in range(3):
        labels = np.full(2, label)
        logp, gradient = poisson_winner(means, labels)
        probabilities.append(np.exp(logp))
        for j in range(3):
            step = np.zeros_like(means)
            step[:, j] = 1e-5
            plus, _ = poisson_winner(means+step, labels)
            minus, _ = poisson_winner(means-step, labels)
            np.testing.assert_allclose(gradient[:, j], -(plus-minus)/2e-5,
                                       rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.sum(probabilities, axis=0), 1, atol=1e-10)


def test_glass_competition_learns_white_tissue_instead_of_border_connectivity():
    yy, xx = np.indices((80, 80))
    coarse = np.where(xx < 40, 1, 2).astype(np.uint8)
    palette = np.array([[190, 40, 40], [210, 180, 60]], np.uint8)
    rgb = palette[coarse-1]
    he = np.full((80, 80, 3), [185, 120, 165], np.uint8)
    he[(xx >= 40) & (yy % 8 < 6)] = 255  # fatty class, mostly white
    he[25:27, 10:25] = 255               # small white cleft in nonwhite class
    c = Evidence(rgb, coarse, np.ones_like(coarse, bool), [1., 1.])
    _, detail = glass_competition(c, palette, he)
    excluded = detail["glass_exclusion"]
    assert excluded[25:27, 10:25].all()
    assert not excluded[:, 40:].any()
    assert detail["tissue_white_rates"][1] > detail["tissue_white_rates"][0]


@pytest.mark.parametrize("key", [c["id"] for c in CHILDREN_TWO])
def test_children_accept_unknown_tissue_without_reading_reference_or_mutating_rgb(key):
    yy, xx = np.indices((48, 56))
    palette = np.array([[200, 40, 30], [25, 45, 210]], np.uint8)
    rgb = palette[(xx > 20+yy//6).astype(int)]
    coarse = np.where(xx > 28, 2, 1).astype(np.uint8)
    coarse[:8, :8] = 0
    original = rgb.copy()
    result, detail = predict(key, rgb, coarse, np.ones_like(coarse, bool), [1., 1.], palette, he=rgb)
    assert set(np.unique(result)) <= {1, 2}
    assert detail["fine_reference_used"] is False
    assert detail["parent"] == CYCLE_TWO[key].parent
    np.testing.assert_array_equal(rgb, original)


def test_next_cycle_has_five_unchanged_survivors_and_three_distinct_children_each():
    assert len(CYCLE_TWO) == 20
    assert validate_children(dict(survivors=list(CYCLE_TWO_SURVIVORS)), CHILDREN_TWO)
