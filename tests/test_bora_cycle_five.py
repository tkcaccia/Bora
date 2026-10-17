"""Independent objective, physical-policy and compiled-parity checks for cycle5."""
import json
from itertools import product

import numpy as np
import pytest
from scipy.optimize import check_grad, minimize
from scipy.special import logsumexp

from experiments.bora_cycles import native
from experiments.bora_cycles.cycle_five import (
    bernstein_basis, cardinal_coordinates, channel_objective, chow_liu_density,
    field_laplacian, fit_bernstein, fit_prevalence, geometric_median_objective,
    information_objective, interface_cavities, learn_palette, noise_objective,
    profiled_plane_cost, trimmed_objective)
from experiments.bora_cycles.methods import predict
from experiments.bora_cycles.registry import CHILDREN_FIVE, CYCLE_FIVE, CYCLE_FIVE_SURVIVORS
from experiments.bora_cycles.review import validate_children


@pytest.mark.parametrize('key', [c['id'] for c in CHILDREN_FIVE])
def test_children_preserve_fixed_inputs_and_known_semantic_ids(key):
    yy, xx = np.indices((40, 48))
    palette = np.array([[200, 40, 30], [25, 45, 210]], np.uint8)
    rgb = palette[(xx > 18+yy//5).astype(int)]
    coarse = np.where(xx >= 24, 2, 1).astype(np.uint8); coarse[:8,:8] = 0
    support = np.ones_like(coarse, bool)
    before = [v.copy() for v in (rgb, coarse, support)]
    labels, detail = predict(key, rgb, coarse, support, [1.,1.], palette, he=rgb)
    assert set(np.unique(labels)) <= {1,2}
    assert labels.shape == coarse.shape
    assert detail['parent'] == CYCLE_FIVE[key].parent
    assert not detail['fine_reference_used'] and not detail['hard_anchors'] and not detail['fixed_margin']
    json.dumps(detail, allow_nan=False)
    for a,b in zip(before,(rgb,coarse,support)):
        np.testing.assert_array_equal(a,b)


def test_twenty_models_have_five_correct_parents_and_fifteen_distinct_interventions():
    assert len(CYCLE_FIVE) == 20
    assert validate_children(dict(survivors=list(CYCLE_FIVE_SURVIVORS)), CHILDREN_FIVE)
    assert len({c.mechanism for c in CYCLE_FIVE.values()}) == 20


def test_cpp_spatial_mixture_matches_independent_logsumexp_and_exact_thread_parity():
    rng = np.random.default_rng(31)
    x = rng.normal(size=(23,17,3)).astype(np.float32)
    coefficient = rng.normal(size=(3,3))
    intercept = np.array([1000.,-1000.,.2])
    mixing = rng.dirichlet(np.ones(3), size=x.shape[:2]).astype(np.float32)
    expected = logsumexp(x.astype(float)@coefficient.T+intercept+np.log(mixing.astype(float)), -1)
    one = native.local_mixture_logits(x, coefficient, intercept, mixing, threads=1)
    two = native.local_mixture_logits(x, coefficient, intercept, mixing, threads=2)
    np.testing.assert_allclose(one,expected,atol=4e-5,rtol=0)
    np.testing.assert_array_equal(one,two)


def test_laplacian_has_neumann_constant_nullspace_and_positive_screened_energy():
    lap = field_laplacian((5,7), [.8,1.4])
    np.testing.assert_allclose(lap.toarray(),lap.T.toarray())
    np.testing.assert_allclose(lap@np.ones(35),0,atol=1e-12)
    assert np.linalg.eigvalsh(lap.toarray()+.015*np.eye(35)).min() > .0149


def test_semantic_cavity_ring_distinguishes_single_class_lumen_and_is_physical():
    he = np.full((40,60,3),[180,100,140],np.uint8);he[12:28,15:35] = 255
    probability = np.zeros((40,60,2));probability[:,:25,0] = 1;probability[:,25:,1] = 1
    support = np.ones((40,60),bool)
    removed, report = interface_cavities(he,probability,support,np.array([1.,1.]))
    assert removed[12:28,15:35].all() and report['accepted_components']==1
    probability[...,0] = 1;probability[...,1] = 0
    removed,_ = interface_cavities(he,probability,support,np.array([1.,1.]))
    assert not removed.any()
    probability[:,:25,1] = 0;probability[:,25:,0] = 0;probability[:,25:,1] = 1
    removed,_ = interface_cavities(he,probability,support,np.array([.25,.25]))
    assert not removed.any()  # same pixels, less than32 physical square microns


def test_blind_palette_coordinate_updates_improve_one_anchored_objective():
    canonical = np.array([[.8,.1,.2],[.1,.3,.8]])
    target = canonical+.04
    x = np.repeat(target,20,axis=0)
    prior = np.repeat(np.eye(2),20,axis=0)
    fitted, report = learn_palette(x,prior,np.ones(40),canonical)
    assert np.linalg.norm(fitted-target) < np.linalg.norm(canonical-target)
    assert np.all((fitted>=0)&(fitted<=1))
    assert np.max(np.diff(report['objective_path'])) < 2e-7


def test_profile_nuisance_matches_independent_constrained_minimization():
    color = np.array([.8,.1,.3]);x = np.array([[.2,.8,.7],[.7,.3,.4]])
    actual = profiled_plane_cost(x,color)
    for value,expected in zip(x,actual):
        objective = lambda ab: np.sum((value-ab[0]*color-ab[1])**2)+.01*((ab[0]-1)**2+ab[1]**2)
        fitted = minimize(objective,[1.,0.],bounds=[(0,None),(None,None)],method='SLSQP',
                          options=dict(ftol=1e-13,maxiter=200))
        assert fitted.success
        assert abs(fitted.fun-expected)<1e-10


def test_cardinal_decoder_reproduces_vertices_constants_and_duplicate_symmetry():
    colors=np.array([[.8,.1,.2],[.1,.3,.8],[.1,.8,.3]])
    q,_=cardinal_coordinates(colors,colors)
    np.testing.assert_allclose(q,np.eye(3),atol=2e-6)
    duplicate=np.vstack([colors,colors[0]])
    q,_=cardinal_coordinates(np.array([[.5,.4,.3],[.2,.2,.9]]),duplicate)
    np.testing.assert_allclose(q.sum(-1),1,atol=1e-12)
    np.testing.assert_allclose(q[:,0],q[:,3],atol=1e-12)


@pytest.mark.parametrize('kind',['trimmed','channel','information','noise'])
def test_new_analytic_objective_gradients(kind):
    rng=np.random.default_rng(51)
    bags=rng.normal(size=(8,7,4));labels=np.arange(8)%3;weight=np.full(8,1/8)
    initial=rng.normal(size=(4,3))*.1
    if kind=='trimmed':
        theta=initial.ravel();f=lambda a:trimmed_objective(a,bags,labels,weight,initial)
    elif kind=='channel':
        channel=np.array([[.8,.1,.1],[.2,.7,.1],[.1,.2,.7]])
        theta=initial.ravel();f=lambda a:channel_objective(a,bags,labels,weight,initial,channel)
    elif kind=='information':
        theta=rng.normal(size=8)*.1
        f=lambda a:information_objective(a,bags.mean(1),labels,weight,3,np.zeros((2,4)))
    else:
        theta=rng.normal(size=12)*.1
        f=lambda a:noise_objective(a,bags.mean(1),labels,weight,3)
    assert check_grad(lambda a:f(a)[0],lambda a:f(a)[1],theta)<3e-5


def test_prevalence_recovers_identifiable_mixture_without_semantic_refitting():
    density=np.array([[.95,.05],[.05,.95]])
    count=density@np.array([.8,.2])
    before=density.copy()
    probability,report=fit_prevalence(np.log(density),count,iterations=300)
    np.testing.assert_allclose(probability,[.8,.2],atol=3e-4)
    np.testing.assert_array_equal(density,before)
    assert report['converged']


def test_geometric_median_resists_a_gross_coarse_tile_outlier():
    data=np.zeros((31,3));data[-1]=100
    f=lambda point:geometric_median_objective(point,data)
    fit=minimize(f,np.mean(data,0),jac=True,method='L-BFGS-B',options=dict(ftol=1e-12))
    assert np.linalg.norm(fit.x)<.001
    assert np.linalg.norm(np.mean(data,0))>5


def test_chow_liu_recovers_an_independent_markov_chain_and_normalizes():
    root=np.array([.4,.6]);middle=np.array([[.9,.1],[.2,.8]]);leaf=np.array([[.85,.15],[.1,.9]])
    joint=np.einsum('a,ab,bc->abc',root,middle,leaf)
    recovered,report=chow_liu_density(joint)
    np.testing.assert_allclose(recovered,joint,atol=1e-14)
    assert abs(report['discrete_normalization']-1)<1e-14


def test_bernstein_components_integrate_to_one_and_em_improves_likelihood():
    point,weight=np.polynomial.legendre.leggauss(2)
    coordinate=(point+1)/2;weight=weight/2
    query=np.array(list(product(coordinate,repeat=3)))
    quadrature=np.array([a*b*c for a,b,c in product(weight,repeat=3)])
    np.testing.assert_allclose(quadrature@bernstein_basis(query),1,atol=1e-14)
    rng=np.random.default_rng(11);basis=bernstein_basis(rng.uniform(0,1,(80,3)))
    theta,report=fit_bernstein(basis,rng.uniform(.1,1,80))
    assert np.all(theta>0) and abs(theta.sum()-1)<1e-12
    assert np.min(np.diff(report['regularized_log_likelihood_path']))>-1e-12
