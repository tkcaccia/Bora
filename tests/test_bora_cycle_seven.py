"""Independent numerical references and invariants for cycle-seven science."""
import json
from itertools import product

import numpy as np
import pytest
from scipy.optimize import check_grad, minimize
from scipy.special import softmax

from experiments.bora_cycles import native
from experiments.bora_cycles.cycle_five import field_laplacian
from experiments.bora_cycles.cycle_seven import (
    POWERS, class_log_risks, empirical_exponential_objective, fit_brier,
    fit_nuclear, fused_field, gaussian_design, gaussian_monomials,
    harmonic_coordinates, logistic_objective, nuclear_prox,
    orientation_laplacian, poisson_field, robust_hinge_objective, solve_vector_field)
from experiments.bora_cycles.methods import predict
from experiments.bora_cycles.registry import (
    CHILDREN_SEVEN, CYCLE_SEVEN, CYCLE_SEVEN_SURVIVORS, candidates)
from experiments.bora_cycles.review import validate_children


@pytest.mark.parametrize('key',[v['id'] for v in CHILDREN_SEVEN])
def test_child_analyzes_unannotated_tissue_without_mutating_fixed_inputs(key):
    yy,xx=np.indices((96,270))
    palette=np.array([[200,40,30],[25,45,210],[30,185,65]],np.uint8)
    pattern=np.clip(((xx+12*np.sin(yy/11))/90).astype(int),0,2)
    rgb=palette[pattern];coarse=(xx//90+1).astype(np.uint8);coarse[:12,:12]=0
    support=np.ones(coarse.shape,bool);before=[a.copy() for a in (rgb,coarse,support)]
    labels,report=predict(key,rgb,coarse,support,[1.,1.],palette,he=rgb)
    assert labels.shape==coarse.shape and set(np.unique(labels))=={1,2,3}
    assert np.all(labels[:12,:12]>0)
    assert report['parent']==CYCLE_SEVEN[key].parent
    assert not report.get('missing_known_bag_class',False)
    assert not report['fine_reference_used'] and not report['hard_anchors'] and not report['fixed_margin']
    json.dumps(report,allow_nan=False)
    for a,b in zip(before,(rgb,coarse,support)):np.testing.assert_array_equal(a,b)


def test_cycle_uses_exact_signed_five_and_fifteen_distinct_interventions():
    assert CYCLE_SEVEN_SURVIVORS==('J01','H02','C17','J07','H10') and len(CYCLE_SEVEN)==20
    assert validate_children(dict(survivors=list(CYCLE_SEVEN_SURVIVORS)),CHILDREN_SEVEN)
    old={v.mechanism for cycle in range(1,7) for v in candidates(cycle).values()}
    new={v['intervention'] for v in CHILDREN_SEVEN}
    assert len(new)==15 and not old.intersection(new)


def test_directional_graph_is_symmetric_positive_and_transports_along_edges():
    guide=np.repeat(np.arange(6)[None,:,None],5,axis=0)*np.ones((5,6,3))
    lap=orientation_laplacian(guide,[1.,1.],length=1.)
    matrix=lap.toarray()
    np.testing.assert_allclose(matrix,matrix.T)
    np.testing.assert_allclose(matrix@np.ones(30),0,atol=1e-12)
    assert np.linalg.eigvalsh(matrix).min()>-1e-12
    # Guide changes horizontally: vertical transport must be more conductive.
    assert -matrix[7,13]>20*(-matrix[7,8])


def test_fused_variational_fit_against_independent_convex_minimization():
    rng=np.random.default_rng(131);lap=field_laplacian((2,3),[1.,1.],length_um=1.)
    diagonal=rng.uniform(.5,1,6);rhs=rng.uniform(.1,.9,(6,3))*diagonal[:,None]
    fitted,report=fused_field(lap,diagonal,rhs,iterations=300)
    a,b=np.where(np.triu(lap.toarray(),1)<0);weight=-lap.toarray()[a,b]
    def objective(flat):
        u=flat.reshape(6,3)
        return .5*np.sum(diagonal[:,None]*u*u)-np.sum(rhs*u)+.02*np.sum(weight*np.sqrt(np.sum((u[a]-u[b])**2,-1)+.0001))
    ref=minimize(objective,(rhs/diagonal[:,None]).ravel(),method='BFGS',options=dict(gtol=1e-8))
    assert abs(objective(fitted.ravel())-ref.fun)<1e-8
    assert np.max(np.diff(report['objective_path']))<1e-10
    assert report['gradient_infinity_norm']<2e-6


def test_covariance_vector_field_matches_full_coupled_normal_equations():
    rng=np.random.default_rng(141);lap=field_laplacian((2,3),[.7,1.2],length_um=1.)
    a=rng.normal(size=(3,3));covariance=a@a.T+.2*np.eye(3)
    diagonal=rng.uniform(.3,1,6);rhs=rng.normal(size=(6,3))
    fitted,report=solve_vector_field(lap,diagonal,rhs,covariance)
    matrix=np.kron(lap.toarray(),covariance/(np.trace(covariance)/3))+np.kron(np.diag(diagonal),np.eye(3))
    expected=np.linalg.solve(matrix,rhs.ravel()).reshape(6,3)
    np.testing.assert_allclose(fitted,expected,atol=1e-11)
    assert max(report['linear_system_max_residual'])<1e-11


def test_positive_field_matches_independent_poisson_objective():
    rng=np.random.default_rng(132);lap=field_laplacian((2,3),[1.,1.],length_um=1.)
    mass=rng.uniform(.2,1,6);sums=rng.uniform(.1,.8,(6,3))*mass[:,None];mean=np.array([.2,.4,.7])
    fitted,report=poisson_field(lap,mass,sums,mean)
    target=sums+.015*mean;linear=mass+.015
    def objective(flat):
        u=flat.reshape(6,3)
        return .5*np.sum(u*(lap@u))+np.sum(linear[:,None]*u-target*np.log(u))
    ref=minimize(objective,np.tile(mean,(6,1)).ravel(),method='L-BFGS-B',bounds=[(1e-7,None)]*18,
                 options=dict(ftol=1e-14,gtol=1e-8,maxiter=300))
    assert abs(objective(fitted.ravel())-ref.fun)<1e-8
    assert report['minimum_intensity']>0 and report['gradient_infinity_norm']<1e-6
    assert np.max(np.diff(report['objective_path']))<1e-10


def test_chebyshev_inverse_certificate_against_independent_epigraph_solver():
    rng=np.random.default_rng(133);colors=rng.uniform(.05,.95,(4,3));x=rng.uniform(.05,.95,(5,3))
    pi=rng.dirichlet(np.ones(4),5);q,report=native.chebyshev_palette(x,pi,colors,iterations=1200)
    for rgb,prior,actual in zip(x.astype(np.float32),pi.astype(np.float32),q):
        def fun(a):return .006*np.sum((a[:4]-prior)**2)+.025*a[-1]
        constraints=[dict(type='eq',fun=lambda a:a[:4].sum()-1),
            dict(type='ineq',fun=lambda a:a[-1]-(a[:4]@colors-rgb)),
            dict(type='ineq',fun=lambda a:a[-1]+(a[:4]@colors-rgb))]
        start=np.r_[prior,np.max(np.abs(prior@colors-rgb))]
        ref=minimize(fun,start,method='SLSQP',bounds=[(0,1)]*4+[(0,None)],constraints=constraints,
            options=dict(ftol=1e-12,maxiter=500))
        assert ref.success
        primal=.006*np.sum((actual-prior)**2)+.025*np.max(np.abs(actual@colors-rgb))
        assert primal-ref.fun<=report['max_primal_dual_gap']+2e-8
        assert abs(primal-ref.fun)<2e-5
    np.testing.assert_allclose(q.sum(-1),1,atol=1e-6)


def test_chebyshev_one_and_two_threads_are_bit_identical():
    rng=np.random.default_rng(134);x=rng.random((31,3));pi=rng.dirichlet(np.ones(5),31);colors=rng.random((5,3))
    a,ra=native.chebyshev_palette(x,pi,colors,threads=1)
    b,rb=native.chebyshev_palette(x,pi,colors,threads=2)
    np.testing.assert_array_equal(a,b)
    assert ra['max_primal_dual_gap']==rb['max_primal_dual_gap']
    assert ra['mean_iterations']==rb['mean_iterations']


def test_harmonic_coordinates_obey_maximum_principle_and_class_permutation():
    rng=np.random.default_rng(135);x=rng.random((50,3));colors=np.array([[.9,.1,.2],[.1,.2,.9],[.2,.8,.2]])
    pi=rng.dirichlet(np.ones(3),50);q,report=harmonic_coordinates(x,pi,colors)
    assert q.min()>=0 and q.max()<=1 and report['linear_system_max_residual']<1e-10
    np.testing.assert_allclose(q.sum(-1),1,atol=1e-10)
    permutation=[2,0,1];other,_=harmonic_coordinates(x,pi[:,permutation],colors[permutation])
    np.testing.assert_allclose(other,q[:,permutation],atol=1e-10)


def test_analytic_gaussian_moments_match_independent_five_point_cubature():
    mean=np.array([[.3,-.5,1.1],[-.6,.2,.4]])
    point,weight=np.polynomial.hermite.hermgauss(5)
    offset=np.array(list(product(point,repeat=3)))*np.sqrt(.08)
    weights=np.array([a*b*c for a,b,c in product(weight,repeat=3)])/np.pi**1.5
    powers=(POWERS[:,None]+POWERS[None,:]).reshape(-1,3)
    expected=np.stack([np.sum(weights[:,None]*np.prod((m+offset)[:,None,:]**powers[None],-1),0) for m in mean])
    np.testing.assert_allclose(gaussian_monomials(mean,powers),expected,rtol=1e-12,atol=1e-12)


def test_brier_normal_equations_match_quadrature_expected_risk():
    rng=np.random.default_rng(136);mean=rng.normal(size=(7,3));labels=np.arange(7)%3;weights=np.full(7,1/7)
    coefficient,report=fit_brier(mean,labels,weights,3)
    assert report['normal_equation_max_residual']<1e-12
    point,weight=np.polynomial.hermite.hermgauss(4)
    offsets=np.array(list(product(point,repeat=3)))*np.sqrt(.08)
    qw=np.array([a*b*c for a,b,c in product(weight,repeat=3)])/np.pi**1.5
    design=np.prod((mean[:,None,None,:]+offsets[None,:,None,:])**POWERS[None,None,:,:],-1)
    matrix=np.einsum('n,q,nqi,nqj->ij',weights,qw,design,design)+.0025*np.eye(10)
    rhs=np.einsum('n,q,nqi,nk->ik',weights,qw,design,np.eye(3)[labels])
    np.testing.assert_allclose(coefficient,np.linalg.solve(matrix,rhs),atol=1e-10)


@pytest.mark.parametrize('kind',['robust','logistic','empirical'])
def test_new_risk_gradients_against_independent_finite_differences(kind):
    rng=np.random.default_rng(137);x=np.c_[rng.normal(size=(6,3)),np.ones(6)]
    labels=np.arange(6)%3;weights=np.full(6,1/6);theta=rng.normal(size=12)*.15
    if kind=='robust':objective=lambda a:robust_hinge_objective(a,x,labels,weights,3)
    elif kind=='logistic':
        design,qw=gaussian_design(x)
        objective=lambda a:logistic_objective(a,design,labels,weights,qw,3)
    else:
        design=np.repeat(x[:,None],7,axis=1);design[...,:3]+=rng.normal(size=(6,7,3))*.2
        objective=lambda a:empirical_exponential_objective(a,design,labels,weights,3)
    error=check_grad(lambda a:objective(a)[0],lambda a:objective(a)[1],theta)
    assert error<3e-6


def test_empirical_risk_is_exact_log_of_explicit_finite_risk_plus_ridge():
    rng=np.random.default_rng(138);design=rng.normal(size=(5,9,4));design[...,-1]=1
    labels=np.arange(5)%3;weight=np.full(5,.2);coefficient=rng.normal(size=(3,4))*.2
    logits=design@coefficient.T;raw=0.
    for n in range(5):
        for j in range(9):
            for rival in range(3):
                if rival!=labels[n]:raw+=weight[n]/9*np.exp(logits[n,j,rival]-logits[n,j,labels[n]])
    value,_=empirical_exponential_objective(coefficient.ravel(),design,labels,weight,3)
    assert abs(value-np.log(raw+.0025*np.sum(coefficient**2)))<1e-12


def test_class_risk_constraint_jacobians_match_independent_differences():
    rng=np.random.default_rng(139);x=np.c_[rng.normal(size=(8,3)),np.ones(8)];labels=np.arange(8)%3
    theta=rng.normal(size=12)*.1
    for g in range(3):
        assert check_grad(lambda a:class_log_risks(a,x,labels,3)[0][g],
                          lambda a:class_log_risks(a,x,labels,3)[1][g],theta)<2e-6


def test_nuclear_prox_matches_independent_rank_one_closed_form():
    matrix=np.array([[3.,0.,0.,2.],[0.,0.,0.,-2.]])
    expected=matrix.copy();expected[0,0]-=.02*.4
    np.testing.assert_allclose(nuclear_prox(matrix,.4),expected)


def test_nuclear_fit_has_monotone_objective_and_honest_residual_flag():
    rng=np.random.default_rng(140);x=np.c_[rng.normal(size=(12,3)),np.ones(12)];labels=np.arange(12)%3
    coefficient,report=fit_nuclear(x,labels,np.full(12,1/12),3)
    assert np.isfinite(coefficient).all() and np.max(np.diff(report['objective_path']))<1e-10
    assert report['converged']==(report['proximal_gradient_residual']<1e-6)
    # Translation-invariant pair risks and the symmetric matrix prior preserve
    # zero mean class contrasts from the specified zero initialization.
    np.testing.assert_allclose(coefficient.mean(0),0,atol=1e-10)
