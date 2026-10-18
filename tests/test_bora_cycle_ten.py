"""Independent mathematical and graph-contract checks, not accuracy claims."""
from itertools import product
import json
import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import check_grad, minimize

from bora.evidence import Evidence
from experiments.bora_cycles.cycle_ten import (envelope_glass,gain_system,guided_lift,
    integral_system,multinomial_objective,noise_probe_fit,residual_noise,sparse_curvature_system)
from experiments.bora_cycles.cycle_nine import projected_inverse
from experiments.bora_cycles.native_ten import tree_potts,seeded_forest,biconnected_white
from experiments.bora_cycles.methods import predict
from experiments.bora_cycles.registry import CHILDREN_TEN,CYCLE_TEN_SURVIVORS,candidates
from experiments.bora_cycles.review import validate_children
from experiments.bora_cycles.expanded_prediction import constant_evidence_fallback
from experiments.bora_cycles.expanded_metrics import glass_only_method


@pytest.mark.parametrize('key',[x['id'] for x in CHILDREN_TEN])
def test_whole_tissue_geometry_eligible_ids_and_input_immutability(key):
    yy,xx=np.indices((96,270))
    palette=np.array([[200,40,30],[25,45,210],[30,185,65]],np.uint8)
    pattern=np.clip(((xx+12*np.sin(yy/11))/90).astype(int),0,2)
    rgb=palette[pattern];coarse=(xx//90+1).astype(np.uint8);coarse[:12,:12]=0
    support=np.ones(coarse.shape,bool);before=[x.copy() for x in (rgb,coarse,support)]
    labels,report=predict(key,rgb,coarse,support,[1.,1.],palette,he=rgb)
    assert labels.shape==coarse.shape and set(np.unique(labels))=={1,2,3}
    assert np.all(labels[:12,:12]>0)
    assert report['parent']==candidates(10)[key].parent
    assert not report['fine_reference_used'] and not report['hard_anchors'] and not report['fixed_margin']
    json.dumps(report,allow_nan=False)
    for a,b in zip(before,(rgb,coarse,support)):np.testing.assert_array_equal(a,b)


def test_lineage_three_distinct_children_per_signed_parent():
    assert CYCLE_TEN_SURVIVORS==('J01','M04','K05','M11','C17')
    assert len(candidates(10))==20
    assert validate_children(dict(survivors=list(CYCLE_TEN_SURVIVORS)),CHILDREN_TEN)
    old={x.mechanism for cycle in range(1,10) for x in candidates(cycle).values()}
    assert not old.intersection(x['intervention'] for x in CHILDREN_TEN)


@pytest.mark.parametrize('key',['N10','N11','N12'])
def test_glass_only_constant_evidence_contract(key):
    he=np.full((24,30,3),[130,40,90],np.uint8);he[:,:8]=255
    coarse=np.ones((24,30),np.uint8);coarse[:,15:]=2;coarse[:4,10:15]=0
    complete=np.where(coarse>0,coarse,1).astype(np.uint8)
    labels,detail=constant_evidence_fallback(key,np.full((*coarse.shape,3),80,np.uint8),
        coarse,complete,np.ones_like(coarse,bool),[1.,1.],he)
    assert np.any(labels==0) and np.all((labels==complete)|(labels==0))
    assert glass_only_method(key,detail) and not detail['refinement_performed']
    assert not glass_only_method(key,dict(mechanism='unregistered'))


def test_tree_dp_equals_exhaustive_multiclass_global_energy():
    rng=np.random.default_rng(1010)
    for k in (2,3):
        parent=np.array([-1,0,0,1,1,2]);order=np.arange(6)
        unary=rng.normal(size=(6,k));penalty=rng.uniform(.1,2,6);penalty[0]=0
        labels,report=tree_potts(parent,order,unary,penalty)
        def energy(z):return sum(unary[i,z[i]] for i in range(6))+sum(penalty[i]*(z[i]!=z[parent[i]]) for i in range(1,6))
        exact=min(energy(z) for z in product(range(k),repeat=6))
        assert abs(energy(labels)-exact)<1e-12 and abs(report['dynamic_program_energy']-exact)<1e-12


def test_marker_forest_cuts_unique_largest_barrier_on_line():
    costs=np.array([.1,.2,.9,.3,.4]);a=np.arange(5);b=a+1;order=np.argsort(costs)
    actual=seeded_forest(a[order],b[order],np.array([1,0,0,0,0,2]))
    np.testing.assert_array_equal(actual,[1,1,1,2,2,2])
    np.testing.assert_array_equal(seeded_forest(a[order],b[order],np.array([2,0,0,0,0,1])),3-actual)


def independent_root_reachable(white,target,deleted=None):
    h,w=white.shape
    todo=[(y,x) for y in range(h) for x in range(w) if white[y,x] and (y in (0,h-1) or x in (0,w-1)) and (y,x)!=deleted]
    seen=set(todo)
    while todo:
        y,x=todo.pop()
        for dy,dx in ((0,1),(1,0),(0,-1),(-1,0)):
            z=(y+dy,x+dx)
            if 0<=z[0]<h and 0<=z[1]<w and white[z] and z!=deleted and z not in seen:
                seen.add(z);todo.append(z)
    return target in seen


def test_biconnectivity_equals_independent_vertex_deletion_oracle():
    rng=np.random.default_rng(1012)
    for _ in range(30):
        white=rng.random((5,6))>.35;actual=biconnected_white(white)
        for y,x in zip(*np.nonzero(white)):
            target=(y,x)
            if y in (0,4) or x in (0,5):expected=True
            else:
                expected=independent_root_reachable(white,target) and all(
                    independent_root_reachable(white,target,(a,b))
                    for a,b in zip(*np.nonzero(white)) if (a,b)!=target)
            assert actual[target]==expected


def test_convex_envelope_preserves_enclosed_lumen_and_removes_outer_white():
    white=np.ones((21,25),bool);white[4:18,5:20]=False;white[8:12,10:14]=True
    mask=envelope_glass(white)
    assert mask[0].all() and not mask[9,11] and not mask[5,6]
    assert np.all(~mask|white)


def test_shared_gain_has_independent_diagonal_normal_equations():
    means=np.array([[.1,.3,.6],[.7,.2,.4]])
    mass=np.array([[.5,.2],[.1,.8],[0,0]])
    sums=mass[...,None]*means[None]*1.7
    u,report=gain_system(sparse.csr_matrix((3,3)),mass,sums,means)
    d=np.sum(mass*np.sum(means*means,-1),-1)
    np.testing.assert_allclose(u,(1.7*d+.015)/(d+.015),atol=1e-14)
    assert report['minimum_gain']>=0 and report['linear_system_max_residual']<1e-14


def test_integral_observation_recovers_constant_and_preserves_total_mass():
    rgb=np.full((40,90,3),[40,130,210],np.uint8)
    c=Evidence(rgb,np.ones((40,90),np.uint8),np.ones((40,90),bool),[1,1])
    shape=(4,9);lap=sparse.csr_matrix((36,36));mean=rgb[0,0]/255
    u,report=integral_system(c,0,shape,lap,mean)
    np.testing.assert_allclose(u,np.broadcast_to(mean,u.shape),atol=1e-7)
    assert abs(report['total_observation_mass']-c.weight.sum()*36/3600)<1e-12
    assert report['linear_system_max_residual']<1e-12


def test_correlated_noise_exact_diagonal_limit_and_positive_variance():
    diagonal=np.linspace(1,3,12);mass=np.linspace(0,1,12);rhs=np.ones((12,3))
    fitted,noise,z,report=noise_probe_fit(sparse.diags(diagonal),mass,rhs,np.repeat(np.arange(3),4))
    np.testing.assert_allclose(noise,np.sqrt(mass)[:,None]*z/diagonal[:,None],atol=1e-15)
    np.testing.assert_allclose(np.mean(noise**2,1),mass/diagonal**2,atol=1e-15)
    assert report['linear_system_max_residual']<1e-14


def test_residual_df_matches_exact_diagonal_smoother_trace():
    mass=np.linspace(.1,1,12);diagonal=mass+.3;rhs=np.ones((12,3))
    fit,noise,z,_=noise_probe_fit(sparse.diags(diagonal),mass,rhs)
    scale,report=residual_noise(mass,np.zeros_like(fit),fit,noise,z)
    expected=np.sum((1-mass/diagonal)**2)
    assert abs(report['residual_df_raw']-expected)<1e-12 and scale>0


def test_guided_lift_preserves_constant_and_reproduces_bilinear_on_flat_guide():
    rng=np.random.default_rng(1019);u=rng.random((4,5,3));guide=np.zeros_like(u)
    actual=guided_lift(u,guide,np.zeros((17,19,3)),(17,19))
    import cv2
    expected=cv2.resize(u,(19,17),interpolation=cv2.INTER_LINEAR)
    # OpenCV computes interpolation coefficients at float precision.
    np.testing.assert_allclose(actual,expected,atol=1e-7)
    const=np.full((4,5,3),.43)
    np.testing.assert_allclose(guided_lift(const,rng.random((4,5,3)),rng.random((17,19,3)),(17,19)),.43,atol=1e-14)


def test_sparse_curvature_against_independent_smooth_constrained_dual():
    rng=np.random.default_rng(1017)
    base=sparse.csr_matrix([[1,-2,1,0],[0,1,-2,1]])*.2
    mass=np.array([.2,.4,.6,.8]);target=rng.random((4,3));sums=mass[:,None]*target;mean=np.array([.2,.5,.7])
    u,report=sparse_curvature_system(base,mass,sums,mean,iterations=3000)
    # The conic primal has a nondifferentiable norm at a straight field. Its
    # independent dual is a smooth quadratic with unit-ball constraints.
    rhs=sums+.015*mean;diagonal=mass+.015
    def dual(flat):
        residual=rhs-base.T@flat.reshape(2,3)
        field=residual/diagonal[:,None]
        return .5*np.sum(residual*field),-(base@field).ravel()
    ref=minimize(dual,np.zeros(6),jac=True,method='SLSQP',constraints=[dict(type='ineq',
        fun=lambda a:1-np.sum(a.reshape(2,3)**2,-1))],options=dict(ftol=1e-12,maxiter=1000))
    assert ref.success
    reference=(rhs-base.T@ref.x.reshape(2,3))/diagonal[:,None]
    np.testing.assert_allclose(u,reference,atol=4e-5)
    assert report['converged']


def test_conditional_multinomial_gradient_and_independent_optimum():
    rng=np.random.default_rng(1024)
    colors=rng.random((4,3))+.1;colors/=colors.sum(-1,keepdims=True)
    x=rng.dirichlet(np.ones(3),3);prior=rng.dirichlet(np.ones(4),3)
    f=lambda a:multinomial_objective(a.reshape(3,4),x,prior,colors)
    assert check_grad(lambda a:f(a)[0],lambda a:f(a)[1].ravel(),prior.ravel())<2e-6
    q,detail=projected_inverse(lambda a:multinomial_objective(a,x,prior,colors),prior,iterations=3000)
    ref=minimize(lambda a:f(a)[0],prior.ravel(),jac=lambda a:f(a)[1].ravel(),method='SLSQP',
        bounds=[(0,1)]*12,constraints=[dict(type='eq',fun=lambda a:a.reshape(3,4).sum(1)-1)],
        options=dict(ftol=1e-12,maxiter=1000))
    assert ref.success and abs(f(q.ravel())[0]-ref.fun)<1e-8 and detail['converged']
