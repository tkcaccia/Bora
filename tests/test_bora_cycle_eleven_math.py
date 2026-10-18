"""Independent finite-dimensional/physical checks for cycle11 hypotheses."""
import numpy as np
from scipy import sparse
from scipy.optimize import check_grad,minimize
from scipy.fft import dctn,idctn

from experiments.bora_cycles.eleven_appearance import (adjoint_edges,affinity_scores,
    clr_objective,clr_solve,folded_normal,fractional_diagonal,fractional_system,
    gradient_edges,matrix_gaussian_score,matrix_log_spd,negative_tail_probabilities,
    order_projection,ordered_system,shared_logvariance,tv_palette)
from experiments.bora_cycles.eleven_support import (alpha_support,convex_support,
    escape_support,escape_times,infer_support,occupancy_support)


def test_halfspace_projection_against_independent_quadratic_program():
    means=np.array([[1.,0,0],[0,1,0],[0,0,1]])
    original=np.array([[[0.,.8,.2],[.9,.1,0],[.2,.7,.1]]])
    projected,report=order_projection(original,means,sweeps=1000)
    pairs=[(i,j,means[i]-means[j]) for i in range(3) for j in range(i+1,3)]
    constraints=[dict(type='ineq',fun=lambda z,i=i,j=j,n=n:(z.reshape(3,3)[i]-z.reshape(3,3)[j])@n-.25*(n@n)) for i,j,n in pairs]
    ref=minimize(lambda z:.5*np.sum((z-original.ravel())**2),original.ravel(),
        jac=lambda z:z-original.ravel(),constraints=constraints,method='SLSQP',options=dict(ftol=1e-12,maxiter=1000))
    assert ref.success
    np.testing.assert_allclose(projected.ravel(),ref.x,atol=2e-6)
    assert report['order_constraint_violation']<1e-7


def test_ordered_field_diagonal_unconstrained_limit():
    means=np.array([[1.,0,0],[0,0,1.]])
    mass=np.array([[.2,.5],[.7,.1]])
    sums=mass[...,None]*means[None]
    actual,report=ordered_system(sparse.csr_matrix((2,2)),mass,sums,means,iterations=500)
    np.testing.assert_allclose(actual,np.broadcast_to(means,actual.shape),atol=1e-12)
    assert report['converged']


def test_fractional_neumann_operator_matches_independent_dense_square_root():
    shape=(3,4);spacing=[1.2,.8];n=np.prod(shape)
    def axis_laplacian(length,step):
        d=np.diag(np.r_[1,np.full(length-2,2),1])-np.diag(np.ones(length-1),1)-np.diag(np.ones(length-1),-1)
        return d/step**2
    lap=np.kron(axis_laplacian(3,spacing[1]),np.eye(4))+np.kron(np.eye(3),axis_laplacian(4,spacing[0]))
    # Eigenvalues at zero are rounded before the independent PSD square root.
    val,vec=np.linalg.eigh(lap);val[np.abs(val)<1e-12]=0
    root=(vec*np.sqrt(np.maximum(val,0)))@vec.T*16
    diagonal=fractional_diagonal(shape,spacing)
    actual=np.stack([idctn(diagonal*dctn(np.eye(n)[i].reshape(shape),norm='ortho'),norm='ortho').ravel() for i in range(n)],-1)
    np.testing.assert_allclose(actual,root,atol=2e-6)
    rng=np.random.default_rng(1102);mass=rng.random(n);rhs=rng.random((n,3))
    fit,report=fractional_system(mass,rhs,shape,spacing)
    np.testing.assert_allclose(fit,np.linalg.solve(root+np.diag(mass+.015),rhs),atol=2e-6)
    assert all(r['converged'] for r in report)


def test_folded_normal_expectation_and_bhattacharyya_closed_forms():
    from scipy.integrate import quad
    from scipy.stats import norm
    for delta,v in [(0.,1.),(.7,.2),(-1.3,2.)]:
        reference=quad(lambda x:abs(x)*norm.pdf(x,loc=delta,scale=np.sqrt(v)),-np.inf,np.inf)[0]
        assert abs(float(folded_normal(np.array(delta),np.array(v)))-reference)<1e-8
    np.testing.assert_allclose(affinity_scores(np.array([0,4.]),np.zeros(2)),[0,-.5])


def test_matrix_covariance_score_matches_full_gaussian_algebra():
    covariance=np.array([[2.,.3,.2],[.3,1.,-.1],[.2,-.1,.7]])
    delta=np.array([[.2,-.4,.9],[1.,.1,-.3]])
    score,clipped=matrix_gaussian_score(delta,np.broadcast_to(matrix_log_spd(covariance),(2,3,3)))
    reference=-.5*(np.einsum('ni,ij,nj->n',delta,np.linalg.inv(covariance),delta)+np.linalg.slogdet(covariance)[1])
    np.testing.assert_allclose(score,reference,atol=1e-14);assert clipped==0


def test_shared_logvariance_matches_independent_full_design_ridge():
    rng=np.random.default_rng(1109);mass=rng.random((4,2));targets=rng.normal(size=(4,2,3))
    a,b,report=shared_logvariance(sparse.csr_matrix((4,4)),mass,targets)
    design=[];values=[];weights=[]
    for i in range(4):
        for g in range(2):
            for j in range(3):
                row=np.zeros(10);row[i]=1;row[4+3*g+j]=1
                design.append(row);values.append(targets[i,g,j]);weights.append(mass[i,g])
    x=np.array(design);v=np.array(values);w=np.array(weights)
    ref=np.linalg.solve((x.T*w)@x+.015*np.eye(10),x.T@(w*v))
    np.testing.assert_allclose(np.r_[a,b.ravel()],ref,atol=1e-13)
    assert report['linear_system_max_residual']<1e-13


def test_physical_gradient_adjoint_and_tv_uncoupled_analytic_solution():
    rng=np.random.default_rng(1113);q=rng.normal(size=(3,4,2));px=rng.normal(size=(3,3,2));py=rng.normal(size=(2,4,2));spacing=[.7,1.3]
    dx,dy=gradient_edges(q,spacing)
    assert abs(np.sum(dx*px)+np.sum(dy*py)-np.sum(q*adjoint_edges(px,py,q.shape,spacing)))<1e-13
    colors=np.array([[1.,0,0],[0,0,1.]])
    x=rng.random((3,4,3));prior=np.full((3,4,2),.5)
    fit,report=tv_palette(x,prior,colors,np.ones((3,3)),np.ones((2,4)),spacing,strength=0,iterations=1000)
    d=colors[0]-colors[1];p=np.clip((np.sum((x-colors[1])*d,axis=-1)+.024*.5)/(d@d+.024),0,1)
    np.testing.assert_allclose(fit[...,0],p,atol=1e-8)
    assert report['converged'] and report['step_condition']<1


def test_clr_gradient_and_stationary_objective_against_independent_solver():
    rng=np.random.default_rng(1114);colors=rng.random((3,3))+.1;x=rng.random((2,3))+.1;prior=np.full((2,3),1/3)
    def f(z):
        values,gradient=clr_objective(z.reshape(2,3),x,prior,colors)
        return values.sum(),gradient.ravel()
    assert check_grad(lambda z:f(z)[0],lambda z:f(z)[1],prior.ravel())<1e-6
    q,report=clr_solve(x,prior,colors,iterations=3000)
    ref=minimize(lambda z:f(z)[0],prior.ravel(),jac=lambda z:f(z)[1],method='SLSQP',
        bounds=[(0,1)]*6,constraints=[dict(type='eq',fun=lambda z:z.reshape(2,3).sum(-1)-1)],options=dict(ftol=1e-12,maxiter=2000))
    assert ref.success and abs(f(q.ravel())[0]-ref.fun)<2e-7
    assert report['objective_final']<=report['objective_initial']


def test_negative_tail_equals_direct_weighted_survival_including_ties():
    q=np.array([[[.2,.8],[.8,.2],[.8,.2],[.95,.05]]]);labels=np.array([[0,1,1,0]]);weight=np.array([[1.,2.,3.,4.]])
    tails,_=negative_tail_probabilities(q,labels,weight)
    for g in range(2):
        negatives=labels!=g
        expected=np.array([(.5+weight[negatives&(q[...,g]>=v)].sum())/(1+weight[negatives].sum()) for v in q[...,g].ravel()])
        np.testing.assert_allclose(tails[...,g].ravel(),expected,atol=1e-14)


def test_poisson_escape_has_exact_line_solution_and_squared_physical_units():
    white=np.zeros((3,5),bool);white[1]=True
    t,reachable,report=escape_times(white,[1,1])
    np.testing.assert_allclose(t[1],[0,1.5,2.,1.5,0],atol=1e-14)
    enlarged,_,_=escape_times(white,[2,2]);np.testing.assert_allclose(enlarged[1],4*t[1],atol=1e-14)
    assert report['poisson_max_residual']<1e-14 and reachable.sum()==5
    closed=np.zeros((7,7),bool);closed[2:5,2:5]=True
    t,reachable,_=escape_times(closed,[1,1]);assert not reachable.any() and np.isinf(t).all()


def test_alpha_can_separate_tissue_pieces_that_convex_control_bridges():
    white=np.ones((120,220),bool);white[40:65,20:40]=False;white[40:65,180:200]=False
    alpha,report=alpha_support(white,[1,1],alpha=20,step=4)
    convex,_=convex_support(white)
    assert convex[50,110] and not alpha[50,110]
    assert np.all(alpha[~white]) and report['retained_triangles']>0


def test_native_support_preserves_nonwhite_and_uses_no_inherited_clip():
    # Leave exterior glass farther than the occupancy model's32um half-window.
    he=np.full((400,448,3),255,np.uint8);he[120:260,140:320]=[120,45,90]
    for key in ('P10','P11','P12','B23'):
        support,report=infer_support(key,he,[.5,.7])
        assert support[120:260,140:320].all()
        assert not report['inherited_common_clip_applied'] and not report['reference_used']
        assert report['removed_white_pixels']>0
