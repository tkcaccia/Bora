"""Independent probability, variational and graph checks for cycle12."""
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import logsumexp,softmax
from scipy.stats import ncx2,norm
from experiments.bora_cycles.native_twelve import reconstruct,bayes_regions
from experiments.bora_cycles.twelve_appearance import gaussian_ball_probability,advection_laplacian,skew_system
from experiments.bora_cycles.twelve_regions import plurality_dual,censored_objective
from experiments.bora_cycles.twelve_support import infer_support,enclosure_depth,texture_evidence,rank_one_bic


def test_gaussian_acceptance_ball_matches_noncentral_chisquare_distribution():
    d=np.array([0.,1e-16,.1,1.,3.,10.,.3,1.01]);v=np.array([1.,.1,.2,.5,2.,.01,1e-12,1e-9])
    got=gaussian_ball_probability(d*d,v)
    np.testing.assert_allclose(got,ncx2.cdf(1/v,3,d*d/v),rtol=1e-7,atol=3e-13)
    assert np.isfinite(got).all() and np.all((got>=0)&(got<=1))


def test_directed_transport_is_constant_reproducing_m_matrix():
    yy,xx=np.indices((4,5));guide=np.stack([xx*yy*.03,xx*.1,yy*.1],-1)
    operator,report=advection_laplacian(guide,[2.,3.])
    dense=operator.toarray();off=dense-np.diag(np.diag(dense))
    assert np.max(off)<=1e-12 and report['constant_reproduction_residual']<1e-12
    assert np.max(np.abs(dense-dense.T))>1e-3
    matrix=dense+np.diag(np.linspace(.1,1.,20))
    np.testing.assert_allclose(np.linalg.solve(matrix,matrix@np.ones(20)),1,atol=1e-12)
    assert np.linalg.inv(matrix).min()>-1e-12


def test_shared_skew_system_matches_independent_dense_observation_design():
    rng=np.random.default_rng(1201);n,k=3,4
    means=rng.normal(size=(k,3));mass=rng.uniform(.1,1,size=(n,k));target=rng.normal(size=(n,k,3))
    lap=np.array([[1.,-1,0],[-1,2,-1],[0,-1,1]])
    omega,_,report=skew_system(sparse.csr_matrix(lap),mass,mass[...,None]*target,means)
    rows=[];rhs=[];bary=means.mean(0)
    for s in range(n):
        for g in range(k):
            design=np.stack([np.cross(axis,means[g]-bary) for axis in np.eye(3)],axis=1)
            block=np.zeros((3,3*n));block[:,s*3:s*3+3]=np.sqrt(mass[s,g])*design
            rows.append(block);rhs.append(np.sqrt(mass[s,g])*(target[s,g]-means[g]))
    design=np.concatenate(rows);observation=np.concatenate(rhs)
    normal=design.T@design+np.kron(lap,np.eye(3))+.015*np.eye(3*n)
    expected=np.linalg.solve(normal,design.T@observation)
    np.testing.assert_allclose(omega.ravel(),expected,atol=2e-13)
    assert report['linear_system_max_residual']<1e-12


def test_gray_reconstruction_matches_all_paths_max_min_algebra():
    ceiling=np.array([[[.9,.3],[.2,.8],[.7,.4]],[[.6,.2],[.5,.9],[.8,.6]]])
    marker=np.zeros_like(ceiling);marker[0,0,0]=.9;marker[1,1,1]=.9
    output=reconstruct(ceiling,marker);n=6
    for g in range(2):
        adjacency=np.zeros((n,n));flat=ceiling[...,g].ravel()
        for y in range(2):
            for x in range(3):
                a=y*3+x;adjacency[a,a]=flat[a]
                for dy,dx in [(1,0),(0,1)]:
                    if y+dy<2 and x+dx<3:
                        b=(y+dy)*3+x+dx;adjacency[a,b]=adjacency[b,a]=min(flat[a],flat[b])
        for pivot in range(n):adjacency=np.maximum(adjacency,np.minimum(adjacency[:,pivot,None],adjacency[None,pivot,:]))
        expected=np.max(np.minimum(marker[...,g].ravel()[:,None],adjacency),axis=0)
        np.testing.assert_array_equal(output[...,g].ravel(),expected)
    np.testing.assert_array_equal(reconstruct(ceiling,output),output)


def test_region_bayes_factor_separates_opposing_certainties_and_merges_equal_ones():
    p=np.array([[[.99,.01],[.99,.01],[.01,.99],[.01,.99]]])
    regions,report=bayes_regions(np.log(p),np.ones((1,4),bool))
    np.testing.assert_array_equal(regions,[[0,0,1,1]])
    assert report['greedy_merges']==2 and report['regions']==2
    expected=2*(logsumexp(2*np.log([.99,.01]))-np.log(2)-2*np.log(.5))
    np.testing.assert_allclose(report['positive_log_bayes_gain'],expected,atol=1e-14)
    isolated,_=bayes_regions(np.log(p),np.array([[1,0,0,1]],bool))
    assert len(np.unique(isolated))==4


def test_plurality_dual_gradient_and_original_primal_dual_gap():
    p=np.array([[.1,.9],[.3,.7],[.8,.2],[.7,.3]])
    logp=np.log(p);tile=np.array([0,0,1,1]);winner=np.array([0,1]);count=np.array([2,2]);start=np.array([[0,.2],[.3,0.]])
    value,gradient=plurality_dual(start,logp,tile,winner,count)
    for a,g in [(0,1),(1,0)]:
        d=np.zeros_like(start);d[a,g]=1e-6
        numeric=(plurality_dual(start+d,logp,tile,winner,count)[0]-plurality_dual(start-d,logp,tile,winner,count)[0])/2e-6
        np.testing.assert_allclose(gradient[a,g],numeric,atol=1e-9)
    def objective(v):
        a,b=plurality_dual(v.reshape(2,2),logp,tile,winner,count);return a,b.ravel()
    result=minimize(objective,np.zeros(4),jac=True,bounds=[(0,0),(0,None),(0,None),(0,0)],method='L-BFGS-B',options={'gtol':1e-12,'ftol':1e-15})
    offsets=result.x.reshape(2,2);shift=-offsets.copy();shift[np.arange(2),winner]=offsets.sum(-1)
    q=softmax(logp+shift[tile],-1);mean=q.reshape(2,2,2).mean(1)
    wrong=mean[np.arange(2),1-winner]-mean[np.arange(2),winner]
    primal=np.sum(q*(np.log(q)-logp)/2)+np.sum(np.maximum(wrong,0)**2)
    np.testing.assert_allclose(primal+result.fun,0,atol=1e-10)


def test_censored_channel_gradient_matches_finite_differences_and_tail_likelihood():
    colours=np.array([[1.,.2,.1],[0.,.8,.9],[.3,.4,.6]])
    q=np.array([[.2,.3,.5],[.4,.3,.3]]);prior=np.ones_like(q)/3;x=np.array([[0.,.5,1.],[.4,.6,.2]])
    value,gradient=censored_objective(q,x,prior,colours)
    direction=np.array([[1.,-1.,0.],[0.,1.,-1.]])
    numeric=(censored_objective(q+1e-6*direction,x,prior,colours)[0]-censored_objective(q-1e-6*direction,x,prior,colours)[0])/2e-6
    np.testing.assert_allclose(numeric,np.sum(gradient*direction,-1),atol=1e-9)
    mu=q[0]@colours
    expected=-.05**2*(norm.logcdf((1/255-mu[0])/.05)+norm.logsf((254/255-mu[2])/.05))+.5*(mu[1]-.5)**2+.006*np.sum((q[0]-prior[0])**2)
    np.testing.assert_allclose(value[0],expected,atol=1e-14)


def test_bayes_heap_matches_exhaustive_greedy_graph_on_random_small_grids():
    rng=np.random.default_rng(1215)
    for _ in range(12):
        logp=np.log(rng.dirichlet([.4,.4,.4],size=6)).reshape(2,3,3)
        actual,_=bayes_regions(logp,np.ones((2,3),bool))
        regions=[{v} for v in range(6)]
        edges={(0,1),(1,2),(3,4),(4,5),(0,3),(1,4),(2,5)}
        flat=logp.reshape(6,3)
        def evidence(region):return logsumexp(flat[list(region)].sum(0))-np.log(3)
        while True:
            offers=[]
            for a in range(len(regions)):
                for b in range(a+1,len(regions)):
                    if any((min(x,y),max(x,y)) in edges for x in regions[a] for y in regions[b]):
                        gain=evidence(regions[a]|regions[b])-evidence(regions[a])-evidence(regions[b])
                        if gain>1e-12:offers.append((gain,a,b))
            if not offers:break
            _,a,b=max(offers);regions[a]|=regions[b];regions.pop(b)
        expected=np.empty(6,int)
        for label,region in enumerate(regions):expected[list(region)]=label
        np.testing.assert_array_equal(actual.ravel()[:,None]==actual.ravel(),expected[:,None]==expected)


def test_adaptive_blank_can_remove_gray_glass_but_does_not_use_reference():
    yy,xx=np.indices((256,256));plane=np.rint(245-22*xx/255-8*yy/255).astype(np.uint8)
    he=np.repeat(plane[...,None],3,-1);he[108:148,108:148]=[180,90,140]
    own,report=infer_support('Q01',he,[1,1])
    assert not own[240,240] and own[128,128]
    assert report['removed_original_nonwhite_pixels']>0 and not report['reference_used']


def test_halfspace_depth_requires_surrounding_stain_and_texture_rejects_constant_blank():
    yy,xx=np.indices((256,256));r=np.hypot(yy-128,xx-128)
    depth,_=enclosure_depth((r>30)&(r<45),[1,1])
    assert depth[128,128]>.025 and depth[10,10]<.001
    evidence,_=texture_evidence(np.full((128,128,3),240,np.uint8),[1,1])
    assert np.max(evidence)<0


def test_texture_bic_matches_full_gaussian_covariance_likelihood_and_parameter_count():
    covariance=np.diag([4.,.8,.6]);blank=np.eye(3);alternative=np.diag([4.,1.,1.]);count=64
    likelihood=lambda model:-count/2*(np.linalg.slogdet(model)[1]+np.trace(np.linalg.solve(model,covariance)))
    # Rank-one strength plus a direction on the two-dimensional unit sphere.
    expected=likelihood(alternative)-likelihood(blank)-3/2*np.log(count)
    np.testing.assert_allclose(rank_one_bic(4.,1.,count),expected,atol=1e-13)


def test_unseeded_quadrature_cells_retain_parent_evidence_instead_of_uniform_class_tie(monkeypatch):
    from types import SimpleNamespace
    from experiments.bora_cycles import twelve_regions as module
    shape=(16,16);q=np.broadcast_to([.1,.9],(*shape,2)).copy()
    c=SimpleNamespace(shape=shape,mpp=np.array([1.,1.]),k=2,
        support=np.zeros(shape,bool),labels=np.zeros(shape,np.uint8),weight=np.zeros(shape))
    # A thin real tissue point is missed by nearest4um support sampling.
    c.support[15,15]=True
    monkeypatch.setattr(module,'edge_appearance',lambda *a:(q.copy(),{}))
    fitted,report=module.fuzzy_regions(c,None,None)
    assert np.all(fitted.argmax(-1)==1)
    np.testing.assert_allclose(fitted,q,atol=1e-7)
