"""Distinct appearance/uncertainty interventions with explicit finite solves."""
import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu,spsolve
from scipy.special import ndtr,softmax
from scipy.stats import ncx2

from .cycle_eight import class_statistics,field_inputs
from .cycle_nine import upsample
from .cycle_six import field_geometry,guide_laplacian
from .cycle_ten import whiten
from .eleven_appearance import sampling_fields


def residual_likelihood(c,palette,he,mode):
    means,cov,shape,spacing,bending=field_inputs(c,he,'bending')
    transform=whiten(cov);values=c.rgb@transform;means=means@transform
    if mode=='scale':
        guide=-np.log((cv2.resize(he,shape[::-1],interpolation=cv2.INTER_AREA).astype(float)+1)/256)
        positive_lap=guide_laplacian(guide,spacing)
    scores=[];reports=[]
    for g in range(c.k):
        mass,sums=class_statistics(c,g,shape,values)
        matrix=bending+sparse.diags(mass+.015);solve=splu(matrix.tocsc()).solve
        rhs=sums+.015*means[g];fitted=solve(rhs);surface=upsample(fitted,c.shape,shape)
        delta=values-surface
        detail=dict(mean_equation_residual=float(np.max(np.abs(matrix@fitted-rhs))))
        if mode=='asymmetric':
            scales=[]
            for sign in (-1,1):
                side=(delta*sign)>=0
                w=c.weight*(c.labels==g)
                side_mass=np.stack([cv2.resize((w*side[...,j]),shape[::-1],interpolation=cv2.INTER_AREA).ravel() for j in range(3)],-1)
                moments=np.stack([cv2.resize(w*side[...,j]*delta[...,j]**2,shape[::-1],interpolation=cv2.INTER_AREA).ravel() for j in range(3)],-1)
                target=.2+.8*moments/np.maximum(side_mass,1e-30)
                target_rhs=mass[:,None]*np.log(target)
                field=solve(target_rhs);detail['negative_scale_residual' if sign<0 else 'positive_scale_residual']=float(np.max(np.abs(matrix@field-target_rhs)))
                scales.append(np.exp(.5*np.clip(upsample(field,c.shape,shape),-30,30)))
            negative,positive=scales;scale=np.where(delta<0,negative,positive)
            score=-np.sum(np.log(negative+positive)+.5*(delta/scale)**2,-1)
        else:
            _,second=class_statistics(c,g,shape,values*values)
            target=sums/np.maximum(mass[:,None],1e-30)
            residual=np.maximum(second/np.maximum(mass[:,None],1e-30)-2*fitted*target+fitted*fitted,0)
            variance=.2+.8*residual
            if mode=='scale':
                scale_matrix=positive_lap+sparse.diags(mass+.015)
                scale_rhs=mass[:,None]*np.sqrt(variance)+.015
                scale_fit=spsolve(scale_matrix.tocsc(),scale_rhs)
                if scale_fit.min()<=0:raise RuntimeError('Positive scale maximum principle failed')
                detail['scale_equation_residual']=float(np.max(np.abs(scale_matrix@scale_fit-scale_rhs)))
                variance=upsample(scale_fit,c.shape,shape)**2;logv=np.log(variance)
            else:
                log_rhs=mass[:,None]*np.log(variance);log_fit=solve(log_rhs)
                detail['variance_equation_residual']=float(np.max(np.abs(matrix@log_fit-log_rhs)))
                logv=np.clip(upsample(log_fit,c.shape,shape),-30,30);variance=np.exp(logv)
            if mode=='conditional':
                z=delta/np.sqrt(variance)
                neighbour=np.stack([(cv2.filter2D(z[...,j],-1,np.array([[0,1,0],[1,0,1],[0,1,0]],float)/4,borderType=cv2.BORDER_REFLECT_101)) for j in range(3)],-1)
                w=c.weight*(c.labels==g);total=max(float(w.sum()),1e-30)
                rho=np.clip(np.sum(w[...,None]*z*neighbour,axis=(0,1))/(np.sum(w[...,None]*neighbour**2,axis=(0,1))+.015),-.8,.8)
                innovation=z-rho*neighbour
                noise=.2+.8*np.sum(w[...,None]*innovation**2,axis=(0,1))/total
                score=-.5*np.sum(innovation**2/noise+np.log(noise)+logv,-1)
                detail.update(autoregression_rgb=rho.tolist(),innovation_variance_rgb=noise.tolist(),
                    normalized_joint_image_likelihood=False,pseudolikelihood=True)
            else:score=-.5*np.sum(delta*delta/variance+logv,-1)
        scores.append(score);reports.append(detail)
    return softmax(np.stack(scores,-1),-1),dict(model=mode+' residual likelihood',fits=reports)


def gaussian_ball_probability(distance,variance,radius=1.):
    """Exact isotropic 3D Gaussian ball probability, with stable tail fallback."""
    distance=np.asarray(distance,float);variance=np.maximum(np.asarray(variance,float),1e-30)
    d=np.sqrt(np.maximum(distance,0));s=np.sqrt(variance)
    a=(radius-d)/s;b=(-radius-d)/s
    phi=lambda z:np.exp(-.5*z*z)/np.sqrt(2*np.pi)
    central=ndtr(a)-ndtr(b)
    term=np.divide(s*(phi(a)-phi(b)),d,out=np.zeros_like(d),where=d>1e-8*s)
    result=central-term
    zero=d<=1e-8*s
    result=np.where(zero,2*ndtr(radius/s)-1-2*radius/s*phi(radius/s),result)
    # Only cancellation cases need a distribution-library fallback.
    unstable=(result<0)|(result>1)|((result<1e-12)&(central>1e-10))
    if unstable.any():result[unstable]=ncx2.cdf(radius*radius/variance[unstable],3,distance[unstable]/variance[unstable])
    if not np.isfinite(result).all():raise ValueError('Nonfinite Gaussian ball probability')
    return np.clip(result,0,1)


def uncertain_risk(c,palette,he,mode):
    values,fields,variance,fits=sampling_fields(c,he)
    distance=np.sum((values[...,None,:]-fields)**2,-1)
    if mode=='credible':scores=-(np.sqrt(distance)+np.sqrt(7.814727903*variance))**2
    else:
        probability=gaussian_ball_probability(distance,variance)
        parent=-.5*distance/(1+variance)-1.5*np.log1p(variance)
        # Sort lexicographically, so parent cannot overturn a resolved probability.
        largest=probability.max(-1,keepdims=True);ties=probability==largest
        winner=np.where(ties,parent,-np.inf).argmax(-1)
        q=np.eye(c.k)[winner]
        return q,dict(model='finite ball prototype probability',fits=fits,radius_whitened_rgb=1.,
            saturated_probability_pixels=int(np.any((probability==0)|(probability==1),axis=-1).sum()),
            tied_maximum_pixels=int((ties.sum(-1)>1).sum()),uncertainty_is_expert_calibrated=False)
    return softmax(scores,-1),dict(model='credible-ball minimax distance',fits=fits,
        nominal_gaussian_ball_mass=.95,uncertainty_is_expert_calibrated=False)


def sandwich_uncertainty(c,palette,he):
    means,cov,shape,spacing,lap=field_inputs(c,he,'uncertain')
    transform=whiten(cov);values=c.rgb@transform;means=means@transform
    probes=np.random.default_rng(1209).choice([-1.,1.],(int(np.prod(shape)),16))
    scores=[];reports=[]
    for g in range(c.k):
        mass,sums=class_statistics(c,g,shape,values);matrix=lap+sparse.diags(mass+.015)
        solve=splu(matrix.tocsc()).solve;rhs=sums+.015*means[g];u=solve(rhs)
        residual=sums/np.maximum(mass[:,None],1e-30)-u
        probe_rhs=(np.sqrt(mass)[:,None,None]*probes[:,:,None]*residual[:,None,:]).reshape(len(mass),48)
        propagated=solve(probe_rhs).reshape(len(mass),16,3)
        covariance=.015*np.einsum('npi,npj->nij',propagated,propagated)/16
        upper=np.triu_indices(3);fine=upsample(covariance[:,upper[0],upper[1]],c.shape,shape)
        V=np.empty((*c.shape,3,3));V[...,upper[0],upper[1]]=fine;V[...,upper[1],upper[0]]=fine
        V+=np.eye(3);sign,logdet=np.linalg.slogdet(V)
        if np.any(sign<=0):raise RuntimeError('Sandwich predictive covariance lost positivity')
        delta=values-upsample(u,c.shape,shape)
        score=-.5*(np.einsum('...i,...ij,...j->...',delta,np.linalg.inv(V),delta)+logdet)
        scores.append(score);reports.append(dict(mean_equation_residual=float(np.max(np.abs(matrix@u-rhs))),
            sandwich_probe_equation_residual=float(np.max(np.abs(matrix@propagated.reshape(len(mass),48)-probe_rhs))),
            maximum_sampling_covariance_eigenvalue=float(np.linalg.eigvalsh(covariance).max())))
    return softmax(np.stack(scores,-1),-1),dict(model='residual influence sandwich covariance',
        fits=reports,probes=16,seed=1209,node_aggregation_loses_within_node_residual_variation=True)


def advection_laplacian(guide,spacing):
    lap=guide_laplacian(guide,spacing);intensity=guide.sum(-1)
    dy,dx=np.gradient(intensity,spacing[1],spacing[0]);scale=max(float(np.max(np.hypot(dx,dy))),1e-12)
    vx=-dy/scale;vy=dx/scale;index=np.arange(intensity.size).reshape(intensity.shape)
    rows=[];columns=[];weights=[]
    for axis,velocity in ((0,vy),(1,vx)):
        a,b=[slice(None)]*2,[slice(None)]*2;a[axis]=slice(None,-1);b[axis]=slice(1,None);a,b=tuple(a),tuple(b)
        drift=16*.5*(velocity[a]+velocity[b])/spacing[1-axis]
        i=index[a].ravel();j=index[b].ravel()
        rows.extend([i,j]);columns.extend([j,i]);weights.extend([np.maximum(drift,0).ravel(),np.maximum(-drift,0).ravel()])
    graph=sparse.coo_matrix((np.concatenate(weights),(np.concatenate(rows),np.concatenate(columns))),shape=lap.shape).tocsr()
    operator=lap+sparse.diags(np.asarray(graph.sum(1)).ravel())-graph
    return operator,dict(constant_reproduction_residual=float(np.max(np.abs(operator@np.ones(lap.shape[0])))),
        maximum_stain_gradient=scale,advection_length_um=16.)


def advected_appearance(c,palette,he):
    means,cov,shape,spacing,_=field_inputs(c,he,'uncertain')
    guide=-np.log((cv2.resize(he,shape[::-1],interpolation=cv2.INTER_AREA).astype(float)+1)/256)
    lap,report=advection_laplacian(guide,spacing);inverse=np.linalg.inv(cov);scores=[];fits=[]
    for g in range(c.k):
        mass,sums=class_statistics(c,g,shape);matrix=lap+sparse.diags(mass+.015);rhs=sums+.015*means[g]
        fitted=spsolve(matrix.tocsc(),rhs);delta=c.rgb-upsample(fitted,c.shape,shape)
        scores.append(-.5*np.einsum('...i,ij,...j->...',delta,inverse,delta))
        fits.append(dict(linear_system_max_residual=float(np.max(np.abs(matrix@fitted-rhs)))))
    return softmax(np.stack(scores,-1),-1),dict(report,model='stain contour upwind appearance',fits=fits)


def cross_matrix(v):
    x,y,z=v
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])


def skew_system(lap,mass,sums,means):
    n,k=mass.shape;barycentre=means.mean(0)
    design=np.stack([-cross_matrix(mu-barycentre) for mu in means])
    blocks=np.einsum('ng,gji,gjk->nik',mass,design,design)+.015*np.eye(3)
    residual=sums-mass[...,None]*means[None]
    rhs=np.einsum('gji,ngj->ni',design,residual)
    row=(np.arange(n)[:,None,None]*3+np.arange(3)[None,:,None])+np.zeros((1,1,3),int)
    col=(np.arange(n)[:,None,None]*3+np.arange(3)[None,None,:])+np.zeros((1,3,1),int)
    matrix=sparse.kron(lap,sparse.eye(3),format='csc')+sparse.coo_matrix((blocks.ravel(),(row.ravel(),col.ravel())),shape=(3*n,3*n)).tocsc()
    omega=spsolve(matrix,rhs.ravel()).reshape(n,3)
    return omega,design,dict(linear_system_max_residual=float(np.max(np.abs(matrix@omega.ravel()-rhs.ravel()))),
        shared_sensor_unknowns=3*n,maximum_rotation_vector_norm=float(np.linalg.norm(omega,axis=-1).max()),
        exact_rotation_matrix=False,linear_skew_model=True)


def skew_appearance(c,palette,he):
    means,cov,shape,spacing,lap=field_inputs(c,he,'uncertain')
    transform=whiten(cov);values=c.rgb@transform;means=means@transform
    stats=[class_statistics(c,g,shape,values) for g in range(c.k)]
    mass=np.stack([s[0] for s in stats],-1);sums=np.stack([s[1] for s in stats],1)
    omega,design,report=skew_system(lap,mass,sums,means)
    field=upsample(omega,c.shape,shape);scores=[]
    for g in range(c.k):
        delta=values-(means[g]+field@design[g].T)
        scores.append(-.5*np.sum(delta*delta,-1))
    return softmax(np.stack(scores,-1),-1),dict(report,model='joint local skew sensor appearance')
