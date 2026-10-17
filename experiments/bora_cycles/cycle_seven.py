"""Fifteen independent cycle-seven models; no fine-reference input or I/O."""
from itertools import product

import cv2
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from scipy.special import logsumexp, softmax

from bora.evidence import Evidence, normalize, project_simplex
from . import native
from .methods import correct_coarse, fit_linear, linear_prob
from .cycle_two import colour_nodes
from .cycle_three import balanced
from .cycle_five import field_laplacian, optimizer_report, whitened_bags, noise_objective
from .cycle_six import field_geometry, guide_laplacian


def graph_from_edges(size, i, j, weight):
    i, j, weight = np.concatenate(i), np.concatenate(j), np.concatenate(weight)
    return sparse.coo_matrix((np.r_[weight, weight, -weight, -weight],
        (np.r_[i,j,i,j], np.r_[i,j,j,i])), shape=(size,size)).tocsr()


def orientation_laplacian(guide, spacing, length=16.):
    """Positive directional graph, not an exact discretization of tensor PDE."""
    gy, gx = np.gradient(guide, spacing[1], spacing[0], axis=(0,1))
    jxx = cv2.GaussianBlur(np.sum(gx*gx,-1), (0,0), 1.)
    jyy = cv2.GaussianBlur(np.sum(gy*gy,-1), (0,0), 1.)
    jxy = cv2.GaussianBlur(np.sum(gx*gy,-1), (0,0), 1.)
    trace = jxx+jyy+.0001
    dxx, dyy, dxy = 1-.98*jxx/trace, 1-.98*jyy/trace, -.98*jxy/trace
    h,w = guide.shape[:2]; index=np.arange(h*w).reshape(h,w)
    ii,jj,ww=[],[],[]
    for dy,dx in [(0,1),(1,0),(1,1),(1,-1)]:
        a=(slice(0,h-dy),slice(max(0,-dx),min(w,w-dx)))
        b=(slice(dy,h),slice(max(0,dx),min(w,w+dx)))
        v=np.array([dx*spacing[0],dy*spacing[1]],float); distance=np.linalg.norm(v);v/=distance
        conductivity=(v[0]**2*dxx+v[1]**2*dyy+2*v[0]*v[1]*dxy)
        weight=.5*(conductivity[a]+conductivity[b])*(length/distance)**2
        if dx and dy:weight*=.5
        ii.append(index[a].ravel());jj.append(index[b].ravel());ww.append(weight.ravel())
    return graph_from_edges(h*w,ii,jj,ww)


def appearance_fields(c, he, mode):
    values=np.log((255*c.rgb+1)/256) if mode=='lognormal' else c.rgb.astype(float)
    _,_,means,covariance=fit_linear(c,values)
    precision=np.linalg.inv(covariance); shape,spacing=field_geometry(c)
    if mode in ('orientation','vector','lognormal'):
        if he is None:raise ValueError('Aligned original H&E is required')
        guide=-np.log((cv2.resize(he,shape[::-1],interpolation=cv2.INTER_AREA).astype(float)+1)/256)
        lap=orientation_laplacian(guide,spacing) if mode=='orientation' else guide_laplacian(guide,spacing)
    else:lap=field_laplacian(shape,spacing)
    if mode=='bending':lap=lap.T@lap
    score=[];reports=[]
    for g in range(c.k):
        weight=c.weight*(c.labels==g)
        mass=cv2.resize(weight,shape[::-1],interpolation=cv2.INTER_AREA).ravel()
        sums=cv2.resize(weight[...,None]*values,shape[::-1],interpolation=cv2.INTER_AREA).reshape(-1,3)
        rhs=sums+.015*means[g]; diagonal=mass+.015
        report={}
        if mode=='vector':
            fitted,report=solve_vector_field(lap,diagonal,rhs,covariance)
        elif mode=='variation':
            fitted,report=fused_field(lap,diagonal,rhs)
        elif mode=='poisson':
            fitted,report=poisson_field(lap,mass,sums,means[g])
        else:
            matrix=lap+sparse.diags(diagonal)
            fitted=spsolve(matrix.tocsc(),rhs)
            report['linear_system_max_residual']=float(np.max(np.abs(matrix@fitted-rhs)))
        surface=cv2.resize(fitted.reshape(*shape,3),c.shape[::-1],interpolation=cv2.INTER_LINEAR)
        if mode=='poisson':
            score.append(np.sum(c.rgb*np.log(np.maximum(surface,1e-12))-surface,-1))
        else:
            delta=values-surface
            score.append(-.5*np.einsum('...i,ij,...j->...',delta,precision,delta))
        reports.append(report)
    return softmax(np.stack(score,-1),-1),dict(model=mode+' appearance fields',field_shape_yx=list(shape),
        field_mpp_xy=spacing.tolist(),fits=reports,
        caveat='Appearance regularity and weak coarse labels do not certify anatomy; no output-label smoothing.')


def solve_vector_field(lap,diagonal,rhs,covariance):
    eigen,rotation=np.linalg.eigh(covariance);projected=rhs@rotation
    fitted=np.empty_like(rhs);residual=[]
    for j in range(3):
        matrix=lap*(eigen[j]/eigen.mean())+sparse.diags(diagonal)
        fitted[:,j]=spsolve(matrix.tocsc(),projected[:,j])
        residual.append(float(np.max(np.abs(matrix@fitted[:,j]-projected[:,j]))))
    return fitted@rotation.T,dict(linear_system_max_residual=residual,
        covariance_eigenvalues=eigen.tolist(),relative_mode_regularities=(eigen/eigen.mean()).tolist())


def fused_field(laplacian, diagonal, rhs, iterations=12):
    graph=sparse.triu(-laplacian,1).tocoo()
    i,j=graph.row,graph.col;weight=graph.data
    incidence=sparse.coo_matrix((np.r_[np.ones(len(i)),-np.ones(len(i))],
        (np.r_[np.arange(len(i)),np.arange(len(i))],np.r_[i,j])),shape=(len(i),len(diagonal))).tocsr()
    fitted=spsolve((laplacian+sparse.diags(diagonal)).tocsc(),rhs)
    history=[];epsilon=.01
    def objective(u):
        difference=incidence@u
        return float(.5*np.sum(diagonal[:,None]*u*u)-np.sum(rhs*u)
                     +.02*np.sum(weight*np.sqrt(np.sum(difference*difference,-1)+epsilon**2)))
    history.append(objective(fitted))
    for used in range(iterations):
        difference=incidence@fitted
        factor=.02*weight/np.sqrt(np.sum(difference*difference,-1)+epsilon**2)
        matrix=incidence.T@sparse.diags(factor)@incidence+sparse.diags(diagonal)
        changed=spsolve(matrix.tocsc(),rhs);change=float(np.max(np.abs(changed-fitted)))
        fitted=changed;history.append(objective(fitted))
        if change<1e-6:break
    difference=incidence@fitted
    gradient=diagonal[:,None]*fitted-rhs+incidence.T@(.02*weight[:,None]*difference/
        np.sqrt(np.sum(difference*difference,-1)+epsilon**2)[:,None])
    residual=float(np.max(np.abs(gradient)))
    return fitted,dict(iterations=used+1,objective_path=history,gradient_infinity_norm=residual,
        converged=residual<1e-6,epsilon=epsilon)


def poisson_field(lap,mass,sums,mean,iterations=15):
    target=sums+.015*np.maximum(mean,1e-6);linear=mass+.015
    fitted=np.maximum(target/linear[:,None],1e-6);history=[];residuals=[]
    def value(u):return float(.5*np.sum(u*(lap@u))+np.sum(linear[:,None]*u-target*np.log(u)))
    history.append(value(fitted))
    for used in range(iterations):
        gradient=lap@fitted+linear[:,None]-target/fitted
        if np.max(np.abs(gradient))<1e-6:break
        direction=np.empty_like(fitted)
        for j in range(3):
            matrix=lap+sparse.diags(target[:,j]/fitted[:,j]**2)
            direction[:,j]=spsolve(matrix.tocsc(),-gradient[:,j])
            residuals.append(float(np.max(np.abs(matrix@direction[:,j]+gradient[:,j]))))
        step=1.;negative=direction<0
        if negative.any():step=min(step,float(np.min(-.99*fitted[negative]/direction[negative])))
        directional=float(np.sum(gradient*direction));old=history[-1]
        while step>1e-12 and value(fitted+step*direction)>old+1e-4*step*directional:step*=.5
        if step<=1e-12:break
        fitted+=step*direction;history.append(value(fitted))
    residual=float(np.max(np.abs(lap@fitted+linear[:,None]-target/fitted)))
    return fitted,dict(iterations=used+1,objective_path=history,gradient_infinity_norm=residual,
        converged=residual<1e-6,linear_system_max_residual=max(residuals,default=0.),
        minimum_intensity=float(fitted.min()),quasi_likelihood='Converted RGB is not physically Poisson counts.')


def worst_channel(c,palette,he):
    x,pi,index=colour_nodes(c);colors=np.asarray(palette,float)[c.ids-1]/255
    q,report=native.chebyshev_palette(x,pi,colors)
    return correct_coarse(q[index],c),dict(model='maximum-channel-error palette inverse',**report)


def tile_edges(n,step):
    return np.unique(np.r_[np.rint(np.arange(0,n,step)).astype(int),n])


def heteroscedastic_palette(c,palette,he):
    colors=np.asarray(palette,float)[c.ids-1]/255;prior=c.lattice();q=np.empty_like(prior)
    report=[];yy,xx=tile_edges(c.shape[0],c.tile_y),tile_edges(c.shape[1],c.tile_x)
    for ya,yb in zip(yy[:-1],yy[1:]):
        for xa,xb in zip(xx[:-1],xx[1:]):
            at=(slice(ya,yb),slice(xa,xb));values=c.rgb[at].reshape(-1,3).astype(float)
            centered=values-values.mean(0)
            covariance=centered.T@centered/max(len(values),1)
            covariance=.8*covariance+.2*np.diag(np.diag(covariance))+np.eye(3)/255**2
            metric=np.linalg.inv(covariance);metric*=3/np.trace(metric)
            hessian=colors@metric@colors.T+.012*np.eye(c.k)
            linear=c.rgb[at]@metric@colors.T+.012*prior[at]
            q[at],fitted=native.quadratic_simplex(hessian,linear,prior[at])
            report.append(fitted['max_kkt_residual'])
    return correct_coarse(q,c),dict(model='tile-specific multichannel palette observation metric',
        tiles=len(report),max_kkt_residual=max(report),metric_trace=3.,
        caveat='Within-tile mixtures are not pure noise; local metrics may introduce tile seams.')


def harmonic_coordinates(x,pi,colors):
    nodes=np.vstack([x,colors]);n=len(x);tree=cKDTree(nodes)
    distance,neighbours=tree.query(nodes,k=min(7,len(nodes)),workers=2)
    distance,neighbours=distance[:,1:],neighbours[:,1:]
    scale=max(float(np.median(distance[:,0])),1/24)
    rows=np.repeat(np.arange(len(nodes)),neighbours.shape[1])
    weights=np.exp(-.5*(distance.ravel()/scale)**2)+1e-4
    adjacency=sparse.coo_matrix((weights,(rows,neighbours.ravel())),shape=(len(nodes),len(nodes))).tocsr()
    adjacency=(adjacency+adjacency.T)*.5
    lap=sparse.diags(np.asarray(adjacency.sum(1)).ravel())-adjacency
    matrix=(lap[:n,:n]+.02*sparse.eye(n)).tocsc()
    rhs=.02*pi-lap[:n,n:]@np.eye(len(colors))
    q=spsolve(matrix,rhs)
    residual=float(np.max(np.abs(matrix@q-rhs)))
    if q.min()<-1e-7 or not np.allclose(q.sum(-1),1,atol=1e-6):
        raise ValueError('Harmonic coordinates violate maximum principle')
    return normalize(np.maximum(q,0)),dict(nodes=len(nodes),edges=adjacency.nnz,graph_bandwidth=scale,
        screening=.02,linear_system_max_residual=residual)


def colour_harmonics(c,palette,he):
    x,pi,index=colour_nodes(c);colors=np.asarray(palette,float)[c.ids-1]/255
    q,report=harmonic_coordinates(x,pi,colors)
    return correct_coarse(q[index],c),dict(model='screened colour-cloud Dirichlet coordinates',**report,
        caveat='Occupied colour connectivity is not anatomical connectivity.')


def bag_design(c):
    data=whitened_bags(c)
    if data is None:return None
    bags,labels,_,_,transform=data
    center=bags[...,:3].mean((0,1))
    means=(bags[...,:3].mean(1)-center)@transform
    return bags,labels,transform,center,np.c_[means,np.ones(len(means))]


def linear_result(c,coefficient,transform,center):
    raw=coefficient[:,:3]@transform.T
    return softmax(native.logits(c.rgb,raw,coefficient[:,3]-raw@center),-1)


def robust_hinge_objective(flat,x,labels,weight,k):
    coefficient=flat.reshape(k,x.shape[1]);gradient=np.zeros_like(coefficient);loss=0.
    for own in range(k):
        take=labels==own
        for rival in range(k):
            if rival==own or not take.any():continue
            delta=coefficient[rival]-coefficient[own]
            norm=np.sqrt(np.sum(delta[:3]**2)+1e-12)
            margin=1+x[take]@delta+.2*norm;positive=np.maximum(margin,0)
            derivative=x[take].copy();derivative[:,:3]+=.2*delta[:3]/norm
            g=(2*weight[take]*positive)@derivative
            gradient[rival]+=g;gradient[own]-=g;loss+=float(weight[take]@(positive**2))
    return loss+.0025*np.sum(coefficient**2),(gradient+.005*coefficient).ravel()


def robust_hinge(c,palette,he):
    data=bag_design(c)
    if data is None:return linear_prob(c),dict(missing_known_bag_class=True)
    _,labels,transform,center,x=data
    objective=lambda a:robust_hinge_objective(a,x,labels,balanced(labels,c.k),c.k)
    fit=minimize(objective,np.zeros(c.k*4),jac=True,method='L-BFGS-B',options=dict(maxiter=60,ftol=1e-9))
    return linear_result(c,fit.x.reshape(c.k,4),transform,center),dict(**optimizer_report(fit,objective),
        model='ellipsoidal worst-case colour hinge risk',radius_whitened=.2,norm_squared_epsilon=1e-12)


POWERS=np.array([(0,0,0),(1,0,0),(0,1,0),(0,0,1),(2,0,0),(0,2,0),(0,0,2),(1,1,0),(1,0,1),(0,1,1)])


def gaussian_monomials(mean,powers,variance=.04):
    """Independent Gaussian raw moments through degree four, no quadrature."""
    moments=[np.ones_like(mean),mean,mean**2+variance,
             mean**3+3*variance*mean,mean**4+6*variance*mean**2+3*variance**2]
    return np.stack([np.prod(np.stack([moments[int(p[j])][:,j] for j in range(3)],-1),-1)
                     for p in powers],-1)


def fit_brier(mean,labels,weight,k,ridge=.0025):
    first=gaussian_monomials(mean,POWERS)
    powers=(POWERS[:,None,:]+POWERS[None,:,:]).reshape(-1,3)
    second=gaussian_monomials(mean,powers).reshape(len(mean),10,10)
    matrix=np.einsum('n,nij->ij',weight,second)+ridge*np.eye(10)
    rhs=first.T@(weight[:,None]*np.eye(k)[labels])
    coefficient=np.linalg.solve(matrix,rhs)
    return coefficient,dict(normal_equation_max_residual=float(np.max(np.abs(matrix@coefficient-rhs))),
        normal_condition_number=float(np.linalg.cond(matrix)))


def quadratic_brier(c,palette,he):
    data=bag_design(c)
    if data is None:return linear_prob(c),dict(missing_known_bag_class=True)
    _,labels,transform,center,x=data
    coefficient,report=fit_brier(x[:,:3],labels,balanced(labels,c.k),c.k)
    nodes,_,index=colour_nodes(c)
    # Variance zero evaluates the fitted polynomial at a point; uncertainty
    # enters its training risk, not a second prediction-time blur.
    features=gaussian_monomials((nodes-center)@transform,POWERS,variance=0.)
    return softmax(features@coefficient,-1)[index],dict(model='analytic Gaussian quadratic Brier risk',
        **report,noise_mean_covariance=.04,polynomial_features=10,
        caveat='One-hot tile-mean regression is a surrogate, not a pixel plurality likelihood.')


def gaussian_design(x):
    point,weight=np.polynomial.hermite.hermgauss(3)
    offsets=np.array(list(product(point,repeat=3)))*np.sqrt(2/25)
    quadrature=np.array([a*b*c for a,b,c in product(weight,repeat=3)])/np.pi**1.5
    design=np.repeat(x[:,None],27,axis=1);design[...,:3]+=offsets
    return design,quadrature


def logistic_objective(flat,design,labels,weight,quadrature,k):
    coefficient=flat.reshape(k,design.shape[-1]);logits=design@coefficient.T
    own=logits[np.arange(len(labels)),:,labels]
    weights=weight[:,None]*quadrature[None]
    loss=float(np.sum(weights*(logsumexp(logits,-1)-own)))
    residual=softmax(logits,-1);residual[np.arange(len(labels)),:,labels]-=1
    gradient=np.einsum('nq,nqk,nqd->kd',weights,residual,design)
    return loss+.0025*np.sum(coefficient**2),(gradient+.005*coefficient).ravel()


def gaussian_logistic(c,palette,he):
    data=bag_design(c)
    if data is None:return linear_prob(c),dict(missing_known_bag_class=True)
    _,labels,transform,center,x=data;design,quadrature=gaussian_design(x)
    objective=lambda a:logistic_objective(a,design,labels,balanced(labels,c.k),quadrature,c.k)
    fit=minimize(objective,np.zeros(c.k*4),jac=True,method='L-BFGS-B',options=dict(maxiter=60,ftol=1e-9))
    return linear_result(c,fit.x.reshape(c.k,4),transform,center),dict(**optimizer_report(fit,objective),
        model='joint multiclass Gaussian logistic risk',quadrature_points=27,
        caveat='Finite cubature; a tile mean is not the observed pixel count winner.')


def empirical_exponential_objective(flat,design,labels,weight,k):
    coefficient=flat.reshape(k,design.shape[-1]);logits=design@coefficient.T
    own=logits[np.arange(len(labels)),:,labels]
    logterms=logits-own[...,None]+np.log(weight)[:,None,None]-np.log(design.shape[1])
    logterms[np.arange(len(labels)),:,labels]=-np.inf
    logrisk=logsumexp(logterms);ridge=.0025*np.sum(coefficient**2)
    value=np.logaddexp(logrisk,np.log(ridge) if ridge>0 else -np.inf)
    derivative=np.exp(logterms-value)
    derivative[np.arange(len(labels)),:,labels]=-derivative.sum(-1)
    gradient=np.einsum('nqk,nqd->kd',derivative,design)+.005*coefficient*np.exp(-value)
    return float(value),gradient.ravel()


def empirical_noise(c,palette,he):
    data=bag_design(c)
    if data is None:return linear_prob(c),dict(missing_known_bag_class=True)
    bags,labels,transform,center,x=data
    residual=(bags[...,:3]-bags[...,:3].mean(1,keepdims=True))@transform/5
    design=np.repeat(x[:,None],25,axis=1);design[...,:3]+=residual
    objective=lambda a:empirical_exponential_objective(a,design,labels,balanced(labels,c.k),c.k)
    fit=minimize(objective,np.zeros(c.k*4),jac=True,method='L-BFGS-B',options=dict(maxiter=60,ftol=1e-9))
    return linear_result(c,fit.x.reshape(c.k,4),transform,center),dict(**optimizer_report(fit,objective),
        model='empirical within-tile residual exponential risk',residual_points=25,
        objective_reporting='log of positive risk plus ridge; same minimizer, not claimed convex after logarithm',
        caveat='Empirical residuals divided by five approximate, not identify, the correlated tile-mean noise law.')


def class_log_risks(flat,x,labels,k):
    coefficient=flat.reshape(k,x.shape[1]);values=[];gradients=[]
    for own in range(k):
        take=labels==own;delta=coefficient-coefficient[own]
        terms=x[take]@delta.T+.5*np.sum(delta[:,:3]**2,-1)[None]/25
        terms[:,own]=-np.inf;logrisk=logsumexp(terms)-np.log(take.sum())
        factor=np.exp(terms-logsumexp(terms));gradient=np.zeros_like(coefficient)
        for rival in range(k):
            if rival==own:continue
            derivative=x[take].copy();derivative[:,:3]+=delta[rival,:3]/25
            g=factor[:,rival]@derivative;gradient[rival]+=g;gradient[own]-=g
        values.append(float(logrisk));gradients.append(gradient.ravel())
    return np.asarray(values),np.asarray(gradients)


def minimax_risk(c,palette,he):
    data=bag_design(c)
    if data is None:return linear_prob(c),dict(missing_known_bag_class=True)
    _,labels,transform,center,x=data
    def objective(a):return a[-1]+.0025*np.sum(a[:-1]**2),np.r_[.005*a[:-1],1.]
    def constraints(a):return a[-1]-class_log_risks(a[:-1],x,labels,c.k)[0]
    def jacobian(a):return np.c_[-class_log_risks(a[:-1],x,labels,c.k)[1],np.ones(c.k)]
    initial=np.r_[np.zeros(c.k*4),np.log(c.k-1)]
    fit=minimize(objective,initial,jac=True,method='SLSQP',
        constraints=[dict(type='ineq',fun=constraints,jac=jacobian)],options=dict(maxiter=80,ftol=1e-9))
    return linear_result(c,fit.x[:-1].reshape(c.k,4),transform,center),dict(
        model='worst-class Gaussian log-exponential risk',optimizer_success=bool(fit.success),
        iterations=int(fit.nit),termination=str(fit.message),objective=float(fit.fun),
        max_constraint_violation=float(max(0.,-constraints(fit.x).min())),
        class_log_risks=class_log_risks(fit.x[:-1],x,labels,c.k)[0].tolist(),
        caveat='A noisy rare class can dominate the fit; objective gradient alone is not a KKT residual.')


def nuclear_prox(matrix,step,penalty=.02):
    result=matrix.copy();u,s,vt=np.linalg.svd(result[:,:3],full_matrices=False)
    result[:,:3]=(u*np.maximum(s-step*penalty,0))@vt
    return result


def fit_nuclear(x,labels,weight,k,iterations=100):
    coefficient=np.zeros((k,4));step=1.;history=[];residual=np.inf
    def objective(a):
        smooth,gradient=noise_objective(a.ravel(),x,labels,weight,k)
        return smooth+.02*np.linalg.svd(a[:,:3],compute_uv=False).sum(),smooth,gradient.reshape(k,4)
    value,smooth,gradient=objective(coefficient);history.append(float(value))
    for used in range(iterations):
        while True:
            changed=nuclear_prox(coefficient-step*gradient,step)
            newvalue,newsmooth,newgradient=objective(changed);delta=changed-coefficient
            upper=smooth+np.sum(gradient*delta)+np.sum(delta*delta)/(2*step)
            if np.isfinite(newvalue) and newsmooth<=upper+1e-12:break
            step*=.5
            if step<1e-14:raise RuntimeError('Nuclear-risk line search failed')
        residual=float(np.max(np.abs(delta))/step)
        coefficient=changed;value,smooth,gradient=newvalue,newsmooth,newgradient
        history.append(float(value))
        if residual<1e-6:break
        step=min(step*1.2,1.)
    # Report the final point's mapping, not a previous iterate's residual.
    mapping=nuclear_prox(coefficient-step*gradient,step)
    residual=float(np.max(np.abs(coefficient-mapping))/step)
    return coefficient,dict(iterations=used+1,objective_path=history,proximal_gradient_residual=residual,
        converged=residual<1e-6,nuclear_penalty=.02,
        contrast_singular_values=np.linalg.svd(coefficient[:,:3],compute_uv=False).tolist())


def shared_contrasts(c,palette,he):
    data=bag_design(c)
    if data is None:return linear_prob(c),dict(missing_known_bag_class=True)
    _,labels,transform,center,x=data
    coefficient,report=fit_nuclear(x,labels,balanced(labels,c.k),c.k)
    return linear_result(c,coefficient,transform,center),dict(model='trace-norm Gaussian exponential discrimination',
        **report,caveat='A low-rank colour geometry can merge a class requiring its own contrast direction.')


IMPLEMENTATIONS=dict(K01=lambda c,p,h:appearance_fields(c,h,'orientation'),
    K02=lambda c,p,h:appearance_fields(c,h,'vector'),K03=lambda c,p,h:appearance_fields(c,h,'lognormal'),
    K04=lambda c,p,h:appearance_fields(c,h,'variation'),K05=lambda c,p,h:appearance_fields(c,h,'bending'),
    K06=lambda c,p,h:appearance_fields(c,h,'poisson'),K07=worst_channel,K08=heteroscedastic_palette,
    K09=colour_harmonics,K10=robust_hinge,K11=quadratic_brier,K12=gaussian_logistic,
    K13=empirical_noise,K14=minimax_risk,K15=shared_contrasts)


def predict(key,rgb,coarse,support,mpp_xy,palette,he=None):
    from .registry import CYCLE_SEVEN
    if key not in IMPLEMENTATIONS:raise ValueError('Unimplemented cycle-seven child')
    c=Evidence(rgb,coarse,support,mpp_xy)
    if not c.support.any():raise ValueError('Empty tissue support')
    if c.k==1:q,detail=np.ones((*c.shape,1),np.float32),dict(single_class=True)
    else:q,detail=IMPLEMENTATIONS[key](c,palette,he)
    if q.shape!=(*c.shape,c.k) or not np.isfinite(q).all() or q.min()<-1e-6 or not np.allclose(q.sum(-1),1,atol=1e-5):
        raise ValueError('Invalid cycle-seven posterior')
    labels=c.ids[q.argmax(-1)].astype(np.uint8)
    detail.update(analysis_mpp_xy=c.mpp.tolist(),analysis_shape_yx=list(c.shape),fine_reference_used=False,
        hard_anchors=False,fixed_margin=False,candidate=key,parent=CYCLE_SEVEN[key].parent,mechanism=CYCLE_SEVEN[key].mechanism)
    return labels,detail
