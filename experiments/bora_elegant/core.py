"""Whole-field evidence models. Prediction code has no reference-mask I/O."""
import time
import cv2
import numpy as np
from scipy import ndimage as ndi, sparse
from scipy.special import softmax
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from skimage.segmentation import slic
from bora.evidence import Evidence,resize,normalize,project_simplex,unmix,confidence_correct,linear,amortized,predict_evidence

TITLES={
 'E01':'Robust full-covariance evidence', 'E02':'Bayesian colour lattice',
 'E03':'Guided posterior regression', 'E04':'Screened region diffusion',
 'E05':'Nonlocal evidence graph', 'E06':'Tile-rank constraints',
 'E07':'Colour-simplex inversion', 'E08':'Confidence-aware simplex correction',
 'E09':'Shared-covariance discriminant', 'E10':'Robust mixture discrimination',
 'E11':'Amortized simplex correction', 'E12':'Majority-consistent calibration',
 'E13':'Quadratic evidence regression', 'E14':'Uncertainty-gated guided evidence',
 'E15':'Coarse-noise deconvolution', 'E16':'Information-geometric diffusion',
}








def robust(c):
    costs=[]
    for i in range(c.k):
        mask=c.known&(c.labels==i);x=c.rgb[mask];w=c.weight[mask]
        if len(x)>6000:
            indices=np.linspace(0,len(x)-1,6000,dtype=int);x=x[indices];w=w[indices]
        if len(x)==0:costs.append(np.full(c.shape,50,np.float32));continue
        for iteration in range(3):
            mean=np.average(x,weights=w,axis=0);v=x-mean
            cov=(v.T*w)@v/max(w.sum(),1e-8)
            cov=.85*cov+.15*np.diag(np.diag(cov))+np.eye(3)*.0025
            inv=np.linalg.inv(cov)
            if iteration<2:
                d=np.einsum('ni,ij,nj->n',v,inv,v)
                w=c.weight[mask][indices] if len(c.rgb[mask])>6000 else c.weight[mask].copy()
                w=w*np.minimum(1.,np.sqrt(7./np.maximum(d,1e-8)))
        delta=c.rgb-mean;distance=np.einsum('...i,ij,...j->...',delta,inv,delta)
        costs.append(4*np.log1p(distance/5)+.5*np.linalg.slogdet(cov)[1])
    return softmax(-np.stack(costs,-1),axis=-1).astype(np.float32)


def box(a,r):return cv2.boxFilter(a,-1,(2*r+1,2*r+1),normalize=True,borderType=cv2.BORDER_REFLECT)


def guided(guide,q,radius,epsilon=.003):
    r=max(1,int(radius));mean=box(guide,r);shape=guide.shape[:2]
    cov=np.empty((*shape,3,3),np.float32)
    for i in range(3):
        for j in range(3):cov[...,i,j]=box(guide[...,i]*guide[...,j],r)-mean[...,i]*mean[...,j]+(epsilon if i==j else 0)
    inverse=np.linalg.inv(cov);out=[]
    for j in range(q.shape[-1]):
        mp=box(q[...,j],r)
        cp=np.stack([box(guide[...,i]*q[...,j],r)-mean[...,i]*mp for i in range(3)],-1)
        a=np.einsum('...ij,...j->...i',inverse,cp);b=mp-(a*mean).sum(-1)
        out.append((box(a,r)*guide).sum(-1)+box(b,r))
    return normalize(np.stack(out,-1))


def region_mean(a,regions,n):
    flat=regions.ravel();count=np.bincount(flat,minlength=n)
    if a.ndim==2:return np.bincount(flat,weights=a.ravel(),minlength=n)/np.maximum(count,1)
    return np.stack([region_mean(a[...,i],regions,n) for i in range(a.shape[-1])],-1)


def diffusion(c,nonlocal_edges=False,spacing=14.,coarse_strength=.10):
    q=c.lattice();n_segments=max(64,min(4096,round(np.prod(c.shape)*c.mpp.prod()/spacing**2)))
    regions=slic(c.rgb,n_segments=n_segments,compactness=3,sigma=0,start_label=0,channel_axis=-1,enforce_connectivity=True)
    n=int(regions.max())+1;means=region_mean(c.rgb,regions,n);prior=region_mean(q,regions,n)
    valid=region_mean(c.known.astype(np.float32),regions,n)
    obs=np.stack([region_mean(c.weight*(c.labels==i),regions,n) for i in range(c.k)],-1)
    obs=normalize(obs)
    pair=np.concatenate([np.stack([regions[:-1].ravel(),regions[1:].ravel()],1),np.stack([regions[:,:-1].ravel(),regions[:,1:].ravel()],1)])
    pair,count=np.unique(np.sort(pair[pair[:,0]!=pair[:,1]],axis=1),axis=0,return_counts=True)
    a,b=pair.T;area=np.bincount(regions.ravel(),minlength=n)
    weights=count/np.sqrt(np.sqrt(area[a]*area[b]))*np.exp(-np.mean((means[a]-means[b])**2,axis=1)/(.10**2))+.0001
    if nonlocal_edges:
        yy,xx=np.indices(c.shape);xy=region_mean(np.stack([yy/c.shape[0],xx/c.shape[1]],-1),regions,n)
        feats=np.concatenate([means/.12,xy*.7],-1);dist,near=cKDTree(feats).query(feats,k=min(9,n))
        aa=np.repeat(np.arange(n),near.shape[1]-1);bb=near[:,1:].ravel();ww=.15*np.exp(-dist[:,1:].ravel()**2/2)
        a=np.r_[a,aa];b=np.r_[b,bb];weights=np.r_[weights,ww]
    graph=sparse.coo_matrix((np.r_[weights,weights],(np.r_[a,b],np.r_[b,a])),shape=(n,n)).tocsr()
    degree=np.asarray(graph.sum(1)).ravel();scale=max(float(np.median(degree)),.001)
    lam=np.full(n,.20*scale);seed=coarse_strength*scale*valid
    system=sparse.diags(degree+lam+seed)-graph
    rhs=lam[:,None]*prior+seed[:,None]*obs
    solved=normalize(spsolve(system.tocsc(),rhs));c.details.update(regions=n,soft_coarse_strength=coarse_strength,hard_anchors=False)
    return solved[regions]


def tile_rank(c):
    q=c.lattice();logits=np.log(np.maximum(q,1e-7));bias=np.zeros_like(q)
    h,w=c.coarse.shape
    # Bounds are defined in the original working-pixel tile coordinates.
    tile_y=c.tile_y;tile_x=c.tile_x
    blocks=[]
    for y in np.arange(0,h,tile_y):
        for x in np.arange(0,w,tile_x):
            ys=slice(round(y),min(h,round(y+tile_y)));xs=slice(round(x),min(w,round(x+tile_x)))
            known=c.known[ys,xs]
            if not known.any():continue
            ids,counts=np.unique(c.labels[ys,xs][known],return_counts=True);label=int(ids[counts.argmax()])
            blocks.append((ys,xs,label))
    for _ in range(12):
        current=softmax(logits+bias,axis=-1)
        for ys,xs,label in blocks:
            means=current[ys,xs].mean((0,1));other=int(means.argmax())
            if means[other]>means[label]+.005:
                step=min(.5,.4*np.log((means[other]+1e-5)/(means[label]+1e-5)))
                bias[ys,xs,label]+=step;bias[ys,xs,other]-=step
        bias=cv2.GaussianBlur(bias,(0,0),1.5/c.um)
        if bias.ndim==2:bias=bias[...,None]
    c.details.update(constraint='coarse winner should have highest mean posterior, not a fixed per-pixel label',iterations=12)
    return softmax(logits+bias,axis=-1).astype(np.float32)


def predict(key,rgb,coarse,support,mpp_xy,palette=None):
    if key in ("E09","E11"):
        return predict_evidence(rgb,coarse,support,mpp_xy,method="linear" if key=="E09" else "simplex",palette=palette)
    c=Evidence(rgb,coarse,support,mpp_xy)
    if c.k==1:return c.ids[np.zeros(c.shape,np.int32)],dict(single_class=True)
    if key=='E01':q=robust(c)
    elif key=='E02':q=c.lattice()
    elif key=='E03':q=guided(c.rgb,c.lattice(),4/c.um)
    elif key=='E04':q=diffusion(c)
    elif key=='E05':q=diffusion(c,True)
    elif key=='E06':q=tile_rank(c)
    elif key=='E07':q=unmix(c,palette)
    elif key=='E08':q=confidence_correct(c,palette)
    elif key=='E09':q=linear(c)
    elif key=='E10':q=mixture(c)
    elif key=='E11':q=amortized(c,palette)
    elif key=='E12':q=bag_calibrate(c)
    elif key=='E13':q=quadratic(c)
    elif key=='E14':q=guided_uncertainty(c,palette)
    elif key=='E15':q=noise_deconvolution(c,palette)
    elif key=='E16':q=semantic_diffusion(c,palette)
    else:raise ValueError(key)
    return c.finish(q),dict(c.details,analysis_shape_yx=list(c.shape),analysis_mpp_xy=c.mpp.tolist(),fixed_margin=False,hard_anchors=False)










def mixture(c):
    from sklearn.mixture import GaussianMixture
    logs=[]
    for i in range(c.k):
        x=c.rgb[c.known&(c.labels==i)]
        if len(x)>2400:x=x[np.linspace(0,len(x)-1,2400,dtype=int)]
        if len(x)<4:logs.append(np.full(c.shape,-50.));continue
        model=GaussianMixture(n_components=min(3,max(1,len(x)//100)),covariance_type='full',reg_covar=.0025,max_iter=35,n_init=1,random_state=430).fit(x)
        logs.append(model.score_samples(c.rgb.reshape(-1,3)).reshape(c.shape))
    return softmax(np.stack(logs,-1),axis=-1).astype(np.float32)




def bag_calibrate(c):
    from scipy.optimize import minimize
    q=linear(c);values=[];labels=[]
    h,w=c.shape
    for y in np.arange(0,h,c.tile_y):
        for x in np.arange(0,w,c.tile_x):
            yy=np.minimum(h-1,np.maximum(0,np.rint(y+(np.arange(7)+.5)*c.tile_y/7).astype(int)))
            xx=np.minimum(w-1,np.maximum(0,np.rint(x+(np.arange(7)+.5)*c.tile_x/7).astype(int)))
            known=c.known[yy[:,None],xx[None,:]]
            if known.mean()<.6:continue
            ls=c.labels[yy[:,None],xx[None,:]][known];hist=np.bincount(ls,minlength=c.k);label=int(hist.argmax())
            if not hist[label]:continue
            values.append(np.log(np.maximum(q[yy[:,None],xx[None,:]].reshape(-1,c.k),1e-6)));labels.append(label)
    if not values:return q
    values=np.array(values);labels=np.array(labels);weights=1/np.maximum(np.bincount(labels,minlength=c.k)[labels],1);weights/=weights.sum();target=np.eye(c.k)[labels]
    def objective(bias):
        p=softmax(values+bias,axis=-1);m=p.mean(1);jac=np.array([np.diag(v) for v in m])-np.einsum('nsi,nsj->nij',p,p)/p.shape[1]
        own=(m*target).sum(1);d=m-own[:,None]+.02;d[np.arange(len(labels)),labels]=-1
        active=1/(1+np.exp(-np.clip(d/.04,-30,30)));active[np.arange(len(labels)),labels]=0
        loss=float(np.sum(weights[:,None]*.04*np.logaddexp(0,d/.04))+.05*(bias*bias).sum())
        grad=np.einsum('n,nj,njl->l',weights,active,jac-jac[np.arange(len(labels)),labels][:,None,:])+.1*bias
        return loss,grad
    fit=minimize(objective,np.zeros(c.k),jac=True,method='L-BFGS-B',options={'maxiter':35,'ftol':1e-8})
    c.details.update(calibration_bias=fit.x.tolist(),training_bags=len(values),optimization_success=bool(fit.success),bag_constraint='mean posterior of coarse class exceeds competing class; soft penalty')
    return softmax(np.log(np.maximum(q,1e-7))+fit.x,axis=-1).astype(np.float32)


def quadratic(c):
    def lift(x):
        r,g,b=x.T
        return np.stack([np.ones(len(x)),r,g,b,r*r,g*g,b*b,r*g,r*b,g*b],-1)
    where=np.flatnonzero(c.known.ravel())
    if len(where)>12000:where=where[np.linspace(0,len(where)-1,12000,dtype=int)]
    x=lift(c.rgb.reshape(-1,3)[where]);labels=c.labels.ravel()[where];weights=c.weight.ravel()[where]
    for i in range(c.k):
        m=labels==i;weights[m]/=max(weights[m].sum(),1e-8)
    gram=(x.T*weights)@x;ridge=.005*np.trace(gram)/len(gram)
    coef=np.linalg.solve(gram+np.eye(len(gram))*ridge,(x.T*weights)@np.eye(c.k)[labels])
    logits=lift(c.rgb.reshape(-1,3))@coef
    c.details.update(feature_dimension=10,ridge=float(ridge),training_pixels=len(where))
    return softmax(logits.reshape(*c.shape,c.k)*4,axis=-1).astype(np.float32)


def guided_uncertainty(c,palette):
    q=amortized(c,palette);filtered=guided(c.rgb,q,6/c.um)
    gap=np.sort(q,axis=-1)[...,-1]-np.sort(q,axis=-1)[...,-2]
    mix=(1-gap)**2
    return normalize(q*(1-mix[...,None])+filtered*mix[...,None])


def noise_deconvolution(c,palette):
    q=amortized(c,palette);transition=[]
    for i in range(c.k):
        mask=c.known&(c.labels==i);transition.append(np.average(q[mask],weights=c.weight[mask],axis=0))
    transition=.65*np.eye(c.k)+.35*np.array(transition)
    corrected=project_simplex(q@np.linalg.inv(transition))
    c.details.update(noise_transition=transition.tolist(),identity_regularizer=.65)
    return normalize(corrected)


def semantic_diffusion(c,palette):
    q=amortized(c,palette)
    for _ in range(2):
        logs=np.log(np.maximum(q,1e-7));total=np.zeros_like(q);mass=np.zeros(c.shape,np.float32)
        for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
            y=np.clip(np.arange(c.shape[0])+dy,0,c.shape[0]-1);x=np.clip(np.arange(c.shape[1])+dx,0,c.shape[1]-1)
            other=q[y[:,None],x[None,:]];middle=(q+other)/2
            js=.5*np.sum(q*np.log(np.maximum(q,1e-7)/np.maximum(middle,1e-7))+other*np.log(np.maximum(other,1e-7)/np.maximum(middle,1e-7)),axis=-1)
            certainty=1+np.sum(other*np.log(np.maximum(other,1e-7)),axis=-1)/np.log(c.k)
            weight=np.exp(-js/.04)*(.1+certainty)
            total+=weight[...,None]*np.log(np.maximum(other,1e-7));mass+=weight
        consensus=total/np.maximum(mass[...,None],1e-7)
        uncertainty=-np.sum(q*logs,axis=-1)/np.log(c.k)
        q=softmax(logs+.35*uncertainty[...,None]*(consensus-logs),axis=-1).astype(np.float32)
    c.details.update(iterations=2,metric='Jensen-Shannon divergence',pooling='confidence-weighted logarithmic opinion pool')
    return q
