"""Whole-field refinement from fixed evidence RGB and original coarse labels.

The linear method uses a shared covariance model. The simplex method recovers
probabilities from the known conversion palette; with more than four dependent
classes, coarse-trained colour likelihoods regularize an underdetermined inverse.
Neither method fixes boundary margins or pixel anchors. This API operates on
ROIs, with explicit pixel spacing and externally supplied tissue support.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.special import softmax


def resize(a,shape,nearest=False):
    return cv2.resize(a,tuple(shape[::-1]),interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)


def normalize(q):
    q=np.maximum(q,1e-8);return (q/q.sum(-1,keepdims=True)).astype(np.float32)


class Evidence:
    def __init__(self,rgb,coarse,support,mpp_xy,target_um=1.,tile_size=90):
        h,w=coarse.shape
        self.shape=(max(8,round(h*mpp_xy[1]/target_um)),max(8,round(w*mpp_xy[0]/target_um)))
        self.mpp=np.array([w*mpp_xy[0]/self.shape[1],h*mpp_xy[1]/self.shape[0]])
        self.um=float(np.sqrt(self.mpp.prod()));self.tile_y=tile_size*self.shape[0]/h;self.tile_x=tile_size*self.shape[1]/w
        self.rgb=resize(rgb,self.shape).astype(np.float32)/255
        self.coarse=resize(coarse,self.shape,True);self.support=resize(support.astype(np.uint8),self.shape,True)>0
        self.ids=np.unique(coarse[coarse>0]);self.k=len(self.ids)
        self.labels=np.searchsorted(self.ids,self.coarse).clip(0,self.k-1)
        self.known=(self.coarse>0)&self.support
        # Continuous weight, positive everywhere: no boundary exclusion or frozen pixels.
        yy=(np.arange(self.shape[0])+.5)*h/self.shape[0]
        xx=(np.arange(self.shape[1])+.5)*w/self.shape[1]
        self.weight=(.15+.85*np.sin(np.pi*(yy%tile_size)/tile_size)[:,None]**2*np.sin(np.pi*(xx%tile_size)/tile_size)[None,:]**2)*self.known
        self.cache={};self.details={}

    def finish(self,q):
        if q.shape!=(*self.shape,self.k) or not np.isfinite(q).all():raise ValueError('Invalid posterior')
        return self.ids[q.argmax(-1)].astype(np.uint8)

    def lattice(self,bins=20,sigma=.85):
        key=('lattice',bins,sigma)
        if key in self.cache:return self.cache[key]
        index=np.minimum((self.rgb*bins).astype(np.int32),bins-1)
        flat=(index[...,0]*bins+index[...,1])*bins+index[...,2]
        logits=[]
        for i in range(self.k):
            weights=self.weight*(self.labels==i)
            hist=np.bincount(flat.ravel(),weights=weights.ravel(),minlength=bins**3).reshape((bins,)*3)
            hist=ndi.gaussian_filter(hist,sigma=sigma,mode='nearest')
            hist=(hist+max(hist.sum(),1)*1e-7);hist/=hist.sum()
            xyz=np.moveaxis(np.clip(self.rgb*bins-.5,0,bins-1),-1,0)
            likelihood=ndi.map_coordinates(hist,xyz,order=1,mode='nearest')
            logits.append(np.log(np.maximum(likelihood,1e-12)))
        q=softmax(np.stack(logits,-1)*.8,axis=-1).astype(np.float32)
        self.cache[key]=q;return q


def project_simplex(x):
    ordered=np.sort(x,axis=-1)[...,::-1];css=np.cumsum(ordered,axis=-1)-1
    rho=(ordered-css/np.arange(1,x.shape[-1]+1)>0).sum(-1)-1
    threshold=np.take_along_axis(css,np.maximum(rho,0)[...,None],axis=-1)[...,0]/(rho+1)
    return np.maximum(x-threshold[...,None],0)


def unmix(c,palette):
    if palette is None:raise ValueError('Known conversion palette required')
    colours=np.asarray(palette,np.float32)[c.ids-1]/255
    affine=np.c_[colours,np.ones(c.k)].T
    flat=np.c_[c.rgb.reshape(-1,3),np.ones(np.prod(c.shape))]
    initial=project_simplex(flat@np.linalg.pinv(affine).T).astype(np.float32).reshape(*c.shape,c.k)
    if np.linalg.matrix_rank(affine)==c.k:
        q=initial;c.details.update(simplex_identifiable=True,palette_rank=int(np.linalg.matrix_rank(affine)),iterations=0)
    else:
        # RGB has only three independent colour coordinates. Resolve ambiguity
        # using coarse-trained likelihood, with a strictly convex quadratic prior.
        prior=c.lattice();q=prior.copy();ridge=.012
        step=.9/(float(np.linalg.norm(colours,2)**2)+ridge)
        for _ in range(18):
            error=q@colours-c.rgb
            q=project_simplex(q-step*(error@colours.T+ridge*(q-prior)))
        c.details.update(simplex_identifiable=False,palette_rank=int(np.linalg.matrix_rank(affine)),iterations=18,prior_regularizer=ridge)
    return normalize(q)


def confidence_correct(c,palette):
    q=unmix(c,palette);corrected=q.copy()
    # Coarse observations only influence uncertain pixels; confident colour
    # evidence can contradict any coarse tile. No pixel is frozen.
    confidence=(np.partition(q,-2,axis=-1)[...,-1]-np.partition(q,-2,axis=-1)[...,-2])
    strength=.35*(1-confidence)**2*c.weight
    for i in range(c.k):corrected[...,i]+=strength*(c.labels==i)
    return normalize(corrected)


def linear(c):
    means=[];covariance=np.zeros((3,3));mass=0
    for i in range(c.k):
        mask=c.known&(c.labels==i);x=c.rgb[mask];w=c.weight[mask]
        if not len(x):means.append(np.zeros(3));continue
        mean=np.average(x,axis=0,weights=w);means.append(mean);delta=x-mean
        covariance+=(delta.T*w)@delta;mass+=w.sum()
    covariance/=max(mass,1e-8);covariance=.8*covariance+.2*np.diag(np.diag(covariance))+np.eye(3)*.0025
    inv=np.linalg.inv(covariance);means=np.array(means);coef=means@inv
    logits=c.rgb@coef.T-.5*np.sum(coef*means,axis=1)
    return softmax(logits,axis=-1).astype(np.float32)


def amortized(c,palette):
    colours=np.asarray(palette,np.float32)[c.ids-1]/255
    if np.linalg.matrix_rank(np.c_[colours,np.ones(c.k)].T)==c.k:return confidence_correct(c,palette)
    prior=c.lattice();bins=24
    index=np.minimum((c.rgb*bins).astype(np.int32),bins-1)
    index=((index[...,0]*bins+index[...,1])*bins+index[...,2]);n=bins**3
    count=np.bincount(index.ravel(),minlength=n);valid=count>0
    rgb=np.stack([np.bincount(index.ravel(),weights=c.rgb[...,i].ravel(),minlength=n)/np.maximum(count,1) for i in range(3)],-1)[valid]
    p=np.stack([np.bincount(index.ravel(),weights=prior[...,i].ravel(),minlength=n)/np.maximum(count,1) for i in range(c.k)],-1)[valid];q=p.copy();ridge=.012
    step=.9/(float(np.linalg.norm(colours,2)**2)+ridge)
    for _ in range(18):q=project_simplex(q-step*((q@colours-rgb)@colours.T+ridge*(q-p)))
    table=np.zeros((n,c.k),np.float32);table[valid]=q;q=normalize(table[index])
    gap=np.sort(q,axis=-1)[...,-1]-np.sort(q,axis=-1)[...,-2];strength=.35*(1-gap)**2*c.weight
    for i in range(c.k):q[...,i]+=strength*(c.labels==i)
    c.details.update(occupied_colour_nodes=int(valid.sum()),quantization_bins=bins,iterations=18,simplex_identifiable=False,prior_regularizer=ridge)
    return normalize(q)


def predict_evidence(rgb,coarse,tissue,mpp_xy,*,method="simplex",palette=None,tile_size=90):
    """Return analysis-grid labels and diagnostics, without background clipping."""
    rgb=np.asarray(rgb);coarse=np.asarray(coarse);tissue=np.asarray(tissue)
    spacing=np.asarray(mpp_xy,float)
    if rgb.dtype!=np.uint8 or rgb.shape!=(*coarse.shape,3) or coarse.ndim!=2 or tissue.shape!=coarse.shape:
        raise ValueError("Expected uint8 RGB and aligned two-dimensional coarse/tissue masks")
    if spacing.shape!=(2,) or not np.isfinite(spacing).all() or (spacing<=0).any():raise ValueError("mpp_xy must contain two positive pixel spacings")
    if not np.isfinite(coarse).all() or (coarse<0).any() or (coarse>255).any() or not np.all(coarse==np.floor(coarse)) or not np.any(coarse>0):raise ValueError("Coarse labels must be integer IDs from 0 to 255, with at least one known class")
    if not np.isfinite(tile_size) or tile_size<=0:raise ValueError("tile_size must be positive")
    if method not in ("simplex","linear"):raise ValueError("Unknown evidence method")
    if method=="simplex" and len(np.unique(coarse[coarse>0]))>1:
        colours=np.asarray(palette,float)
        if colours.ndim!=2 or colours.shape[1]!=3 or colours.shape[0]<coarse.max() or not np.isfinite(colours).all() or (colours<0).any() or (colours>255).any():raise ValueError("Simplex method needs the RGB conversion palette indexed by label ID minus one")
    c=Evidence(rgb,coarse,tissue.astype(bool),spacing,tile_size=tile_size)
    if c.k==1:labels=np.full(c.shape,c.ids[0],np.uint8)
    else:labels=c.finish(linear(c) if method=="linear" else amortized(c,palette))
    return labels,dict(c.details,method=method,analysis_shape_yx=list(c.shape),analysis_mpp_xy=c.mpp.tolist(),fixed_margin=False,hard_anchors=False)


def refine_evidence(rgb,coarse,tissue,mpp_xy,*,method="simplex",palette=None,tile_size=90):
    """Return a mask at input geometry, assigning all tissue and removing glass."""
    labels,details=predict_evidence(rgb,coarse,tissue,mpp_xy,method=method,palette=palette,tile_size=tile_size)
    result=resize(labels,np.asarray(coarse).shape,True);result[~np.asarray(tissue,bool)]=0
    return result,details
