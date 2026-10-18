"""Bora-AF: physically calibrated, H&E-guided multiclass appearance fields."""
from dataclasses import dataclass
import math
import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.special import softmax
from .evidence import Evidence


@dataclass(frozen=True)
class AppearanceConfig:
    analysis_um: float=1.
    field_um: float=8.
    length_um: float=16.
    prior_strength: float=.015
    od_contrast: float=.1
    conductance_floor: float=.02


def class_statistics(c):
    xs=[];labels=[];weights=[]
    for g in range(c.k):
        at=np.flatnonzero(c.known.ravel()&(c.labels.ravel()==g)&(c.weight.ravel()>0))
        if len(at)>max(1,16000//c.k):at=at[np.linspace(0,len(at)-1,max(1,16000//c.k),dtype=int)]
        if not len(at):raise ValueError('A coarse class has no supported tissue observations')
        xs.append(c.rgb.reshape(-1,3)[at]);labels.append(np.full(len(at),g));weights.append(c.weight.ravel()[at])
    x=np.concatenate(xs);labels=np.concatenate(labels);weights=np.concatenate(weights)
    scale=np.maximum(np.std(x,axis=0),.05);x=x/scale
    means=[];covariance=np.zeros((3,3));mass=0.
    for g in range(c.k):
        at=labels==g;mean=np.average(x[at],axis=0,weights=weights[at]);means.append(mean);delta=x[at]-mean
        covariance+=(delta.T*weights[at])@delta;mass+=weights[at].sum()
    covariance/=mass;covariance=.8*covariance+.2*np.diag(np.diag(covariance))+.0025*np.eye(3)
    return np.asarray(means)*scale,covariance*scale[:,None]*scale[None,:]


def laplacian(shape,spacing,length,guide=None,contrast=.1,floor=.02):
    index=np.arange(np.prod(shape)).reshape(shape);rows=[];cols=[];values=[]
    for axis in (0,1):
        a,b=[slice(None)]*2,[slice(None)]*2;a[axis]=slice(None,-1);b[axis]=slice(1,None);a,b=tuple(a),tuple(b)
        weight=np.full(index[a].shape,(length/spacing[1-axis])**2)
        if guide is not None:weight*=floor+np.exp(-np.sum((guide[a]-guide[b])**2,-1)/(2*contrast**2))
        weight=weight.ravel();i,j=index[a].ravel(),index[b].ravel()
        rows.extend([i,j,i,j]);cols.extend([i,j,j,i]);values.extend([weight,weight,-weight,-weight])
    return sparse.coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),shape=(index.size,index.size)).tocsr()


def appearance_scores(c,he,config):
    means,covariance=class_statistics(c)
    shape=tuple(max(2,math.ceil(n*v/config.field_um)) for n,v in zip(c.shape,c.mpp[::-1]))
    spacing=c.mpp*np.array(c.shape[::-1])/np.array(shape[::-1])
    original=cv2.resize(he,shape[::-1],interpolation=cv2.INTER_AREA)
    guide=-np.log((original.astype(float)+1)/256)
    regularizer=laplacian(shape,spacing,config.length_um,guide,config.od_contrast,config.conductance_floor)
    values=c.rgb;precision=np.linalg.inv(covariance)
    scores=[];fits=[];rho=config.prior_strength
    def reduce(a):return cv2.resize(a,shape[::-1],interpolation=cv2.INTER_AREA).reshape(-1,3)
    def up(a):return cv2.resize(a.reshape(*shape,3),c.shape[::-1],interpolation=cv2.INTER_LINEAR)
    for g in range(c.k):
        weight=c.weight*(c.labels==g);mass=cv2.resize(weight,shape[::-1],interpolation=cv2.INTER_AREA).ravel()
        sums=reduce(weight[...,None]*values);matrix=regularizer+sparse.diags(mass+rho);rhs=sums+rho*means[g]
        fitted=spsolve(matrix.tocsc(),rhs)
        detail=dict(linear_system_max_residual=float(np.max(np.abs(matrix@fitted-rhs))))
        delta=values-up(fitted)
        score=-.5*np.einsum('...i,ij,...j->...',delta,precision,delta)
        scores.append(score);fits.append(detail)
    return softmax(np.stack(scores,-1),axis=-1),dict(field_shape_yx=list(shape),field_mpp_xy=spacing.tolist(),fits=fits)


def refine_appearance(rgb,coarse,tissue,mpp_xy,*,he,tile_size=90,
                      config=None,output_shape_yx=None,output_tissue=None):
    """Return a categorical mask and diagnostics on the requested output grid.

    RGB evidence, original H&E and coarse/tissue masks must be aligned. Pixel
    spacing belongs to that input grid. For native source output after image
    resampling, supply its shape and aligned H&E-derived tissue support; labels
    are projected directly from the analysis grid, avoiding double rounding.
    Background exclusion is applied after refinement. Support is not truth.
    """
    config=config or AppearanceConfig();rgb=np.asarray(rgb);he=np.asarray(he);coarse=np.asarray(coarse);tissue=np.asarray(tissue)
    spacing=np.asarray(mpp_xy,float)
    if (rgb.dtype!=np.uint8 or he.dtype!=np.uint8 or coarse.ndim!=2 or
        rgb.shape!=(*coarse.shape,3) or he.shape!=rgb.shape or tissue.shape!=coarse.shape):raise ValueError('Aligned uint8 RGB/H&E and categorical/tissue masks required')
    if spacing.shape!=(2,) or not np.isfinite(spacing).all() or np.any(spacing<=0):raise ValueError('Positive per-axis pixel spacing required')
    if (not np.isfinite(coarse).all() or np.any(coarse<0) or np.any(coarse>255) or
        np.any(coarse!=np.floor(coarse)) or not np.any(coarse>0)):raise ValueError('Coarse mask must contain positive integer class IDs at most255')
    if not np.isfinite(tile_size) or tile_size<=0:raise ValueError('Positive tile width required')
    positive=(config.analysis_um,config.field_um,config.length_um,config.prior_strength,config.od_contrast)
    if not np.isfinite(positive).all() or min(positive)<=0 or not 0<config.conductance_floor<=1:raise ValueError('Invalid field configuration')
    c=Evidence(rgb,coarse,tissue>0,spacing,target_um=config.analysis_um,tile_size=tile_size)
    if not c.support.any():raise ValueError('No tissue support')
    if c.k==1:q=np.ones((*c.shape,1));details=dict(single_class=True)
    else:q,details=appearance_scores(c,he,config)
    if not np.isfinite(q).all():raise ValueError('Nonfinite appearance scores')
    shape=tuple(output_shape_yx or coarse.shape)
    if len(shape)!=2 or min(shape)<1:raise ValueError('Positive output shape required')
    if output_tissue is None:
        if shape!=coarse.shape:raise ValueError('Supply aligned output tissue support for a different output grid')
        support=tissue>0
    else:
        support=np.asarray(output_tissue)>0
        if support.shape!=shape:raise ValueError('Output tissue shape mismatch')
    labels=cv2.resize(c.ids[q.argmax(-1)].astype(np.uint8),shape[::-1],interpolation=cv2.INTER_NEAREST)
    labels[~support]=0
    details.update(method='Bora-AF',analysis_shape_yx=list(c.shape),analysis_mpp_xy=c.mpp.tolist(),
        output_shape_yx=list(shape),fine_annotation_used=False,hard_anchors=False,fixed_margin=False,
        original_he_conductance=True,background_removed_after_refinement=True)
    return labels,details
