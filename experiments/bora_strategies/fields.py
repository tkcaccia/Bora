"""Retained whole-field probabilistic, watershed, geodesic and TV strategies."""
import cv2
import numpy as np
from scipy.special import softmax
from skimage.segmentation import watershed
from skimage.graph import MCP_Geometric
from .context import gaussian,simplex,average
from .registry import PARAMETERS as P


def multiscale(c):
    logp=np.zeros_like(c.u)
    for um in (1.,4.,12.):
        values=gaussian(c.lab,um/c.um);cost=c.appearance(values);logp+=np.log(softmax(-cost,axis=-1)+1e-7)/3
    return c.fix(logp.argmax(-1))




def flooding(c):
    markers=np.where(c.anchors>=0,c.anchors+1,0).astype(np.int32)
    return watershed(c.gradient,markers=markers,compactness=.0001).astype(np.int32)-1


def shifted(a,dy,dx):
    h,w=a.shape[:2];ys=np.clip(np.arange(h)+dy,0,h-1);xs=np.clip(np.arange(w)+dx,0,w-1)
    return a[ys[:,None],xs[None,:]]


def crf(c):
    q=c.p.copy();weights=[]
    for um in P['crf_radii_um']:
        step=max(1,round(um/c.um))
        for dy,dx in ((step,0),(-step,0),(0,step),(0,-step)):
            w=np.exp(-np.mean((c.lab-shifted(c.lab,dy,dx))**2,axis=-1)/(12**2))+.001
            weights.append((dy,dx,w))
    norm=sum(w for _,_,w in weights)[...,None]
    for _ in range(P['crf_iterations']):
        messages=sum(w[...,None]*shifted(q,dy,dx) for dy,dx,w in weights)/norm
        q=softmax(-c.u+P['crf_weight']*messages,axis=-1).astype(np.float32)
        a=c.anchors>=0;q[a]=np.eye(c.k,dtype=np.float32)[c.anchors[a]]
    return q.argmax(-1)


def geodesic(c):
    # Equal zero start cost at every class anchor; path length is cumulative.
    best=np.full(c.shape,np.inf);answer=np.zeros(c.shape,np.int32)
    for i in range(c.k):
        travel=.03+c.u[...,i]+.3*np.clip(c.gradient,0,3)
        starts=np.argwhere(c.anchors==i)
        cost,_=MCP_Geometric(travel,fully_connected=True,sampling=tuple(c.mpp_xy[::-1])).find_costs(starts)
        update=cost<best;answer[update]=i;best[update]=cost[update]
    return c.fix(answer)


def tv(c):
    q=c.p.copy();qbar=q.copy();px=np.zeros_like(q);py=np.zeros_like(q);weight=P['tv_weight'];tau=.2;sigma=.2
    for _ in range(P['tv_iterations']):
        gx=np.diff(qbar,axis=1,append=qbar[:,-1:]);gy=np.diff(qbar,axis=0,append=qbar[-1:])
        px+=sigma*gx;py+=sigma*gy;norm=np.maximum(1,np.sqrt(px*px+py*py)/weight);px/=norm;py/=norm
        px[:,-1]=0;py[-1]=0
        divx=px-np.pad(px[:,:-1],((0,0),(1,0),(0,0)))
        divy=py-np.pad(py[:-1],((1,0),(0,0),(0,0)))
        previous=q;q=simplex(q+tau*(divx+divy-c.u)).astype(np.float32)
        known=c.anchors>=0;q[known]=np.eye(c.k,dtype=np.float32)[c.anchors[known]]
        qbar=2*q-previous
    return q.argmax(-1)
