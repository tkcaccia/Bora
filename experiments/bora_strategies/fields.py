"""Retained whole-field probabilistic strategies."""
import cv2
import numpy as np
from scipy.special import softmax
from .context import gaussian,simplex,average
from .registry import PARAMETERS as P


def multiscale(c):
    logp=np.zeros_like(c.u)
    for um in (1.,4.,12.):
        values=gaussian(c.lab,um/c.um);cost=c.appearance(values);logp+=np.log(softmax(-cost,axis=-1)+1e-7)/3
    return c.fix(logp.argmax(-1))




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
