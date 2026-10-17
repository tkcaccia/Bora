"""Shared inputs and coarse-only statistics; no reference-mask I/O."""
import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.special import softmax
from .registry import PARAMETERS as P


def resize(a,shape,nearest=False):
    return cv2.resize(a,tuple(shape[::-1]),interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)


def average(a,regions):
    n=int(regions.max())+1;flat=regions.ravel();count=np.bincount(flat,minlength=n)
    if a.ndim==2:return np.bincount(flat,weights=a.ravel(),minlength=n)/np.maximum(count,1)
    return np.stack([average(a[...,i],regions) for i in range(a.shape[-1])],axis=1)


def gaussian(a,sigma):return cv2.GaussianBlur(a,(0,0),max(float(sigma),.1))


def simplex(a):
    ordered=np.sort(a,axis=-1)[...,::-1];s=np.cumsum(ordered,axis=-1)-1
    rho=(ordered-s/np.arange(1,a.shape[-1]+1)>0).sum(axis=-1)-1
    theta=np.take_along_axis(s,np.maximum(rho,0)[...,None],axis=-1)[...,0]/(rho+1)
    return np.maximum(a-theta[...,None],0)


class Context:
    def __init__(self,rgb,he,original,completed,support,mpp_xy):
        self.input_shape=completed.shape;h,w=completed.shape
        self.shape=(max(8,round(h*mpp_xy[1]/P['physical_target_um'])),max(8,round(w*mpp_xy[0]/P['physical_target_um'])))
        self.mpp_xy=np.array([w*mpp_xy[0]/self.shape[1],h*mpp_xy[1]/self.shape[0]])
        self.um=float(np.sqrt(np.prod(self.mpp_xy)));self.rgb=resize(rgb,self.shape);self.he=resize(he,self.shape)
        self.original=resize(original,self.shape,True);self.completed=resize(completed,self.shape,True)
        self.support=resize(support.astype(np.uint8),self.shape,True)>0
        self.ids=np.unique(original[original>0]);self.k=len(self.ids)
        self.initial=np.searchsorted(self.ids,self.completed).clip(0,self.k-1).astype(np.int32)
        self.lab=cv2.cvtColor(self.rgb,cv2.COLOR_RGB2LAB).astype(np.float32)
        self.lab[...,0]*=.55;self.lab[...,1:]*=.8
        self.lab=gaussian(self.lab,.7/self.um)
        self.core=np.full(self.shape,-1,np.int32)
        for i,g in enumerate(self.ids):
            region=(self.original==g)&self.support
            distance=ndi.distance_transform_edt(region,sampling=self.mpp_xy[::-1])
            chosen=region&(distance>=P['training_core_um'])
            if chosen.sum()<16:chosen=region&(distance>=max(.01,float(distance.max())*.5))
            if not chosen.any():chosen=self.original==g
            self.core[chosen]=i
        self.anchors=np.full(self.shape,-1,np.int32)
        for y in range(0,h,90):
            for x in range(0,w,90):
                crop=original[y:min(y+90,h),x:min(x+90,w)];ids,counts=np.unique(crop[crop>0],return_counts=True)
                if not len(ids):continue
                raw=ids[counts.argmax()];i=int(np.searchsorted(self.ids,raw))
                cy=min(self.shape[0]-1,round((y+crop.shape[0]/2)*self.shape[0]/h));cx=min(self.shape[1]-1,round((x+crop.shape[1]/2)*self.shape[1]/w))
                if self.original[cy,cx]==raw and self.support[cy,cx]:self.anchors[cy,cx]=i
        for i in range(self.k):
            if not np.any(self.anchors==i):
                yy,xx=np.nonzero(self.core==i);j=len(yy)//2;self.anchors[yy[j],xx[j]]=i
        self.u=self.appearance(self.lab);self.p=softmax(-self.u,axis=-1).astype(np.float32)
        gx=np.diff(self.lab,axis=1,append=self.lab[:,-1:]);gy=np.diff(self.lab,axis=0,append=self.lab[-1:])
        self.gradient=np.sqrt(np.mean(gx*gx+gy*gy,axis=-1));self.gradient/=max(float(np.percentile(self.gradient,95)),1e-4)
        self.cache={};self.outputs={};self.diagnostics={}

    def appearance(self,values):
        costs=[]
        for i in range(self.k):
            samples=values[self.core==i];mean=np.mean(samples,axis=0);var=np.maximum(np.var(samples,axis=0),64)
            cost=np.mean((values-mean)**2/var+np.log(var),axis=-1);costs.append(cost)
        u=np.stack(costs,axis=-1).astype(np.float32);u-=u.min(axis=-1,keepdims=True)
        u/=max(float(np.median(u.max(axis=-1)[self.support])),.1)
        return np.clip(u,0,30)

    def fix(self,labels):
        labels=labels.astype(np.int32);known=self.anchors>=0;labels[known]=self.anchors[known];return labels

    def probabilities(self,labels):return np.eye(self.k,dtype=np.float32)[labels]

    def finish(self,labels):
        if labels.shape!=self.shape or not np.isfinite(labels).all() or labels.min()<0 or labels.max()>=self.k:raise ValueError('Invalid strategy labels')
        return self.ids[labels.astype(np.int32)].astype(np.uint8)
