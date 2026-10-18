"""Calibrated input grid and soft coarse-tile weights for Bora-AF."""
import cv2
import numpy as np


def resize(a, shape, nearest=False):
    return cv2.resize(a, tuple(shape[::-1]), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)


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

