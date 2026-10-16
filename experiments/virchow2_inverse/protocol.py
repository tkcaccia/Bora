"""Shared frozen protocol; this module never reads the fine reference mask."""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import tifffile
from .common import PATCH,coarse_tile_observations,digest,pad_image,save_json

COLOURS=np.array([[220,45,65],[55,165,75],[55,110,230],[200,100,220],[220,180,55]],np.float32)
TITLES=["Selective feature inversion","Minimal visual evidence","Nearest-class counterfactual",
"Class-specific concepts","Worst-class colour geometry","Boundary-specific experts",
"Colour-explained subspace removal","Stain-perturbation invariance","Scale-persistent evidence",
"Context chooses, detail places","Context-disagreement gating","Geodesic evidence transport",
"Semantic boundary barriers","Native-image superpixels","Object relationships",
"Soft tissue unmixing","Topology-aware refinement","Feature-matched tissue quilting",
"Predictable latent structure","Feature-guided wavelet detail"]

def expand(values,shape):
    """Interpolate token centres in the padded canvas, then crop to native frame."""
    h,w=shape;gh,gw=values.shape[:2]
    result=cv2.resize(np.asarray(values,np.float32),(gw*PATCH,gh*PATCH),interpolation=cv2.INTER_LINEAR)
    return result[:h,:w]

def align_scale(values,scaled_shape,native_shape):
    small=expand(values,scaled_shape)
    return cv2.resize(small,(native_shape[1],native_shape[0]),interpolation=cv2.INTER_LINEAR)

def rgb_scale(values):
    a=np.asarray(values,np.float32)
    if a.ndim==2:a=a[...,None]
    lo,hi=np.percentile(a,(1,99),axis=(0,1))
    out=np.clip((a-lo)/np.maximum(hi-lo,1e-6),0,1)
    if out.shape[2]==1:out=np.repeat(out,3,axis=2)
    return np.rint(out[...,:3]*255).astype(np.uint8)

def probability_rgb(p):return np.rint(np.clip(p,0,1)@COLOURS).clip(0,255).astype(np.uint8)

def fit_classifier(values,labels):
    return LogisticRegression(C=.3,class_weight="balanced",max_iter=2000,random_state=430).fit(values,labels)

def coarse_summaries(a,records,tile=90):
    result=[]
    for row in records:
        y,x=row["y"],row["x"];result.append(a[y+tile//4:min(y+3*tile//4,a.shape[0]),x+tile//4:min(x+3*tile//4,a.shape[1])].mean((0,1)))
    return np.asarray(result)

def train_on_map(a,ctx):
    clf=fit_classifier(coarse_summaries(a,ctx.records),ctx.labels)
    shape=a.shape[:2]
    return clf.predict_proba(a.reshape(-1,a.shape[-1])).reshape(*shape,5).astype(np.float32),clf

def border(labels):
    edge=np.zeros(labels.shape,bool)
    d=(labels[1:]!=labels[:-1])&(labels[1:]>0)&(labels[:-1]>0);edge[1:]|=d;edge[:-1]|=d
    d=(labels[:,1:]!=labels[:,:-1])&(labels[:,1:]>0)&(labels[:,:-1]>0);edge[:,1:]|=d;edge[:,:-1]|=d
    return edge

def bora_rgb(image,coarse,config=None):
    from bora.boundary import BoundaryConfig,_interior
    config=config or BoundaryConfig();h,w=coarse.shape
    size=(int(np.ceil(w/config.downsample)),int(np.ceil(h/config.downsample)))
    lab=cv2.resize(coarse,size,interpolation=cv2.INTER_NEAREST)
    ans=_interior(cv2.resize(image,size,interpolation=cv2.INTER_AREA),lab,90/config.downsample,config)
    ans=cv2.resize(ans,(w,h),interpolation=cv2.INTER_NEAREST)
    ans[coarse==0]=0
    return ans

class Context:
    def __init__(self,source,root):
        self.source=Path(source);self.root=Path(root)
        self.image=np.asarray(Image.open(self.source/"image.original.png").convert("RGB"))
        self.coarse=tifffile.imread(self.source/"tile_90/coarse.tif")
        self.info=json.loads((self.source/"results.json").read_text())
        self.inference=json.loads((self.root/"features/inference.json").read_text())
        if digest(self.source/"image.original.png")!=self.inference["image_sha256"]:raise ValueError("Image checksum mismatch")
        path=self.root/"features/dense_patch_features.npy"
        if digest(path)!=self.inference["dense_features_sha256"]:raise ValueError("Feature checksum mismatch")
        f=np.load(path)
        self.selection=json.loads((self.root/"initial_feature_selection.json").read_text())
        self.selected=np.array(self.selection["indices"],int)
        self.summaries,self.labels,self.groups,self.records=coarse_tile_observations(f,self.coarse)
        self.scaler=StandardScaler().fit(self.summaries[:,self.selected])
        self.scaler.scale_=np.maximum(self.scaler.scale_,.05)
        self.tile_z=self.scaler.transform(self.summaries[:,self.selected]).astype(np.float32)
        self.z=self.scaler.transform(f[...,self.selected].reshape(-1,len(self.selected))).reshape(*f.shape[:2],-1).astype(np.float32)
        self.grid=self.z.shape[:2];self.shape=self.coarse.shape
        self.classifier=fit_classifier(self.tile_z,self.labels)
        self.p=self.classifier.predict_proba(self.z.reshape(-1,self.z.shape[2])).reshape(*self.grid,5).astype(np.float32)
        self.native_p=expand(self.p,self.shape)
        self.feature_rgb=probability_rgb(self.native_p)
        self.baseline=bora_rgb(self.image,self.coarse)
        cy=np.minimum(((np.arange(self.grid[0])+.5)*14).astype(int),self.shape[0]-1)
        cx=np.minimum(((np.arange(self.grid[1])+.5)*14).astype(int),self.shape[1]-1)
        yy,xx=np.meshgrid(cy,cx,indexing="ij")
        self.token_labels=self.coarse[yy,xx]
        self.token_seeds=((yy%90>=23)&(yy%90<67)&(xx%90>=23)&(xx%90<67)&(self.token_labels>0))
        np.savez(self.root/"classifier_parameters.npz",selected=self.selected,mean=self.scaler.mean_,scale=self.scaler.scale_,
                 coefficients=self.classifier.coef_,intercept=self.classifier.intercept_,classes=self.classifier.classes_)

    def grid_from_native(self,a):
        """Sample a native map at true token centres; clamp only padded edge cells."""
        yy,xx=np.meshgrid((np.arange(self.grid[0])+.5)*14-.5,(np.arange(self.grid[1])+.5)*14-.5,indexing="ij")
        if a.ndim==2:return ndi.map_coordinates(a,[yy,xx],order=1,mode="nearest")
        return np.stack([ndi.map_coordinates(a[...,c],[yy,xx],order=1,mode="nearest") for c in range(a.shape[-1])],-1)

    def output(self,number,rgb,details,prediction=None,extras=None):
        folder=self.root/f"{number:02d}";folder.mkdir(exist_ok=True)
        if rgb.shape!=self.image.shape or rgb.dtype!=np.uint8 or not np.isfinite(rgb).all():raise ValueError("Invalid experiment RGB frame")
        if prediction is None:prediction=bora_rgb(rgb,self.coarse)
        if prediction.shape!=self.shape or not set(np.unique(prediction)).issubset(set(np.unique(self.coarse))):raise ValueError("Invalid prediction frame/labels")
        prediction[self.coarse==0]=0
        Image.fromarray(rgb).save(folder/"representation.png")
        tifffile.imwrite(folder/"bora.tif",prediction.astype(np.uint8),compression="deflate")
        for name,a in (extras or {}).items():
            if a.dtype==np.uint8 and a.ndim in (2,3):Image.fromarray(a).save(folder/(name+".png"))
            else:np.save(folder/(name+".npy"),a)
        record={"id":number,"title":TITLES[number-1],"status":"completed","details":details,
                "image_sha256":digest(folder/"representation.png"),"prediction_sha256":digest(folder/"bora.tif"),
                "fine_reference_used":False,"changed_pixels_from_coarse":int(np.sum(prediction!=self.coarse)),
                "native_mpp_xy":[.5,.5],"shape_yx":list(self.shape)}
        save_json(folder/"result.json",record)
        return record
