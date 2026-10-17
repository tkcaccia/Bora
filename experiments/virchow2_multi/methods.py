"""The seven selected hypotheses, generalized to each image's coarse classes.

Presets match the earlier five-class study. Probability colours are fixed per
dataset/label; classifier columns follow actual observed coarse class IDs.
"""
from itertools import combinations
import numpy as np
from scipy import ndimage as ndi
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from experiments.virchow2_inverse.common import coarse_tile_observations
from experiments.virchow2_inverse.selection import select_features,round_robin
from experiments.virchow2_inverse.protocol import expand,rgb_scale,fit_classifier,coarse_summaries
from experiments.virchow2_inverse.algorithms import object_maps

METHODS={9:"Scale-persistent evidence",10:"Context chooses, detail places",15:"Object relationships",19:"Predictable latent structure"}


class Context:
    def __init__(self,image,coarse,features,colours,*,analysis_support=None):
        self.image=image;self.coarse=coarse;self.shape=coarse.shape;self.grid=features.shape[:2]
        self.analysis_support=coarse>0 if analysis_support is None else np.asarray(analysis_support,bool)
        if self.analysis_support.shape!=self.shape:raise ValueError("Analysis support geometry mismatch")
        self.summaries,self.labels,_,self.records=coarse_tile_observations(features,coarse)
        self.classes=np.unique(self.labels);self.colours=np.asarray(colours,np.float32)[self.classes-1]
        self.selection=select_features(self.summaries,self.labels,"balanced_mrmr",64)
        self.selected=np.array(self.selection["indices"])
        self.scaler=StandardScaler().fit(self.summaries[:,self.selected]);self.scaler.scale_=np.maximum(self.scaler.scale_,.05)
        self.tile_z=self.scaler.transform(self.summaries[:,self.selected]).astype(np.float32)
        self.z=self.scaler.transform(features[...,self.selected].reshape(-1,64)).reshape(*self.grid,64).astype(np.float32)
        # Match the original head exactly: normalize the float64 tile means,
        # rather than re-averaging the already rounded float32 dense z map.
        self.classifier=fit_classifier(self.tile_z,self.labels)
        self.p=self.classifier.predict_proba(self.z.reshape(-1,64)).reshape(*self.grid,len(self.classes)).astype(np.float32)
        self.native_p=expand(self.p,self.shape)

    def probability_rgb(self,p):return np.rint(np.clip(p,0,1)@self.colours).clip(0,255).astype(np.uint8)


def grid_prob(z,c):
    summaries,labels,_,_=coarse_tile_observations(z,c.coarse)
    clf=fit_classifier(summaries,labels)
    if not np.array_equal(clf.classes_,c.classes):raise ValueError("Classifier class-order changed")
    return clf.predict_proba(z.reshape(-1,z.shape[-1])).reshape(*z.shape[:2],len(c.classes)).astype(np.float32)


def balanced_features(values,labels,penalty=None,k=32):
    classes=np.unique(labels)
    means=[values[labels==g].mean(0) for g in classes];var=[values[labels==g].var(0) for g in classes]
    lists=[]
    for a,b in combinations(range(len(classes)),2):
        score=(means[a]-means[b])**2/(var[a]+var[b]+.05)
        if penalty is not None:score*=penalty
        lists.append(np.argsort(-score))
    return round_robin(lists,min(k,values.shape[1]))




def align_variant(values,variant_shape,c):
    sh,sw=variant_shape;h,w=c.shape
    y=(np.arange(c.grid[0])+.5)*14*sh/h/14-.5;x=(np.arange(c.grid[1])+.5)*14*sw/w/14-.5
    yy,xx=np.meshgrid(y,x,indexing="ij")
    return np.stack([ndi.map_coordinates(values[...,i],[yy,xx],order=1,mode="nearest") for i in range(values.shape[-1])],-1)


def scale_z(values,c):
    summaries,_,_,_=coarse_tile_observations(values,c.coarse)
    scaler=StandardScaler().fit(summaries);scaler.scale_=np.maximum(scaler.scale_,.05)
    return scaler.transform(values.reshape(-1,values.shape[-1])).reshape(values.shape).astype(np.float32),scaler.transform(summaries)


def run_method(number,c,variants,*,return_probabilities=False):
    z=c.z;extras={};details=dict(preset="original fixed hyperparameters, generalized class count",fine_reference_used=False)
    if number==9:
        middle,mt=scale_z(variants["scale_1"],c);wide,wt=scale_z(variants["scale_2"],c)
        correlations=np.stack([np.array([np.corrcoef(c.tile_z[:,i],other[:,i])[0,1] for i in range(64)]) for other in (mt,wt)])
        persistence=np.maximum(np.nan_to_num(correlations.min(0)),0)
        keep=balanced_features(c.tile_z,c.labels,.05+persistence,32)
        consensus=(z[...,keep]+middle[...,keep]+wide[...,keep])/3
        p=expand(grid_prob(consensus,c),c.shape);rgb=c.probability_rgb(p)
        details.update(selected_dimensions=c.selected[keep].tolist(),cross_scale_correlation=correlations.tolist(),nominal_sampling_um=[.5,1,2])
    elif number==10:
        middle,_=scale_z(variants["scale_1"],c);wide,_=scale_z(variants["scale_2"],c)
        p1,p2=grid_prob(middle,c),grid_prob(wide,c);context=np.sqrt(p1*p2)
        top=np.argsort(context,axis=2)[...,-2:];gate=np.full_like(context,.15);np.put_along_axis(gate,top,1,axis=2)
        combined=c.p*np.sqrt(context+1e-5)*gate;combined/=combined.sum(2,keepdims=True)
        p=expand(combined,c.shape);rgb=c.probability_rgb(p)
        details.update(context_top_classes=min(2,len(c.classes)),other_class_weight=.15)
    elif number==15:
        objects,nuclei,lumens,threshold=object_maps(c.image,c.analysis_support.astype(np.uint8))
        a=np.concatenate([c.native_p,objects],-1)
        scaler=StandardScaler().fit(coarse_summaries(a,c.records));scaler.scale_=np.maximum(scaler.scale_,1e-4)
        a=scaler.transform(a.reshape(-1,a.shape[2])).reshape(a.shape).astype(np.float32)
        clf=fit_classifier(coarse_summaries(a,c.records),c.labels)
        p=clf.predict_proba(a.reshape(-1,a.shape[2])).reshape(*c.shape,len(c.classes)).astype(np.float32);rgb=c.probability_rgb(p)
        details.update(original_rgb_supplies_geometry=True,candidate_objects_are_not_expert_masks=True,hematoxylin_threshold=threshold,
                       object_scales_um=[6,20,12],coefficients=clf.coef_.tolist())
        extras={"candidate_nuclei":(nuclei*255).astype(np.uint8),"candidate_lumens":(lumens*255).astype(np.uint8)}
    elif number==19:
        neighbours=np.empty((*c.grid,192),np.float32)
        for j,k in enumerate((1,2,4)):
            padded=np.pad(z,((k,k),(k,k),(0,0)),mode="reflect")
            neighbours[...,j*64:(j+1)*64]=(padded[:-2*k,k:-k]+padded[2*k:,k:-k]+padded[k:-k,:-2*k]+padded[k:-k,2*k:])/4
        yy,xx=np.indices(c.grid);folds=(yy>=c.grid[0]//2)*2+(xx>=c.grid[1]//2)
        completed=np.empty_like(z);records=[]
        for fold in range(4):
            test=folds==fold;train=~ndi.binary_dilation(test,iterations=4)
            model=Ridge(alpha=100).fit(neighbours[train],z[train]);completed[test]=model.predict(neighbours[test])
            records.append(dict(fold=fold,train=int(train.sum()),test=int(test.sum()),training_buffer_tokens=4))
        p=expand(grid_prob(completed,c),c.shape);rgb=c.probability_rgb(p)
        details.update(folds=records,own_token_in_predictor=False,encoder_contexts_overlap_between_tokens=True)
    else:raise ValueError(number)
    raw=c.classes[p.argmax(-1)].astype(np.uint8);raw[c.coarse==0]=0
    if rgb.shape!=(*c.shape,3) or not np.isfinite(rgb).all():raise ValueError("Invalid method RGB")
    if return_probabilities:
        return rgb,raw,details,extras,p
    return rgb,raw,details,extras
