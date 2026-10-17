"""Whole-field probabilistic, variational, geodesic and retrieval strategies."""
import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.special import softmax
from scipy.spatial import cKDTree
from skimage.segmentation import watershed
from skimage.graph import MCP_Geometric
from skimage.color import rgb2hed
from skimage.filters import threshold_otsu
from sklearn.ensemble import ExtraTreesClassifier
from .context import gaussian,simplex,average
from .registry import PARAMETERS as P


def multiscale(c):
    logp=np.zeros_like(c.u)
    for um in (1.,4.,12.):
        values=gaussian(c.lab,um/c.um);cost=c.appearance(values);logp+=np.log(softmax(-cost,axis=-1)+1e-7)/3
    return c.fix(logp.argmax(-1))


def region_snap(c):
    g=c.region_graph();votes=g['votes'];label=g['u'].argmin(-1);known=votes.sum(-1)>.05;label[known]=votes[known].argmax(-1)
    return label[g['regions']]


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


def contour(c):
    answer=c.initial.copy();values=c.lab/128
    for _ in range(P['contour_iterations']):
        costs=[]
        for i in range(c.k):
            region=(answer==i)&c.support;sample=values[region] if region.any() else values[c.core==i]
            mean=sample.mean(0)
            cost=np.mean((values-mean)**2,axis=-1)
            same=cv2.boxFilter((answer==i).astype(np.float32),-1,(3,3),normalize=True)
            costs.append(cost+P['contour_curvature']*(1-same))
        updated=c.fix(np.stack(costs,-1).argmin(-1))
        if np.array_equal(updated,answer):break
        answer=updated
    return answer


def transport(c):
    target=np.array([np.mean(c.initial[c.support]==i) for i in range(c.k)])
    target=np.maximum(target,1e-5);target/=target.sum()
    kernel=np.exp(-c.u.reshape(-1,c.k)/P['transport_temperature']).astype(np.float64)+1e-10
    support=c.support.ravel();b=np.ones(c.k)
    for _ in range(P['transport_iterations']):
        q=kernel*b;q/=q.sum(-1,keepdims=True);mass=q[support].mean(0);b*=target/np.maximum(mass,1e-12);b/=max(b.max(),1e-12)
    q=kernel*b;q/=q.sum(-1,keepdims=True)
    c.diagnostics['B16']=dict(target_soft_mass=target.tolist(),achieved_soft_mass=q[support].mean(0).tolist())
    return c.fix(q.argmax(-1).reshape(c.shape))


def descriptor(c):
    channels=[]
    for um in (1.,4.,12.):
        channels.append(gaussian(c.lab,um/c.um)/128)
    channels.append(c.gradient[...,None])
    return np.concatenate(channels,-1).astype(np.float32)


def nonlocal_vote(c):
    values=descriptor(c);step=max(1,round(P['patch_stride_um']/c.um));sample=values[::step,::step];labels=c.core[::step,::step]
    rng=np.random.default_rng(P['random_seed']);data=[];truth=[]
    for i in range(c.k):
        a=sample[labels==i]
        if not len(a):a=values[c.core==i]
        take=rng.choice(len(a),min(800,len(a)),replace=False);data.append(a[take]);truth.extend([i]*len(take))
    data=np.concatenate(data);truth=np.array(truth);scale=np.maximum(data.std(0),.03)
    tree=cKDTree(data/scale);dist,near=tree.query(sample.reshape(-1,sample.shape[-1])/scale,k=min(P['patch_neighbors'],len(data)),workers=2)
    if dist.ndim==1:dist=dist[:,None];near=near[:,None]
    weights=1/np.maximum(dist,.05);p=np.stack([(weights*(truth[near]==i)).sum(-1) for i in range(c.k)],-1);p/=p.sum(-1,keepdims=True)
    p=p.reshape(*sample.shape[:2],c.k).astype(np.float32)
    p=cv2.resize(p,tuple(c.shape[::-1]),interpolation=cv2.INTER_LINEAR)
    if p.ndim==2:p=p[...,None]
    return c.fix(p.argmax(-1))


def object_context(c):
    hematoxylin=rgb2hed(c.he)[...,0].astype(np.float32);observed=hematoxylin[c.support]
    threshold=float(threshold_otsu(observed)) if observed.max()>observed.min() else float(observed.mean())
    nuclei=(hematoxylin>threshold)&c.support;cc,n=ndi.label(nuclei);areas=np.bincount(cc.ravel())*np.prod(c.mpp_xy)
    keep=(areas>=3)&(areas<=300);keep[0]=False;nuclei=keep[cc]
    lumens=(c.he.min(-1)>210)&c.support;cc,n=ndi.label(lumens);areas=np.bincount(cc.ravel())*np.prod(c.mpp_xy)
    keep=(areas>=10)&(areas<=3000);keep[0]=False;keep[np.unique(np.r_[cc[0],cc[-1],cc[:,0],cc[:,-1]])]=False;lumens=keep[cc]
    features=np.concatenate([c.lab/128,gaussian(nuclei.astype(np.float32),6/c.um)[...,None],
        gaussian(nuclei.astype(np.float32),20/c.um)[...,None],gaussian(lumens.astype(np.float32),12/c.um)[...,None],
        np.minimum(ndi.distance_transform_edt(~lumens,sampling=c.mpp_xy[::-1]),50)[...,None]/50],axis=-1)
    g=c.region_graph('he');x=average(features,g['regions']);votes=g['votes'];known=votes.sum(-1)>.1
    y=votes.argmax(-1);purity=votes.max(-1)/np.maximum(votes.sum(-1),1e-8);known&=purity>=.75
    train_x=[x[known]];train_y=[y[known]]
    # Ensure each coarse class is represented without borrowing fine labels.
    for i in range(c.k):train_x.append(features[c.core==i].mean(0,keepdims=True));train_y.append(np.array([i]))
    model=ExtraTreesClassifier(n_estimators=32,max_depth=10,min_samples_leaf=2,class_weight='balanced',random_state=P['random_seed'],n_jobs=2)
    model.fit(np.concatenate(train_x),np.concatenate(train_y));labels=model.predict(x)
    c.diagnostics['B15']=dict(training_regions=int(known.sum()),hematoxylin_threshold=threshold,candidate_nuclei_not_expert=True)
    return labels[g['regions']]
