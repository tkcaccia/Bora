"""Distinct spatial, geometry and retrieval interventions for the twenty ideas."""
from __future__ import annotations
import cv2
import numpy as np
from scipy import ndimage as ndi,sparse
from scipy.sparse.linalg import spsolve
from .protocol import border,expand,rgb_scale,bora_rgb

def graph_transport(z,prior,labels,seeds):
    h,w=z.shape[:2];index=np.arange(h*w).reshape(h,w)
    a=np.r_[index[:-1].ravel(),index[:,:-1].ravel()]
    b=np.r_[index[1:].ravel(),index[:,1:].ravel()]
    flat=z.reshape(h*w,-1);distance=np.mean((flat[a]-flat[b])**2,axis=1)
    weights=np.exp(-distance/max(float(np.median(distance)),1e-4))+.001
    adjacency=sparse.coo_matrix((np.r_[weights,weights],(np.r_[a,b],np.r_[b,a])),shape=(h*w,h*w)).tocsr()
    lap=sparse.diags(np.asarray(adjacency.sum(1)).ravel())-adjacency
    anchor=seeds.ravel().astype(float)*30
    rhs=.03*prior.reshape(h*w,5)
    for g in range(1,6):rhs[:,g-1]+=anchor*(labels.ravel()==g)
    ans=spsolve(lap+sparse.diags(.03+anchor),rhs)
    ans=np.maximum(ans,0);ans/=np.maximum(ans.sum(1,keepdims=True),1e-8)
    return ans.reshape(h,w,5).astype(np.float32)

def pair_experts(ctx):
    from bora.boundary import BoundaryConfig,_interior
    cfg=BoundaryConfig();size=(int(np.ceil(ctx.shape[1]/2)),int(np.ceil(ctx.shape[0]/2)))
    coarse=cv2.resize(ctx.coarse,size,interpolation=cv2.INTER_NEAREST)
    yy,xx=np.indices(coarse.shape);best=np.full(coarse.shape,np.inf);owner=np.full(coarse.shape,-1,int);tied=np.zeros(coarse.shape,bool)
    active=[];radius=round(45*cfg.internal_search_fraction)
    rows=ctx.selection["top_dimensions_by_pair"]
    raw=np.load(ctx.root/"features/dense_patch_features.npy",mmap_mode="r")
    previews={};selected={};pair_images=[]
    for row in rows:
        a,b=row["class_a"],row["class_b"];edge=np.zeros(coarse.shape,bool)
        for axis in (0,1):
            left=coarse[:-1] if axis==0 else coarse[:,:-1];right=coarse[1:] if axis==0 else coarse[:,1:]
            d=((left==a)&(right==b))|((left==b)&(right==a))
            if axis==0:edge[:-1]|=d;edge[1:]|=d
            else:edge[:,:-1]|=d;edge[:,1:]|=d
        if not edge.any():continue
        k=len(active);distance=ndi.distance_transform_edt(~edge)
        eligible=((coarse==a)|(coarse==b))&(distance<=radius)
        closer=eligible&(distance<best-1e-6);equal=eligible&np.isclose(distance,best,atol=1e-6,rtol=0)
        owner[closer]=k;best[closer]=distance[closer];tied[closer]=False;tied[equal]=True
        dims=row["indices"][:3];rgb=rgb_scale(expand(np.asarray(raw[...,dims]),ctx.shape))
        previews[f"pair_{a}_{b}"]=rgb;selected[f"{a}-{b}"]=dims
        pair_images.append(cv2.resize(rgb,size,interpolation=cv2.INTER_AREA));active.append((a,b))
    result=coarse.copy();composite=cv2.resize(ctx.feature_rgb,size)
    for k,(a,b) in enumerate(active):
        restricted=np.where((coarse==a)|(coarse==b),coarse,0).astype(np.uint8)
        proposal=_interior(pair_images[k],restricted,45,cfg)
        use=(owner==k)&~tied&((proposal==a)|(proposal==b));result[use]=proposal[use]
        composite[owner==k]=pair_images[k][owner==k]
    return cv2.resize(composite,(ctx.shape[1],ctx.shape[0])),cv2.resize(result,(ctx.shape[1],ctx.shape[0]),interpolation=cv2.INTER_NEAREST),previews,selected

def feature_gradient(z):
    dy=np.diff(z,axis=0,append=z[-1:]);dx=np.diff(z,axis=1,append=z[:,-1:])
    return np.sqrt(np.mean(dx**2+dy**2,axis=2))

def barrier_refine(ctx,energy):
    from skimage.segmentation import watershed
    size=(int(np.ceil(ctx.shape[1]/2)),int(np.ceil(ctx.shape[0]/2)))
    coarse=cv2.resize(ctx.coarse,size,interpolation=cv2.INTER_NEAREST)
    edge=border(coarse);band=ndi.distance_transform_edt(~edge)<=round(45*.28)
    energy=cv2.resize(energy,size,interpolation=cv2.INTER_AREA)
    markers=np.where(~band,coarse,0).astype(np.int32)
    for g in range(1,6):
        if not np.any(markers==g):
            distance=ndi.distance_transform_edt(coarse==g)
            if distance.max()>0:markers[np.unravel_index(distance.argmax(),distance.shape)]=g
    proposed=watershed(energy,markers=markers,mask=coarse>0,compactness=.01).astype(np.uint8)
    answer=np.where(band,proposed,coarse)
    return cv2.resize(answer,(ctx.shape[1],ctx.shape[0]),interpolation=cv2.INTER_NEAREST)

def superpixel_probabilities(image,p):
    from skimage.segmentation import slic
    segments=slic(image,n_segments=4500,compactness=8,sigma=1,start_label=0,channel_axis=-1)
    n=int(segments.max()+1);ids=segments.ravel();counts=np.bincount(ids,minlength=n)
    probs=np.stack([np.bincount(ids,weights=p[...,i].ravel(),minlength=n)/counts for i in range(5)],1)
    colours=np.stack([np.bincount(ids,weights=image[...,i].ravel()/255,minlength=n)/counts for i in range(3)],1)
    pairs=np.concatenate([np.stack([segments[:-1].ravel(),segments[1:].ravel()],1),np.stack([segments[:,:-1].ravel(),segments[:,1:].ravel()],1)])
    pairs=np.unique(np.sort(pairs[pairs[:,0]!=pairs[:,1]],axis=1),axis=0);a,b=pairs.T
    d=np.mean((probs[a]-probs[b])**2,axis=1)+.2*np.mean((colours[a]-colours[b])**2,axis=1)
    weights=np.exp(-d/max(float(np.median(d)),1e-5))
    adj=sparse.coo_matrix((np.r_[weights,weights],(np.r_[a,b],np.r_[b,a])),shape=(n,n)).tocsr()
    lap=sparse.diags(np.asarray(adj.sum(1)).ravel())-adj
    smoothed=spsolve(sparse.eye(n)+.3*lap,probs)
    smoothed=np.maximum(smoothed,0);smoothed/=np.maximum(smoothed.sum(1,keepdims=True),1e-8)
    return smoothed[segments].astype(np.float32),segments

def object_maps(image,coarse):
    from skimage.color import rgb2hed
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects
    h=rgb2hed(image)[...,0];support=coarse>0
    threshold=float(threshold_otsu(h[support]))
    nuclei=remove_small_objects((h>threshold)&support,min_size=12)
    lab,n=ndi.label(nuclei);sizes=np.bincount(lab.ravel());keep=(sizes<=1200)&(sizes>=12);keep[0]=False;nuclei=keep[lab]
    white=(image.min(2)>210)&support
    lab,n=ndi.label(white);sizes=np.bincount(lab.ravel());keep=(sizes>=40)&(sizes<=12000);keep[0]=False
    keep[np.unique(np.r_[lab[0],lab[-1],lab[:,0],lab[:,-1]])]=False
    lumen=keep[lab]
    density=ndi.gaussian_filter(nuclei.astype(float),12)
    wider=ndi.gaussian_filter(nuclei.astype(float),40)
    lumen_context=ndi.gaussian_filter(lumen.astype(float),24)
    distance=np.minimum(ndi.distance_transform_edt(~lumen),100)/100
    # Structure tensor distinguishes oriented nuclei arrangements from dense clusters.
    gy,gx=np.gradient(ndi.gaussian_filter(nuclei.astype(float),3));xx=ndi.gaussian_filter(gx*gx,16);yy=ndi.gaussian_filter(gy*gy,16);xy=ndi.gaussian_filter(gx*gy,16)
    anisotropy=np.sqrt((xx-yy)**2+4*xy**2)/(xx+yy+1e-8)
    return np.stack([density,wider,lumen_context,distance,anisotropy],-1).astype(np.float32),nuclei,lumen,threshold

def soft_transport(z,tile_z,labels,prior_strength=.25):
    means=np.stack([tile_z[labels==g].mean(0) for g in range(1,6)])
    var=np.stack([tile_z[labels==g].var(0) for g in range(1,6)])+.25
    cost=np.stack([np.mean((z-mu)**2/v,axis=2) for mu,v in zip(means,var)],-1)
    logits=-cost/max(float(np.median(cost)),.1)*4;logits-=logits.max(2,keepdims=True)
    kernel=np.exp(logits).reshape(-1,5)+1e-9
    free=kernel/kernel.sum(1,keepdims=True)
    counts=np.array([np.mean(labels==g) for g in range(1,6)])
    target=.5*counts+.5*free.mean(0);b=np.ones(5)
    for _ in range(80):
        prob=kernel*b;prob/=prob.sum(1,keepdims=True)
        b*=np.power(target/np.maximum(prob.mean(0),1e-8),prior_strength)
        b/=np.maximum(b.mean(),1e-8)
    p=kernel*b;p/=p.sum(1,keepdims=True)
    return p.reshape(*z.shape[:2],5).astype(np.float32),{"weak_target_mass":target.tolist(),"actual_mass":p.mean(0).tolist(),"strength":prior_strength}

def topology_project(coarse,proposal,p):
    """Penalize local topology changes, while allowing changes with strong evidence."""
    from skimage.measure import euler_number
    size=(int(np.ceil(coarse.shape[1]/2)),int(np.ceil(coarse.shape[0]/2)))
    c=cv2.resize(coarse,size,interpolation=cv2.INTER_NEAREST);target=cv2.resize(proposal,size,interpolation=cv2.INTER_NEAREST)
    probabilities=cv2.resize(p,size,interpolation=cv2.INTER_AREA);answer=c.copy()
    edge=border(c);band=ndi.distance_transform_edt(~edge)<=round(45*.28)
    yy,xx=np.nonzero((c!=target)&band&(c>0)&(target>0))
    gains=probabilities[yy,xx,target[yy,xx]-1]-probabilities[yy,xx,c[yy,xx]-1]
    accepted=0;penalized=0
    for index in np.argsort(-gains,kind="stable"):
        y,x=yy[index],xx[index]
        if y<2 or x<2 or y>=c.shape[0]-2 or x>=c.shape[1]-2:continue
        old,new=int(answer[y,x]),int(target[y,x]);patch=answer[y-2:y+3,x-2:x+3]
        before=[euler_number(patch==g,connectivity=2) for g in (old,new)]
        changed=patch.copy();changed[2,2]=new
        after=[euler_number(changed==g,connectivity=2) for g in (old,new)]
        penalty=sum(abs(a-b) for a,b in zip(before,after))
        penalized+=penalty>0
        if gains[index]>.02+.15*penalty:answer[y,x]=new;accepted+=1
    return cv2.resize(answer,(coarse.shape[1],coarse.shape[0]),interpolation=cv2.INTER_NEAREST),{"candidates":len(yy),"accepted":accepted,"topology_penalized":int(penalized),"gain_threshold":.02,"topology_penalty":.15}

def minimum_seam(error):
    cost=np.asarray(error,float).copy();parent=np.zeros(error.shape,np.int32)
    for row in range(1,len(cost)):
        for col in range(cost.shape[1]):
            lo=max(0,col-1);hi=min(cost.shape[1],col+2)
            parent[row,col]=lo+np.argmin(cost[row-1,lo:hi]);cost[row,col]+=cost[row-1,parent[row,col]]
    path=np.empty(len(cost),int);path[-1]=np.argmin(cost[-1])
    for row in range(len(cost)-1,0,-1):path[row-1]=parent[row,path[row]]
    return path

def quilt(ctx):
    from .common import pad_image
    padded=pad_image(ctx.image);gh,gw=ctx.grid
    ys=sorted(set([*range(0,gh-2,2),gh-3]));xs=sorted(set([*range(0,gw-2,2),gw-3]))
    locations=np.array([(y,x) for y in ys for x in xs]);features=np.stack([ctx.z[y:y+3,x:x+3].mean((0,1)) for y,x in locations])
    windows=np.array(ctx.inference["window_coordinates_yx"])
    boxes=[]
    for y,x in locations*14:
        hit=(windows[:,0]<y+42)&(windows[:,0]+224>y)&(windows[:,1]<x+42)&(windows[:,1]+224>x)
        boxes.append([windows[hit,0].min(),windows[hit,1].min(),windows[hit,0].max()+224,windows[hit,1].max()+224])
    boxes=np.array(boxes);canvas=np.zeros_like(padded,dtype=np.float32);donors=np.full(padded.shape[:2],-1,np.int32);records=[]
    source_yx=np.full((*padded.shape[:2],2),-1,np.int16)
    native_donors=(locations[:,0]*14+42<=ctx.shape[0])&(locations[:,1]*14+42<=ctx.shape[1])
    for i,(gy,gx) in enumerate(locations):
        y,x=gy*14,gx*14;box=boxes[i]
        disjoint=(boxes[:,2]<=box[0])|(boxes[:,0]>=box[2])|(boxes[:,3]<=box[1])|(boxes[:,1]>=box[3])
        candidates=np.flatnonzero(disjoint&native_donors)
        if not len(candidates):raise ValueError("No context-disjoint quilt donor")
        d=np.mean((features[candidates]-features[i])**2,1);chosen=candidates[np.argsort(d)[:8]]
        column=int(np.flatnonzero(np.array(xs)==gx)[0]);row=int(np.flatnonzero(np.array(ys)==gy)[0])
        left=42-(gx-xs[column-1])*14 if column else 0;top=42-(gy-ys[row-1])*14 if row else 0
        existing=canvas[y:y+42,x:x+42]
        scores=[]
        for j in chosen:
            sy,sx=locations[j]*14;patch=padded[sy:sy+42,sx:sx+42].astype(np.float32)
            seam=0
            if left:seam+=np.mean(((existing[:,:left]-patch[:,:left])/255)**2)
            if top:seam+=np.mean(((existing[:top]-patch[:top])/255)**2)
            scores.append(np.mean((features[j]-features[i])**2)+.5*seam)
        j=int(chosen[np.argmin(scores)]);sy,sx=locations[j]*14;patch=padded[sy:sy+42,sx:sx+42].astype(np.float32)
        use=np.ones((42,42),bool)
        if left:
            seam=minimum_seam(np.mean((existing[:,:left]-patch[:,:left])**2,2))
            use[:,:left]&=np.arange(left)[None,:]>=seam[:,None]
        if top:
            seam=minimum_seam(np.mean((existing[:top]-patch[:top])**2,2).T)
            use[:top]&=np.arange(top)[:,None]>=seam[None,:]
        existing[use]=patch[use];donors[y:y+42,x:x+42][use]=j
        dy,dx=np.indices((42,42));source_yx[y:y+42,x:x+42][use]=np.stack([dy+sy,dx+sx],-1)[use]
        records.append({"target_yx":[int(y),int(x)],"donor_yx":[int(sy),int(sx)],"target_context_box":box.tolist(),"donor_context_box":boxes[j].tolist(),"context_disjoint":True})
    if np.any(donors<0):raise AssertionError("Unfilled quilt canvas")
    return np.rint(canvas[:ctx.shape[0],:ctx.shape[1]]).astype(np.uint8),donors[:ctx.shape[0],:ctx.shape[1]],source_yx[:ctx.shape[0],:ctx.shape[1]],records
