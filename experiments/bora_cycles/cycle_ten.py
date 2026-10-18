"""Fifteen cycle-ten models; no fine references, encoder calls or RGB edits."""
import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order
from scipy.sparse.linalg import splu, spsolve
from scipy.special import log_ndtr, softmax

from bora.evidence import Evidence
from . import native
from .native_ten import tree_potts, seeded_forest, biconnected_white
from .methods import correct_coarse, fit_linear
from .cycle_two import colour_nodes
from .cycle_eight import class_statistics, field_inputs, resize_field
from .cycle_nine import upsample, projected_inverse
from .cycle_six import edge_appearance
from .diagnostics import white_regions


def whiten(covariance):
    values,vectors=np.linalg.eigh(covariance)
    return (vectors/np.sqrt(values))@vectors.T


def tile_ids(c,shape=None):
    shape=shape or c.shape
    yy,xx=np.indices(shape)
    columns=int(np.ceil(c.shape[1]/c.tile_x))
    return (((yy+.5)*c.shape[0]/shape[0]/c.tile_y).astype(int)*columns
            +((xx+.5)*c.shape[1]/shape[1]/c.tile_x).astype(int))


def integral_system(c,g,shape,lap,mean):
    ids=tile_ids(c).ravel();weight=(c.weight*(c.labels==g)).ravel()
    use=weight>0
    if not use.any():
        return np.broadcast_to(mean,(int(np.prod(shape)),3)).copy(),dict(
            integral_observations=0,total_observation_mass=0.,
            linear_system_max_residual=0.,empty_supported_class=True)
    yy,xx=np.indices(c.shape)
    node=(np.minimum(((yy+.5)*shape[0]/c.shape[0]).astype(int),shape[0]-1)*shape[1]
          +np.minimum(((xx+.5)*shape[1]/c.shape[1]).astype(int),shape[1]-1)).ravel()
    _,row=np.unique(ids[use],return_inverse=True)
    count=int(row.max())+1
    mass=np.bincount(row,weights=weight[use],minlength=count)
    averaging=sparse.coo_matrix((weight[use]/mass[row],(row,node[use])),shape=(count,np.prod(shape))).tocsr()
    target=np.stack([np.bincount(row,weights=weight[use]*c.rgb[...,ch].ravel()[use],minlength=count)/mass
                     for ch in range(3)],-1)
    observed_mass=mass*np.prod(shape)/np.prod(c.shape)
    matrix=lap+averaging.T@sparse.diags(observed_mass)@averaging+sparse.eye(lap.shape[0])*.015
    rhs=averaging.T@(observed_mass[:,None]*target)+.015*mean
    fitted=spsolve(matrix.tocsc(),rhs)
    return fitted,dict(integral_observations=count,
        total_observation_mass=float(observed_mass.sum()),
        linear_system_max_residual=float(np.max(np.abs(matrix@fitted-rhs))))


def integral_field(c,palette,he):
    means,cov,shape,spacing,lap=field_inputs(c,he,'uncertain')
    precision=np.linalg.inv(cov);scores=[];fits=[]
    for g in range(c.k):
        u,report=integral_system(c,g,shape,lap,means[g]);fits.append(report)
        delta=c.rgb-upsample(u,c.shape,shape)
        scores.append(-.5*np.einsum('...i,ij,...j->...',delta,precision,delta))
    return softmax(np.stack(scores,-1),-1),dict(model='integral original-tile field',fits=fits)


def gain_system(lap,mass,sums,means):
    diagonal=np.sum(mass*np.sum(means*means,-1)[None],-1)
    rhs=np.einsum('ngc,gc->n',sums,means)+.015
    matrix=lap+sparse.diags(diagonal+.015)
    gain=spsolve(matrix.tocsc(),rhs)
    if gain.min()<-1e-10:raise RuntimeError('Positive gain maximum principle failed')
    return gain,dict(linear_system_max_residual=float(np.max(np.abs(matrix@gain-rhs))),
                     minimum_gain=float(gain.min()),maximum_gain=float(gain.max()))


def gain_field(c,palette,he):
    means,cov,shape,spacing,lap=field_inputs(c,he,'uncertain')
    stats=[class_statistics(c,g,shape) for g in range(c.k)]
    mass=np.stack([x[0] for x in stats],-1);sums=np.stack([x[1] for x in stats],1)
    gain,report=gain_system(lap,mass,sums,means)
    fine=cv2.resize(gain.reshape(shape),c.shape[::-1],interpolation=cv2.INTER_LINEAR)
    delta=c.rgb[...,None,:]-fine[...,None,None]*means
    return softmax(-.5*np.einsum('...gi,ij,...gj->...g',delta,np.linalg.inv(cov),delta),-1),dict(
        report,model='common positive multiplicative sensor field')


def grid_geometry(c,step=4.):
    shape=tuple(max(2,int(np.ceil(n*s/step))) for n,s in zip(c.shape,c.mpp[::-1]))
    return shape,c.mpp*np.array(c.shape[::-1])/np.array(shape[::-1])


def grid_edges(guide):
    h,w=guide.shape[:2];index=np.arange(h*w).reshape(h,w)
    aa=[];bb=[];cost=[];axes=[]
    for axis in (0,1):
        a,b=[slice(None)]*2,[slice(None)]*2;a[axis]=slice(None,-1);b[axis]=slice(1,None)
        a,b=tuple(a),tuple(b)
        aa.append(index[a].ravel());bb.append(index[b].ravel())
        cost.append(np.sum((guide[a]-guide[b])**2,-1).ravel())
        axes.append(np.full(index[a].size,axis,np.int8))
    return tuple(np.concatenate(x) for x in (aa,bb,cost,axes))


def anatomical_tree(c,palette,he):
    q,report=edge_appearance(c,palette,he)
    shape,spacing=grid_geometry(c)
    guide=-np.log((cv2.resize(he,shape[::-1],interpolation=cv2.INTER_AREA).astype(float)+1)/256)
    a,b,cost,axis=grid_edges(guide);n=int(np.prod(shape))
    graph=sparse.coo_matrix((cost+1e-8,(a,b)),shape=(n,n)).tocsr()
    mst=minimum_spanning_tree(graph)
    tree=mst+mst.T
    order,parent=breadth_first_order(tree,0,directed=False,return_predecessors=True)
    if len(order)!=n:raise RuntimeError('Anatomical tree disconnected')
    # Child-edge contrasts are looked up from the same OD grid.
    flat=guide.reshape(-1,3)
    penalty=np.zeros(n);v=order[1:];p=parent[v]
    length=np.where(np.abs(v-p)==shape[1],spacing[0],spacing[1])
    penalty[v]=6*length*np.exp(-np.sum((flat[v]-flat[p])**2,-1)/.02)
    unary=resize_field(-np.log(np.maximum(q,1e-12)),shape).reshape(n,c.k)*np.prod(spacing)
    labels,detail=tree_potts(parent,order,unary,penalty)
    fine=cv2.resize(labels.reshape(shape),c.shape[::-1],interpolation=cv2.INTER_NEAREST)
    report.update(detail,model='anatomical tree Potts partition',partition_shape_yx=list(shape),
                  partition_spacing_xy=spacing.tolist(),projection='nearest categorical cells')
    return np.eye(c.k)[fine],report


def noise_probe_fit(matrix,mass,rhs,groups=None):
    rng=np.random.default_rng(1004)
    if groups is None:z=rng.choice([-1.,1.],(len(mass),16))
    else:z=rng.choice([-1.,1.],(int(groups.max())+1,16))[groups]
    all_rhs=np.c_[rhs,np.sqrt(mass)[:,None]*z]
    solution=splu(matrix.tocsc()).solve(all_rhs)
    return solution[:,:3],solution[:,3:],z,dict(
        linear_system_max_residual=float(np.max(np.abs(matrix@solution-all_rhs))),noise_probes=16)


def residual_noise(mass,target,fitted,noise,z):
    smooth=np.sqrt(mass)[:,None]*noise
    trace=float(np.mean(np.sum(z*smooth,0)))
    trace_square=float(np.mean(np.sum(smooth*smooth,0)))
    raw_df=float(np.count_nonzero(mass)-2*trace+trace_square)
    df=max(raw_df,1e-6)
    rss=float(np.sum(mass[:,None]*(fitted-target)**2))
    scale=(rss+6*.015)/(3*df+6)
    return scale,dict(residual_sum_squares=rss,residual_df_raw=raw_df,residual_df_used=df,
        residual_df_clipped=raw_df<1e-6,smoother_trace=trace,smoother_squared_trace=trace_square,
        estimated_noise_variance=float(scale),calibration='Coarse node means, not expert labels')


def uncertainty_field(c,palette,he,mode):
    means,cov,shape,spacing,lap=field_inputs(c,he,'uncertain')
    transform=whiten(cov);values=c.rgb@transform;means=means@transform
    fields=[];perturbations=[];variances=[];distances=[];fits=[]
    groups=tile_ids(c,shape).ravel() if mode=='blocks' else None
    for g in range(c.k):
        mass,sums=class_statistics(c,g,shape,values)
        matrix=lap+sparse.diags(mass+.015);rhs=sums+.015*means[g]
        fit,noise,z,detail=noise_probe_fit(matrix,mass,rhs,groups)
        scale=.015
        if mode=='residual':
            target=sums/np.maximum(mass[:,None],1e-30)
            scale,extra=residual_noise(mass,target,fit,noise,z);detail.update(extra)
        surface=upsample(fit,c.shape,shape)
        variance=cv2.resize((scale*np.mean(noise*noise,1)).reshape(shape),c.shape[::-1],interpolation=cv2.INTER_LINEAR)
        fields.append(surface);perturbations.append(noise)
        variances.append(variance);distances.append(np.sum((values-surface)**2,-1));fits.append(detail)
    if mode=='contrasts':
        scores=np.zeros((*c.shape,c.k))
        for g in range(c.k):
            for h in range(g+1,c.k):
                cross=.015*np.mean(perturbations[g]*perturbations[h],1)
                cross=cv2.resize(cross.reshape(shape),c.shape[::-1],interpolation=cv2.INTER_LINEAR)
                delta_g,delta_h=values-fields[g],values-fields[h]
                variance=(np.sum((fields[g]-fields[h])**2,-1)+variances[g]*distances[g]
                          +variances[h]*distances[h]-2*cross*np.sum(delta_g*delta_h,-1))
                contrast=.5*(distances[h]-distances[g])/np.sqrt(np.maximum(variance,1e-12))
                scores[...,g]+=log_ndtr(contrast);scores[...,h]+=log_ndtr(-contrast)
        scores/=max(c.k-1,1)
    else:
        scores=np.stack([-.5*d/(1+v)-1.5*np.log1p(v) for d,v in zip(distances,variances)],-1)
    return softmax(scores,-1),dict(model=mode+' appearance uncertainty',fits=fits,
        uncertainty_is_expert_calibrated=False,linearized_contrasts=mode=='contrasts')


def sparse_curvature_system(base,mass,sums,mean,iterations=60):
    matrix=sparse.diags(mass+.015)+base.T@base
    solve=splu(matrix.tocsc()).solve;rhs=sums+.015*mean
    u=solve(rhs);z=base@u;dual=np.zeros_like(z)
    for used in range(iterations):
        u=solve(rhs+base.T@(z-dual));curve=base@u
        old=z;v=curve+dual
        z=v*np.maximum(1-1/np.maximum(np.linalg.norm(v,axis=1),1e-30),0)[:,None]
        dual=dual+curve-z
        primal=float(np.max(np.abs(curve-z)))
        dual_residual=float(np.max(np.abs(base.T@(z-old))))
        gradient=(mass+.015)[:,None]*u-rhs+base.T@dual
        stationarity=float(np.max(np.abs(gradient)))
        if max(primal,dual_residual,stationarity)<1e-5:break
    objective=float(.5*np.sum(mass[:,None]*u*u)-np.sum(sums*u)
                    +.0075*np.sum((u-mean)**2)+np.linalg.norm(base@u,axis=1).sum())
    return u,dict(iterations=used+1,primal_residual=primal,dual_residual=dual_residual,
        gradient_infinity_norm=stationarity,objective=objective,
        converged=max(primal,dual_residual,stationarity)<1e-5)


def guided_lift(fitted,guide,he,shape):
    h,w=shape;gh,gw=guide.shape[:2]
    yy,xx=np.indices(shape)
    fy=(yy+.5)*gh/h-.5;fx=(xx+.5)*gw/w-.5
    y0=np.floor(fy).astype(int);x0=np.floor(fx).astype(int)
    ay=fy-y0;ax=fx-x0
    out=np.zeros((h,w,3));total=np.zeros((h,w))
    for dy,dx in ((0,0),(0,1),(1,0),(1,1)):
        y=np.clip(y0+dy,0,gh-1);x=np.clip(x0+dx,0,gw-1)
        weight=(ay if dy else 1-ay)*(ax if dx else 1-ax)
        weight*=.02+np.exp(-np.sum((he-guide[y,x])**2,-1)/.02)
        out+=weight[...,None]*fitted[y,x];total+=weight
    return out/total[...,None]


def bending_field(c,palette,he,mode):
    from .cycle_five import field_laplacian
    means,cov,shape,spacing,bending=field_inputs(c,he,'bending')
    transform=whiten(cov);values=c.rgb@transform;means=means@transform
    base=field_laplacian(shape,spacing);scores=[];fits=[]
    guide=-np.log((cv2.resize(he,shape[::-1],interpolation=cv2.INTER_AREA).astype(float)+1)/256)
    original=-np.log((cv2.resize(he,c.shape[::-1],interpolation=cv2.INTER_AREA).astype(float)+1)/256)
    for g in range(c.k):
        mass,sums=class_statistics(c,g,shape,values)
        matrix=bending+sparse.diags(mass+.015);rhs=sums+.015*means[g]
        if mode=='sparse':u,detail=sparse_curvature_system(base,mass,sums,means[g])
        else:
            solver=splu(matrix.tocsc());u=solver.solve(rhs)
            detail=dict(linear_system_max_residual=float(np.max(np.abs(matrix@u-rhs))))
        if mode=='lift':surface=guided_lift(u.reshape(*shape,3),guide,original,c.shape)
        else:surface=upsample(u,c.shape,shape)
        delta=values-surface
        if mode=='logvariance':
            _,second=class_statistics(c,g,shape,values*values)
            target=sums/np.maximum(mass[:,None],1e-30)
            residual=second/np.maximum(mass[:,None],1e-30)-2*u*target+u*u
            variance=.2+.8*np.maximum(residual,0)
            log_rhs=mass[:,None]*np.log(variance)
            log_fit=solver.solve(log_rhs)
            detail['log_variance_system_residual']=float(np.max(np.abs(matrix@log_fit-log_rhs)))
            logv=upsample(log_fit,c.shape,shape)
            # exp clipping is numerical overflow protection, recorded explicitly.
            detail['log_variance_values_clipped']=int(np.count_nonzero((logv<-30)|(logv>30)))
            logv=np.clip(logv,-30,30)
            score=-.5*np.sum(delta*delta*np.exp(-logv)+logv,-1)
        else:score=-.5*np.sum(delta*delta,-1)
        scores.append(score);fits.append(detail)
    return softmax(np.stack(scores,-1),-1),dict(model=mode+' bending appearance',fits=fits)


GLASS_MODES={'N10':'hull','N11':'forest','N12':'biconnected'}


def envelope_glass(white):
    stained=~white
    if not stained.any():return white.copy()
    ys=np.flatnonzero(stained.any(1))
    left=stained.argmax(1)[ys];right=stained.shape[1]-1-stained[:,::-1].argmax(1)[ys]
    points=np.r_[np.c_[left,ys],np.c_[right,ys]].astype(np.int32)
    hull=cv2.convexHull(points);inside=np.zeros(white.shape,np.uint8)
    cv2.fillConvexPoly(inside,hull,1)
    return white & ~inside.astype(bool)


def topological_glass(he,spacing,mode):
    white,_,policy=white_regions(he)
    if mode=='hull':return envelope_glass(white),dict(white_policy=policy,geometry='convex stained-pixel hull')
    shape=tuple(max(2,int(np.ceil(n*s/4))) for n,s in zip(he.shape[:2],np.asarray(spacing)[::-1]))
    image=cv2.resize(he,shape[::-1],interpolation=cv2.INTER_AREA)
    coarse_white,_,_=white_regions(image)
    if mode=='biconnected':excluded=biconnected_white(coarse_white)
    else:
        od=-np.log((image.astype(float)+1)/256)
        a,b,cost,_=grid_edges(od);order=np.lexsort((b,a,cost))
        seeds=np.zeros(shape,np.int32);seeds[od.sum(-1)>=.45]=2
        border=np.zeros(shape,bool);border[[0,-1]]=True;border[:,[0,-1]]=True
        seeds[coarse_white & border]=1
        assigned=seeded_forest(a[order],b[order],seeds.ravel()).reshape(shape)
        excluded=coarse_white & (assigned==1)
    lifted=cv2.resize(excluded.astype(np.uint8),he.shape[1::-1],interpolation=cv2.INTER_NEAREST).astype(bool)
    return white & lifted,dict(white_policy=policy,glass_grid_shape_yx=list(shape),
        glass_grid_spacing_xy=(np.asarray(spacing)*np.asarray(he.shape[1::-1])/np.asarray(shape[::-1])).tolist(),
        graph_neighbours=4,geometry=mode)


def background_field(c,palette,he,mode):
    q,detail=edge_appearance(c,palette,he)
    excluded,report=topological_glass(cv2.resize(he,c.shape[::-1],interpolation=cv2.INTER_AREA),c.mpp,mode)
    detail.update(report,glass_exclusion=excluded,proposed_glass_pixels=int((excluded & c.support).sum()))
    return q,detail


def multinomial_objective(r,x,prior,colors):
    prediction=r@colors
    value=-np.sum(x*np.log(prediction))+.006*np.sum((r-prior)**2)
    gradient=-(x/prediction)@colors.T+.012*(r-prior)
    return float(value),gradient


def palette_field(c,palette,he,mode):
    colors=np.asarray(palette,float)[c.ids-1]/255
    if mode=='categorical':
        distance=np.sum((c.rgb[...,None,:]-colors)**2,-1)
        return np.eye(c.k)[distance.argmin(-1)],dict(model='categorical codeword least squares',mixture_fractions_fitted=False)
    x,prior,index=colour_nodes(c)
    if mode=='multinomial':
        positive=colors+.5/255;brightness=positive.sum(-1)
        observed=x+.5/255;observed/=observed.sum(-1,keepdims=True)
        rprior=prior*brightness;rprior/=rprior.sum(-1,keepdims=True)
        normalized=positive/brightness[:,None]
        r,detail=projected_inverse(lambda r:multinomial_objective(r,observed,rprior,normalized),rprior)
        solved=r/brightness;solved/=solved.sum(-1,keepdims=True)
    else:
        od=lambda a:-np.log((a+1/255)/(1+1/255))
        solved,detail=native.simplex_qp(od(x),prior,od(colors))
    return correct_coarse(solved[index],c),dict(detail,model=mode+' palette inverse',occupied_colour_nodes=len(x),
        forward_model_is_verified_physical_sensor=False,
        branch_difference='Occupied-node aggregation and prior differ from direct-pixel affine C17')


IMPLEMENTATIONS=dict(N01=integral_field,N02=gain_field,N03=anatomical_tree,
    N04=lambda c,p,h:uncertainty_field(c,p,h,'blocks'),
    N05=lambda c,p,h:uncertainty_field(c,p,h,'contrasts'),
    N06=lambda c,p,h:uncertainty_field(c,p,h,'residual'),
    N07=lambda c,p,h:bending_field(c,p,h,'sparse'),
    N08=lambda c,p,h:bending_field(c,p,h,'logvariance'),
    N09=lambda c,p,h:bending_field(c,p,h,'lift'),
    N10=lambda c,p,h:background_field(c,p,h,'hull'),
    N11=lambda c,p,h:background_field(c,p,h,'forest'),
    N12=lambda c,p,h:background_field(c,p,h,'biconnected'),
    N13=lambda c,p,h:palette_field(c,p,h,'categorical'),
    N14=lambda c,p,h:palette_field(c,p,h,'multinomial'),
    N15=lambda c,p,h:palette_field(c,p,h,'optical-density'))


def predict(key,rgb,coarse,support,mpp_xy,palette,he=None):
    from .registry import CYCLE_TEN
    if key not in IMPLEMENTATIONS:raise ValueError('Unimplemented cycle-ten child')
    c=Evidence(rgb,coarse,support,mpp_xy)
    if not c.support.any():raise ValueError('Empty tissue support')
    if c.k==1:
        q=np.ones((*c.shape,1));detail=dict(single_class=True)
        if key in GLASS_MODES:
            excluded,report=topological_glass(cv2.resize(he,c.shape[::-1],interpolation=cv2.INTER_AREA),c.mpp,GLASS_MODES[key])
            detail.update(report,glass_exclusion=excluded)
    else:q,detail=IMPLEMENTATIONS[key](c,palette,he)
    if q.shape!=(*c.shape,c.k) or not np.isfinite(q).all() or q.min()<-1e-6 or not np.allclose(q.sum(-1),1,atol=1e-5):
        raise ValueError('Invalid normalized cycle-ten scores')
    labels=c.ids[q.argmax(-1)].astype(np.uint8)
    excluded=detail.pop('glass_exclusion',None)
    if excluded is not None:labels[excluded]=0
    detail.update(analysis_mpp_xy=c.mpp.tolist(),analysis_shape_yx=list(c.shape),
        fine_reference_used=False,hard_anchors=False,fixed_margin=False,candidate=key,
        parent=CYCLE_TEN[key].parent,mechanism=CYCLE_TEN[key].mechanism)
    return labels,detail
