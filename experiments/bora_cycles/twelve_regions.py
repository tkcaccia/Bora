"""Palette observation, tile-integral and contiguous-region hypotheses."""
import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr,logsumexp,softmax
from bora.evidence import normalize,project_simplex
from .cycle_two import colour_nodes
from .cycle_six import edge_appearance
from .cycle_ten import tile_ids
from .cycle_eight import resize_field
from .eleven_support import grid_geometry
from .methods import correct_coarse,palette_active_set
from .native_twelve import reconstruct,bayes_regions


def fuzzy_regions(c,palette,he):
    q,report=edge_appearance(c,palette,he)
    shape,spacing=grid_geometry(c.shape,c.mpp,4.)
    parent_grid=resize_field(q,shape)
    ceiling=parent_grid.copy()
    supported=cv2.resize(c.support.astype(np.uint8),shape[::-1],interpolation=cv2.INTER_NEAREST)>0
    ceiling*=supported[...,None]
    label=cv2.resize(c.labels.astype(np.uint8),shape[::-1],interpolation=cv2.INTER_NEAREST)
    trust=cv2.resize(c.weight,shape[::-1],interpolation=cv2.INTER_AREA)
    marker=ceiling*(np.eye(c.k)[label]*(trust>.8)[...,None])
    closure=reconstruct(ceiling,marker)
    no_seed=closure.sum(-1)==0
    # A4um support cell can miss a thin real native tissue fragment. Its zero
    # graph ceiling is a propagation barrier, not evidence of equal classes.
    # Retain the parent's nonzero evidence there before lifting to finer pixels.
    closure[no_seed]=parent_grid[no_seed]
    q=normalize(resize_field(closure,c.shape,cv2.INTER_LINEAR))
    report.update(model='class gray-ceiling geodesic reconstruction',connectivity_grid_mpp_xy=spacing.tolist(),
        seeded_cells_per_class=np.count_nonzero(marker,axis=(0,1)).tolist(),
        no_connected_seed_cells=int(no_seed.sum()),unseeded_fallback='original parent class probabilities',
        unsupported_quadrature_cells=int((~supported).sum()),
        primitive='C++ maximum-priority widest paths',primitive_threads=2)
    return q,report


def plurality_dual(offsets,logp,tile,winner,count,tau=2.):
    """Smooth convex dual for per-tile mean KL plus squared plurality violations."""
    t,k=offsets.shape
    shift=-offsets.copy();shift[np.arange(t),winner]=offsets.sum(-1)
    logit=logp+shift[tile];logz=logsumexp(logit,axis=-1)
    q=softmax(logit,axis=-1)
    # Equal weight per tile, then mean over its quadrature cells.
    value=float(np.sum(logz/count[tile])+.5*np.sum(offsets*offsets)/tau)
    mean=np.stack([np.bincount(tile,weights=q[:,g]/count[tile],minlength=t) for g in range(k)],-1)
    gradient=mean[np.arange(t),winner,None]-mean+offsets/tau
    gradient[np.arange(t),winner]=0
    return value,gradient


def tile_plurality(c,palette,he):
    q,parent=palette_active_set(c,palette)
    shape,spacing=grid_geometry(c.shape,c.mpp,4.)
    p=normalize(resize_field(q,shape));grid_tiles=tile_ids(c,shape)
    observed=cv2.resize(c.coarse,shape[::-1],interpolation=cv2.INTER_NEAREST)
    grid_support=cv2.resize(c.support.astype(np.uint8),shape[::-1],interpolation=cv2.INTER_NEAREST)>0
    ids,inverse=np.unique(grid_tiles,return_inverse=True);inverse=inverse.ravel();t=len(ids)
    votes=np.stack([np.bincount(inverse,weights=((observed==v)&grid_support).ravel(),minlength=t) for v in c.ids],-1)
    winner=votes.argmax(-1);valid=votes.sum(-1)>0;count=np.bincount(inverse,minlength=t)
    logp=np.log(np.maximum(p.reshape(-1,c.k),1e-30))
    bounds=[(0.,0.) if (not valid[a] or g==winner[a]) else (0.,None) for a in range(t) for g in range(c.k)]
    def objective(flat):
        value,grad=plurality_dual(flat.reshape(t,c.k),logp,inverse,winner,count)
        return value,grad.ravel()
    result=minimize(objective,np.zeros(t*c.k),jac=True,bounds=bounds,method='L-BFGS-B',options=dict(maxiter=100,ftol=1e-12,gtol=1e-6))
    offsets=result.x.reshape(t,c.k);shift=-offsets.copy();shift[np.arange(t),winner]=offsets.sum(-1)
    full_ids=tile_ids(c);lookup=np.searchsorted(ids,full_ids)
    lookup=np.minimum(lookup,len(ids)-1);sampled=ids[lookup]==full_ids
    full_shift=shift[lookup]*sampled[...,None]
    final=softmax(np.log(np.maximum(q,1e-30))+full_shift,axis=-1)
    grad=objective(result.x)[1].reshape(t,c.k)
    projected=offsets-np.maximum(offsets-grad,0);projected[~valid]=0;projected[np.arange(t),winner]=0
    residual=float(np.max(np.abs(projected)))
    return final,dict(model='soft tile-integral plurality KL projection',palette_fit=parent,
        quadrature_mpp_xy=spacing.tolist(),observed_tiles=int(valid.sum()),
        unsampled_partial_tiles=len(np.unique(full_ids[~sampled])),
        unsampled_partial_tile_rule='Keep parent probabilities without a tile offset',
        dual_iterations=int(result.nit),dual_projected_residual=residual,
        converged=residual<1e-5,optimizer_success=bool(result.success),optimizer_message=str(result.message),
        plurality_penalty=1.,all_pixel_labels_remain_free=True)


def censored_objective(q,x,prior,colours,sigma=.05,ridge=.012):
    mu=q@colours;low=x<=1/255;high=x>=254/255
    residual=mu-x;cost=.5*residual*residual;derivative=residual.copy()
    for flag,z,direction in ((low,(1/255-mu)/sigma,1.),(high,(mu-254/255)/sigma,-1.)):
        logcdf=log_ndtr(z)
        cost=np.where(flag,-sigma*sigma*logcdf,cost)
        mills=np.exp(-.5*z*z-.5*np.log(2*np.pi)-logcdf)
        derivative=np.where(flag,direction*sigma*mills,derivative)
    value=np.sum(cost,axis=-1)+.5*ridge*np.sum((q-prior)**2,-1)
    gradient=derivative@colours.T+ridge*(q-prior)
    return value,gradient


def censored_palette(c,palette,he):
    x,prior,index=colour_nodes(c);colours=np.asarray(palette,float)[c.ids-1]/255
    q=prior.copy();ridge=.012;step=.95/(float(np.linalg.norm(colours,2)**2)+ridge)
    initial=float(censored_objective(q,x,prior,colours)[0].sum())
    for used in range(160):
        _,gradient=censored_objective(q,x,prior,colours)
        changed=project_simplex(q-step*gradient)
        if np.max(np.abs(changed-q))<1e-8:q=changed;break
        q=changed
    value,gradient=censored_objective(q,x,prior,colours)
    residual=float(np.max(np.abs(q-project_simplex(q-gradient))))
    return correct_coarse(q[index],c),dict(model='censored-channel Gaussian palette observation',
        iterations=used+1,projected_stationarity=residual,converged=residual<1e-6,
        initial_objective=initial,final_objective=float(value.sum()),
        censored_colour_channels=int(((x<=1/255)|(x>=254/255)).sum()),
        sigma=.05,prior_half_ridge=.006,occupied_colour_nodes=len(x))


def bayesian_regions(c,palette,he):
    q,parent=palette_active_set(c,palette);shape,spacing=grid_geometry(c.shape,c.mpp,4.)
    logp=resize_field(np.log(np.maximum(q,1e-30)),shape)
    support=cv2.resize(c.support.astype(np.uint8),shape[::-1],interpolation=cv2.INTER_NEAREST)>0
    regions,report=bayes_regions(logp,support);n=int(regions.max())+1
    pooled=np.stack([np.bincount(regions.ravel(),weights=logp[...,g].ravel(),minlength=n) for g in range(c.k)],-1)
    posterior=softmax(pooled,axis=-1)[regions]
    lifted=resize_field(posterior,c.shape,cv2.INTER_NEAREST)
    return lifted,dict(report,model='integrated shared-class region evidence',palette_fit=parent,
        region_grid_mpp_xy=spacing.tolist(),greedy_partition_is_global_optimum=False,
        independent_probability_cells_assumed=True,primitive='C++ deterministic Bayes-factor priority queue')
