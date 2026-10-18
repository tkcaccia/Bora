"""Cycle11 field models and explicit, finite numerical contracts."""
import cv2
import numpy as np
from scipy import sparse
from scipy.fft import dctn, idctn
from scipy.sparse.linalg import LinearOperator, cg, splu, spsolve
from scipy.special import ndtr, ndtri, softmax
from scipy.stats import qmc

from .cycle_eight import class_statistics, field_inputs
from .cycle_five import field_laplacian
from .cycle_nine import sampling_variance, upsample
from .cycle_six import edge_appearance,field_geometry
from .cycle_ten import grid_edges, whiten
from bora.evidence import project_simplex
from .cycle_two import colour_nodes
from .methods import correct_coarse,palette_active_set,fit_linear


def order_projection(values, means, sweeps=20, tolerance=1e-7):
    """Euclidean projection onto oriented pairwise halfspaces by Dykstra."""
    z=values.copy();pairs=[]
    for g in range(len(means)):
        for h in range(g+1,len(means)):
            normal=means[g]-means[h];length=float(normal@normal)
            if length>1e-12:pairs.append((g,h,normal,length))
    corrections=np.zeros((len(pairs),len(z)))
    change=0.
    for used in range(sweeps):
        old=z.copy()
        for i,(g,h,normal,length) in enumerate(pairs):
            # Each stored correction is a scalar along the halfspace normal.
            a=z[:,g]+corrections[i,:,None]*normal
            b=z[:,h]-corrections[i,:,None]*normal
            shift=np.maximum(.25*length-(a-b)@normal,0)/(2*length)
            z[:,g]=a+shift[:,None]*normal;z[:,h]=b-shift[:,None]*normal
            corrections[i]=-shift
        change=float(np.max(np.abs(z-old)))
        if change<tolerance:break
    violation=max([float(np.maximum(.25*l-(z[:,g]-z[:,h])@n,0).max())
                   for g,h,n,l in pairs] or [0.])
    return z,dict(projection_sweeps=used+1,projection_change=change,
                  order_constraint_violation=violation)


def ordered_system(lap,masses,sums,means,iterations=80):
    n,k=masses.shape;rho=.5
    matrices=[lap+sparse.diags(masses[:,g]+.015) for g in range(k)]
    rhs=[sums[:,g]+.015*means[g] for g in range(k)]
    solvers=[splu((a+sparse.eye(n)*rho).tocsc()).solve for a in matrices]
    z=np.broadcast_to(means,(n,k,3)).copy();dual=np.zeros_like(z);u=z.copy()
    primal=dual_residual=np.inf;projection={}
    for used in range(iterations):
        for g in range(k):u[:,g]=solvers[g](rhs[g]+rho*(z[:,g]-dual[:,g]))
        old=z;z,projection=order_projection(u+dual,means)
        dual+=u-z;primal=float(np.max(np.abs(u-z)))
        dual_residual=float(rho*np.max(np.abs(z-old)))
        if max(primal,dual_residual,projection['order_constraint_violation'])<1e-5:break
    # z is the returned constrained iterate; do not call an unfinished solve exact.
    stationarity=max(float(np.max(np.abs(matrices[g]@z[:,g]-rhs[g]+rho*dual[:,g]))) for g in range(k))
    return z,dict(iterations=used+1,primal_residual=primal,dual_residual=dual_residual,
        gradient_infinity_norm=stationarity,**projection,
        converged=max(primal,dual_residual,stationarity,projection['order_constraint_violation'])<1e-5)


def ordered_field(c,palette,he):
    means,cov,shape,spacing,lap=field_inputs(c,he,'uncertain')
    transform=whiten(cov);values=c.rgb@transform;means=means@transform
    stats=[class_statistics(c,g,shape,values) for g in range(c.k)]
    masses=np.stack([s[0] for s in stats],-1);sums=np.stack([s[1] for s in stats],1)
    fitted,report=ordered_system(lap,masses,sums,means)
    scores=[-.5*np.sum((values-upsample(fitted[:,g],c.shape,shape))**2,-1) for g in range(c.k)]
    return softmax(np.stack(scores,-1),-1),dict(report,model='convex oriented appearance order')


def fractional_diagonal(shape,spacing):
    y,x=[np.arange(n) for n in shape]
    eigen=(2*np.sin(np.pi*y/(2*shape[0]))/spacing[1])[:,None]**2
    eigen=eigen+(2*np.sin(np.pi*x/(2*shape[1]))/spacing[0])[None,:]**2
    return 16*np.sqrt(eigen)


def fractional_system(mass,rhs,shape,spacing,iterations=160):
    diagonal=fractional_diagonal(shape,spacing)
    def action(a):
        field=a.reshape(shape)
        return ((mass.reshape(shape)+.015)*field+idctn(diagonal*dctn(field,norm='ortho'),norm='ortho')).ravel()
    def precondition(a):
        return idctn(dctn(a.reshape(shape),norm='ortho')/(diagonal+mass.mean()+.015),norm='ortho').ravel()
    matrix=LinearOperator((len(mass),len(mass)),matvec=action)
    pre=LinearOperator(matrix.shape,matvec=precondition)
    fitted=[];reports=[]
    for j in range(rhs.shape[1]):
        count=[0]
        def callback(a):count[0]+=1
        u,status=cg(matrix,rhs[:,j],M=pre,rtol=1e-7,atol=1e-10,maxiter=iterations,callback=callback)
        residual=float(np.max(np.abs(action(u)-rhs[:,j])))
        fitted.append(u);reports.append(dict(iterations=count[0],cg_status=int(status),
            linear_system_max_residual=residual,converged=status==0))
    return np.stack(fitted,-1),reports


def fractional_field(c,palette,he):
    _,_,means,cov=fit_linear(c);shape,spacing=field_geometry(c)
    precision=np.linalg.inv(cov);scores=[];fits=[]
    for g in range(c.k):
        mass,sums=class_statistics(c,g,shape)
        u,reports=fractional_system(mass,sums+.015*means[g],shape,spacing);fits.extend(reports)
        delta=c.rgb-upsample(u,c.shape,shape)
        scores.append(-.5*np.einsum('...i,ij,...j->...',delta,precision,delta))
    return softmax(np.stack(scores,-1),-1),dict(model='fractional Neumann appearance transport',fits=fits,
        prior_uses_original_he=False,field_shape_yx=list(shape))


def stain_regions(c,palette,he):
    from .native_eleven import contrast_components
    q,report=edge_appearance(c,palette,he)
    od=-np.log((cv2.resize(he,c.shape[::-1],interpolation=cv2.INTER_AREA).astype(float)+1)/256)
    a,b,squared,_=grid_edges(od);cost=np.sqrt(squared);order=np.lexsort((b,a,cost))
    regions=contrast_components(a[order],b[order],cost[order],np.prod(c.shape),float(c.mpp.prod()),16.)
    count=np.bincount(regions);number=len(count)
    logq=np.log(np.maximum(q.reshape(-1,c.k),1e-12))
    scores=np.stack([np.bincount(regions,weights=logq[:,g],minlength=number)/count for g in range(c.k)],-1)
    report.update(model='adaptive internal-contrast stain components',stain_components=number,
        contrast_scale_area=16.,region_evidence='geometric mean',extra_minimum_size_merge=False)
    return softmax(scores,-1)[regions].reshape(*c.shape,c.k),report


def sampling_fields(c,he):
    means,cov,shape,spacing,lap=field_inputs(c,he,'uncertain')
    transform=whiten(cov);means=means@transform;values=c.rgb@transform
    fields=[];variances=[];fits=[]
    for g in range(c.k):
        mass,sums=class_statistics(c,g,shape,values)
        matrix=lap+sparse.diags(mass+.015)
        u,v,report=sampling_variance(matrix,mass,sums+.015*means[g])
        fields.append(upsample(u,c.shape,shape));fits.append(report)
        variances.append(cv2.resize(v.reshape(shape),c.shape[::-1],interpolation=cv2.INTER_LINEAR))
    return values,np.stack(fields,-2),np.stack(variances,-1),fits


def folded_normal(delta,variance):
    sd=np.sqrt(np.maximum(variance,1e-30));z=delta/sd
    return sd*np.sqrt(2/np.pi)*np.exp(-.5*z*z)+delta*(2*ndtr(z)-1)


def affinity_scores(distance,variance):
    mean_variance=1+.5*variance
    return -.125*distance/mean_variance-1.5*np.log(mean_variance)+.75*np.log1p(variance)


def uncertain_decision(c,palette,he,mode):
    values,fields,variance,fits=sampling_fields(c,he)
    delta=values[...,None,:]-fields
    report=dict(model=mode+' uncertain appearance decision',fits=fits,
        uncertainty_is_expert_calibrated=False)
    if mode=='absolute':scores=-folded_normal(delta,variance[...,None]).sum(-1)
    elif mode=='affinity':scores=affinity_scores(np.sum(delta*delta,-1),variance)
    else:
        points=qmc.Sobol(3*c.k,scramble=True,seed=1104).random_base2(4)
        normals=ndtri(np.clip(points,1e-12,1-1e-12)).reshape(16,c.k,3)
        normals=np.concatenate([normals,-normals]);votes=np.zeros((*c.shape,c.k))
        sd=np.sqrt(variance)[...,None]
        for z in normals:
            winner=np.sum((delta-sd*z)**2,-1).argmin(-1)
            for g in range(c.k):votes[...,g]+=winner==g
        report.update(winner_draws=32,spatial_joint_posterior_sampled=False,
                      assumed_independent_local_prototype_noise=True)
        return (votes+.5)/(32+.5*c.k),report
    return softmax(scores,-1),report


def matrix_log_spd(a):
    values,vectors=np.linalg.eigh(a)
    if np.any(values<=0):raise ValueError('Positive covariance target required')
    return np.einsum('...ij,...j,...kj->...ik',vectors,np.log(values),vectors)


def matrix_gaussian_score(delta,log_covariance):
    values,vectors=np.linalg.eigh(log_covariance)
    clipped=int(np.count_nonzero((values<-30)|(values>30)));values=np.clip(values,-30,30)
    rotated=np.einsum('...ji,...j->...i',vectors,delta)
    return -.5*(np.sum(rotated*rotated*np.exp(-values),-1)+values.sum(-1)),clipped


def shared_logvariance(lap,masses,targets):
    """One scalar spatial field plus identifiable class-channel offsets."""
    n,k=masses.shape
    coupling=np.repeat(masses,3,axis=1)
    spatial=lap+sparse.diags(3*masses.sum(-1)+.015)
    offsets=sparse.diags(coupling.sum(0)+.015)
    matrix=sparse.bmat([[spatial,sparse.csr_matrix(coupling)],
                       [sparse.csr_matrix(coupling.T),offsets]],format='csc')
    weighted=masses[...,None]*targets
    rhs=np.r_[weighted.sum((1,2)),weighted.sum(0).ravel()]
    answer=spsolve(matrix,rhs)
    return answer[:n],answer[n:].reshape(k,3),dict(
        linear_system_max_residual=float(np.max(np.abs(matrix@answer-rhs))),
        common_noise_nodes=n,class_channel_offsets=3*k)


def variance_field(c,palette,he,mode):
    means,cov,shape,spacing,bending=field_inputs(c,he,'bending')
    transform=whiten(cov);values=c.rgb@transform;means=means@transform
    fields=[];logfields=[];masses=[];targets=[];fits=[]
    for g in range(c.k):
        mass,sums=class_statistics(c,g,shape,values)
        matrix=bending+sparse.diags(mass+.015);solve=splu(matrix.tocsc()).solve
        rhs=sums+.015*means[g];u=solve(rhs)
        report=dict(linear_system_max_residual=float(np.max(np.abs(matrix@u-rhs))))
        target=sums/np.maximum(mass[:,None],1e-30)
        if mode=='matrix':
            outer=(values[..., :,None]*values[...,None,:]).reshape(*c.shape,9)
            _,second=class_statistics(c,g,shape,outer)
            residual=second.reshape(-1,3,3)/np.maximum(mass[:,None,None],1e-30)
            residual-=u[:,:,None]*target[:,None,:]+target[:,:,None]*u[:,None,:]
            residual+=u[:,:,None]*u[:,None,:]
            ev,vec=np.linalg.eigh(.5*(residual+residual.transpose(0,2,1)))
            residual=np.einsum('nij,nj,nkj->nik',vec,np.maximum(ev,0),vec)
            logtarget=matrix_log_spd(.2*np.eye(3)+.8*residual)
            upper=np.triu_indices(3)
            logrhs=mass[:,None]*logtarget[:,upper[0],upper[1]];logfit=solve(logrhs)
            report['log_variance_system_residual']=float(np.max(np.abs(matrix@logfit-logrhs)))
            lifted=upsample(logfit,c.shape,shape);symmetric=np.empty((*c.shape,3,3))
            symmetric[...,upper[0],upper[1]]=lifted;symmetric[...,upper[1],upper[0]]=lifted
            logfields.append(symmetric);report['transported_symmetric_components']=6
        else:
            _,second=class_statistics(c,g,shape,values*values)
            residual=second/np.maximum(mass[:,None],1e-30)-2*u*target+u*u
            logtarget=np.log(.2+.8*np.maximum(residual,0))
            if mode=='shared':masses.append(mass);targets.append(logtarget)
            else:
                logrhs=mass[:,None]*logtarget;logfit=solve(logrhs)
                report['log_variance_system_residual']=float(np.max(np.abs(matrix@logfit-logrhs)))
                logfields.append(upsample(logfit,c.shape,shape))
        fields.append(upsample(u,c.shape,shape));fits.append(report)
    if mode=='shared':
        a,b,report=shared_logvariance(bending,np.stack(masses,-1),np.stack(targets,1))
        surface=cv2.resize(a.reshape(shape),c.shape[::-1],interpolation=cv2.INTER_LINEAR)
        logfields=[surface[...,None]+b[g] for g in range(c.k)];fits.append(report)
    scores=[];clipped=0
    for g in range(c.k):
        delta=values-fields[g];logv=logfields[g]
        if mode=='matrix':score,n=matrix_gaussian_score(delta,logv);clipped+=n
        else:
            clipped+=int(np.count_nonzero((logv<-30)|(logv>30)));logv=np.clip(logv,-30,30)
            if mode=='hyvarinen':score=-.5*np.sum(delta*delta*np.exp(-2*logv),-1)+np.exp(-logv).sum(-1)
            else:score=-.5*np.sum(delta*delta*np.exp(-logv)+logv,-1)
        scores.append(score)
    return softmax(np.stack(scores,-1),-1),dict(model=mode+' appearance variance',fits=fits,
        log_variance_values_clipped=clipped,uncertainty_is_expert_calibrated=False)


def gradient_edges(q,spacing):
    return (q[:,1:]-q[:,:-1])/spacing[0],(q[1:]-q[:-1])/spacing[1]


def adjoint_edges(px,py,shape,spacing):
    result=np.zeros(shape,dtype=px.dtype)
    result[:,:-1]-=px/spacing[0];result[:,1:]+=px/spacing[0]
    result[:-1]-=py/spacing[1];result[1:]+=py/spacing[1]
    return result


def tv_palette(x,prior,colors,wx,wy,spacing,iterations=120,strength=.02):
    """One convex spatial inverse, with a finite primal-dual solve.

    The explicit smooth-step condition is tau*L/2+tau*sigma*||K||²<1.
    Vector TV is a group norm across classes on each physical grid edge.
    """
    q=np.asarray(prior,float).copy();bar=q.copy();px=np.zeros((*q.shape[:1],q.shape[1]-1,q.shape[2]));py=np.zeros((q.shape[0]-1,*q.shape[1:]))
    lipschitz=float(np.linalg.norm(colors,2)**2+.012)
    bound=4/spacing[0]**2+4/spacing[1]**2
    tau=.9/(lipschitz+np.sqrt(bound));sigma=.9/np.sqrt(bound)
    def clip_dual(p,weight):
        radius=strength*weight
        return p*np.minimum(1,radius/np.maximum(np.linalg.norm(p,axis=-1),1e-30))[...,None]
    def smooth_gradient(a):return (a@colors-x)@colors.T+.012*(a-prior)
    def objective(a):
        dx,dy=gradient_edges(a,spacing)
        return float(.5*np.sum((a@colors-x)**2)+.006*np.sum((a-prior)**2)
            +strength*(np.sum(wx*np.linalg.norm(dx,axis=-1))+np.sum(wy*np.linalg.norm(dy,axis=-1))))
    initial=objective(q)
    for used in range(iterations):
        dx,dy=gradient_edges(bar,spacing)
        px=clip_dual(px+sigma*dx,wx);py=clip_dual(py+sigma*dy,wy)
        changed=project_simplex(q-tau*(smooth_gradient(q)+adjoint_edges(px,py,q.shape,spacing)))
        bar=2*changed-q;q=changed
    dx,dy=gradient_edges(q,spacing)
    primal=float(np.max(np.abs(q-project_simplex(q-smooth_gradient(q)-adjoint_edges(px,py,q.shape,spacing)))))
    dual=max(float(np.max(np.abs(px-clip_dual(px+dx,wx)),initial=0)),
             float(np.max(np.abs(py-clip_dual(py+dy,wy)),initial=0)))
    return q,dict(iterations=used+1,objective_initial=initial,objective_final=objective(q),
        primal_projected_residual=primal,dual_fixed_point_residual=dual,
        converged=max(primal,dual)<1e-5,simplex_sum_residual=float(np.max(np.abs(q.sum(-1)-1))),
        tv_strength=strength,primal_step=tau,dual_step=sigma,
        step_condition=tau*lipschitz/2+tau*sigma*bound)


def spatial_palette(c,palette,he):
    colors=np.asarray(palette,float)[c.ids-1]/255
    original=cv2.resize(he,c.shape[::-1],interpolation=cv2.INTER_AREA)
    od=-np.log((original.astype(float)+1)/256)
    dx,dy=gradient_edges(od,c.mpp)
    wx=np.exp(-np.sum(dx*dx,axis=-1)/(.2**2*2));wy=np.exp(-np.sum(dy*dy,axis=-1)/(.2**2*2))
    q,report=tv_palette(c.rgb,c.lattice(),colors,wx,wy,c.mpp)
    return correct_coarse(q,c),dict(report,model='joint physical H&E-weighted vector-TV palette inverse',
        dense_fraction_field=True,posthoc_label_smoothing=False,
        coarse_additive_correction='same weak correction as parent C17; solver residuals describe the fitted fractions before this shared correction')


def clr_objective(q,x,prior,colors):
    observation=q@colors+1/255
    logs=np.log(observation);target=np.log(x+1/255)
    delta=logs-logs.mean(-1,keepdims=True)-target+target.mean(-1,keepdims=True)
    value=.5*np.sum(delta*delta,axis=-1)+.006*np.sum((q-prior)**2,axis=-1)
    gradient=((delta-delta.mean(-1,keepdims=True))/observation)@colors.T+.012*(q-prior)
    return value,gradient


def clr_solve(x,prior,colors,iterations=180):
    q=np.asarray(prior,float).copy();step=1.;backtracks=0
    initial=float(clr_objective(q,x,prior,colors)[0].sum())
    for used in range(iterations):
        value,gradient=clr_objective(q,x,prior,colors)
        for attempt in range(40):
            changed=project_simplex(q-step*gradient);delta=changed-q
            trial=clr_objective(changed,x,prior,colors)[0]
            bound=value+np.sum(gradient*delta,axis=-1)+np.sum(delta*delta,axis=-1)/(2*step)
            if np.all(trial<=bound+1e-12):break
            step*=.5;backtracks+=1
        else:raise RuntimeError('CLR projected line search did not find a finite majorizer')
        change=float(np.max(np.abs(changed-q)));q=changed;step=min(1.,step*1.25)
        if change<1e-8:break
    value,gradient=clr_objective(q,x,prior,colors)
    residual=float(np.max(np.abs(q-project_simplex(q-gradient))))
    return q,dict(iterations=used+1,line_search_backtracks=backtracks,
        objective_initial=initial,objective_final=float(value.sum()),
        projected_stationarity=residual,converged=residual<1e-5,
        observation_epsilon=1/255,global_optimum_claimed=False)


def compositional_palette(c,palette,he):
    x,prior,index=colour_nodes(c)
    colors=np.asarray(palette,float)[c.ids-1]/255
    q,report=clr_solve(x,prior,colors)
    return correct_coarse(q[index],c),dict(report,model='centred-logratio palette observation inverse',
        occupied_colour_nodes=len(x),quantization_bins=24)


def negative_tail_probabilities(q,labels,weight):
    scores=[];sizes=[]
    for g in range(q.shape[-1]):
        take=(labels!=g)&(weight>0)
        values=q[...,g][take];weights=weight[take]
        if not len(values):raise ValueError('No negative coarse evidence for a class')
        order=np.argsort(values,kind='stable');values=values[order];weights=weights[order]
        cumulative=np.r_[0,np.cumsum(weights)];mass=float(cumulative[-1])
        index=np.searchsorted(values,q[...,g],side='left')
        tail=(.5+mass-cumulative[index])/(1+mass)
        scores.append(tail);sizes.append(dict(negative_samples=len(values),negative_weight=mass))
    return np.stack(scores,-1),sizes


def negative_tail(c,palette,he):
    q,detail=palette_active_set(c,palette)
    tail,counts=negative_tail_probabilities(q,c.labels,c.weight)
    # Lexicographic choice: parent fraction breaks an exact null-tail tie;
    # class ID resolves the remaining exact tie. No additive epsilon can
    # reverse a genuinely smaller tail probability.
    best=tail.min(-1,keepdims=True)
    tied=tail==best;winner=np.argmax(np.where(tied,q,-np.inf),axis=-1)
    result=np.eye(c.k,dtype=np.float32)[winner]
    return result,dict(detail,model='negative-class empirical survival decision',
        negative_evidence=counts,expert_error_rate_calibrated=False,
        tie_rule='parent palette fraction, then lowest class ID',half_count=.5)
