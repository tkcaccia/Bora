"""H&E-only native support hypotheses; fine reference labels are never read."""
import cv2
import numpy as np
from .diagnostics import white_regions
from .eleven_support import grid_geometry,occupancy_support


def blank_plane(he,spacing):
    shape,step=grid_geometry(he.shape[:2],spacing,4.)
    image=cv2.resize(he,shape[::-1],interpolation=cv2.INTER_AREA).astype(float)
    yy,xx=np.indices(shape);x=(xx+.5)/shape[1]-.5;y=(yy+.5)/shape[0]-.5
    border=(yy==0)|(xx==0)|(yy==shape[0]-1)|(xx==shape[1]-1)
    neutral=(np.ptp(image,axis=-1)<=24)&(image.min(-1)>180)&border
    white,_,policy=white_regions(he)
    if neutral.sum()<12:
        return white,dict(blank_fit_fallback=True,blank_samples=int(neutral.sum()),
                          reason='Fewer than12 neutral border samples',white_policy=policy)
    bright=image.mean(-1);neutral&=bright>=np.quantile(bright[neutral],.5)
    design=np.c_[np.ones(neutral.sum()),x[neutral],y[neutral]];target=image[neutral]
    if np.linalg.matrix_rank(design)<3:
        return white,dict(blank_fit_fallback=True,blank_samples=int(neutral.sum()),
            reason='Neutral border samples do not identify an affine plane',white_policy=policy)
    beta=np.linalg.lstsq(design,target,rcond=None)[0]
    for _ in range(8):
        residual=target-design@beta
        scale=np.maximum(1.4826*np.median(np.abs(residual-np.median(residual,axis=0)),axis=0),2.)
        weight=np.minimum(1.,1.5/np.maximum(np.sqrt(np.mean((residual/scale)**2,axis=1)),1e-12))
        beta=np.linalg.lstsq(design*np.sqrt(weight[:,None]),target*np.sqrt(weight[:,None]),rcond=None)[0]
    residual=target-design@beta
    scale=np.maximum(1.4826*np.median(np.abs(residual-np.median(residual,axis=0)),axis=0),2.)
    h,w=he.shape[:2];ny=(np.arange(h)+.5)/h-.5;nx=(np.arange(w)+.5)/w-.5
    distance=np.zeros((h,w))
    for j in range(3):
        plane=beta[0,j]+nx[None,:]*beta[1,j]+ny[:,None]*beta[2,j]
        distance+=((he[...,j].astype(float)-plane)/scale[j])**2
    blank=(distance<=16)&(np.ptp(he.astype(float),axis=-1)<=24)
    return blank,dict(blank_fit_fallback=False,blank_samples=int(neutral.sum()),
        blank_affine_rgb_coefficients=beta.tolist(),blank_residual_scale_rgb=scale.tolist(),
        standardized_squared_radius=16.,robust_iterations=8,fit_step_um=step.tolist(),
        original_fixed_white_pixels=int(white.sum()),adaptive_blank_pixels=int(blank.sum()),
        original_white_policy=policy,calibration_is_not_optical_resolution=True)


def enclosure_depth(stain,spacing,radius=64.):
    shape,step=grid_geometry(stain.shape,spacing,4.)
    grid=cv2.resize(stain.astype(float),shape[::-1],interpolation=cv2.INTER_AREA)
    rx,ry=np.ceil(radius/step).astype(int)
    yy,xx=np.mgrid[-ry:ry+1,-rx:rx+1];px=xx*step[0];py=yy*step[1]
    disk=px*px+py*py<=radius*radius
    depth=np.ones(shape)
    for angle in np.arange(8)*np.pi/4:
        kernel=(disk&(px*np.cos(angle)+py*np.sin(angle)>=-1e-9)).astype(float)
        # Absent image area is observed as no stain, not renormalized away.
        value=cv2.filter2D(grid,-1,kernel/kernel.sum(),borderType=cv2.BORDER_CONSTANT)
        depth=np.minimum(depth,np.maximum(value,0))
    return cv2.resize(depth,stain.shape[::-1],interpolation=cv2.INTER_LINEAR),dict(
        enclosure_radius_um=radius,oriented_halfdisks=8,occupancy_step_um=step.tolist(),
        outside_image_stain_fraction=0.,minimum_halfdisk_fraction=.025)


def rank_one_bic(largest,noise,count):
    # Strength plus two independent directions: three extra parameters, since
    # blank isotropic noise is estimated globally and held fixed in both models.
    ratio=np.maximum(np.asarray(largest)/noise,1.)
    return .5*count*(ratio-1-np.log(ratio))-1.5*np.log(np.maximum(count,1))


def texture_evidence(he,spacing,width=64.):
    shape,step=grid_geometry(he.shape[:2],spacing,4.)
    image=cv2.resize(he,shape[::-1],interpolation=cv2.INTER_AREA).astype(float)/255
    neutral=(image.min(-1)>.7)&(np.ptp(image,axis=-1)<.1)
    border=np.zeros(shape,bool);border[[0,-1]]=True;border[:,[0,-1]]=True
    noise=[]
    for axis in (0,1):
        a,b=[slice(None)]*2,[slice(None)]*2;a[axis]=slice(None,-1);b[axis]=slice(1,None)
        a,b=tuple(a),tuple(b);use=neutral[a]&neutral[b]&border[a]&border[b]
        if use.any():noise.extend(np.mean((image[a][use]-image[b][use])**2,-1)/2)
    variance=max(float(np.median(noise)) if noise else 0.,(1/255)**2)
    window=tuple(max(1,round(width/v)) for v in step)
    count=cv2.boxFilter(np.ones(shape),-1,window,normalize=False,borderType=cv2.BORDER_CONSTANT)
    def average(values):
        return cv2.boxFilter(values,-1,window,normalize=False,borderType=cv2.BORDER_CONSTANT)/count
    mean=np.stack([average(image[...,j]) for j in range(3)],-1)
    covariance=np.empty((*shape,3,3))
    for a in range(3):
        for b in range(a,3):
            covariance[...,a,b]=covariance[...,b,a]=average(image[...,a]*image[...,b])-mean[...,a]*mean[...,b]
    largest=np.linalg.eigvalsh(covariance)[...,-1]
    evidence=rank_one_bic(largest,variance,count)
    return cv2.resize(evidence,he.shape[1::-1],interpolation=cv2.INTER_LINEAR),dict(
        blank_noise_variance=variance,blank_difference_samples=len(noise),
        blank_noise_floor=(1/255)**2,texture_grid_mpp_xy=step.tolist(),
        window_um_xy=(np.asarray(window)*step).tolist(),bic_extra_parameters=3,
        patch_independence_is_an_approximation=True)


def infer_support(key,he,spacing):
    white,_,policy=white_regions(he)
    if key=='Q01':
        blank,fit=blank_plane(he,spacing)
        support,report=occupancy_support(blank,spacing);report.update(fit)
    elif key=='Q02':
        depth,report=enclosure_depth(~white,spacing);support=(~white)|(depth>.025)
    elif key=='Q03':
        evidence,report=texture_evidence(he,spacing);support=(~white)|(evidence>0)
    else:raise ValueError('Unknown cycle12 native-support intervention')
    if support.shape!=white.shape:raise ValueError('Native support geometry changed')
    return support,dict(report,model=key+' native support',white_policy=policy,
        own_native_support=True,inherited_common_clip_applied=False,reference_used=False,
        photometrically_nonwhite_can_be_removed=key=='Q01',
        native_mpp_xy=list(map(float,spacing)),native_shape_yx=list(white.shape),
        retained_pixels=int(support.sum()),removed_pixels=int((~support).sum()),
        removed_original_nonwhite_pixels=int(((~white)&~support).sum()))
