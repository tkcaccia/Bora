"""RGB-only tissue support and coarse-only extension. No reference access."""
import cv2
import numpy as np
from scipy import ndimage as ndi

POLICY=dict(od_mean_min=.025,saturation_min=.04,dark_od_mean=.30,closing_radius_um=2.,min_component_um2=4.,
    enclosed_holes='fill to preserve enclosed gland lumina and adipose spaces',
    outside='RGB-only estimated glass/background; no annotation or class mask used')


def tissue_support(image,mpp_xy):
    a=image.astype(np.float32);maximum=a.max(2);minimum=a.min(2)
    saturation=(maximum-minimum)/np.maximum(maximum,1)
    od=-np.log((a+1)/256).mean(2)
    raw=((od>POLICY['od_mean_min'])&(saturation>POLICY['saturation_min']))|(od>POLICY['dark_od_mean'])
    rx=max(1,round(POLICY['closing_radius_um']/mpp_xy[0]));ry=max(1,round(POLICY['closing_radius_um']/mpp_xy[1]))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*rx+1,2*ry+1))
    closed=cv2.morphologyEx(raw.astype(np.uint8),cv2.MORPH_CLOSE,kernel,borderType=cv2.BORDER_REPLICATE)>0
    count,labels,stats,_=cv2.connectedComponentsWithStats(closed.astype(np.uint8),8)
    keep=stats[:,cv2.CC_STAT_AREA]*mpp_xy[0]*mpp_xy[1]>=POLICY['min_component_um2'];keep[0]=False
    support=ndi.binary_fill_holes(keep[labels])
    return support,dict(**POLICY,native_shape_yx=list(image.shape[:2]),native_mpp_xy=mpp_xy,
        tissue_pixels=int(support.sum()),background_pixels=int((~support).sum()),stain_positive_pixels=int(raw.sum()))


def complete_tiles(original,probabilities,classes,tissue,tile=90):
    """Keep known coarse labels; fill zero tiles from mean tissue class evidence.

    Background receives temporary labels for Bora and is removed afterward.
    All algorithms receive this same completed input, with no pseudo-label fitting.
    """
    if probabilities.shape[:2]!=original.shape or tissue.shape!=original.shape:raise ValueError('Geometry mismatch')
    out=original.copy();filled=0
    for y in range(0,out.shape[0],tile):
        for x in range(0,out.shape[1],tile):
            block=out[y:y+tile,x:x+tile];unknown=block==0
            if not unknown.any():continue
            evidence=probabilities[y:y+tile,x:x+tile];support=tissue[y:y+tile,x:x+tile]
            average=evidence[support].mean(0) if support.any() else evidence.mean((0,1))
            block[unknown]=classes[average.argmax()];filled+=1
    if np.any(out==0) or not np.array_equal(out[original>0],original[original>0]):raise ValueError('Coarse extension invariant failed')
    return out,dict(filled_tiles=filled,original_labelled_pixels=int((original>0).sum()),extended_pixels=int((original==0).sum()),
        training='original coarse labels only; pseudo labels never train feature selection or classifiers',
        completion='90 px tile mean of plain selected-feature class probabilities on RGB-detected tissue; whole tile if empty')


def encoder_coverage(metadata,grid,shape):
    coverage=np.zeros(grid,np.uint16)
    for y,x in metadata['window_coordinates_yx']:
        coverage[y//14:y//14+16,x//14:x//14+16]+=1
    required=coverage[:int(np.ceil(shape[0]/14)),:int(np.ceil(shape[1]/14))]
    if not np.all(required>0):raise ValueError('Encoder did not cover the full original image')
    return dict(all_original_tokens_covered=True,minimum_windows_per_token=int(required.min()),maximum_windows_per_token=int(required.max()),
        original_shape_yx=list(shape),token_grid_yx=list(grid),window_count=len(metadata['window_coordinates_yx']),
        encoder_input='entire original RGB field; annotations never gate encoding')
