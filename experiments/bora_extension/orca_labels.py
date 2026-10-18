"""Decode provider ORCA grayscale/antialiased black-grey-white masks.

The original paper defines black=non-tissue, grey=non-carcinoma tissue,
white=carcinoma. Released PNGs contain grey129/130 and interpolated edge values,
not just0/127/255. Quantize to the nearest published semantic intensity0/129/255;
do not discard hard boundary pixels or treat grey shades as separate classes.
"""
import numpy as np


def decode_orca(array):
    a=np.asarray(array)
    if a.dtype!=np.uint8:raise ValueError('Expected byte-valued ORCA mask')
    if a.ndim==3:
        if a.shape[2] not in (3,4) or not np.all(a[...,:3]==a[...,0,None]):
            raise ValueError('Original ORCA mask is not grayscale')
        if a.shape[2]==4 and np.any(a[...,3]!=255):raise ValueError('Unexpected annotation transparency')
        a=a[...,0]
    if a.ndim!=2:raise ValueError('Expected2D grayscale mask')
    output=np.zeros(a.shape,np.uint8)
    output[(a>=65)&(a<=192)]=127
    output[a>=193]=255
    info=dict(source_levels=np.unique(a).tolist(),normalization='nearest semantic intensity0/129/255; uint8 cutpoints65,193; output0/127/255',
        noncanonical_source_fraction=float(np.mean(~np.isin(a,[0,129,130,255]))),
        source='https://doi.org/10.3390/app10228285',edge_pixels_retained=True)
    return output,info
