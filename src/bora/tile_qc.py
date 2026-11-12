"""Exact area-weighted GrandQC percentages and strict clustering-tile admission.

QC pixels represent rectangles over the complete calibrated image. Integer
cross-grid coordinates preserve exact threshold comparisons despite different
mask/image resolutions. Artificial image padding counts as tile background.
"""
from fractions import Fraction
import numpy as np


def select_clustering_tiles(bad_area,unknown_area,area_denominator,max_bad_percent=10):
    limit=Fraction(str(max_bad_percent))/100
    if not 0 <= limit <= 1:raise ValueError('QC percentage threshold must lie in [0,100]')
    bad=np.asarray(bad_area);unknown=np.asarray(unknown_area)
    if bad.dtype.kind not in 'iu' or unknown.shape!=bad.shape or np.any(bad<0) or np.any(bad>area_denominator):raise ValueError('Invalid normalized QC areas')
    maximum=np.iinfo(np.int64).max
    if int(area_denominator)*max(limit.denominator,limit.numerator)>maximum:
        allowed=bad.astype(object)*limit.denominator < int(area_denominator)*limit.numerator
    else:allowed=bad*limit.denominator < int(area_denominator)*limit.numerator
    return np.asarray(allowed,bool)&(unknown==0)


def measure_tile_qc(labels,frame_shape_yx,true_size_wh,*,tile_pixels=224,max_bad_percent=10):
    fh,fw=map(int,frame_shape_yx);tw,th=map(int,true_size_wh);tile=int(tile_pixels)
    if tile<1 or min(fh,fw,tw,th)<1 or tw>fw or th>fh or fh%tile or fw%tile:raise ValueError('Use a complete padded image frame and positive true bounds')
    if isinstance(labels,np.ndarray):
        if labels.ndim!=2 or labels.dtype.kind not in 'iu':raise ValueError('Use a 2D GrandQC categorical mask')
        mh,mw=labels.shape
        def read(y0,y1):return labels[y0:y1]
    else:
        if labels.bands!=1 or labels.format!='uchar':raise ValueError('Use a single-channel uint8 GrandQC categorical mask')
        mh,mw=labels.height,labels.width
        def read(y0,y1):return np.frombuffer(labels.crop(0,y0,mw,y1-y0).write_to_memory(),np.uint8).reshape(y1-y0,mw)
    if min(mh,mw)<1:raise ValueError('Empty QC mask')
    denominator=tile*tile*int(mw)*int(mh)
    if max(denominator,fw*min(tile,th)*mw*mh)>np.iinfo(np.int64).max:raise OverflowError('Image/QC area exceeds exact int64 working capacity')
    gh,gw=fh//tile,fw//tile
    areas={name:np.zeros((gh,gw),np.int64) for name in ('background','artifact','clean','unknown')}
    x0=np.minimum(np.arange(gw,dtype=np.int64)*tile,tw)*mw
    x1=np.minimum((np.arange(gw,dtype=np.int64)+1)*tile,tw)*mw
    def integral(columns,positions):
        prefix=np.empty(mw+1,np.int64);prefix[0]=0;np.cumsum(columns,out=prefix[1:])
        index,remainder=np.divmod(positions,fw)
        return prefix[index]*fw+columns[np.minimum(index,mw-1)]*remainder
    padding_total=0
    for row in range(gh):
        y0=min(row*tile,th)*mh;y1=min((row+1)*tile,th)*mh
        body=(x1-x0)*(y1-y0);padding=denominator-body
        areas['background'][row]=padding;padding_total+=sum(int(x) for x in padding)
        if y1==y0:continue
        first=y0//fh;last=(y1+fh-1)//fh
        source_rows=np.arange(first,last,dtype=np.int64)
        weights=np.minimum((source_rows+1)*fh,y1)-np.maximum(source_rows*fh,y0)
        slab=read(first,last)
        predicates=dict(background=(slab==7),artifact=((slab>=2)&(slab<=6)),clean=(slab==1),unknown=((slab<1)|(slab>7)))
        for name,bits in predicates.items():
            columns=np.einsum('i,ij->j',weights,bits,dtype=np.int64,optimize=False)
            areas[name][row]+=integral(columns,x1)-integral(columns,x0)
    total=sum(areas.values())
    if not np.all(total==denominator):raise AssertionError('QC classes do not partition each tile')
    bad=areas['background']+areas['artifact']
    percentages={name:values.astype(np.float64)*100./denominator for name,values in areas.items()}
    percentages['background_or_artifact']=bad.astype(np.float64)*100./denominator
    eligible=select_clustering_tiles(bad,areas['unknown'],denominator,max_bad_percent)
    image_area=int(tw)*int(th)*int(mw)*int(mh)
    whole={name:sum(int(x) for x in values.ravel()) for name,values in areas.items()};whole['background']-=padding_total
    image_percentages={name:value*100./image_area for name,value in whole.items()}
    image_percentages['background_or_artifact']=(whole['background']+whole['artifact'])*100./image_area
    return dict(areas=areas,bad_area=bad,percentages=percentages,eligible=eligible,area_denominator=denominator,
        image_percentages=image_percentages,frame_shape_yx=[fh,fw],true_size_wh=[tw,th],mask_shape_yx=[mh,mw],grid_shape_yx=[gh,gw],tile_pixels=tile,
        max_background_artifact_percent=float(max_bad_percent),comparison='<',unclassified_policy='exclude any tile with unclassified QC area',
        artificial_padding='counts as background in tile percentages; excluded from real-image percentages')
