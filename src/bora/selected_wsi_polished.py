"""Versioned whole-slide Bora solver without a square tile-label prior.

Virchow2 and KODAMA are upstream. The complete frame is projected once and
refined with continuous probabilities; no diagnostic crop is fitted alone.
The final physical-scale contour polish and 2000 µm² cleanup are performed by
the separate versioned entry point in ``wsi_polished``.
"""
from __future__ import annotations
import ctypes, gc, hashlib, json, math, os, time
from pathlib import Path
import cv2
import numpy as np
import tifffile
from rasterio.features import rasterize, shapes
from rasterio import Affine
from shapely import make_valid
from shapely.geometry import Polygon, shape, mapping
from shapely.affinity import scale
from shapely.ops import unary_union
from shapely.strtree import STRtree
from .multicpu_geojson import convert
from .compact_geojson import geometry_agreement, boundary_stats
from .wsi_coverage_lookup import fit_coverage_indexed as fit_coverage
from .mask_cleanup import remove_small_vector_parts, fill_small_vector_holes

PROFILE = "bora-wholeslide-contour-polished-2026-10-30-v2"


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def atomic(p,d):
    q=p.with_name(p.name+'.partial');q.write_text(json.dumps(d,indent=2)+'\n');os.replace(q,p)

def release():
    gc.collect()
    try:ctypes.CDLL('libc.so.6').malloc_trim(0)
    except OSError:pass

def pixels(image,x,y,w,h):
    p=image.crop(x,y,w,h)
    if p.bands>3:p=p.extract_band(0,n=3)
    if p.format!='uchar':p=p.cast('uchar')
    return np.ndarray(buffer=p.write_to_memory(),dtype=np.uint8,shape=(h,w,3)).copy()

def connected(binary,connectivity=8):
    if binary.size > np.iinfo(np.int32).max:
        from .large_components import connected_large
        return connected_large(binary,connectivity)
    return cv2.connectedComponentsWithStats(np.asarray(binary,np.uint8),connectivity=connectivity)

def tissue_support(image,shape_yx,mpp,out):
    h,w=shape_yx
    support=np.lib.format.open_memmap(out/'support.npy',mode='w+',dtype=np.uint8,shape=(h,w))
    for y in range(0,h,1024):
        for x in range(0,w,1024):
            x1,y1=min(w,x+1024),min(h,y+1024);xa,ya=max(0,x-4),max(0,y-4);xb,yb=min(w,x1+4),min(h,y1+4)
            he=cv2.GaussianBlur(pixels(image,xa,ya,xb-xa,yb-ya),(0,0),1)
            raw=(he.max(2).astype(np.int16)-he.min(2).astype(np.int16)>25)|(he.min(2)<180)
            support[y:y1,x:x1]=raw[y-ya:y1-ya,x-xa:x1-xa]
    # These are global components; crop or block borders do not count as slide edges.
    n,cc,stats,_=connected(support,4);keep=stats[:,cv2.CC_STAT_AREA]*np.prod(mpp)>=100;keep[0]=False
    for y in range(0,h,256):support[y:y+256]=keep[cc[y:y+256]]
    del cc,stats;release()
    n,cc,stats,_=connected(support==0,4);fill=stats[:,cv2.CC_STAT_AREA]*np.prod(mpp)<900
    border=np.unique(np.concatenate((cc[0],cc[-1],cc[:,0],cc[:,-1])));fill[border]=False;fill[0]=False
    for y in range(0,h,256):
        a=support[y:y+256];a[fill[cc[y:y+256]]]=1
    del cc,stats;release();support.flush()
    return support

def diffuse(p,he,prior,inside):
    k=p.shape[-1];p=np.asarray(p,np.float32).copy();p[~inside]=0
    p/=np.maximum(p.sum(2,keepdims=True),1e-6)
    # The KODAMA grid trains the continuous response model. Reusing it here
    # as a piecewise-constant pixel prior imprints exact 224-pixel tile edges
    # into the final mask, even where image evidence crosses those edges.
    unary=p
    guide=cv2.GaussianBlur(he,(0,0),1);od=-np.log((guide.astype(np.float32)+1)/256)
    wx=np.exp(-np.sum((od[:,1:]-od[:,:-1])**2,2)/(2*.18**2))
    wy=np.exp(-np.sum((od[1:]-od[:-1])**2,2)/(2*.18**2))
    wx=(.02+.98*wx)*(inside[:,1:]&inside[:,:-1]);wy=(.02+.98*wy)*(inside[1:]&inside[:-1])
    conduct=np.zeros(inside.shape,np.float32)
    conduct[:,1:]+=wx;conduct[:,:-1]+=wx;conduct[1:]+=wy;conduct[:-1]+=wy
    den=1+3*conduct;q=unary
    for _ in range(16):
        u=unary.copy();u[:,1:]+=3*wx[...,None]*q[:,:-1];u[:,:-1]+=3*wx[...,None]*q[:,1:]
        u[1:]+=3*wy[...,None]*q[:-1];u[:-1]+=3*wy[...,None]*q[1:];q=u/den[...,None]
    mask=(q.argmax(2)+1).astype(np.uint8);mask[~inside]=0
    return mask

def diffuse_window(image,prob,grid,support,box):
    x0,y0,x1,y1=box;h,w=support.shape
    he=pixels(image,2*x0,2*y0,min(2*x1,w)-2*x0,min(2*y1,h)-2*y0)
    if he.shape[:2]!=(2*(y1-y0),2*(x1-x0)):
        he=cv2.copyMakeBorder(he,0,2*(y1-y0)-he.shape[0],0,2*(x1-x0)-he.shape[1],cv2.BORDER_REPLICATE)
    he=cv2.resize(he,(x1-x0,y1-y0),interpolation=cv2.INTER_AREA)
    xx=np.arange(x0,x1,dtype=np.float32);yy=np.arange(y0,y1,dtype=np.float32)
    mx,my=np.meshgrid((2*xx+1)/14-.5,(2*yy+1)/14-.5)
    p=cv2.remap(prob,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    # The encoder's overlapping windows can leave a weak, periodic class
    # transition at the original tile lattice. Blur probabilities over four
    # 1-um analysis pixels before H&E-guided diffusion; support still comes
    # from the reviewed tissue GeoJSON and is applied below.
    p=cv2.GaussianBlur(p,(0,0),4)
    prior=grid[(2*yy.astype(int))[:,None]//224,(2*xx.astype(int))[None,:]//224]
    inside=np.asarray(support[2*y0:min(2*y1,h):2,2*x0:min(2*x1,w):2],bool)
    return diffuse(p,he,prior,inside)

def predict(image,prob,grid,support,out,mpp):
    h,w=support.shape;sh,sw=(h+1)//2,(w+1)//2
    small=np.lib.format.open_memmap(out/'diffusion-1um.npy',mode='w+',dtype=np.uint8,shape=(sh,sw))
    prob=np.asarray(prob,np.float32);halo=24;blocks=[]
    for y in range(0,sh,768):
        for x in range(0,sw,768):
            x1,y1=min(sw,x+768),min(sh,y+768);xa,ya=max(0,x-halo),max(0,y-halo);xb,yb=min(sw,x1+halo),min(sh,y1+halo)
            # Skip only blocks consisting entirely of source glass.
            if not np.any(support[2*y: min(2*y1,h),2*x:min(2*x1,w)]):
                small[y:y1,x:x1]=0
            else:
                labels=diffuse_window(image,prob,grid,support,(xa,ya,xb,yb))
                small[y:y1,x:x1]=labels[y-ya:y1-ya,x-xa:x1-xa]
            blocks.append([x,y,x1,y1]);small.flush()
        atomic(out/'PROGRESS.json',{'stage':'global_diffusion','completed_small_rows':y1,'total_small_rows':sh})
        print('DIFFUSION_ROWS',y1,'/',sh,flush=True)
    # Re-evaluate strips across several computation boundaries. Their cores
    # must be identical to the stored global raster, without any blending.
    checks=[]
    for x,y in ((768,768),(1536,768),(768,1536)):
        if x+80>=sw or y+80>=sh:continue
        core=(x-48,y-48,x+48,y+48);box=(x-72,y-72,x+72,y+72)
        local=diffuse_window(image,prob,grid,support,box)[24:-24,24:-24]
        mismatch=int(np.count_nonzero(local!=small[core[1]:core[3],core[0]:core[2]]))
        checks.append({'core_xyxy':list(core),'different_pixels':mismatch})
        if mismatch:raise AssertionError('Global diffusion block-boundary equivalence failed')
    atomic(out/'SEAM_AUDIT.json',{'status':'COMPLETE','halo_analysis_pixels':halo,'iterations':16,'checks':checks})
    del prob;release()
    # The accepted 1.5 µm one-hot contour smoothing is also globally aligned.
    smoothed=np.lib.format.open_memmap(out/'smoothed-mask.npy',mode='w+',dtype=np.uint8,shape=(h,w))
    halo=16;sigma=(1.5/float(mpp[0]),1.5/float(mpp[1]))
    for y in range(0,h,1024):
        for x in range(0,w,1024):
            x1,y1=min(w,x+1024),min(h,y+1024);xa,ya=max(0,x-halo),max(0,y-halo);xb,yb=min(w,x1+halo),min(h,y1+halo)
            labels=np.array(small[np.arange(ya,yb)[:,None]//2,np.arange(xa,xb)[None,:]//2]);labels[np.asarray(support[ya:yb,xa:xb])==0]=0
            best=np.full(labels.shape,-1.,np.float32);result=np.zeros(labels.shape,np.uint8)
            for g in np.unique(labels):
                response=cv2.GaussianBlur((labels==g).astype(np.float32),(0,0),sigmaX=sigma[0],sigmaY=sigma[1],borderType=cv2.BORDER_REFLECT_101)
                better=response>best;result[better]=g;best[better]=response[better]
            result[np.asarray(support[ya:yb,xa:xb])==0]=0
            smoothed[y:y1,x:x1]=result[y-ya:y1-ya,x-xa:x1-xa]
        smoothed.flush()
    return smoothed

def clean_global(labels,mpp,output,minimum_area_um2=2000):
    """Exact global 8-connected areas, with bounded integer CC storage."""
    if not math.isfinite(minimum_area_um2) or minimum_area_um2 <= 0:
        raise ValueError('Positive physical minimum area required')
    out=np.lib.format.open_memmap(output,mode='w+',dtype=np.uint8,shape=labels.shape);out[:]=labels
    threshold=math.ceil(minimum_area_um2/np.prod(mpp));kernel=np.ones((3,3),np.uint8);counts={'parts':0,'holes':0,'pixels':0}
    for g in [int(v) for v in np.unique(labels) if v>0]+[0]:
        n,cc,stats,_=connected(out==g,8)
        small=np.flatnonzero((stats[:,cv2.CC_STAT_AREA]<threshold)&(np.arange(n)>0))
        for cid in small:
            x,y,w,h,area=map(int,stats[cid]);xa,ya=max(0,x-1),max(0,y-1);xb,yb=min(out.shape[1],x+w+1),min(out.shape[0],y+h+1)
            if g==0 and (x==0 or y==0 or x+w==out.shape[1] or y+h==out.shape[0]):continue
            at=cc[ya:yb,xa:xb]==cid;ring=(cv2.dilate(at.astype(np.uint8),kernel)>0)&~at
            view=out[ya:yb,xa:xb];values=view[ring];values=values[values!=g]
            if not len(values):continue
            winner=int(np.bincount(values).argmax());view[at]=winner
            counts['holes' if g==0 else 'parts']+=1;counts['pixels']+=area
        del cc,stats;release();out.flush();print('GLOBAL_AREA_CLEANUP',g,counts,flush=True)
    return out,counts

def fit_vectors(mask,mpp,out):
    tifffile.imwrite(out/'global-vector-input.tif',mask,tile=(512,512),compression='deflate',bigtiff=True)
    del mask;release()
    report=convert(out/'global-vector-input.tif',out/'initial.geojson',page=0,min_area=2000/np.prod(mpp),simplify=2.,workers=1,striped_native=False)
    data=json.loads((out/'initial.geojson').read_text());data['native_mpp_xy']=list(mpp)
    before=boundary_stats(data);attempts=[];chosen=data
    for tolerance in (5.,10/3,5/3,5/6,5/12):
        candidate=fit_coverage(data,mpp,tolerance);agreement=geometry_agreement(data,candidate)
        okay=(agreement['all_classes_preserved'] and agreement['all_polygons_valid'] and agreement['interclass_overlap_area_pixels']<1e-7 and agreement['minimum_class_iou']>=.95 and agreement['changed_tissue_area_fraction']<=.025)
        attempts.append({'tolerance_um':tolerance,'admissible':okay,'agreement':agreement})
        if okay:chosen=candidate;break
    chosen=remove_small_vector_parts(chosen,2000/np.prod(mpp));chosen=fill_small_vector_holes(chosen,2000/np.prod(mpp))
    for f in chosen['features']:
        g=int(f['properties']['value']);f['properties'].update(name=f'group_{g:02d}',objectType='annotation',identity='anonymous KODAMA group')
    chosen['compact_boundary_report']={'before':before,'after':boundary_stats(chosen),'attempts':attempts,'shared_arcs_fitted_once':True}
    chosen['coordinate_space']='calibrated_0p5um_pixel_edges';chosen['processing_scope']='entire slide; no field inference or field cleanup'
    atomic(out/'global-fitted.working.geojson',chosen)
    return chosen,report

def raster_global(data,shape_yx,path):
    result=np.lib.format.open_memmap(path,mode='w+',dtype=np.uint8,shape=tuple(shape_yx))
    for y in range(0,shape_yx[0],512):
        hh=min(512,shape_yx[0]-y)
        result[y:y+hh]=rasterize([(f['geometry'],int(f['properties']['value'])) for f in data['features']],out_shape=(hh,shape_yx[1]),fill=0,dtype=np.uint8,transform=Affine(1,0,0,0,1,y))
    result.flush();return result

def repair_raster(data,original,cleaned):
    patches={}
    for y in range(0,original.shape[0],512):
        old=np.asarray(original[y:y+512]);new=np.asarray(cleaned[y:y+512]);difference=old!=new
        if not difference.any():continue
        for geom,value in shapes(new.astype(np.int16),mask=difference,connectivity=4,transform=Affine(1,0,0,0,1,y)):
            patches.setdefault(int(value),[]).append(shape(geom))
    if not patches:return data,0
    merged={v:unary_union(gg) for v,gg in patches.items()};all_patches=unary_union(list(merged.values()))
    for f in data['features']:
        v=int(f['properties']['value']);g=shape(f['geometry']).difference(all_patches)
        if v in merged:g=g.union(merged[v])
        g=make_valid(g)
        if g.geom_type=='GeometryCollection':g=unary_union([q for q in g.geoms if q.geom_type in ('Polygon','MultiPolygon')])
        f['geometry']=mapping(g)
    return data,sum(len(v) for v in patches.values())

def audit(data,mask,mpp):
    min_area=float('inf');minhole=float('inf');geoms=[shape(f['geometry']) for f in data['features']]
    polygon_list=[];polygon_owner=[]
    for owner,g in enumerate(geoms):
        for p in ([g] if g.geom_type=='Polygon' else g.geoms):
            polygon_list.append(p);polygon_owner.append(owner)
    tree=STRtree(polygon_list)
    for i,g in enumerate(geoms):
        if not g.is_valid:raise ValueError('Invalid global geometry')
        for p in ([g] if g.geom_type=='Polygon' else g.geoms):
            min_area=min(min_area,p.area*np.prod(mpp))
            if p.area*np.prod(mpp)<2000-1e-6:raise ValueError('Global polygon below physical minimum')
            for r in p.interiors:
                hole=Polygon(r)
                occupants=[polygon_list[j] for j in tree.query(hole) if polygon_owner[j]!=i]
                unoccupied=hole.difference(unary_union(occupants)) if occupants else hole
                if not unoccupied.is_empty:
                    for h in ([unoccupied] if unoccupied.geom_type=='Polygon' else getattr(unoccupied,'geoms',[])):
                        if h.geom_type!='Polygon':continue
                        if h.area*np.prod(mpp)<1e-6:continue
                        minhole=min(minhole,h.area*np.prod(mpp))
                        if h.area*np.prod(mpp)<2000-1e-6:raise ValueError('Global hole below physical minimum')
    overlap=sum(geoms[i].intersection(geoms[j]).area for i in range(len(geoms)) for j in range(i))
    if overlap>1e-7:raise ValueError('Global classes overlap')
    threshold=math.ceil(2000/np.prod(mpp));counts=[]
    for g in [int(v) for v in np.unique(mask) if v>0]+[0]:
        n,cc,stats,_=connected(mask==g,8);a=stats[1:,cv2.CC_STAT_AREA]
        if g==0:
            border=np.unique(np.concatenate((cc[0],cc[-1],cc[:,0],cc[:,-1])))
            ids=np.setdiff1d(np.arange(1,n),border);a=stats[ids,cv2.CC_STAT_AREA]
        bad=int(np.count_nonzero(a<threshold));counts.append({'group':g,'components':len(a),'below_minimum':bad})
        del cc,stats;release()
        if bad:raise ValueError('Global raster still contains a small component')
    return {'status':'COMPLETE','global_component_audits':counts,'minimum_polygon_um2':min_area,
            'minimum_enclosed_background_hole_um2':minhole if math.isfinite(minhole) else None,
            'interclass_overlap_pixels2':overlap,'boundaries':boundary_stats(data)}
