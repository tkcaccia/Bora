"""Fill microscopic label islands without extending segmentation onto glass."""
import numpy as np
from scipy import ndimage
from skimage.measure import label


def fill_small_regions(mask,minimum_pixels=20):
    """Replace connected positive regions with area strictly below the limit.

    The surrounding positive tissue with the longest 8-neighbor contact wins;
    ties prefer the larger neighboring component, then its first raster pixel.
    Enclosed background holes below the same threshold are filled; exterior
    background is preserved. An isolated tissue island surrounded
    entirely by background is filled with that surrounding background.
    The threshold counts native pixels of area, not length on either axis.
    """
    if not isinstance(minimum_pixels,(int,np.integer)) or minimum_pixels<1:raise ValueError('Positive integer pixel-area threshold required')
    original=np.asarray(mask)
    if original.ndim!=2 or original.dtype.kind not in 'iu':raise ValueError('A 2D integer mask is required')
    result=original.copy();components=label(original,background=0,connectivity=2)
    sizes=np.bincount(components.ravel());small=np.flatnonzero((sizes<minimum_pixels)&(np.arange(len(sizes))>0))
    locations=ndimage.find_objects(components);filled_pixels=filled_regions=isolated=0
    parent=np.arange(len(sizes));values=np.zeros(len(sizes),original.dtype);adj={}
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for component in range(1,len(sizes)):
        sl=locations[component-1];values[component]=original[sl][components[sl]==component][0]
    for component in small:
        ys,xs=locations[component-1];y0=max(0,ys.start-1);y1=min(result.shape[0],ys.stop+1);x0=max(0,xs.start-1);x1=min(result.shape[1],xs.stop+1)
        cc=components[y0:y1,x0:x1];at=cc==component
        ring=ndimage.binary_dilation(at,np.ones((3,3)))&~at
        ids,counts=np.unique(cc[ring],return_counts=True)
        adj[int(component)]={int(n):int(c) for n,c in zip(ids,counts)}
    for component in sorted(small,key=lambda c:(-sizes[c],c)):
        component=int(component)
        if root(component)!=component or sizes[component]>=minimum_pixels:continue
        contacts={}
        for n,c in adj.get(component,{}).items():
            n=root(n)
            if n!=component:contacts[n]=contacts.get(n,0)+c
        positive={n:c for n,c in contacts.items() if n>0}
        if positive:winner=max(positive,key=lambda n:(positive[n],sizes[n],-n))
        else:winner=0;isolated+=1
        parent[component]=winner;sizes[winner]+=sizes[component];filled_pixels+=int(sizes[component]);sizes[component]=0;filled_regions+=1
        if winner in adj:
            for n,c in adj.get(component,{}).items():adj[winner][n]=adj[winner].get(n,0)+c
    lookup=values.copy()
    for i in range(1,len(lookup)):lookup[i]=values[root(i)]
    result=lookup[components]
    holes,n_holes=ndimage.label(original==0,np.ones((3,3)))
    hole_sizes=np.bincount(holes.ravel());border=set(np.r_[holes[0,:],holes[-1,:],holes[:,0],holes[:,-1]].tolist())
    hole_slices=ndimage.find_objects(holes);filled_holes=hole_pixels=0
    for h in np.flatnonzero((hole_sizes<minimum_pixels)&(np.arange(len(hole_sizes))>0)):
        if h in border:continue
        ys,xs=hole_slices[h-1];y0=max(0,ys.start-1);y1=min(result.shape[0],ys.stop+1);x0=max(0,xs.start-1);x1=min(result.shape[1],xs.stop+1)
        local=result[y0:y1,x0:x1];at=holes[y0:y1,x0:x1]==h
        ring=ndimage.binary_dilation(at,np.ones((3,3)))&~at&(local>0)
        values,counts=np.unique(local[ring],return_counts=True)
        if not len(values):continue
        # Contact ties use the first touching raster pixel, rather than IDs.
        winners=set(values[counts==counts.max()].tolist());target=next(int(v) for v in local[ring] if int(v) in winners)
        local[at]=target;filled_holes+=1;hole_pixels+=int(at.sum())
    exterior=np.isin(holes,list(border-{0}))&(original==0)
    if np.any(result[exterior]!=0):raise AssertionError('Cleanup painted exterior background')
    return result,dict(minimum_region_area_pixels=int(minimum_pixels),strictly_less_than_threshold=True,
        filled_regions=filled_regions,filled_pixels=filled_pixels,isolated_regions_filled_with_background=isolated,
        filled_holes=filled_holes,hole_pixels=hole_pixels,exterior_background_unchanged=True,connectivity=8)


def fill_small_vector_holes(data,minimum_pixels=20):
    """Fill enclosed GeoJSON holes introduced or retained during simplification.

    Areas are in native pixel squared coordinates. A hole occupied by another
    predicted tissue feature is never filled across that feature's boundary.
    The exterior rings and all shared tissue borders are unchanged.
    """
    from shapely.geometry import shape,Polygon,MultiPolygon,mapping
    from shapely.strtree import STRtree
    geometries=[shape(f['geometry']) for f in data['features']];tree=STRtree(geometries)
    filled=0;area=0.
    for i,(feature,g) in enumerate(zip(data['features'],geometries)):
        polygons=[g] if g.geom_type=='Polygon' else list(g.geoms);rebuilt=[]
        for p in polygons:
            keep=[]
            for ring in p.interiors:
                hole=Polygon(ring);occupied=any(j!=i and hole.intersection(geometries[j]).area>1e-7 for j in tree.query(hole))
                if hole.area<minimum_pixels and not occupied:filled+=1;area+=hole.area
                else:keep.append(ring.coords[:])
            rebuilt.append(Polygon(p.exterior.coords[:],keep))
        result=rebuilt[0] if g.geom_type=='Polygon' else MultiPolygon(rebuilt)
        if not result.is_valid:raise ValueError('Small-hole filling produced invalid geometry')
        feature['geometry']=mapping(result)
        if 'area_pixels' in feature['properties']:feature['properties']['area_pixels']=float(result.area)
        if 'area_um2' in feature['properties'] and 'native_mpp_xy' in data:
            feature['properties']['area_um2']=float(result.area*np.prod(data['native_mpp_xy']))
    data['small_vector_hole_cleanup']=dict(minimum_hole_area_pixels=minimum_pixels,filled_holes=filled,filled_area_pixels=area,
        exterior_rings_unchanged=True,other_predicted_tissue_not_overwritten=True)
    return data


def remove_small_vector_parts(data,minimum_pixels=20):
    """Remove polygon parts below a native pixel-area threshold.

    A part shares its area with the adjacent tissue having the longest common
    edge. An isolated part surrounded by glass is removed to background. This
    checks continuous polygon area, including subpixel slivers that a native
    raster component check cannot detect. Exactly-threshold parts are retained.
    """
    from shapely.geometry import shape,mapping,Polygon
    from shapely.ops import unary_union
    geometries=[shape(f['geometry']) for f in data['features']]
    removed=pixels=0
    def parts(g):
        if g.is_empty:return []
        return [g] if g.geom_type=='Polygon' else list(g.geoms)
    # Each step removes one component; merging cannot create a new component.
    while True:
        candidates=[(p.area,i,p) for i,g in enumerate(geometries) for p in parts(g) if p.area<minimum_pixels]
        if not candidates:break
        _,i,p=max(candidates,key=lambda x:(x[0],-x[1]))
        neighbors=[]
        for j,g in enumerate(geometries):
            if j==i or g.is_empty:continue
            contact=p.boundary.intersection(g.boundary).length
            if contact>1e-8:neighbors.append((contact,g.area,-j,j))
        geometries[i]=unary_union([q for q in parts(geometries[i]) if q is not p and not q.equals(p)])
        if neighbors:
            j=max(neighbors)[-1];geometries[j]=geometries[j].union(p)
        removed+=1;pixels+=p.area
    features=[]
    for feature,g in zip(data['features'],geometries):
        if g.is_empty:continue
        if not g.is_valid:raise ValueError('Invalid geometry after small polygon removal')
        feature['geometry']=mapping(g);feature['properties']['area_pixels']=float(g.area)
        if 'native_mpp_xy' in data:feature['properties']['area_um2']=float(g.area*np.prod(data['native_mpp_xy']))
        features.append(feature)
    data['features']=features
    data['small_polygon_part_cleanup']=dict(minimum_area_native_pixels=minimum_pixels,
        strictly_less_than_threshold=True,removed_parts=removed,removed_area_pixels=float(pixels),
        adjacent_tissue_filled=True,isolated_parts_removed_to_background=True)
    return fill_small_vector_holes(data,minimum_pixels)


def finalize_vector_regions(data,native_shape,minimum_pixels=20):
    """Enforce both continuous polygon area and native raster pixel counts."""
    from shapely.geometry import shape,Polygon
    totals=dict(removed_parts=0,removed_area_pixels=0.)
    pixel_totals={};hole_totals=dict(filled_holes=0,filled_area_pixels=0.)
    for iteration in range(10):
        data=remove_small_vector_parts(data,minimum_pixels)
        for k in totals:totals[k]+=data['small_polygon_part_cleanup'][k]
        for k in hole_totals:hole_totals[k]+=data['small_vector_hole_cleanup'][k]
        data=_finalize_vector_pixels(data,native_shape,minimum_pixels)
        report=data['final_native_pixel_region_cleanup']
        for k,v in report.items():
            if k in ('filled_regions','filled_pixels','filled_holes','hole_pixels','isolated_regions_filled_with_background','final_vector_raster_pixels_changed','numerical_boundary_tie_repair_pixels','repair_passes'):
                pixel_totals[k]=pixel_totals.get(k,0)+v
        parts=[]
        for f in data['features']:
            g=shape(f['geometry']);parts.extend([g] if g.geom_type=='Polygon' else list(g.geoms))
        if all(p.area>=minimum_pixels and all(Polygon(r).area>=minimum_pixels for r in p.interiors) for p in parts):break
    else:raise ValueError('Continuous and raster area cleanup did not converge')
    data['small_polygon_part_cleanup'].update(totals,iterations=iteration+1)
    data['small_vector_hole_cleanup'].update(hole_totals)
    data['final_native_pixel_region_cleanup'].update(pixel_totals,continuous_polygon_area_checked=True,
        positive_polygon_parts_below_threshold=0,polygon_holes_below_threshold=0)
    return data


def _finalize_vector_pixels(data,native_shape,minimum_pixels=20):
    """Enforce the pixel-area rule on the FINAL rasterized vector partition.

    Only changed tiny regions are repaired. All other simplified boundaries
    retain the published converter's coordinates. Pixel-cell patches are local
    repairs, not a replacement of the full mask by pixel-edge polygons.
    """
    from rasterio.features import rasterize,shapes
    from shapely.geometry import shape,mapping,Polygon,MultiPolygon
    from shapely import make_valid
    from shapely.ops import unary_union
    def render(features):
        pairs=[(f['geometry'],int(f['properties']['value'])) for f in features]
        return rasterize(pairs,out_shape=tuple(native_shape),fill=0,dtype=np.uint8) if pairs else np.zeros(native_shape,np.uint8)
    data=fill_small_vector_holes(data,minimum_pixels);original=render(data['features']);cleaned,report=fill_small_regions(original,minimum_pixels)
    # DSU merging is monotone, but a cleanup can expose another small region.
    for _ in range(3):
        again,extra=fill_small_regions(cleaned,minimum_pixels)
        if np.array_equal(again,cleaned):break
        cleaned=again
        for k in ('filled_regions','filled_pixels','filled_holes','hole_pixels','isolated_regions_filled_with_background'):report[k]+=extra[k]
    changed=cleaned!=original
    current=original;tie_pixels=0;repair_passes=0
    for repair_pass in range(5):
        difference=current!=cleaned
        if not difference.any():break
        tie_pixels+=int((difference&~changed).sum());repair_passes+=1;patches={}
        for g,value in shapes(cleaned.astype(np.int16),mask=difference,connectivity=4):patches.setdefault(int(value),[]).append(shape(g))
        patches={v:unary_union(gs) for v,gs in patches.items()};all_patches=unary_union(list(patches.values()));features=[]
        for f in data['features']:
            value=int(f['properties']['value']);g=shape(f['geometry']).difference(all_patches)
            if value in patches:g=g.union(patches[value])
            g=make_valid(g)
            if g.geom_type=='GeometryCollection':g=unary_union([p for p in g.geoms if p.geom_type in ('Polygon','MultiPolygon')])
            if g.is_empty:continue
            if g.geom_type not in ('Polygon','MultiPolygon') or not g.is_valid:raise ValueError('Invalid final tiny-region repair')
            f['geometry']=mapping(g);f['properties']['area_pixels']=float(g.area)
            if 'native_mpp_xy' in data:f['properties']['area_um2']=float(g.area*np.prod(data['native_mpp_xy']))
            features.append(f)
        data['features']=features;current=render(features)
    if not np.array_equal(current,cleaned):raise AssertionError('Final repaired GeoJSON differs from cleaned native mask')
    report.update(final_vector_raster_pixels_changed=int(changed.sum()),numerical_boundary_tie_repair_pixels=tie_pixels,repair_passes=repair_passes,all_boundaries_outside_local_repairs_unchanged=True)
    data['final_native_pixel_region_cleanup']=report
    return data
