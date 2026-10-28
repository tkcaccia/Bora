"""Equivalent shared-RDP fitting with prepared per-polygon area lookups.

The global boundary network, simplification, faces and maximum-overlap class
assignment are unchanged. Indexing the constituent polygons avoids repeated
intersection with every complete-slide MultiPolygon for each small face.
"""
from copy import deepcopy
from collections import defaultdict
from shapely import line_merge,make_valid
from shapely.affinity import scale
from shapely.geometry import shape,mapping,Polygon
from shapely.ops import unary_union,polygonize
from shapely.prepared import prep
from shapely.strtree import STRtree


def fit_coverage_indexed(data,mpp_xy,tolerance_um):
    if tolerance_um<=0 or len(mpp_xy)!=2 or min(mpp_xy)<=0:raise ValueError('Positive physical axes and tolerance required')
    originals=[scale(shape(f['geometry']),xfact=mpp_xy[0],yfact=mpp_xy[1],origin=(0,0)) for f in data['features']]
    if not originals:return deepcopy(data)
    network=unary_union([g.boundary for g in originals]);merged=line_merge(network)
    arcs=list(merged.geoms) if merged.geom_type=='MultiLineString' else [merged]
    fitted_arcs=[a.simplify(tolerance_um,preserve_topology=True) for a in arcs]
    faces=list(polygonize(unary_union([a for a in fitted_arcs if not a.is_empty])))
    pieces=[];owners=[]
    for i,g in enumerate(originals):
        for p in ([g] if g.geom_type=='Polygon' else g.geoms):
            if not p.is_empty:pieces.append(p);owners.append(i)
    tree=STRtree(pieces);prepared=[prep(p) for p in pieces];buckets=[[] for _ in originals]
    for face in faces:
        mass_by_owner=defaultdict(float)
        for j in tree.query(face):
            if not prepared[j].intersects(face):continue
            area=face.area if prepared[j].covers(face) else face.intersection(pieces[j]).area
            mass_by_owner[owners[j]]+=area
        if not mass_by_owner:continue
        label,mass=max(mass_by_owner.items(),key=lambda item:(item[1],-item[0]))
        background=max(face.area-sum(mass_by_owner.values()),0.)
        if mass>background and mass>0:buckets[label].append(face)
    result=deepcopy(data);features=[]
    for old,polys in zip(data['features'],buckets):
        g=make_valid(unary_union(polys))
        if g.geom_type not in ('Polygon','MultiPolygon'):
            g=unary_union([p for p in g.geoms if p.geom_type in ('Polygon','MultiPolygon')])
        if g.is_empty:continue
        f=deepcopy(old);native=scale(g,xfact=1/mpp_xy[0],yfact=1/mpp_xy[1],origin=(0,0))
        f['geometry']=mapping(native);f['properties']['area_pixels']=float(native.area);f['properties']['area_um2']=float(g.area);features.append(f)
    result['features']=features
    result['boundary_fitting']={'method':'shared-rdp','tolerance_um':float(tolerance_um),'native_mpp_xy':list(mpp_xy),
        'junctions_fixed':True,'shared_arcs_fitted_once':True,'polygon_faces_partitioned':True,
        'source':'Published Bora coverage converter followed by calibrated shared-boundary fitting',
        'area_lookup':'prepared constituent polygons; same maximum-overlap face assignment'}
    return result
