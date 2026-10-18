"""Fit shared polygon boundaries with straight chords rather than pixel stairs.

The input is the published Bora converter's planar coverage. Each maximal
shared arc is fitted once; polygonized faces therefore have identical borders.
The tolerance is in micrometres, using both calibrated native pixel axes.
"""
from copy import deepcopy
from time import perf_counter
import numpy as np
from shapely import line_merge,make_valid
from shapely.affinity import scale
from shapely.geometry import shape,mapping,Polygon,MultiPolygon
from shapely.ops import unary_union,polygonize
from shapely.strtree import STRtree


def boundary_stats(data):
    edges=vertices=parts=stairs=axis_stairs=diagonal=0;diagonal_length=total_length=0.
    for feature in data['features']:
        g=shape(feature['geometry']);ps=[g] if g.geom_type=='Polygon' else list(g.geoms);parts+=len(ps)
        for p in ps:
            for ring in [p.exterior,*p.interiors]:
                xy=np.asarray(ring.coords);d=np.diff(xy,axis=0);length=np.linalg.norm(d,axis=1)
                diag=(np.abs(d[:,0])>1e-8)&(np.abs(d[:,1])>1e-8)
                horizontal=(np.abs(d[:,1])<1e-8)&(np.abs(d[:,0])>1e-8)
                vertical=(np.abs(d[:,0])<1e-8)&(np.abs(d[:,1])>1e-8)
                short=length<=2.01;perpendicular=(horizontal&np.roll(vertical,1))|(vertical&np.roll(horizontal,1))
                stairs+=int(np.count_nonzero(perpendicular&short&np.roll(short,1)));axis_stairs+=int(perpendicular.sum())
                edges+=len(d);vertices+=len(xy);diagonal+=int(diag.sum());diagonal_length+=float(length[diag].sum());total_length+=float(length.sum())
    return dict(boundary_segments=edges,vertices=vertices,polygon_parts=parts,short_stair_corners=stairs,axis_aligned_stair_corners=axis_stairs,
                diagonal_segments=diagonal,diagonal_length_fraction=diagonal_length/max(total_length,1e-20))


def _polygons(g):
    if g.is_empty:return g
    if g.geom_type in ('Polygon','MultiPolygon'):return g
    return unary_union([p for p in g.geoms if p.geom_type in ('Polygon','MultiPolygon')])


def fit_coverage(data,mpp_xy,tolerance_um):
    """No reference annotations, raster tracing, or tissue relabeling model."""
    if tolerance_um<=0 or len(mpp_xy)!=2 or min(mpp_xy)<=0:raise ValueError('Positive calibrated axes and tolerance required')
    originals=[scale(shape(f['geometry']),xfact=mpp_xy[0],yfact=mpp_xy[1],origin=(0,0)) for f in data['features']]
    if not originals:return deepcopy(data)
    network=unary_union([g.boundary for g in originals]);merged=line_merge(network)
    arcs=list(merged.geoms) if merged.geom_type=='MultiLineString' else [merged]
    # RDP minimizes redundant bends under a perpendicular-distance budget.
    # Junction endpoints are fixed because line_merge stops at degree !=2.
    fitted_arcs=[a.simplify(tolerance_um,preserve_topology=True) for a in arcs]
    faces=list(polygonize(unary_union([a for a in fitted_arcs if not a.is_empty])))
    tree=STRtree(originals);buckets=[[] for _ in originals]
    for face in faces:
        candidates=tree.query(face);intersections=[(int(i),face.intersection(originals[i]).area) for i in candidates]
        if not intersections:continue
        label,mass=max(intersections,key=lambda p:(p[1],-p[0]));background=max(face.area-sum(a for _,a in intersections),0.)
        if mass>background and mass>0:buckets[label].append(face)
    fitted=[unary_union(polys) for polys in buckets]
    result=deepcopy(data);features=[]
    for old,g in zip(data['features'],fitted):
        g=_polygons(make_valid(g))
        if g.is_empty:continue
        f=deepcopy(old);native=scale(g,xfact=1/mpp_xy[0],yfact=1/mpp_xy[1],origin=(0,0))
        f['geometry']=mapping(native);f['properties']['area_pixels']=float(native.area);f['properties']['area_um2']=float(g.area);features.append(f)
    result['features']=features
    result['boundary_fitting']=dict(method='shared-rdp',tolerance_um=float(tolerance_um),native_mpp_xy=list(mpp_xy),
        junctions_fixed=True,shared_arcs_fitted_once=True,polygon_faces_partitioned=True,
        source='Published Bora coverage converter followed by calibrated shared-boundary fitting')
    return result


def geometry_agreement(reference,candidate):
    old={int(f['properties']['value']):shape(f['geometry']) for f in reference['features']}
    new={int(f['properties']['value']):shape(f['geometry']) for f in candidate['features']}
    per_class={};difference=0.
    for value,g in old.items():
        p=new.get(value,Polygon());intersection=g.intersection(p).area;union=g.area+p.area-intersection
        per_class[value]=intersection/max(union,1e-20);difference+=g.symmetric_difference(p).area
    for value,g in new.items():
        if value not in old:difference+=g.area
    # Half the total per-class symmetric difference counts a shared-edge shift
    # once; external-background displacement is conservatively counted in full.
    tissue=unary_union(list(old.values()));proposed=unary_union(list(new.values())) if new else Polygon()
    exterior=tissue.symmetric_difference(proposed).area
    changed=(difference+exterior)*.5/max(tissue.area,1e-20)
    gs=list(new.values());overlap=sum(gs[i].intersection(gs[j]).area for i in range(len(gs)) for j in range(i))
    return dict(minimum_class_iou=min(per_class.values(),default=1.),per_class_iou=per_class,
        changed_tissue_area_fraction=float(changed),all_classes_preserved=set(old)==set(new),
        interclass_overlap_area_pixels=float(overlap),all_polygons_valid=all(g.is_valid for g in gs))


def compact(data,mpp_xy,native_shape,*,tolerance_um=3.,minimum_class_iou=.97,maximum_change_fraction=.015,minimum_pixels=20):
    """Choose the strongest admissible chord fit using predicted geometry only.

    Quality guards never inspect expert annotations. The previous lossless
    export remains an explicit fallback. The tiny-region rule is enforced after
    fitting, and all other boundaries remain diagonals/straight fitted chords.
    """
    from bora.mask_cleanup import finalize_vector_regions
    start=perf_counter();before=boundary_stats(data);attempts=[]
    for tolerance in [tolerance_um,tolerance_um*2/3,tolerance_um/3,tolerance_um/6,tolerance_um/12]:
        candidate=fit_coverage(data,mpp_xy,tolerance);agreement=geometry_agreement(data,candidate)
        admissible=agreement['all_classes_preserved'] and agreement['all_polygons_valid'] and agreement['interclass_overlap_area_pixels']<1e-7 and agreement['minimum_class_iou']>=minimum_class_iou and agreement['changed_tissue_area_fraction']<=maximum_change_fraction
        if admissible:
            candidate=finalize_vector_regions(candidate,native_shape,minimum_pixels);agreement=geometry_agreement(data,candidate)
            admissible=agreement['minimum_class_iou']>=minimum_class_iou and agreement['changed_tissue_area_fraction']<=maximum_change_fraction and agreement['all_classes_preserved'] and agreement['interclass_overlap_area_pixels']<1e-7
        after=boundary_stats(candidate);admissible=admissible and after['boundary_segments']<before['boundary_segments']
        attempts.append(dict(tolerance_um=tolerance,admissible=bool(admissible),agreement=agreement,after=after))
        if admissible:break
    else:candidate=deepcopy(data);agreement=geometry_agreement(data,candidate);after=before
    candidate['compact_boundary_report']=dict(requested_tolerance_um=tolerance_um,selected_tolerance_um=candidate.get('boundary_fitting',{}).get('tolerance_um',0.),
        before=before,after=after,agreement=agreement,attempts=attempts,runtime_seconds=perf_counter()-start,
        expert_annotations_read=False,minimum_class_iou_guard=minimum_class_iou,maximum_change_fraction_guard=maximum_change_fraction,
        original_export_fallback=after==before)
    return candidate
