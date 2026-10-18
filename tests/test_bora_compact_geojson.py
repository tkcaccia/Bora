import numpy as np
import pytest
pytest.importorskip('shapely');pytest.importorskip('rasterio')
from shapely.geometry import shape
from rasterio.features import shapes
from bora.compact_geojson import fit_coverage,boundary_stats,geometry_agreement,compact


def data(mask):return dict(type='FeatureCollection',features=[dict(type='Feature',geometry=g,properties=dict(value=int(v))) for g,v in shapes(mask,mask=mask>0)])


def test_shared_staircase_becomes_long_diagonal_with_matching_neighbors():
    yy,xx=np.mgrid[:180,:200];mask=np.where(xx>yy+10,1,2).astype(np.uint8);source=data(mask)
    result=fit_coverage(source,[.5,.5],.6)
    before,after=boundary_stats(source),boundary_stats(result)
    assert after['boundary_segments']<before['boundary_segments']*.1
    assert after['short_stair_corners']<before['short_stair_corners']*.05
    assert after['diagonal_segments']>0
    geometries=[shape(f['geometry']) for f in result['features']]
    assert all(g.is_valid for g in geometries)
    assert geometries[0].intersection(geometries[1]).area==0
    assert geometries[0].boundary.intersection(geometries[1].boundary).length>150
    assert geometry_agreement(source,result)['minimum_class_iou']>.98


def test_calibration_preserves_same_physical_result_on_anisotropic_grid():
    from shapely.affinity import scale
    from shapely.geometry import mapping
    yy,xx=np.mgrid[:100,:130];source=data(np.where(xx>yy+10,1,2).astype(np.uint8))
    stretched=data(np.where(xx>yy+10,1,2).astype(np.uint8))
    for f in stretched['features']:f['geometry']=mapping(scale(shape(f['geometry']),xfact=2,yfact=1,origin=(0,0)))
    a=fit_coverage(source,[.5,.5],1.);b=fit_coverage(stretched,[.25,.5],1.)
    for f,g in zip(a['features'],b['features']):assert shape(f['geometry']).hausdorff_distance(scale(shape(g['geometry']),xfact=.5,yfact=1,origin=(0,0)))<1e-7


def test_guard_keeps_classes_and_final_twenty_pixel_rule():
    yy,xx=np.mgrid[:160,:180];source=data(np.where(xx>yy+10,1,2).astype(np.uint8))
    result=compact(source,[.5,.5],(160,180),tolerance_um=2.)
    assert {f['properties']['value'] for f in result['features']}=={1,2}
    assert result['compact_boundary_report']['after']['vertices']<result['compact_boundary_report']['before']['vertices']
    assert result['compact_boundary_report']['agreement']['minimum_class_iou']>=.98
