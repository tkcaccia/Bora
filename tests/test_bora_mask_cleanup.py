import numpy as np
from bora.mask_cleanup import fill_small_regions


def test_fill_nineteen_but_preserve_twenty_native_pixels():
    x=np.ones((30,30),np.uint8);x[5,3:22]=2;x[12,3:23]=3
    y,report=fill_small_regions(x,20)
    assert np.all(y[5,3:22]==1)
    assert np.all(y[12,3:23]==3)
    assert report['filled_pixels']==19


def test_background_is_preserved_and_surrounding_tissue_wins():
    x=np.zeros((20,20),np.uint8);x[2:18,2:18]=1;x[8:10,8:10]=2;x[4,4]=0
    y,_=fill_small_regions(x)
    assert np.all(y[0,:]==0)
    assert y[4,4]==1
    assert np.all(y[8:10,8:10]==1)


def test_label_renaming_does_not_change_geometric_choice():
    x=np.ones((30,30),np.uint8);x[:,16:]=2;x[12:14,14:16]=3
    y,_=fill_small_regions(x);permutation=np.array([0,7,4,9],np.uint8)
    z,_=fill_small_regions(permutation[x])
    assert np.array_equal(z,permutation[y])


def test_twenty_pixel_hole_stays_open_and_edge_glass_is_not_filled():
    x=np.ones((30,40),np.uint8);x[5,3:23]=0;x[12,3:22]=0;x[0,0]=0
    y,report=fill_small_regions(x)
    assert np.all(y[5,3:23]==0)
    assert np.all(y[12,3:22]==1)
    assert y[0,0]==0 and report['hole_pixels']==19


def test_vector_holes_after_simplification_and_other_classes():
    import pytest
    pytest.importorskip('shapely')
    from shapely.geometry import Polygon,mapping,shape
    from bora.mask_cleanup import fill_small_vector_holes
    shell=[(0,0),(30,0),(30,30),(0,30),(0,0)]
    hole=[(3,3),(6,3),(6,6),(3,6),(3,3)]
    occupied=[(12,3),(15,3),(15,6),(12,6),(12,3)]
    data={'features':[{'geometry':mapping(Polygon(shell,[hole,occupied])),'properties':{}},
                      {'geometry':mapping(Polygon(occupied)),'properties':{}}]}
    result=fill_small_vector_holes(data)
    assert len(shape(result['features'][0]['geometry']).interiors)==1
    assert result['small_vector_hole_cleanup']['filled_holes']==1
    assert shape(result['features'][0]['geometry']).intersection(shape(result['features'][1]['geometry'])).area==0


def test_final_vectors_merge_tiny_class_fill_hole_and_keep_twenty_pixels():
    import pytest
    pytest.importorskip('rasterio')
    from shapely.geometry import Polygon,mapping
    from rasterio.features import rasterize
    from bora.mask_cleanup import finalize_vector_regions
    shell=[(2,2),(28,2),(28,28),(2,28),(2,2)]
    empty=[(4,4),(7,4),(7,7),(4,7),(4,4)]
    small=[(12,4),(14,4),(14,7),(12,7),(12,4)]
    twenty=[(20,4),(24,4),(24,9),(20,9),(20,4)]
    pairs=[(Polygon(shell,[empty,small,twenty]),1),(Polygon(small),2),(Polygon(twenty),3)]
    data={'features':[{'geometry':mapping(p),'properties':{'value':v}} for p,v in pairs]}
    result=finalize_vector_regions(data,(30,30),20)
    raster=rasterize([(f['geometry'],f['properties']['value']) for f in result['features']],out_shape=(30,30),dtype=np.uint8)
    assert np.all(raster[4:7,4:7]==1) and np.all(raster[4:7,12:14]==1)
    assert np.count_nonzero(raster==3)==20 and np.count_nonzero(raster==2)==0
    assert np.all(raster[0,:]==0)


def test_continuous_small_parts_removed_even_when_raster_has_twenty_pixels():
    import pytest
    pytest.importorskip('rasterio')
    from shapely.geometry import box,mapping,shape
    from bora.mask_cleanup import finalize_vector_regions
    # 19.6 square native pixels, but 20 pixel centres: raster-only misses this.
    small=box(5.1,5,9.02,10);exact=box(15,5,19,10);large=box(0,0,30,30).difference(small.union(exact))
    parts=[(large,1),(small,2),(exact,3),(box(32,5,33,6),4)]
    d={'features':[{'geometry':mapping(g),'properties':{'value':v}} for g,v in parts]}
    result=finalize_vector_regions(d,(35,35),20)
    values={f['properties']['value'] for f in result['features']}
    assert values=={1,3}
    for f in result['features']:
        g=shape(f['geometry'])
        for p in ([g] if g.geom_type=='Polygon' else g.geoms):assert p.area>=20
    assert result['final_native_pixel_region_cleanup']['positive_polygon_parts_below_threshold']==0
