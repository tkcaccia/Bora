import numpy as np
import pytest
from bora.tile_qc import measure_tile_qc,select_clustering_tiles

@pytest.mark.parametrize('bad,accepted',[(9,True),(10,False),(11,False)])
def test_strict_ten_percent_at_different_pixel_resolution(bad,accepted):
    labels=np.ones((10,10),np.uint8);labels.ravel()[:bad]=7
    q=measure_tile_qc(labels,[224,224],[224,224])
    assert q['percentages']['background_or_artifact'][0,0]==bad
    assert bool(q['eligible'][0,0])==accepted

def test_combined_background_and_all_artifact_types():
    labels=np.ones((10,10),np.uint8);labels.ravel()[:5]=7;labels.ravel()[5:10]=np.arange(2,7)
    q=measure_tile_qc(labels,[224,224],[224,224])
    assert q['percentages']['background'][0,0]==5
    assert q['percentages']['artifact'][0,0]==5
    assert q['percentages']['background_or_artifact'][0,0]==10
    assert not q['eligible'][0,0]

def test_fractional_qc_pixel_at_tile_boundary_has_exact_area():
    q=measure_tile_qc(np.array([[7,2,1]],np.uint8),[10,20],[20,10],tile_pixels=10)
    assert q['areas']['background'][0].tolist()==[200,0]
    assert q['areas']['artifact'][0].tolist()==[100,100]
    assert q['area_denominator']==300
    assert np.allclose(q['percentages']['background_or_artifact'],[[100,100/3]])
    assert q['image_percentages']['background_or_artifact']==pytest.approx(200/3)

def test_padding_is_background_and_exact_ten_percent_edge_is_excluded():
    q=measure_tile_qc(np.ones((1,2),np.uint8),[10,20],[19,10],tile_pixels=10)
    assert np.array_equal(q['percentages']['background'],[[0,10]])
    assert q['eligible'].tolist()==[[True,False]]
    assert q['image_percentages']['background']==0

def test_unknown_is_reported_separately_and_never_admitted_as_clean():
    labels=np.ones((10,10),np.uint8);labels[0,0]=0
    q=measure_tile_qc(labels,[224,224],[224,224])
    assert q['percentages']['background_or_artifact'][0,0]==0
    assert q['percentages']['unknown'][0,0]==1
    assert not q['eligible'][0,0]

def test_vips_streaming_matches_array_and_can_reapply_cutoff():
    import pyvips
    a=np.random.default_rng(4).integers(0,8,size=(7,11),dtype=np.uint8)
    image=pyvips.Image.new_from_memory(a.tobytes(),11,7,1,'uchar')
    x=measure_tile_qc(a,[20,30],[29,19],tile_pixels=10)
    y=measure_tile_qc(image,[20,30],[29,19],tile_pixels=10)
    for key in x['areas']:assert np.array_equal(x['areas'][key],y['areas'][key])
    assert np.array_equal(select_clustering_tiles(x['bad_area'],x['areas']['unknown'],x['area_denominator'],10),x['eligible'])

def test_high_precision_cutoff_does_not_overflow_or_round_at_boundary():
    denominator=10**14;bad=np.array([10**13-1,10**13,10**13+1],np.int64)
    assert select_clustering_tiles(bad,np.zeros(3,np.int64),denominator,10).tolist()==[True,False,False]
    assert select_clustering_tiles(bad,np.zeros(3,np.int64),denominator,'10.000000001').all()
