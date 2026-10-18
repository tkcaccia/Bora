import numpy as np
import pytest
from experiments.bora_extension.orca_labels import decode_orca
from experiments.bora_extension.prepare_puma import rasterize
from experiments.bora_extension.prepare_gleasonxai import consensus


def test_orca_semantics_preserve_ambiguous_boundaries():
    source=np.array([[0,64,65,127,128,129,130,192,193,255]],np.uint8)
    labels,receipt=decode_orca(source)
    assert labels.tolist()==[[0,0,127,127,127,127,127,127,255,255]]
    assert receipt['edge_pixels_retained']


def test_puma_polygon_hole_remains_unknown():
    p=dict(type='FeatureCollection',features=[dict(properties=dict(classification=dict(name='tissue_tumor')),
      geometry=dict(type='Polygon',coordinates=[[[100,100],[500,100],[500,500],[100,500],[100,100]],[[200,200],[300,200],[300,300],[200,300],[200,200]]]))])
    mask,overlap=rasterize(p)
    assert mask[150,150]==1 and mask[250,250]==0 and mask[50,50]==0 and overlap==0


def test_gleason_plurality_ignores_rater_tie():
    coords='[[0,0],[0.5,0],[0.5,1],[0,1]]'
    rows=[dict(annotator='a',grade='3',coords=coords),dict(annotator='b',grade='4',coords=coords)]
    mask,info=consensus(rows,(20,20));assert not mask.any() and info['disagreement_tie_pixels']>0
    rows.append(dict(annotator='c',grade='3',coords=coords))
    mask,info=consensus(rows,(20,20));assert mask[5,5]==3 and mask[5,15]==0


def test_puma_one_pixel_edge_excursion_is_clipped_without_coordinate_scaling():
    def document(left):
        return dict(features=[dict(properties=dict(classification=dict(name='tissue_stroma')),
            geometry=dict(type='Polygon',coordinates=[[[left,0],[100,0],[100,100],[left,100],[left,0]]]))])
    mask,_=rasterize(document(-1))
    assert mask.shape==(1024,1024) and mask[50,0]==2 and mask[50,150]==0
    with pytest.raises(ValueError,match='outside original annotated ROI'):
        rasterize(document(-20))
