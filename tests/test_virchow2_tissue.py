import numpy as np
import pytest
from experiments.virchow2_tissue.support import tissue_support,complete_tiles,encoder_coverage
from experiments.virchow2_multi.score import Evaluator


@pytest.mark.parametrize('glass',[(255,255,255),(223,224,226)])
def test_rgb_support_removes_external_glass_but_preserves_enclosed_lumina(glass):
    image=np.empty((160,160,3),np.uint8);image[:]=glass;image[30:130,30:130]=[170,90,150];image[65:95,65:95]=255
    tissue,_=tissue_support(image,[.5,.5])
    assert not tissue[:20].any()
    assert tissue[30:130,30:130].all()
    assert tissue[65:95,65:95].all()
    assert tissue.dtype==bool


def test_coarse_extension_uses_tissue_evidence_and_keeps_original_labels():
    original=np.zeros((90,180),np.uint8);original[:,:90]=7
    p=np.zeros((90,180,2),np.float32);p[:]=[.9,.1];p[:,90:]=[.1,.9];p[0,90:]=[.99,.01]
    support=np.zeros(original.shape,bool);support[1:,90:]=True
    result,_=complete_tiles(original,p,np.array([1,7]),support)
    assert np.all(result==7)
    p[1:,90:]=[.9,.1]
    result,_=complete_tiles(original,p,np.array([1,7]),support)
    assert np.all(result[:,:90]==7) and np.all(result[:,90:]==1)


def test_full_frame_encoder_coverage_cannot_silently_skip_unlabelled_tiles():
    meta=dict(window_coordinates_yx=[[0,0],[0,224]])
    assert encoder_coverage(meta,(16,32),(220,440))['all_original_tokens_covered']
    with pytest.raises(ValueError,match='full original image'):
        encoder_coverage(dict(window_coordinates_yx=[[0,0]]),(16,32),(220,440))


def test_removing_annotated_tissue_is_penalized_and_unlabelled_predictions_are_ignored():
    truth=np.zeros((40,40),np.uint8);truth[10:30,:20]=1;truth[10:30,20:]=2
    pred=truth.copy();pred[truth==0]=7
    score=Evaluator(truth,[.5,.5]).score(pred)
    assert score['macro_iou']==1 and score['valid_pixels']==800
    pred[10:15,:20]=0
    score=Evaluator(truth,[.5,.5]).score(pred)
    assert score['valid_pixels']==800 and score['macro_iou']<1
