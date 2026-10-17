from dataclasses import replace
import numpy as np
import pytest
from bora.boundary import BoundaryConfig, _regularize_classes


@pytest.mark.parametrize('mode', ['none','majority','edge_potts'])
def test_categorical_regularization_is_equivariant_to_class_ids_and_keeps_fixed_pixels(mode):
    rng=np.random.default_rng(823); ids=np.array([1,2,3]); labels=rng.integers(1,4,(16,20)).astype(np.uint8)
    editable=np.zeros(labels.shape,bool);editable[3:-3,3:-3]=True
    lab=rng.uniform(0,100,(*labels.shape,3)).astype(np.float32)
    unary=rng.uniform(0,2,(3,*labels.shape)).astype(np.float32)
    cfg=replace(BoundaryConfig(),internal_regularizer=mode,internal_median_size=3)
    result=_regularize_classes(labels.copy(),editable,ids,lab,cfg,unary)
    new_ids=np.array([7,25,4]); lookup=np.zeros(256,np.uint8);lookup[ids]=new_ids
    permuted=_regularize_classes(lookup[labels],editable,new_ids,lab,cfg,unary)
    assert np.array_equal(permuted,lookup[result])
    assert np.array_equal(result[~editable],labels[~editable])
    assert set(np.unique(result))<=set(ids)


def test_majority_ties_preserve_current_label():
    labels=np.array([[1,2,1],[2,3,2],[1,2,1]],np.uint8)
    cfg=replace(BoundaryConfig(),internal_regularizer='majority',internal_median_size=3)
    result=_regularize_classes(labels.copy(),np.ones((3,3),bool),np.array([1,2,3]),np.zeros((3,3,3)),cfg)
    assert result[1,1]==3


def test_edge_regularizer_never_crosses_infinite_unary_or_overrides_exact_ties():
    labels=np.ones((10,10),np.uint8);labels[:,5:]=2
    unary=np.ones((2,10,10),np.float32);unary[1,:,:5]=np.inf;unary[0,:,5:]=np.inf
    cfg=replace(BoundaryConfig(),internal_regularizer='edge_potts')
    result=_regularize_classes(labels.copy(),np.ones(labels.shape,bool),np.array([1,2]),np.zeros((10,10,3)),cfg,unary)
    assert np.array_equal(result,labels)


def test_edge_updates_reduce_the_declared_potts_energy():
    rng=np.random.default_rng(948);ids=np.array([1,2,3]);labels=rng.integers(1,4,(20,24)).astype(np.uint8)
    unary=rng.uniform(0,1,(3,*labels.shape)).astype(np.float32)
    lab=np.zeros((*labels.shape,3),np.float32);editable=np.ones(labels.shape,bool);editable[:2]=False
    cfg=replace(BoundaryConfig(),internal_regularizer='edge_potts')
    def energy(a):
        return float(sum(unary[i][a==g].sum() for i,g in enumerate(ids)) + cfg.internal_regularizer_weight *
            (np.count_nonzero(a[:,1:]!=a[:,:-1])+np.count_nonzero(a[1:]!=a[:-1])))
    result=_regularize_classes(labels.copy(),editable,ids,lab,cfg,unary)
    assert energy(result)<energy(labels)
    assert np.array_equal(result[~editable],labels[~editable])


def test_indexed_visual_masks_preserve_class_colours_and_transparency(tmp_path):
    from PIL import Image
    from experiments.virchow2_smoothing.publish import save_palette
    mask=np.array([[0,1,3],[2,1,0]],np.uint8);colours=[[250,30,50],[40,200,80],[30,60,230]]
    path=tmp_path/'mask.png';save_palette(mask,colours,path)
    assert np.array_equal(np.array(Image.open(path)),mask)
    rgba=np.array(Image.open(path).convert('RGBA'))
    assert np.array_equal(rgba[...,3]>0,mask>0)
    for i,colour in enumerate(colours,1):assert np.all(rgba[mask==i,:3]==colour)
