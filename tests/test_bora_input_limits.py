"""Check the optimistic bound against exhaustive small prediction spaces."""
from itertools import product

import numpy as np
import pytest

from experiments.bora_cycles.input_limits import summarize
from experiments.bora_cycles.metrics import RegionEvaluator


def test_absent_class_and_support_errors_are_counted_without_changing_accuracy():
    truth=np.array([[1,1,2],[2,0,0]],np.uint8)
    support=np.array([[1,0,1],[1,1,0]],bool)
    report=summarize(truth,support,np.array([[1]],np.uint8),{'1':'tumour','2':'stroma'},[.25,.5])
    assert report['positive_reference_pixels']==4 and report['deleted_positive_pixels']==1
    assert report['deleted_positive_area_um2']==.125 and report['unannotated_supported_pixels']==1
    assert report['reference_classes_absent_from_coarse']==[2]
    assert report['optimistic_macro_iou_upper_bound']==.25
    score=RegionEvaluator(truth,[.25,.5])
    for values in product([0,1],repeat=int(support.sum())):
        prediction=np.zeros_like(truth);prediction[support]=values
        assert score.score(prediction)['macro_iou']<=report['optimistic_macro_iou_upper_bound']


def test_complete_supported_class_information_gives_unit_optimistic_bound():
    truth=np.array([[1,2],[0,2]],np.uint8)
    report=summarize(truth,np.ones_like(truth,bool),truth,{},[1.,1.])
    assert report['optimistic_macro_iou_upper_bound']==1
    assert report['unannotated_supported_pixels']==1


def test_native_support_and_spacing_must_be_valid():
    with pytest.raises(ValueError):summarize(np.ones((2,2)),np.ones((1,2)),np.ones((2,2)),{},[1.,1.])
    with pytest.raises(ValueError):summarize(np.ones((2,2)),np.ones((2,2)),np.ones((2,2)),{},[1.,0.])
