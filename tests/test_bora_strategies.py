import numpy as np
import pytest
from experiments.bora_strategies.context import Context,simplex
from experiments.bora_strategies.methods import run
from experiments.bora_strategies.registry import STRATEGIES,CONSENSUS,REJECTED_STRATEGIES


def test_simplex_projection_respects_class_mass():
    a=np.random.default_rng(4).normal(size=(12,10,5));p=simplex(a)
    assert np.all(p>=0) and np.allclose(p.sum(-1),1)


def test_context_never_trains_on_completed_unlabelled_tiles():
    he=np.full((100,180,3),150,np.uint8);original=np.ones((100,180),np.uint8);original[:,90:]=0;original[:50,:45]=3
    completed=original.copy();completed[:,90:]=3
    c=Context(he,he,original,completed,np.ones(original.shape,bool),[.5,.5])
    assert np.all(c.core[c.original==0]==-1) and np.all(c.anchors[c.original==0]==-1)
    assert set(c.ids)=={1,3}


def test_retained_solvers_keep_nonconsecutive_label_geometry_and_cover_the_field():
    rgb=np.zeros((180,270,3),np.uint8);rgb[:,:135]=[210,50,80];rgb[:,135:]=[50,170,80]
    original=np.ones((180,270),np.uint8);original[:,135:]=7
    c=Context(rgb,rgb,original,original,np.ones(original.shape,bool),[.48,.52])
    for key in STRATEGIES:
        result=c.finish(run(key,c))
        assert result.shape==c.shape and set(np.unique(result))=={1,7}
    assert set(CONSENSUS)<=set(STRATEGIES) and set(STRATEGIES).isdisjoint(REJECTED_STRATEGIES)
    for key in REJECTED_STRATEGIES:
        with pytest.raises(ValueError):run(key,c)


def test_correlated_conversions_do_not_inflate_case_bootstrap():
    from experiments.bora_strategies.score import summarize
    cases=[]
    for j in range(6):
        values={k:dict(score=dict(macro_iou=.5+j*.02+(i%3)*.01,boundary_f1_at_um={'8':.7})) for i,k in enumerate(STRATEGIES)}
        cases.append(dict(dataset='a' if j<3 else 'b',tiles=dict(macro_iou=.45+j*.02),conversions={m:dict(strategies=values) for m in ('09','19')}))
    single=summarize(cases,['09']);duplicate=summarize(cases,['09','19'])
    for a,b in zip(single,duplicate):assert a['strategy']==b['strategy'] and a['paired_delta_bootstrap95']==b['paired_delta_bootstrap95']
