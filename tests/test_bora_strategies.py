from itertools import product
import numpy as np
from scipy import ndimage as ndi
from experiments.bora_strategies.graphs import cut_binary,expansion
from experiments.bora_strategies.geometry import SIMPLE,topology
from experiments.bora_strategies.context import Context,simplex


def test_binary_graph_cut_matches_exhaustive_energy():
    rng=np.random.default_rng(83);pairs=np.array([[0,1],[1,2],[2,3],[0,3],[1,3]]);w=rng.uniform(.1,1,len(pairs));d0=rng.uniform(-1,1,4);d1=rng.uniform(-1,1,4)
    def energy(x):return np.where(x,d1,d0).sum()+np.sum(w*(x[pairs[:,0]]!=x[pairs[:,1]]))
    actual=cut_binary(d0,d1,pairs,w)
    assert energy(actual)<=min(energy(np.array(x)) for x in product((False,True),repeat=4))+1e-3


def test_alpha_expansion_never_increases_declared_energy():
    rng=np.random.default_rng(23);pairs=np.array([[0,1],[1,2],[2,3],[0,3]]);u=rng.uniform(0,1,(4,3));w=np.ones(4)*.4
    result,history=expansion(u,pairs,w,np.array([0,1,2,0]))
    assert np.all(np.diff(history)<0)
    assert set(result)<=set(range(3))


def test_simple_point_table_protects_isolated_objects_and_new_holes():
    assert not SIMPLE[0]
    assert not SIMPLE[255]
    assert SIMPLE[sum(1<<i for i in (0,1,2))]


def test_simplex_projection_respects_class_mass():
    a=np.random.default_rng(4).normal(size=(12,10,5));p=simplex(a)
    assert np.all(p>=0)
    assert np.allclose(p.sum(-1),1)


def test_topology_moves_preserve_components_and_holes():
    h,w=80,100;yy,xx=np.indices((h,w));initial=((yy-40)**2+(xx-45)**2<24**2).astype(np.int32)
    class C:pass
    c=C();c.initial=initial;c.shape=initial.shape;c.k=2;c.anchors=np.full(initial.shape,-1,np.int32);c.diagnostics={}
    target=((yy-40)**2+(xx-50)**2<26**2).astype(np.int32);c.u=np.stack([target==1,target==0],-1).astype(float)
    result=topology(c)
    for i in (0,1):
        assert ndi.label(result==i,np.ones((3,3)))[1]==ndi.label(initial==i,np.ones((3,3)))[1]
        assert ndi.label(result!=i)[1]==ndi.label(initial!=i)[1]


def test_context_never_trains_on_completed_unlabelled_tiles():
    he=np.full((100,180,3),150,np.uint8);original=np.ones((100,180),np.uint8);original[:,90:]=0;original[:50,:45]=3
    completed=original.copy();completed[:,90:]=3
    c=Context(he,he,original,completed,np.ones(original.shape,bool),[.5,.5])
    assert np.all(c.core[c.original==0]==-1)
    assert np.all(c.anchors[c.original==0]==-1)
    assert set(c.ids)=={1,3}


def test_correlated_conversions_do_not_inflate_case_bootstrap():
    from experiments.bora_strategies.score import summarize
    from experiments.bora_strategies.registry import STRATEGIES
    cases=[]
    for j in range(6):
        values={k:dict(score=dict(macro_iou=.5+j*.02+(i%3)*.01,boundary_f1_at_um={'8':.7})) for i,k in enumerate(STRATEGIES)}
        cases.append(dict(dataset='a' if j<3 else 'b',conversions={m:dict(strategies=values) for m in ('09','19')}))
    single=summarize(cases,['09']);duplicate=summarize(cases,['09','19'])
    for a,b in zip(single,duplicate):
        assert a['strategy']==b['strategy']
        assert a['paired_delta_bootstrap95']==b['paired_delta_bootstrap95']
