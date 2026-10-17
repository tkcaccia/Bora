"""Mechanism and leakage checks, rather than twenty implementation snapshots."""
from itertools import product
import numpy as np
import pytest
from experiments.virchow2_inverse.protocol import expand
from experiments.virchow2_inverse.algorithms import minimum_seam,graph_transport,soft_transport,topology_project

def test_native_interpolation_uses_physical_token_centres_before_cropping():
    values=np.broadcast_to(np.arange(3,dtype=np.float32),(2,3))
    output=expand(values,(25,38))
    expected=np.clip((np.arange(38)+.5)/14-.5,0,2)
    np.testing.assert_allclose(output,np.broadcast_to(expected,(25,38)),atol=1e-6)

def test_quilting_seam_is_globally_minimal_among_valid_paths():
    error=np.random.default_rng(3).random((4,3))
    path=minimum_seam(error)
    oracle=min(sum(error[r,p[r]] for r in range(4)) for p in product(range(3),repeat=4) if all(abs(p[i]-p[i-1])<=1 for i in range(1,4)))
    assert np.all(abs(np.diff(path))<=1)
    assert np.isclose(error[np.arange(4),path].sum(),oracle)

def test_harmonic_transport_is_normalized_and_anchors_each_class():
    labels=np.repeat(np.arange(1,6)[None,:],5,axis=0)
    z=np.eye(5)[labels-1];prior=np.full((5,5,5),.2);seeds=np.zeros((5,5),bool);seeds[2]=True
    p=graph_transport(z,prior,labels,seeds)
    np.testing.assert_allclose(p.sum(2),1,atol=1e-6)
    assert np.all(p>=0)
    assert np.array_equal(p[2].argmax(1)+1,labels[2])
    # Soft seed weight is 30; at most four neighbours each have weight 1.001.
    assert np.all(p[2,np.arange(5),np.arange(5)]>=30/(30+4*1.001+.03))

def test_unmixing_allows_rare_classes_and_returns_normalized_weights():
    labels=np.repeat(np.arange(1,6),[30,10,4,4,12]);rng=np.random.default_rng(2)
    tiles=np.eye(5)[labels-1]*3+rng.normal(0,.05,(60,5))
    z=tiles[:50].reshape(5,10,5)
    p,record=soft_transport(z,tiles,labels)
    assert np.isfinite(p).all() and np.all(p>0)
    np.testing.assert_allclose(p.sum(2),1,atol=1e-6)
    assert np.all(p.mean((0,1))>0)

def test_topology_intervention_cannot_relabel_distant_coarse_interiors():
    pytest.importorskip("bora")
    coarse=np.ones((128,128),np.uint8);coarse[:,64:]=2
    proposal=np.full_like(coarse,2);p=np.zeros((128,128,5),np.float32);p[...,1]=1
    result,info=topology_project(coarse,proposal,p)
    np.testing.assert_array_equal(result[:,:30],coarse[:,:30])
    assert set(np.unique(result)).issubset({1,2})

def test_feature_only_inversion_cannot_copy_original_pixels():
    torch=pytest.importorskip("torch")
    if not torch.cuda.is_available():pytest.skip("CUDA-specific optimization")
    from experiments.virchow2_inverse.gpu_ideas import optimize_window,selected_forward
    class Tiny(torch.nn.Module):
        def forward(self,x):
            pooled=torch.nn.functional.avg_pool2d(x,14).flatten(2).transpose(1,2)
            channels=torch.cat([pooled,pooled**2,pooled[...,:2]**3],-1)
            return torch.cat([channels.new_zeros((len(x),5,8)),channels],1)
    model=Tiny().cuda().eval().requires_grad_(False)
    original=torch.full((1,3,224,224),.65,device="cuda")
    selected=list(range(8))
    with torch.inference_mode():target=selected_forward(model,original,selected)
    mean=torch.zeros(8,device="cuda");scale=torch.ones(8,device="cuda")
    coef=torch.arange(40,device="cuda",dtype=torch.float32).reshape(5,8)/50;bias=torch.zeros(5,device="cuda")
    first,_,_=optimize_window(model,original,target,selected,mean,scale,coef,bias,1,42,4)
    second,_,_=optimize_window(model,original*.2,target,selected,mean,scale,coef,bias,1,42,4)
    np.testing.assert_array_equal(first,second)
    assert np.isfinite(first).all()
