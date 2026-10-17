"""Scientific invariants for the paired RGB reconstruction experiment."""
import numpy as np
import pytest
from experiments.virchow2_paired.prepare import case_key
from experiments.virchow2_paired.reconstruct import edit_tokens,contrast_basis
from experiments.virchow2_paired.model import Decoder,reconstruction_loss


def test_case_exclusion_handles_all_related_rois():
    assert case_key("114S_[1, 2, 3, 4].png")==case_key("114B_[3, 4, 5, 6].png")=="114"
    assert case_key("TCGA-A1-A0SB-DX1_[1, 2].png")=="TCGA-A1-A0SB"
    assert case_key("TC_S01_P000159_C001_[1, 2].png")=="TC_S01_P000159"


def test_identity_edit_and_nondiscriminative_features_preserved():
    rng=np.random.default_rng(1);tokens=rng.normal(size=(2,16,16,1280)).astype(np.float32)
    selected=np.arange(64);basis=np.linalg.qr(rng.normal(size=(64,4)))[0].astype(np.float32)
    mean=np.zeros(1280,np.float32);scale=np.ones(1280,np.float32);centre=np.zeros(64,np.float32)
    identity,stats=edit_tokens(tokens,mean,scale,selected,centre,basis,1)
    assert np.array_equal(identity,tokens)
    edited,stats=edit_tokens(tokens,mean,scale,selected,centre,basis,2)
    assert np.array_equal(edited[...,64:],tokens[...,64:])
    assert stats["max_abs_edit_training_sd"]<=.500001
    assert stats["max_relative_token_norm_edit"]<=.050001
    assert not np.array_equal(edited,tokens)
    # Neither amplification nor suppression may remove the subspace orthogonal
    # to the class contrasts when the component/norm caps are inactive.
    small,stats=edit_tokens(tokens,mean,scale,selected,centre,basis,1.001)
    remainder=(small-tokens)[...,selected]
    assert np.max(abs(remainder-remainder@basis@basis.T))<3e-7


def test_decoder_uses_all_spatial_channels_and_has_no_rgb_argument():
    import torch
    torch.set_num_threads(2);torch.manual_seed(4)
    model=Decoder(np.zeros(1280),np.ones(1280),width=16,blocks=1).eval()
    x=torch.randn(1,1280,16,16,requires_grad=True)
    decoded=model(x)
    assert decoded.shape==(1,3,224,224)
    decoded.mean().backward()
    assert torch.all(x.grad.abs().sum((0,2,3))>0)
    loss,parts=reconstruction_loss(decoded.detach(),decoded.detach())
    assert float(loss)<1e-5


def test_nucleus_matching_cannot_match_two_reconstructions_to_one_nucleus():
    from experiments.virchow2_paired.evaluate import match_nuclei
    truth=np.array([[10.,10.],[30.,30.]])
    predicted=np.array([[10.,11.],[10.,12.],[30.,31.]])
    metrics=match_nuclei(truth,predicted)
    assert metrics["matches"]==2
    assert metrics["recall"]==1
    assert metrics["precision"]==pytest.approx(2/3)
    assert metrics["median_displacement_um"]==.5
