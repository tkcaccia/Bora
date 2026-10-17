import numpy as np
import pytest

from experiments.virchow2_inverse.common import SELECTORS,assemble_patches,pad_image,pixel_patches,reconstruction_folds,starts
from experiments.virchow2_inverse.selection import select_features


def test_native_pixels_roundtrip_through_padded_token_patches():
    image=np.random.default_rng(4).integers(0,256,(59,67,3),dtype=np.uint8)
    padded=pad_image(image);patches=pixel_patches(padded)
    restored=assemble_patches(patches,(padded.shape[0]//14,padded.shape[1]//14),image.shape[:2])
    np.testing.assert_array_equal(restored,image)


def test_all_window_starts_are_patch_aligned_and_cover_the_padded_frame():
    for length in [1176,1232]:
        coordinates=starts(length)
        assert coordinates[0]==0 and coordinates[-1]+224==length
        assert all(x%14==0 for x in coordinates)
        covered=np.zeros(length,bool)
        for x in coordinates:covered[x:x+224]=True
        assert covered.all()


def test_reconstruction_folds_keep_whole_spatial_blocks_together():
    folds=reconstruction_folds((24,24)).reshape(24,24)
    for y in range(0,24,8):
        for x in range(0,24,8):assert np.unique(folds[y:y+8,x:x+8]).size==1
    assert set(np.unique(folds))=={0,1,2,3}


@pytest.mark.parametrize("selector",SELECTORS)
def test_selectors_retain_information_for_every_class(selector):
    rng=np.random.default_rng(5);labels=np.repeat(np.arange(1,6),20)
    values=rng.normal(0,.03,(100,24))
    for g in range(1,6):values[labels==g,g-1]+=3
    result=select_features(values,labels,selector,top_k=10)
    assert set(range(5)).issubset(result["indices"])
    assert len(set(result["indices"]))==len(result["indices"])
    assert result["fine_annotation_used"] is False


@pytest.mark.parametrize("device",["cpu","mps"])
def test_streamed_forward_matches_ordinary_forward_on_actual_transformer_blocks(tmp_path,device):
    torch=pytest.importorskip("torch")
    pytest.importorskip("timm")
    if device=="mps" and not torch.backends.mps.is_available():pytest.skip("MPS unavailable")
    from timm.models.vision_transformer import VisionTransformer
    from timm.layers import SwiGLUPacked
    from safetensors.torch import save_file
    from experiments.virchow2_inverse.extract import streamed_forward
    torch.manual_seed(6);torch.set_num_threads(1)
    model=VisionTransformer(img_size=28,patch_size=14,embed_dim=32,depth=2,num_heads=4,mlp_ratio=2,
                           num_classes=0,global_pool="",reg_tokens=4,mlp_layer=SwiGLUPacked,act_layer=torch.nn.SiLU).eval()
    weights=tmp_path/"model.safetensors";save_file(model.state_dict(),str(weights))
    data=np.random.default_rng(6).normal(size=(3,3,28,28)).astype(np.float32)
    model=model.to(device)
    from contextlib import nullcontext
    with torch.inference_mode(),torch.autocast(device_type=device,dtype=torch.float16) if device=="mps" else nullcontext():
        expected=model(torch.from_numpy(data).to(device)).float().cpu().numpy()
    actual,metadata=streamed_forward(model,weights,data,device,batch_size=1)
    np.testing.assert_allclose(actual,expected,atol=.003 if device=="mps" else 2e-6,rtol=.003 if device=="mps" else 2e-6)
    assert metadata["all_state_tensors_consumed"] and metadata["strict_state_load"]
