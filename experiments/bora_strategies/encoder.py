"""Frozen Virchow2 spatial tokens on the entire RGB field."""
import numpy as np
import torch
from experiments.virchow2_inverse.common import pad_image,starts
from experiments.virchow2_inverse.gpu_ideas import normalize

def encode(model,image,selected=None):
    p=pad_image(image);coords=[(y,x) for y in starts(p.shape[0]) for x in starts(p.shape[1])]
    channels=1280 if selected is None else len(selected)
    accum=np.zeros((p.shape[0]//14,p.shape[1]//14,channels),np.float32);mass=np.zeros(accum.shape[:2],np.float32)
    axis=np.maximum(np.hanning(16),.05);blend=axis[:,None]*axis[None,:]
    with torch.inference_mode():
        for start in range(0,len(coords),4):
            batch=np.stack([p[y:y+224,x:x+224].transpose(2,0,1) for y,x in coords[start:start+4]])
            x=torch.from_numpy(batch.copy()).cuda().float()/255
            with torch.autocast("cuda",dtype=torch.float16):values=model(normalize(x))[:,5:]
            if selected is not None:values=values[:,:,selected]
            values=values.float().cpu().numpy().reshape(-1,16,16,channels)
            for (y,x),v in zip(coords[start:start+4],values):
                accum[y//14:y//14+16,x//14:x//14+16]+=v*blend[...,None];mass[y//14:y//14+16,x//14:x//14+16]+=blend
    result=(accum/mass[...,None]).astype(np.float16)
    if not np.isfinite(result).all():raise FloatingPointError("Nonfinite embeddings")
    return result,dict(window_coordinates_yx=coords,window_px=224,stride_px=112,padded_shape_yx=list(p.shape[:2]),
        spatial_tokens="5:261; CLS and registers excluded",patch_px=14,channels=channels,selected_dimensions=None if selected is None else selected.tolist())

