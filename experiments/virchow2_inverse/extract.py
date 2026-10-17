#!/usr/bin/env python3
"""Pinned Virchow2 inference, with block-major streaming for bounded memory.

Streaming is the same forward graph with independent images batched through
one transformer block before loading the next. No model fine-tuning occurs.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.virchow2_inverse.common import MODEL_REPOSITORY,MODEL_REVISION,MODEL_SHA256,PATCH,WINDOW,digest,pad_image,save_json,starts


def streamed_forward(model, weights, normalized_images, device, batch_size=1, mixed_precision=True, progress=None):
    import torch
    from safetensors import safe_open
    device=torch.device(device)
    with safe_open(str(weights),framework="pt",device="cpu") as reader:available=set(reader.keys())
    expected=set(model.state_dict())
    if available!=expected:
        raise ValueError(f"Checkpoint mismatch: missing={sorted(expected-available)[:5]}, unexpected={sorted(available-expected)[:5]}")
    consumed=set()

    def load_module(module,prefix):
        keys=set(module.state_dict())
        with safe_open(str(weights),framework="pt",device="cpu") as reader:
            state={key:reader.get_tensor(prefix+key).to(device=device) for key in keys}
        module.load_state_dict(state,strict=True,assign=True)
        module.eval().requires_grad_(False)
        consumed.update(prefix+key for key in keys)

    def clear_module(module):
        module.to("meta")
        gc.collect()
        if device.type=="mps":torch.mps.empty_cache()
        elif device.type=="cuda":torch.cuda.empty_cache()

    def amp():
        return torch.autocast(device_type=device.type,dtype=torch.float16) if mixed_precision and device.type in {"mps","cuda"} else nullcontext()

    # Keep only small root parameters (position, CLS, registers) resident.
    with safe_open(str(weights),framework="pt",device="cpu") as reader:
        for name,_ in list(model.named_parameters(recurse=False)):
            value=reader.get_tensor(name).to(device=device)
            setattr(model,name,torch.nn.Parameter(value,requires_grad=False));consumed.add(name)
        for name,_ in list(model.named_buffers(recurse=False)):
            if name in available:
                setattr(model,name,reader.get_tensor(name).to(device=device));consumed.add(name)
    load_module(model.patch_embed,"patch_embed.")
    load_module(model.norm_pre,"norm_pre.")
    n=len(normalized_images);tokens=None
    for start in range(0,n,batch_size):
        with torch.inference_mode(),amp():
            x=torch.as_tensor(normalized_images[start:start+batch_size]).to(device)
            x=model.patch_embed(x);x=model._pos_embed(x);x=model.patch_drop(x);x=model.norm_pre(x)
            result=x.float().cpu()
            if tokens is None:tokens=torch.empty((n,*result.shape[1:]),dtype=torch.float32)
            tokens[start:start+len(result)]=result
        del x,result
        if device.type=="mps":
            torch.mps.synchronize();torch.mps.empty_cache()
    clear_module(model.patch_embed);clear_module(model.norm_pre)
    if progress:
        print(json.dumps({"stage":"embedding_complete","device_allocated_bytes":torch.mps.current_allocated_memory() if device.type=="mps" else None}),flush=True)
    for index,block in enumerate(model.blocks):
        began=time.perf_counter();load_module(block,f"blocks.{index}.")
        with torch.inference_mode(),amp():
            for start in range(0,n,batch_size):
                value=block(tokens[start:start+batch_size].to(device))
                tokens[start:start+len(value)]=value.float().cpu()
        clear_module(block)
        if progress:progress(index,len(model.blocks),time.perf_counter()-began)
    load_module(model.norm,"norm.")
    load_module(model.fc_norm,"fc_norm.");load_module(model.head,"head.")
    with torch.inference_mode(),amp():
        for start in range(0,n,batch_size):
            value=model.norm(tokens[start:start+batch_size].to(device))
            # global_pool='' and num_classes=0 retain the token sequence.
            value=model.forward_head(value)
            tokens[start:start+len(value)]=value.float().cpu()
    if consumed!=available:raise AssertionError(f"Unconsumed state tensors: {sorted(available-consumed)}")
    return tokens.numpy(),{"strict_state_load":True,"state_tensors":len(consumed),"all_state_tensors_consumed":True,
                           "block_major_streaming":True,"whole_model_resident_on_device":False}


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--model-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",choices=["mps","cuda","cpu"],required=True)
    parser.add_argument("--batch-size",type=int,default=1)
    parser.add_argument("--execution",choices=["standard","streamed"],default="standard")
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    import torch,timm
    from timm.layers import SwiGLUPacked
    torch.set_num_threads(1)
    if args.device=="mps":
        if not torch.backends.mps.is_available():raise RuntimeError("Requested MPS is unavailable")
        torch.mps.set_per_process_memory_fraction(.18)
    elif args.device=="cuda" and not torch.cuda.is_available():raise RuntimeError("Requested CUDA is unavailable")
    if args.device=="cuda":torch.cuda.set_per_process_memory_fraction(.4)
    config=json.loads((args.model_dir/"config.json").read_text())
    if config["architecture"]!="vit_huge_patch14_224" or config["model_args"]["reg_tokens"]!=4:
        raise ValueError("Unexpected Virchow2 architecture")
    weights=args.model_dir/"model.safetensors"
    if digest(weights)!=MODEL_SHA256:raise ValueError("Checkpoint differs from the pinned released Virchow2 weights")
    image=np.asarray(Image.open(args.source/"image.original.png").convert("RGB"));padded=pad_image(image)
    coordinates=[(y,x) for y in starts(padded.shape[0]) for x in starts(padded.shape[1])]
    images=np.stack([padded[y:y+WINDOW,x:x+WINDOW] for y,x in coordinates]).astype(np.float32)/255
    mean=np.asarray(config["pretrained_cfg"]["mean"],np.float32);std=np.asarray(config["pretrained_cfg"]["std"],np.float32)
    normalized=np.ascontiguousarray(((images-mean)/std).transpose(0,3,1,2));del images
    with torch.device("meta"):
        model=timm.create_model(config["architecture"],pretrained=False,**config["model_args"],mlp_layer=SwiGLUPacked,act_layer=torch.nn.SiLU)
    model.eval().requires_grad_(False)
    parameters=sum(p.numel() for p in model.parameters())
    began=time.perf_counter()
    if args.execution=="streamed":
        outputs,loading=streamed_forward(model,weights,normalized,args.device,args.batch_size,progress=lambda i,n,s:print(json.dumps({"block":i+1,"of":n,"seconds":s}),flush=True))
    else:
        from safetensors.torch import load_file
        state=load_file(str(weights),device="cpu")
        model.load_state_dict(state,strict=True,assign=True)
        n_state=len(state);model=model.to(args.device).eval().requires_grad_(False);del state
        outputs=np.empty((len(normalized),261,1280),np.float32)
        with torch.inference_mode(),torch.autocast(device_type=args.device,dtype=torch.float16) if args.device in {"cuda","mps"} else nullcontext():
            for start in range(0,len(normalized),args.batch_size):
                values=model(torch.from_numpy(normalized[start:start+args.batch_size]).to(args.device))
                outputs[start:start+len(values)]=values.float().cpu().numpy()
                if start%10==0:print(json.dumps({"windows_complete":start+len(values),"total":len(normalized)}),flush=True)
        loading={"strict_state_load":True,"state_tensors":n_state,"all_state_tensors_consumed":True,
                 "block_major_streaming":False,"whole_model_resident_on_device":True,
                 "cuda_peak_allocated_bytes":int(torch.cuda.max_memory_allocated()) if args.device=="cuda" else None}
    if outputs.shape!=(len(coordinates),261,1280) or not np.isfinite(outputs).all():raise ValueError("Unexpected/nonfinite output token tensor")
    tokens=outputs[:,5:].reshape(-1,16,16,1280)
    gh,gw=padded.shape[0]//PATCH,padded.shape[1]//PATCH
    accum=np.zeros((gh,gw,1280),np.float32)
    squares=np.zeros_like(accum);mass=np.zeros((gh,gw),np.float32)
    window=np.maximum(np.hanning(16).astype(np.float32),.05);blend=window[:,None]*window[None,:]
    for (y,x),values in zip(coordinates,tokens):
        row,col=y//PATCH,x//PATCH
        accum[row:row+16,col:col+16]+=values*blend[...,None]
        squares[row:row+16,col:col+16]+=values**2*blend[...,None]
        mass[row:row+16,col:col+16]+=blend
    if np.any(mass<=0):raise AssertionError("Dense token grid has uncovered cells")
    dense=(accum/mass[...,None]).astype(np.float16)
    np.save(out/"dense_patch_features.npy",dense)
    variance=np.maximum(squares/mass[...,None]-(accum/mass[...,None])**2,0)
    np.save(out/"overlap_view_variance.npy",variance.astype(np.float16))
    np.save(out/"window_patch_tokens.npy",tokens.astype(np.float16))
    np.save(out/"overlap_weight_mass.npy",mass)
    np.save(out/"window_tile_embeddings.npy",np.concatenate([outputs[:,0],outputs[:,5:].mean(1)],axis=1).astype(np.float16))
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    info={"repository":MODEL_REPOSITORY,"revision":MODEL_REVISION,"checkpoint_sha256":digest(weights),"config_sha256":digest(args.model_dir/"config.json"),
          "image_sha256":digest(args.source/"image.original.png"),"parameters":parameters,"device":args.device,"batch_size":args.batch_size,
          "precision":"FP32 checkpoint and layernorm/residuals; autocast FP16 linear/attention" if args.device!="cpu" else "FP32",
          "native_shape_yx":list(image.shape[:2]),"padded_shape_yx":list(padded.shape[:2]),"native_mpp_xy":[.5,.5],
          "model_input_px":224,"model_input_mpp_xy":[.5,.5],"model_window_fov_um_xy":[112.,112.],
          "patch_px":14,"patch_fov_um_xy":[7.,7.],"dense_grid_shape_yx":[gh,gw],"dense_grid_centres_native_px":"(column+0.5)*14,(row+0.5)*14",
          "padding":"reflect only at right/bottom to a multiple of 14; crop outputs back to native frame",
          "window_stride_px":112,"window_coordinates_yx":coordinates,"window_count":len(coordinates),
          "stitching":"patch-aligned weighted overlap-add of final patch tokens, separable Hann with axis floor 0.05",
          "tokens":"CLS index 0; four registers indices 1:5 excluded; dense spatial tokens indices 5:261",
          "tile_embedding":"concatenate CLS and mean spatial tokens (2560); retained for audit, not reconstruction inputs",
          "dense_feature_channels":1280,"dense_features_sha256":digest(out/"dense_patch_features.npy"),
          "overlap_variance_file":"overlap_view_variance.npy",
          "overlap_variance_interpretation":"weighted variation across overlapping contexts, not calibrated epistemic uncertainty; zero where only one context covers a token",
          "unblended_window_tokens_file":"window_patch_tokens.npy",
          "fine_annotation_used":False,"microscope_optical_resolution_um":None,"microscope_objective_magnification":None,
          "runtime":{"torch":torch.__version__,"timm":timm.__version__,"python":platform.python_version()},
          "source_sha256":digest(__file__),"forward_source":inspect.getsource(type(model).forward_features),
          "seconds":time.perf_counter()-began,"max_rss_bytes":rss if sys.platform=="darwin" else rss*1024,**loading}
    save_json(out/"inference.json",info);print(json.dumps({"complete":True,"seconds":info["seconds"],"max_rss_bytes":info["max_rss_bytes"]}),flush=True)


if __name__=="__main__":main()
