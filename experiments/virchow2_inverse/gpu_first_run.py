"""Additional physical-scale/stain inference and genuine gradient experiments 1–3."""
from __future__ import annotations
import argparse,json,time,gc
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import timm
from timm.layers import SwiGLUPacked
from safetensors.torch import load_file
from .common import MODEL_SHA256,digest,pad_image,starts,save_json

def load_model(root):
    if digest(root/"model.safetensors")!=MODEL_SHA256:raise ValueError("Wrong checkpoint")
    cfg=json.loads((root/"config.json").read_text())
    with torch.device("meta"):
        model=timm.create_model(cfg["architecture"],pretrained=False,**cfg["model_args"],mlp_layer=SwiGLUPacked,act_layer=torch.nn.SiLU)
    model.load_state_dict(load_file(str(root/"model.safetensors")),strict=True,assign=True)
    return model.to("cuda").eval().requires_grad_(False)

def windows(image):
    p=pad_image(image);coords=[(y,x) for y in starts(p.shape[0]) for x in starts(p.shape[1])]
    return p,coords,np.stack([p[y:y+224,x:x+224] for y,x in coords])

def normalize(x):
    mean=x.new_tensor([.485,.456,.406])[None,:,None,None];std=x.new_tensor([.229,.224,.225])[None,:,None,None]
    return (x-mean)/std

def selected_forward(model,x,selected):
    with torch.autocast("cuda",dtype=torch.float16):return model(normalize(x))[:,5:,selected].float()

def encode(model,image,selected):
    p,coords,patches=windows(image);gh,gw=p.shape[0]//14,p.shape[1]//14
    accum=np.zeros((gh,gw,len(selected)),np.float32);mass=np.zeros((gh,gw),np.float32)
    axis=np.maximum(np.hanning(16),.05);blend=axis[:,None]*axis[None,:]
    with torch.inference_mode():
        for start in range(0,len(coords),2):
            x=torch.from_numpy(patches[start:start+2].transpose(0,3,1,2).copy()).cuda().float()/255
            result=selected_forward(model,x,selected).cpu().numpy().reshape(-1,16,16,len(selected))
            for (y,xx),v in zip(coords[start:start+2],result):
                accum[y//14:y//14+16,xx//14:xx//14+16]+=v*blend[...,None]
                mass[y//14:y//14+16,xx//14:xx//14+16]+=blend
    return (accum/mass[...,None]).astype(np.float16),coords

def tv(x):return (x[...,1:,:]-x[...,:-1,:]).abs().mean()+(x[...,:,1:]-x[...,:,:-1]).abs().mean()

def optimize_window(model,original,target,selected,mean,scale,coef,intercept,mode,seed,steps):
    torch.manual_seed(seed)
    if mode==1:param=torch.nn.Parameter(torch.randn_like(original)*.35+.7)
    elif mode==2:param=torch.nn.Parameter(torch.full((1,1,16,16),2.,device="cuda"))
    else:param=torch.nn.Parameter(torch.logit(original.clamp(.005,.995)))
    optimizer=torch.optim.Adam([param],lr=.1 if mode!=3 else .035)
    neutral=original.new_tensor([.92,.82,.88])[None,:,None,None]
    target_z=(target-mean)/scale
    original_logits=target_z@coef.T+intercept
    original_prob=original_logits.softmax(-1)
    ordering=original_logits.argsort(-1,descending=True)
    first,second=ordering[...,0],ordering[...,1]
    history=[]
    def current():
        if mode==2:
            mask=F.interpolate(param.sigmoid(),size=(224,224),mode="bilinear",align_corners=False)
            return original*mask+neutral*(1-mask),mask
        return param.sigmoid(),None
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True);x,mask=current()
        feat=selected_forward(model,x,selected);z=(feat-mean)/scale;logits=z@coef.T+intercept
        if mode==1:
            primary=(z-target_z).square().mean();loss=primary+.03*tv(x)
        elif mode==2:
            primary=F.kl_div(logits.log_softmax(-1),original_prob,reduction="sum")/256
            loss=primary+.15*mask.mean()+.03*tv(mask)
        else:
            a=logits.gather(-1,first[...,None]).squeeze(-1);b=logits.gather(-1,second[...,None]).squeeze(-1)
            primary=F.relu(.5+a-b).mean();loss=primary+4*(x-original).abs().mean()+.05*tv(x-original)
        if not torch.isfinite(loss):raise FloatingPointError("Nonfinite gradient optimization loss")
        loss.backward();torch.nn.utils.clip_grad_norm_([param],5.);optimizer.step()
        if step in (0,steps//2,steps-1):history.append({"step":step,"total":float(loss.detach()),"primary":float(primary.detach())})
    with torch.inference_mode():
        x,mask=current();feat=selected_forward(model,x,selected);logits=((feat-mean)/scale)@coef.T+intercept
        pred=logits.argmax(-1)
        record={"history":history,"feature_mse_final":float((((feat-mean)/scale-target_z)**2).mean()),
                "class_preservation_fraction":float((pred==first).float().mean()),"counterfactual_target_fraction":float((pred==second).float().mean()),
                "mean_abs_pixel_edit":float((x-original).abs().mean()),"mask_fraction":float(mask.mean()) if mask is not None else None}
        return x[0].permute(1,2,0).cpu().numpy(),None if mask is None else mask[0,0].cpu().numpy(),record

def main():
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--root",type=Path,required=True)
    p.add_argument("--model-dir",type=Path,required=True);p.add_argument("--steps",type=int,default=32);a=p.parse_args()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.4)
    root=a.root;extra=root/"additional_inference";extra.mkdir(exist_ok=True)
    params=np.load(root/"classifier_parameters.npz");selected=params["selected"].tolist()
    image=np.asarray(Image.open(a.source/"image.original.png").convert("RGB"));model=load_model(a.model_dir)
    # Genuine re-encodings, completed before the slower optimization experiments.
    transforms={"stain_a":np.array([[1.12,.03,0],[0,.88,.03],[.02,0,1.03]]),"stain_b":np.array([[.88,0,.04],[.03,1.12,0],[0,.03,.97]])}
    metadata=[]
    for name in ["stain_a","stain_b","scale_1","scale_2"]:
        if name.startswith("stain"):
            od=-np.log((image.astype(float)+1)/256);out=np.rint(256*np.exp(-(od@transforms[name].T))-1).clip(0,255).astype(np.uint8)
            physical=[.5,.5]
        else:
            factor=2 if name=="scale_1" else 4
            out=np.asarray(Image.fromarray(image).resize((round(image.shape[1]/factor),round(image.shape[0]/factor)),Image.Resampling.LANCZOS))
            physical=[.5*image.shape[1]/out.shape[1],.5*image.shape[0]/out.shape[0]]
        destination=extra/(name+".npy")
        if not destination.exists():
            values,coords=encode(model,out,selected);np.save(destination,values);Image.fromarray(out).save(extra/(name+".png"))
        metadata.append({"name":name,"shape_yx":list(out.shape[:2]),"effective_mpp_xy":physical,"window_fov_um_xy":[224*v for v in physical],
                         "selected_dimensions":selected,"data_sha256":digest(destination),"original_pixel_detail_added":False,
                         "stain_matrix":transforms[name].tolist() if name in transforms else None})
        save_json(extra/"manifest.json",metadata);print(json.dumps({"additional_inference":name,"complete":True}),flush=True)
    save_json(extra/"READY.json",{"complete":True,"fine_reference_used":False})
    model.set_grad_checkpointing(True)
    padded,coords,patches=windows(image)
    target=np.load(root/"features/window_patch_tokens.npy",mmap_mode="r")
    tensors=[torch.tensor(params[key],dtype=torch.float32,device="cuda") for key in ["mean","scale","coefficients","intercept"]]
    mean,scale,coef,intercept=tensors
    native_mask=np.zeros(padded.shape[:2],np.float32);tile_root=root/"gpu_tiles";tile_root.mkdir(exist_ok=True)
    axis=np.maximum(np.hanning(224),.05);blend=axis[:,None]*axis[None,:]
    for mode in (1,2,3):
        seeds=(42,314) if mode==1 else (42,)
        for seed in seeds:
            accum=np.zeros_like(padded,dtype=np.float32);mass=np.zeros(padded.shape[:2],np.float32);records=[]
            native_mask.fill(0)
            for index,((y,x),patch) in enumerate(zip(coords,patches)):
                cache=tile_root/f"{mode:02d}_{seed}_{index:03d}_{a.steps}.npz";record_path=cache.with_suffix(".json")
                if cache.exists() and record_path.exists():
                    saved=np.load(cache);result=saved["image"];mask=saved["mask"];record=json.loads(record_path.read_text())
                else:
                    original=torch.from_numpy(patch.transpose(2,0,1).copy()).cuda().float()[None]/255
                    feat=torch.tensor(np.asarray(target[index,:,:,selected]).transpose(1,2,0).reshape(1,256,len(selected)).copy(),device="cuda",dtype=torch.float32)
                    # Advanced NumPy indexing puts selected dimensions first above.
                    start=time.perf_counter()
                    result,mask,record=optimize_window(model,original,feat,selected,mean,scale,coef,intercept,mode,seed+index,a.steps)
                    mask=np.ones((224,224),np.float32) if mask is None else mask
                    record.update({"window_index":index,"yx":[y,x],"seconds":time.perf_counter()-start,"steps":a.steps,"seed":seed+index})
                    np.savez_compressed(cache,image=result,mask=mask);save_json(record_path,record)
                accum[y:y+224,x:x+224]+=result*blend[...,None];mass[y:y+224,x:x+224]+=blend
                native_mask[y:y+224,x:x+224]+=mask*blend;records.append(record)
                if index%5==0:print(json.dumps({"idea":mode,"seed":seed,"windows_complete":index+1,"total":len(coords),"last":record}),flush=True)
            out=np.rint(np.clip(accum/mass[...,None],0,1)*255).astype(np.uint8)[:image.shape[0],:image.shape[1]]
            folder=root/f"{mode:02d}";folder.mkdir(exist_ok=True)
            filename=f"optimized_seed_{seed}.png";Image.fromarray(out).save(folder/filename)
            mask=(native_mask/mass)[:image.shape[0],:image.shape[1]]
            if mode==2:np.save(folder/"evidence_mask.npy",mask);Image.fromarray(np.rint(mask*255).astype(np.uint8)).save(folder/"evidence_mask.png")
            if mode==3:
                edit=np.abs(out.astype(np.float32)-image).mean(2)/255
                np.save(folder/"edit_magnitude.npy",edit)
            # Evaluate actual stitched pixels by running the frozen encoder again.
            actual,_=encode(model,out,selected)
            reference=np.load(root/"features/dense_patch_features.npy",mmap_mode="r")[...,selected].astype(np.float32)
            np.save(folder/f"stitched_features_seed_{seed}.npy",actual)
            metadata={"idea":mode,"seed":seed,"steps_per_window":a.steps,"windows":records,"image":filename,"original_rgb_at_inference":mode!=1,
                "checkpoint_sha256":MODEL_SHA256,"selected_dimensions":selected,"all_parameters_frozen":True,"gradient_checkpointing":True,
                "stitched_normalized_feature_mse":float(np.mean(((actual.astype(np.float32)-reference)/params["scale"])**2)),
                "cuda_peak_bytes":int(torch.cuda.max_memory_allocated()),"fine_reference_used":False}
            save_json(folder/f"gpu_seed_{seed}.json",metadata)
            if seed==42:save_json(folder/"GPU_READY.json",metadata)
            gc.collect();torch.cuda.empty_cache()
            print(json.dumps({"idea":mode,"seed":seed,"complete":True,"stitched_feature_mse":metadata["stitched_normalized_feature_mse"]}),flush=True)
    save_json(root/"GPU_COMPLETE.json",{"complete":True,"steps":a.steps})

if __name__=="__main__":main()
