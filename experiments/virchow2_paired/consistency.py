"""Focal-tile diagnostic: refine decoder pixels using full-token targets only.

This is separate from the already frozen primary decoder experiment. It asks
whether decoder approximation error accounts for lost detail. The optimization
function receives no original image pixels and does not use expert masks.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from experiments.virchow2_inverse.gpu_ideas import load_model,normalize,tv
from experiments.virchow2_inverse.common import save_json,digest
from .reconstruct import edit_tokens


def refine(model,initial,target,scale,steps,lr=.002):
    param=torch.nn.Parameter(torch.logit(initial.clamp(.002,.998)))
    optimizer=torch.optim.Adam([param],lr=lr)
    best=float("inf");best_pixels=None;history=[]
    for step in range(steps):
        for group in optimizer.param_groups:group["lr"]=lr*(.1+.9*(1+np.cos(np.pi*step/steps))/2)
        optimizer.zero_grad(set_to_none=True);image=param.sigmoid()
        # Input-pixel gradients through a frozen 632M encoder are small: use
        # FP32 here, rather than unscaled FP16 gradients that may underflow.
        features=model(normalize(image))[:,5:].float()
        mse=(((features-target)/scale)**2).mean()
        # Both regularizers reference the learned decoder output, never original RGB.
        loss=mse+.001*tv(image)+.001*(image-initial).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError("Invalid feature consistency loss")
        if float(loss.detach())<best:
            best=float(loss.detach());best_pixels=image.detach().clone();best_step=step
        loss.backward();torch.nn.utils.clip_grad_norm_([param],5);optimizer.step()
        if step%32==0 or step==steps-1:
            history.append(dict(step=step,normalized_feature_mse=float(mse.detach()),objective=float(loss.detach())))
    return best_pixels,dict(history=history,best_objective=best,best_step=best_step)


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--previous",type=Path,required=True);p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--steps",type=int,default=400);p.add_argument("--lr",type=float,default=.002);a=p.parse_args()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.4)
    protocol=json.loads((a.root/"PROTOCOL.json").read_text());index=protocol["focal"]["index"]
    basis=np.load(a.root/"contrast_basis.npz")
    raw=np.load(a.previous/"features/window_patch_tokens.npy",mmap_mode="r")
    original=np.array(raw[index:index+1],dtype=np.float32)
    model=load_model(a.model_dir);model.set_grad_checkpointing(True)
    save_json(a.root/"CONSISTENCY_PROTOCOL.json",dict(steps=a.steps,lr=a.lr,lr_schedule="cosine down to0.1initial LR",seed="deterministic decoder initialization",
        original_rgb_supplied=False,initialization="frozen decoder's direct-window RGB output, quantized8bit",
        targets="all1280 final spatial channels, train-SD-normalized MSE",same_procedure_for_both=True,
        precision="FP32 forward and input gradients; frozen encoder with activation checkpointing",
        objective="feature MSE +0.001TV +0.001MSE to decoder initialization",focal_index=index,
        scope="single-tile diagnostic only; primary ROI/Bora experiment unchanged",implementation_sha256=digest(__file__)))
    records={};hashes={}
    for name,gain in [("full",1.),("selective",1.25)]:
        target,_=edit_tokens(original,basis["mean"],basis["scale"],basis["selected"],basis["centre"],basis["basis"],gain)
        target=torch.tensor(target.reshape(1,256,1280),device="cuda")
        scale=torch.tensor(basis["scale"],device="cuda").reshape(1,1,1280)
        initial=torch.from_numpy(np.array(Image.open(a.root/f"tile.{name}.png")).transpose(2,0,1).copy()).cuda().float()[None]/255
        start=time.monotonic();pixels,record=refine(model,initial,target,scale,a.steps,a.lr)
        result=np.rint(pixels[0].permute(1,2,0).cpu().numpy()*255).clip(0,255).astype(np.uint8)
        filename=f"tile.consistency.{name}.png";Image.fromarray(result).save(a.root/filename)
        hashes[filename]=digest(a.root/filename);record["seconds"]=time.monotonic()-start;records[name]=record
        print(json.dumps(dict(name=name,**record)),flush=True)
    save_json(a.root/"CONSISTENCY_FROZEN.json",dict(files_sha256=hashes,records=records,fine_reference_used=False,
        original_rgb_target_used=False,timestamp_unix=time.time()))

if __name__=="__main__":main()
