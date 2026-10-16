"""Finite-difference audit of actual Virchow2 input-pixel gradients."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from experiments.virchow2_inverse.gpu_ideas import load_model,normalize
from experiments.virchow2_inverse.common import save_json,digest


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--previous",type=Path,required=True);p.add_argument("--model-dir",type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.4)
    model=load_model(a.model_dir);model.set_grad_checkpointing(True)
    index=json.loads((a.root/"PROTOCOL.json").read_text())["focal"]["index"]
    values=np.load(a.previous/"features/window_patch_tokens.npy",mmap_mode="r")
    target=torch.tensor(np.array(values[index]).reshape(1,256,1280),device="cuda",dtype=torch.float32)
    scale=torch.tensor(np.load(a.root/"contrast_basis.npz")["scale"],device="cuda")
    pixels=torch.tensor(np.array(Image.open(a.root/"tile.full.png")).transpose(2,0,1).copy(),device="cuda",dtype=torch.float32)[None]/255
    param=torch.logit(pixels.clamp(.002,.998)).requires_grad_(True)
    def objective(x,half=False):
        with torch.autocast("cuda",dtype=torch.float16,enabled=half):features=model(normalize(x.sigmoid()))[:,5:].float()
        return (((features-target)/scale)**2).mean()
    full=objective(param);gradient=torch.autograd.grad(full,param)[0]
    half=objective(param,True);ghalf=torch.autograd.grad(half,param)[0]
    direction=gradient/gradient.norm();eps=.05
    with torch.no_grad():finite=float((objective(param+eps*direction)-objective(param-eps*direction))/(2*eps))
    analytic=float((gradient*direction).sum());relative=abs(finite-analytic)/max(abs(analytic),1e-12)
    record=dict(analytic_directional_derivative=analytic,finite_difference_derivative=finite,relative_error=relative,
        gradient_cosine_fp16_vs_fp32=float(torch.nn.functional.cosine_similarity(gradient.ravel(),ghalf.ravel(),dim=0)),
        fp32_loss=float(full.detach()),fp16_loss=float(half.detach()),passed=relative<.1,finite_difference_epsilon=eps,
        scope="actual frozen Virchow2 input-gradient audit on decoder initialization; no original RGB target",implementation_sha256=digest(__file__))
    save_json(a.root/"GRADIENT_AUDIT.json",record);print(json.dumps(record),flush=True)
    if not record["passed"]:raise RuntimeError("FP32 gradient check failed")

if __name__=="__main__":main()
