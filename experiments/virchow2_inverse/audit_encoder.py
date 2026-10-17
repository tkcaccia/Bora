"""Re-encode actual pixels and verify the cached selected-window targets."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .gpu_ideas import load_model,selected_forward,windows
from .common import save_json

def main():
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--root",type=Path,required=True);p.add_argument("--model-dir",type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.3)
    model=load_model(a.model_dir);params=np.load(a.root/"classifier_parameters.npz");selected=params["selected"].tolist()
    image=np.asarray(Image.open(a.source/"image.original.png").convert("RGB"));_,coords,patches=windows(image)
    cached=np.load(a.root/"features/window_patch_tokens.npy",mmap_mode="r");records=[]
    with torch.inference_mode():
        for index in [0,len(coords)//2-3,len(coords)-1]:
            pixels=torch.from_numpy(patches[index].transpose(2,0,1).copy())[None].cuda().float()/255
            actual=selected_forward(model,pixels,selected).cpu().numpy()
            target=np.asarray(cached[index,:,:,selected]).transpose(1,2,0).reshape(1,256,len(selected))
            error=float(np.mean((actual-target)**2));records.append({"window":index,"mse":error})
            if error>=1e-5:raise AssertionError(f"Cached target differs from actual image encoding: {error}")
    save_json(a.root/"encoder_target_audit.json",{"passed":True,"raw_feature_mse":records,"fine_reference_used":False,"tolerance":1e-5,
        "note":"small differences reflect cached FP16 token storage and mixed-precision batch-size effects"})

if __name__=="__main__":main()
