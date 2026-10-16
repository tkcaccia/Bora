#!/usr/bin/env python3
"""Fixed scale-sensitivity screen; experimental scales are never called calibration."""
from pathlib import Path
import argparse,json,sys
import cv2
import numpy as np
from PIL import Image
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.gigatime_flash.run_pilot import load_model
from experiments.gigatime_flash.audit_transfer import full_image,he_agreement


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--flash",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);cv2.setNumThreads(2)
    model,_=load_model(a.flash);model.cuda().eval()
    original=np.asarray(Image.open(a.results/"transfer/tiger_114S.original.png").convert("RGB"))[512:1024,256:768]
    normalized=np.asarray(Image.open(a.results/"stains/tiger_114S.macenko.input.png").convert("RGB"))[512:1024,256:768]
    records=[]
    fig,axs=plt.subplots(2,5,figsize=(20,9))
    for row,(name,image) in enumerate([("original",original),("macenko",normalized)]):
        axs[row,0].imshow(original);axs[row,0].set_title("Original H&E | "+name+" input")
        for col,scale in enumerate([.125,.25,.5,.75],1):
            maps,plan=full_image(image,model,[.5,.5],target=scale)
            np.save(a.output/f"{name}.{scale}.dapi.npy",maps[0].astype(np.float16))
            proxy=he_agreement(original,maps[0]);records.append({"input":name,"sampling":plan,**proxy})
            print(name,scale,proxy,flush=True)
            axs[row,col].imshow(original)
            color=np.zeros((*image.shape[:2],4),np.float32);color[...,:3]=[0,1,1];color[...,3]=.45*maps[0]
            axs[row,col].imshow(color);axs[row,col].contour(maps[0],levels=[.5],colors="red",linewidths=.4)
            axs[row,col].set_title(f"Flash {scale} µm/px; DAPI > 0.5")
        for ax in axs[row]:ax.axis("off")
    fig.suptitle("Scale sensitivity on one fixed native 256 × 256 µm ROI; only 0.25 µm/px follows the clarified author workflow")
    fig.tight_layout();fig.savefig(a.output/"scale_sensitivity.png",dpi=180);plt.close(fig)
    (a.output/"scale_sensitivity.json").write_text(json.dumps({"native_crop_xywh":[256,512,512,512],"results":records},indent=2))


if __name__=="__main__":main()
