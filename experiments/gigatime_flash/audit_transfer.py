#!/usr/bin/env python3
"""Cross-check Flash/CNN DAPI on paired references and unchanged benchmark RGB."""
from pathlib import Path
import argparse
import json
import sys

import cv2
import numpy as np
from PIL import Image
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.color import rgb2hed
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.gigatime_flash.audit_dapi import predict, reference, metrics
from experiments.gigatime_flash.original_cnn import gigatime
from experiments.gigatime_flash.run_pilot import load_model, digest
from bora.virtual_markers import sampling_plan


def full_image(rgb, model, mpp_xy, target=.25):
    plan = sampling_plan(rgb.shape[:2], mpp_xy, target)
    nh, nw = plan["inference_shape_yx"]
    image = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC if nh > rgb.shape[0] else cv2.INTER_AREA)
    ph, pw = max(256, nh), max(256, nw)
    image = np.pad(image, ((0, ph-nh), (0, pw-nw), (0, 0)), mode="reflect")
    values = (image.astype(np.float32)/255-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    def starts(n):
        return sorted(set([*range(0,n-255,128),n-256]))
    coords=[(y,x) for y in starts(ph) for x in starts(pw)]
    hann=np.maximum(np.hanning(256).astype(np.float32),.01)
    weight=hann[:,None]*hann[None,:]
    accum=np.zeros((23,ph,pw),np.float32)
    mass=np.zeros((ph,pw),np.float32)
    with torch.inference_mode():
        for k in range(0,len(coords),2):
            batch=coords[k:k+2]
            x=np.stack([values[y:y+256,x:x+256].transpose(2,0,1) for y,x in batch])
            maps=model(torch.from_numpy(x).cuda()).sigmoid().cpu().numpy()
            for (y,x),p in zip(batch,maps):
                accum[:,y:y+256,x:x+256]+=p*weight
                mass[y:y+256,x:x+256]+=weight
    result=np.stack([cv2.resize(p[:nh,:nw]/mass[:nh,:nw],rgb.shape[1::-1],interpolation=cv2.INTER_LINEAR) for p in accum])
    return result,plan


def he_agreement(rgb,dapi):
    """Stain-derived proxy only; never a replacement for measured DAPI."""
    he=rgb2hed(rgb)[...,0]
    tissue=(rgb.min(2)<220)
    return {"spearman_with_HE_hematoxylin_proxy":float(spearmanr(he[tissue],dapi[tissue]).statistic),
            "status":"descriptive H&E proxy, not independent DAPI ground truth"}


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--data-root",type=Path,default=Path("/mnt/sata_ssd/Bora-benchmark"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--flash",type=Path,required=True)
    p.add_argument("--cnn",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);cv2.setNumThreads(2)
    flash,info=load_model(a.flash)
    if digest(a.cnn)!="d200308302c7fefd13b259c62de546f2c859c28d1079e9f67dc8808915d95d0c":raise ValueError("CNN checksum")
    cnn=gigatime(23)
    cnn.load_state_dict(torch.load(a.cnn,map_location="cpu",weights_only=True),strict=True)
    models={"flash":flash.cuda().eval(),"original_cnn_control":cnn.cuda().eval()}
    source=a.data_root/"raw/gigatime-official-test/extracted/sample_test_data/data"
    images=sorted(source.glob("*_he.png"))[:5]
    scores=[]
    for name,model in models.items():
        for image in images:
            key=image.name.replace("_he.png","")
            rgb=np.asarray(Image.open(image).convert("RGB"))
            target,nuclei=reference(source/f"{key}_comet_binary_thres_labels.pkl.gz")
            pred=predict(rgb,model)
            scores.append({"model":name,"sample":key,**metrics(pred[0],nuclei)})
            if image==images[0]:
                fig,axs=plt.subplots(1,4,figsize=(16,4))
                for ax,im,title in zip(axs,[rgb,nuclei,target[...,0],pred[0]],["Original H&E","Released nuclear reference","Released DAPI binary","Predicted DAPI: "+name]):
                    ax.imshow(im,cmap="gray",vmin=0,vmax=1) if im.ndim==2 else ax.imshow(im)
                    ax.set_title(title);ax.axis("off")
                fig.tight_layout();fig.savefig(a.output/f"paired_reference_{name}.png",dpi=180);plt.close(fig)
        subset=[r for r in scores if r["model"]==name]
        print(name,{k:float(np.mean([r[k] for r in subset])) for k in ["auc","average_precision","pearson_8px","dice_05"]},flush=True)
    cases=[("tiger_114S",a.data_root/"results/gigatime-flash-pilot-2026-10-16/image.original.png",[.5,.5]),
           ("camelyon16_tumor091",a.data_root/"prepared/camelyon16-subset/native_examples/tumor_091.png",[.226321,.226316])]
    transfer=[]
    for key,path,mpp in cases:
        rgb=np.asarray(Image.open(path).convert("RGB"))
        # First fixed central native crop; retains original RGB and physical sampling.
        if key.startswith("camelyon"):
            h,w=rgb.shape[:2];rgb=rgb[h//2-512:h//2+512,w//2-512:w//2+512]
        Image.fromarray(rgb).save(a.output/f"{key}.original.png")
        fig,axs=plt.subplots(2,3,figsize=(14,9))
        for row,(name,model) in enumerate(models.items()):
            maps,plan=full_image(rgb,model,mpp)
            np.save(a.output/f"{key}.{name}.native.npy",maps.astype(np.float16))
            proxy=he_agreement(rgb,maps[0])
            transfer.append({"sample":key,"model":name,"sampling":plan,**proxy})
            print(key,name,proxy,flush=True)
            axs[row,0].imshow(rgb);axs[row,0].set_title("Original H&E")
            axs[row,1].imshow(maps[0],cmap="gray",vmin=0,vmax=1);axs[row,1].set_title(name+": DAPI")
            axs[row,2].imshow(rgb)
            color=np.zeros((*rgb.shape[:2],4),np.float32);color[...,:3]=[0,1,1];color[...,3]=.75*maps[0]
            axs[row,2].imshow(color);axs[row,2].set_title("DAPI overlay")
            for ax in axs[row]:ax.axis("off")
        fig.suptitle(key+" | Corrected input sampling: 0.25 µm/pixel | Model outputs, no stain-derived replacement")
        fig.tight_layout();fig.savefig(a.output/f"{key}.model_comparison.png",dpi=180);plt.close(fig)
    (a.output/"transfer.json").write_text(json.dumps({"paired_calibration":scores,"transfer":transfer,"flash_model":info},indent=2))


if __name__=="__main__":main()
