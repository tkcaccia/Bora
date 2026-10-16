#!/usr/bin/env python3
"""Test two explicit stain-transfer hypotheses without changing image geometry."""
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

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.gigatime_flash.audit_transfer import full_image,he_agreement
from experiments.gigatime_flash.audit_dapi import predict,reference,metrics
from experiments.gigatime_flash.run_pilot import load_model


def optical_density(rgb):
    return -np.log((rgb.astype(np.float64)+1)/256)


def macenko_basis(rgb):
    od=optical_density(rgb).reshape(-1,3)
    tissue=od[(od>.15).all(1)]
    if len(tissue)<100:raise ValueError("Too few stained pixels for Macenko")
    _,v=np.linalg.eigh(np.cov(tissue,rowvar=False))
    plane=v[:,[2,1]]
    for j in range(2):
        if plane[0,j]<0:plane[:,j]*=-1
    proj=tissue@plane
    angles=np.arctan2(proj[:,1],proj[:,0])
    low,high=np.percentile(angles,[1,99])
    a=plane@np.array([np.cos(low),np.sin(low)])
    b=plane@np.array([np.cos(high),np.sin(high)])
    matrix=np.stack([a,b] if a[0]>b[0] else [b,a],axis=1)
    concentration=np.maximum(np.linalg.lstsq(matrix,od.T,rcond=None)[0],0)
    return matrix,np.percentile(concentration,99,axis=1)


def normalize(rgb,target,method):
    if method=="macenko":
        source,maximum=macenko_basis(rgb)
        dest,target_max=macenko_basis(target)
        concentration=np.maximum(np.linalg.lstsq(source,optical_density(rgb).reshape(-1,3).T,rcond=None)[0],0)
        concentration*= (target_max/np.maximum(maximum,1e-6))[:,None]
        output=(256*np.exp(-(dest@concentration))-1).T.reshape(rgb.shape)
        return np.clip(output,0,255).astype(np.uint8),{"source_basis":source.tolist(),"target_basis":dest.tolist(),"source_99":maximum.tolist(),"target_99":target_max.tolist()}
    if method=="reinhard_tissue":
        lab=cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
        target_lab=cv2.cvtColor(target.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
        support=optical_density(rgb).mean(2)>.15
        target_support=optical_density(target).mean(2)>.15
        sm,ss=lab[support].mean(0),lab[support].std(0)
        tm,ts=target_lab[target_support].mean(0),target_lab[target_support].std(0)
        norm=(lab-sm)/np.maximum(ss,1e-6)*ts+tm
        output=cv2.cvtColor(norm.astype(np.float32),cv2.COLOR_LAB2RGB)
        # White background is retained rather than forcing glass to tissue colour.
        output[~support]=rgb[~support]/255
        return np.clip(output*255,0,255).astype(np.uint8),{"source_mean":sm.tolist(),"source_std":ss.tolist(),"target_mean":tm.tolist(),"target_std":ts.tolist()}
    raise ValueError(method)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--data-root",type=Path,default=Path("/mnt/sata_ssd/Bora-benchmark"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--flash",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);cv2.setNumThreads(2)
    model,info=load_model(a.flash);model.cuda().eval()
    source=a.data_root/"raw/gigatime-official-test/extracted/sample_test_data/data"
    calib=sorted(source.glob("*_he.png"))[:5]
    target=np.concatenate([np.asarray(Image.open(p).convert("RGB")) for p in calib],axis=0)
    rgb=np.asarray(Image.open(a.data_root/"results/gigatime-flash-pilot-2026-10-16/image.original.png").convert("RGB"))
    rows=[]
    for method in ["macenko","reinhard_tissue"]:
        normalized,params=normalize(rgb,target,method)
        Image.fromarray(normalized).save(a.output/f"tiger_114S.{method}.input.png")
        maps,plan=full_image(normalized,model,[.5,.5])
        np.save(a.output/f"tiger_114S.{method}.native.npy",maps.astype(np.float16))
        agreement=he_agreement(rgb,maps[0]);print(method,agreement,flush=True)
        row={"method":method,"sampling":plan,"parameters":params,**agreement,"paired_control":[]}
        for path in calib:
            other=np.concatenate([np.asarray(Image.open(q).convert("RGB")) for q in calib if q!=path],axis=0)
            control=np.asarray(Image.open(path).convert("RGB"))
            control,_=normalize(control,other,method)
            pred=predict(control,model)
            _,nuclei=reference(source/path.name.replace("_he.png","_comet_binary_thres_labels.pkl.gz"))
            row["paired_control"].append({"sample":path.name,**metrics(pred[0],nuclei)})
        print(method,"paired AUC",np.mean([r["auc"] for r in row["paired_control"]]),flush=True)
        rows.append(row)
        fig,axs=plt.subplots(1,4,figsize=(17,5))
        for ax,image,title in zip(axs,[rgb,normalized,maps[0],rgb],["Original H&E",method+" input","Flash predicted DAPI","DAPI on original H&E"]):
            ax.imshow(image,cmap="gray",vmin=0,vmax=1) if image.ndim==2 else ax.imshow(image)
            ax.set_title(title);ax.axis("off")
        overlay=np.zeros((*rgb.shape[:2],4),np.float32);overlay[...,:3]=[0,1,1];overlay[...,3]=.65*maps[0]
        axs[-1].imshow(overlay)
        fig.suptitle("Color normalization hypothesis, no geometric edits; all maps at original coordinates")
        fig.tight_layout();fig.savefig(a.output/f"tiger_114S.{method}.comparison.png",dpi=180);plt.close(fig)
    (a.output/"stain_audit.json").write_text(json.dumps({"target_images":[p.name for p in calib],"results":rows},indent=2))


if __name__=="__main__":main()
