#!/usr/bin/env python3
"""Show fixed reference examples and the failed/corrected benchmark DAPI crop."""
import argparse,json,sys,tempfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.gigatime_flash.run_pilot import load_model,digest,predict_maps,CHANNELS
from experiments.gigatime_flash.audit_dapi import predict,reference
from bora.virtual_markers import sampling_plan


def overlay(ax,rgb,prob,color=(0,1,1)):
    ax.imshow(rgb)
    rgba=np.zeros((*rgb.shape[:2],4),np.float32);rgba[...,:3]=color;rgba[...,3]=.45*prob
    ax.imshow(rgba)


def scale_bar(ax,shape,mpp,length=20):
    h,w=shape;x,y=w*.05,h*.94
    ax.plot([x,x+length/mpp],[y,y],color="black",lw=3)
    ax.text(x,y-h*.025,f"{length} µm",fontsize=9,bbox={"facecolor":"white","edgecolor":"none","pad":1})


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--data-root",type=Path,default=Path("/mnt/sata_ssd/Bora-benchmark"))
    p.add_argument("--results",type=Path,required=True);p.add_argument("--weights",type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);cv2.setNumThreads(2)
    audit=a.data_root/"results/gigatime-flash-audit-2026-10-16"
    confirmation=json.loads((audit/"confirmation/confirmation.json").read_text())
    model,info=load_model(a.weights,"cnn");model.cuda().eval()
    source=a.data_root/"raw/gigatime-official-test/extracted/sample_test_data/data"
    chosen=confirmation["confirmation_files"][:3]
    fig,axs=plt.subplots(3,4,figsize=(16,12))
    for row,filename in enumerate(chosen):
        rgb=np.asarray(Image.open(source/filename).convert("RGB"))
        rgb512=cv2.resize(rgb,(512,512),interpolation=cv2.INTER_LINEAR)
        key=filename.replace("_he.png","")
        _,nuclei=reference(source/f"{key}_comet_binary_thres_labels.pkl.gz")
        target=cv2.resize(nuclei.astype(np.uint8),(512,512),interpolation=cv2.INTER_NEAREST)
        config={"channels":CHANNELS,"normalization":{"mean":[.485,.456,.406],"std":[.229,.224,.225]}}
        plan=sampling_plan(rgb.shape[:2],[128/556]*2,.25)
        with tempfile.TemporaryDirectory(dir=a.results) as scratch:
            predict_maps(rgb,model,plan,config,Path(scratch),4,"cuda",info)
            pred=np.load(Path(scratch)/"virtual_markers.model_grid.npy")[0].astype(np.float32)
        axs[row,0].imshow(rgb512);axs[row,0].set_title("H&E: "+key,fontsize=10)
        axs[row,1].imshow(target,cmap="gray",vmin=0,vmax=1);axs[row,1].set_title("Released mIF-derived nuclei",fontsize=11)
        axs[row,2].imshow(pred,cmap="gray",vmin=0,vmax=1);axs[row,2].set_title("CNN virtual DAPI (0–1)",fontsize=11)
        overlay(axs[row,3],rgb512,pred);axs[row,3].set_title("DAPI on the same H&E frame",fontsize=11)
        for ax in axs[row]:ax.axis("off");scale_bar(ax,target.shape,.25)
    fig.suptitle("First three unused reference patches, chosen by filename | Original GigaTIME CNN | 0.25 µm/pixel",fontsize=14)
    fig.subplots_adjust(top=.94,bottom=.035,left=.015,right=.985,hspace=.15,wspace=.08)
    fig.savefig(a.results/"dapi_paired_reference_check.png",dpi=180);plt.close(fig)
    r=json.loads((a.results/"results.json").read_text())
    rgb=np.asarray(Image.open(a.data_root/r["sample"]["image"]).convert("RGB"))
    maps=np.load(a.results/"inference_0.250000/virtual_markers.native.npy",mmap_mode="r")
    old=np.load(a.data_root/"results/gigatime-flash-pilot-2026-10-16/inference_0.500000/virtual_markers.native.npy",mmap_mode="r")
    x,y,w,h=280,590,320,320;image=rgb[y:y+h,x:x+w]
    dapi=np.asarray(maps[0,y:y+h,x:x+w],np.float32)
    ck=np.asarray(maps[16,y:y+h,x:x+w],np.float32)
    failed=np.asarray(old[0,y:y+h,x:x+w],np.float32)
    fig,axs=plt.subplots(2,3,figsize=(15,10))
    axs[0,0].imshow(image);axs[0,0].set_title("Unchanged H&E; fixed 160 × 160 µm crop")
    axs[0,1].imshow(failed,cmap="gray",vmin=0,vmax=1);axs[0,1].set_title("Earlier Flash DAPI: rejected")
    axs[0,2].imshow(dapi,cmap="gray",vmin=0,vmax=1);axs[0,2].set_title("CNN DAPI at corrected 0.25 µm/pixel")
    overlay(axs[1,0],image,failed);axs[1,0].set_title("Earlier Flash overlay: rejected")
    overlay(axs[1,1],image,dapi);axs[1,1].set_title("CNN DAPI overlay")
    overlay(axs[1,2],image,ck,color=(1,0,1));axs[1,2].set_title("CNN CK overlay")
    for ax in axs.flat:ax.axis("off");scale_bar(ax,image.shape[:2],.5)
    fig.suptitle("TIGER: CNN improves nuclear localization; remaining errors visible | This ROI has no measured DAPI truth",fontsize=13)
    fig.subplots_adjust(top=.93,bottom=.03,left=.015,right=.985,hspace=.12,wspace=.07)
    fig.savefig(a.results/"dapi_correction_zoom.png",dpi=200);plt.close(fig)
    (a.results/"dapi_qa_figures.json").write_text(json.dumps({"reference_examples":chosen,"selection":"first three unused audit files, no score-based selection",
        "benchmark_crop_xywh":[x,y,w,h],"original_native_mpp":.5,"cnn_inference_mpp":.25,
        "benchmark_measured_DAPI_available":False,"plot_script_sha256":digest(__file__)},indent=2))


if __name__=="__main__":main()
