#!/usr/bin/env python3
"""Fixed first-three BCSS native-resolution transfer check; no case cherry-picking."""
import argparse,json,sys
from pathlib import Path
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
from experiments.gigatime_flash.audit_stains import normalize
from experiments.gigatime_flash.original_cnn import gigatime


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--data-root",type=Path,default=Path("/mnt/sata_ssd/Bora-benchmark"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--flash",type=Path,required=True);p.add_argument("--cnn",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);cv2.setNumThreads(2)
    flash,_=load_model(a.flash);flash.cuda().eval()
    cnn=gigatime(23);cnn.load_state_dict(torch.load(a.cnn,map_location="cpu",weights_only=True),strict=True);cnn.cuda().eval()
    source=a.data_root/"raw/gigatime-official-test/extracted/sample_test_data/data"
    target=np.concatenate([np.asarray(Image.open(p).convert("RGB")) for p in sorted(source.glob("*_he.png"))[:5]],axis=0)
    rows=[json.loads(line) for line in (a.data_root/"prepared/bcss.jsonl").open()][:3]
    out=[]
    for i,row in enumerate(rows):
        rgb=np.asarray(Image.open(a.data_root/row["image"]).convert("RGB"))
        h,w=rgb.shape[:2];y,x=h//2-384,w//2-384
        rgb=rgb[y:y+768,x:x+768]
        expert=np.asarray(Image.open(a.data_root/row["mask"]))[y:y+768,x:x+768]
        Image.fromarray(rgb).save(a.output/f"case_{i}.original.png")
        Image.fromarray(expert).save(a.output/f"case_{i}.expert.png")
        norm,_=normalize(rgb,target,"macenko")
        fig,axs=plt.subplots(2,4,figsize=(16,9))
        for col,(method,model,image) in enumerate([("flash_raw",flash,rgb),("flash_macenko",flash,norm),("cnn_raw",cnn,rgb)],1):
            maps,plan=full_image(image,model,[row["mpp_um"]]*2)
            np.save(a.output/f"case_{i}.{method}.native.npy",maps.astype(np.float16))
            proxy=he_agreement(rgb,maps[0])
            out.append({"case":i,"sample":row,"crop_xywh":[x,y,768,768],"method":method,"sampling":plan,**proxy})
            print(i,method,proxy,flush=True)
            axs[0,col].imshow(maps[0],cmap="gray",vmin=0,vmax=1);axs[0,col].set_title(method+": virtual DAPI")
            axs[1,col].imshow(rgb)
            overlay=np.zeros((*rgb.shape[:2],4),np.float32);overlay[...,:3]=[0,1,1];overlay[...,3]=.55*maps[0]
            axs[1,col].imshow(overlay);axs[1,col].set_title("DAPI on unchanged H&E")
        axs[0,0].imshow(rgb);axs[0,0].set_title("BCSS native H&E; mirror normalized")
        axs[1,0].imshow(expert,cmap="tab20",interpolation="nearest");axs[1,0].set_title("Expert tissue mask (not DAPI truth)")
        for ax in axs.flat:ax.axis("off")
        fig.suptitle(row["id"]+" | Fixed central crop, source and inference 0.25 µm/pixel")
        fig.tight_layout();fig.savefig(a.output/f"case_{i}.comparison.png",dpi=180);plt.close(fig)
    (a.output/"highres.json").write_text(json.dumps(out,indent=2))


if __name__=="__main__":main()
