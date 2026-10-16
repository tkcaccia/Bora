#!/usr/bin/env python3
"""Confirm the fixed inference hypotheses on the 45 unused official patches."""
from pathlib import Path
import argparse,json,sys
import cv2
import numpy as np
from PIL import Image
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.gigatime_flash.run_pilot import load_model,digest
from experiments.gigatime_flash.audit_dapi import predict,reference,metrics
from experiments.gigatime_flash.audit_stains import normalize
from experiments.gigatime_flash.original_cnn import gigatime


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--data",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--flash",type=Path,required=True);p.add_argument("--cnn",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);cv2.setNumThreads(2)
    model,info=load_model(a.flash);model.cuda().eval()
    cnn=gigatime(23);cnn.load_state_dict(torch.load(a.cnn,map_location="cpu",weights_only=True),strict=True);cnn.cuda().eval()
    paths=sorted(a.data.glob("*_he.png"));calib,confirmation=paths[:5],paths[5:]
    target=np.concatenate([np.asarray(Image.open(p).convert("RGB")) for p in calib],axis=0)
    results=[]
    for i,p in enumerate(confirmation):
        rgb=np.asarray(Image.open(p).convert("RGB"));key=p.name.replace("_he.png","")
        ref,nuclei=reference(a.data/f"{key}_comet_binary_thres_labels.pkl.gz")
        norm,_=normalize(rgb,target,"macenko")
        for method,net,image in [("flash_raw",model,rgb),("flash_macenko",model,norm),("original_cnn_raw",cnn,rgb)]:
            maps=predict(image,net)
            results.append({"sample":key,"method":method,"reference_nuclei":metrics(maps[0],nuclei),
                "released_DAPI_binary":metrics(maps[0],ref[...,0])})
        if (i+1)%10==0:print(json.dumps({"completed_patches":i+1,"total":len(confirmation)}),flush=True)
    summary={}
    for method in ["flash_raw","flash_macenko","original_cnn_raw"]:
        rows=[r["reference_nuclei"] for r in results if r["method"]==method]
        summary[method]={k:{"mean":float(np.mean([r[k] for r in rows])),"median":float(np.median([r[k] for r in rows]))}
                         for k in rows[0] if all(r[k] is not None for r in rows)}
    report={"calibration_files":[p.name for p in calib],"confirmation_files":[p.name for p in confirmation],
        "n_confirmation":len(confirmation),"no_reference_labels_used_for_stain_fit":True,
        "patient_independence_of_patches_unknown":True,"input_resize":"556 to 512, bilinear",
        "physical_sampling_um_per_pixel":.25,"inference_windows":"256x256, stride 256, exact official disjoint-window control",
        "reference":"released nuclei and unmodified released DAPI binary; not full downstream cell-label reconstruction",
        "model":info,"cnn_sha256":digest(a.cnn),"results":results,"summary":summary}
    (a.output/"confirmation.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":main()
