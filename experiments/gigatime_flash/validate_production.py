#!/usr/bin/env python3
"""Check the actual overlap-add/quantized production pipeline on paired data."""
import argparse,gzip,json,sys,tempfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.gigatime_flash.run_pilot import load_model,predict_maps,CHANNELS
from experiments.gigatime_flash.audit_dapi import reference,NumpyOnlyUnpickler,metrics
from bora.virtual_markers import sampling_plan


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--data",type=Path,required=True);p.add_argument("--weights",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);cv2.setNumThreads(2)
    model,info=load_model(a.weights,"cnn");model.cuda().eval()
    config={"channels":CHANNELS,"normalization":{"mean":[.485,.456,.406],"std":[.229,.224,.225]}}
    paths=sorted(a.data.glob("*_he.png"))[5:]
    dapi,rows=[],[]
    for i,path in enumerate(paths):
        rgb=np.asarray(Image.open(path).convert("RGB"));key=path.name.replace("_he.png","")
        lp=a.data/f"{key}_comet_binary_thres_labels.pkl.gz"
        ref,nuclei=reference(lp)
        with gzip.open(lp,"rb") as f: data=NumpyOnlyUnpickler(f).load()
        ref=ref.copy();ref[data["labels_dapi_expanded"]==0]=0;ref[...,0]*=nuclei
        plan=sampling_plan(rgb.shape[:2],[128/556]*2,.25)
        with tempfile.TemporaryDirectory(dir=a.output) as scratch:
            _,meta=predict_maps(rgb,model,plan,config,Path(scratch),4,"cuda",info)
            maps=np.load(Path(scratch)/"virtual_markers.model_grid.npy").astype(np.float32)
        dapi.append({"sample":key,**metrics(maps[0],nuclei)})
        for c,name in enumerate(CHANNELS):
            truth=cv2.resize(ref[...,c],(512,512),interpolation=cv2.INTER_NEAREST)
            t8=cv2.resize(truth.astype(np.float32),(64,64),interpolation=cv2.INTER_AREA)
            p8=cv2.resize(maps[c],(64,64),interpolation=cv2.INTER_AREA)
            corr=float(np.corrcoef(t8.ravel(),p8.ravel())[0,1]) if t8.std()>1e-8 and p8.std()>1e-8 else None
            rows.append({"sample":key,"channel":name,"pearson_8px":corr})
        if (i+1)%10==0:print("production reference patches",i+1,flush=True)
    summary={k:{"mean":float(np.mean([r[k] for r in dapi])),"median":float(np.median([r[k] for r in dapi]))}
             for k in dapi[0] if k!="sample" and all(r[k] is not None for r in dapi)}
    common={"model":info,"n_confirmation":len(paths),"cnn_sha256":info["checkpoint_sha256"],
            "production_pipeline":"run_pilot.predict_maps: AREA 556->512, 256 windows, stride128, Hann, float16 saved maps",
            "original_reference_mpp":128/556,"model_mpp":.25,"no_stain_normalization":True,
            "biological_independence_of_patches_unknown":True}
    (a.output/"production_dapi_validation.json").write_text(json.dumps({**common,
        "summary":{"original_cnn_raw":summary},"results":dapi,"reference":"released mIF-derived nuclei resized nearest to the inference frame"},indent=2))
    channel_summary=[]
    for name in CHANNELS:
        vals=[r["pearson_8px"] for r in rows if r["channel"]==name and r["pearson_8px"] is not None]
        mean=float(np.mean(vals)) if vals else None
        eligible=name not in ["DAPI","TRITC","Cy5"] and len(vals)>=20 and mean>=.3
        channel_summary.append({"channel":name,"mean_pearson_8px":mean,"n_variable_patches":len(vals),
                                "eligible_for_exploratory_boundary_features":eligible})
    (a.output/"production_channel_validation.json").write_text(json.dumps({**common,"n_reference_patches":len(paths),
        "summary":channel_summary,"rows":rows,
        "eligible_channels":[r["channel"] for r in channel_summary if r["eligible_for_exploratory_boundary_features"]],
        "rule":"non-DAPI/nonbackground; mean 8-pixel-window Pearson >=0.30 and >=20 variable reference patches",
        "threshold_scope":"exploratory feature screen, not a clinical validation threshold",
        "reference_processing":"released 23-channel binary masks restricted to cell support; DAPI restricted to nuclei; nearest resize",
        "publication_comparison":"diagnostic, not exact paper metric reproduction"},indent=2))
    print(json.dumps({"DAPI":summary,"channels":channel_summary},indent=2),flush=True)


if __name__=="__main__":main()
