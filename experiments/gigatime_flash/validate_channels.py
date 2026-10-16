#!/usr/bin/env python3
"""Reference screen for eligible CNN channels; diagnostic, not paper reproduction."""
import argparse,gzip,json,sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.gigatime_flash.audit_dapi import predict,reference,NumpyOnlyUnpickler
from experiments.gigatime_flash.run_pilot import load_model,CHANNELS


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--data",type=Path,required=True);p.add_argument("--weights",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);cv2.setNumThreads(2)
    model,info=load_model(a.weights,"cnn");model.cuda().eval()
    paths=sorted(a.data.glob("*_he.png"))[5:]
    rows=[]
    for p in paths:
        key=p.name.replace("_he.png","")
        labelpath=a.data/f"{key}_comet_binary_thres_labels.pkl.gz"
        ref,nuclei=reference(labelpath)
        with gzip.open(labelpath,"rb") as f: data=NumpyOnlyUnpickler(f).load()
        support=data["labels_dapi_expanded"]>0
        ref=ref.copy();ref[~support]=0
        ref[...,0]*=nuclei
        maps=predict(np.asarray(Image.open(p).convert("RGB")),model)
        for c,name in enumerate(CHANNELS):
            truth=cv2.resize(ref[...,c],(512,512),interpolation=cv2.INTER_NEAREST)
            t8=cv2.resize(truth.astype(np.float32),(64,64),interpolation=cv2.INTER_AREA)
            p8=cv2.resize(maps[c],(64,64),interpolation=cv2.INTER_AREA)
            corr=float(np.corrcoef(t8.ravel(),p8.ravel())[0,1]) if t8.std()>1e-8 and p8.std()>1e-8 else None
            rows.append({"sample":key,"channel":name,"pearson_8px":corr,"positive_reference_pixels":int(truth.sum())})
    summary=[]
    for name in CHANNELS:
        correlations=[r["pearson_8px"] for r in rows if r["channel"]==name and r["pearson_8px"] is not None]
        mean=float(np.mean(correlations)) if correlations else None
        eligible=(name not in ["DAPI","TRITC","Cy5"] and len(correlations)>=20 and mean>=.3)
        summary.append({"channel":name,"n_variable_patches":len(correlations),"mean_pearson_8px":mean,
                        "median_pearson_8px":float(np.median(correlations)) if correlations else None,
                        "eligible_for_exploratory_boundary_features":eligible})
    report={"model":info,"n_reference_patches":len(paths),"summary":summary,"rows":rows,
        "eligible_channels":[r["channel"] for r in summary if r["eligible_for_exploratory_boundary_features"]],
        "rule":"non-DAPI/nonbackground; mean 8-pixel-window Pearson >= 0.30 and >=20 variable reference patches",
        "threshold_scope":"pragmatic exploratory feature filter, not a clinical validation threshold",
        "reference_processing":"released 23-channel binary reference, masked by released cell support; DAPI restricted to nuclear support; nearest resize 556 to 512",
        "comparison_to_publication":"diagnostic only; does not reproduce full cell-convex-hull relabeling, target smoothing or paper metrics",
        "biological_independence_of_patches_unknown":True}
    (a.output/"channel_validation.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":main()
