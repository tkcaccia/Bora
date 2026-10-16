#!/usr/bin/env python3
"""Display every checked CNN output channel without changing its alignment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.gigatime_flash.run_pilot import CHANNELS,digest,save_json
from experiments.gigatime_flash.plot_overlays import bar


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); source,out=args.source,args.output
    out.mkdir(parents=True,exist_ok=True)
    previous=json.loads((source/"results.json").read_text())
    info=previous["inference"]["0.250000"]
    path=source/"inference_0.250000/virtual_markers.native.npy"
    if previous["model"]["model_family"]!="cnn" or info["channel_order"]!=CHANNELS or digest(path)!=info["native_maps_sha256"]:
        raise ValueError("Atlas requires checksummed CNN maps in canonical channel order")
    maps=np.load(path,mmap_mode="r")
    he=np.asarray(Image.open(source/"image.original.png").convert("RGB"))
    mpp=info["sampling"]["native_mpp_xy"]
    metadata=[]
    for index,name in enumerate(CHANNELS):
        values=maps[index].astype(np.float32)
        low,high=np.percentile(values,[1,99])
        high=max(high,low+.01)
        status=("nuclear context" if name=="DAPI" else "background channel" if name in {"TRITC","Cy5"}
                else "eligible protein feature" if name in previous["eligible_marker_channels"] else "below reference screen")
        filename=f"{index:02d}-{re.sub(r'[^a-zA-Z0-9-]', '-', name)}.png"
        contrast=np.rint(np.clip((values-low)/(high-low),0,1)*255).astype(np.uint8)
        Image.fromarray(contrast).save(out/filename)
        metadata.append({"index":index,"channel":name,"status":status,"file":filename,
                         "display_low":float(low),"display_high":float(high),"display_rule":"linear P1-P99; minimum range 0.01",
                         "activation_mean":float(values.mean()),"activation_min":float(values.min()),"activation_max":float(values.max())})
    for robust in [True,False]:
        fig,axes=plt.subplots(6,4,figsize=(16,24),constrained_layout=True)
        axes.flat[0].imshow(he)
        axes.flat[0].set_title("Original H&E\nTIGER breast tissue ROI",fontsize=12)
        bar(axes.flat[0],he.shape[:2],mpp); axes.flat[0].axis("off")
        for ax,meta in zip(list(axes.flat)[1:],metadata):
            values=maps[meta["index"]]
            low,high=(meta["display_low"],meta["display_high"]) if robust else (0,1)
            ax.imshow(values,cmap="gray",vmin=low,vmax=high,interpolation="nearest")
            color=("#2073a3" if meta["status"]=="nuclear context" else "#777777" if meta["status"]=="background channel"
                   else "#18713a" if meta["status"]=="eligible protein feature" else "#a94c12")
            ax.set_title(f"{meta['index']:02d}  {meta['channel']} — {meta['status']}\nDisplay range {low:.4f}–{high:.4f}",fontsize=10,color=color)
            bar(ax,he.shape[:2],mpp); ax.axis("off")
        mode="Individual P1–P99 contrast scales (brightness cannot be compared across channels)" if robust else "Shared 0–1 activation scale for every channel"
        fig.suptitle("All 23 GigaTIME CNN output channels, aligned to the original image\n"+mode+
                     "\n1224 × 1166 px at ≈0.5 µm/px; inference 0.25 µm/px. Virtual predictions, not measured stains.",fontsize=14)
        fig.savefig(out/("all_markers.contrast.png" if robust else "all_markers.fixed_scale.png"),dpi=160)
        plt.close(fig)
    save_json(out/"manifest.json",{"model":previous["model"],"sample":previous["sample"],
        "maps_sha256":info["native_maps_sha256"],"native_mpp_xy":mpp,"native_shape_yx":list(he.shape[:2]),
        "inference_mpp_xy":info["sampling"]["effective_inference_mpp_xy"],
        "microscope_optical_resolution_um":None,"all_panels_same_original_frame":True,
        "channels":metadata,"source_sha256":digest(__file__),
        "interpretation":"virtual activations; individual contrast scales are for morphology, not quantitative cross-marker brightness comparisons"})
    print(json.dumps({"channels":len(metadata),"output":str(out)},indent=2))


if __name__=="__main__":
    main()
