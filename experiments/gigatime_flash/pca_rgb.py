#!/usr/bin/env python3
"""Original H&E RGB plus class-selected virtual markers -> PCA -> RGB -> Bora."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bora.boundary import BoundaryConfig, smooth_tile_boundaries
from bora.virtual_markers import select_markers
from experiments.gigatime_flash.group_markers import group_selections
from experiments.gigatime_flash.top3_rgb import rank_markers, write_csv


def class_marker_columns(selection):
    assignments = [{"group": int(g), "marker": row["ranking"][0]["channel"],
                    **{k: v for k, v in row["ranking"][0].items() if k != "channel"}}
                   for g, row in sorted(selection["per_group"].items(), key=lambda item: int(item[0]))]
    unique = list(dict.fromkeys(row["marker"] for row in assignments))
    return assignments, unique


def fit_pca_image(features, support, names):
    """Exact three-PC projection, standardized using coarse support only.

    No tissue reference is accepted. Each feature has unit variance before
    PCA; whitening is off. Component signs use the largest absolute loading,
    fixed before any RGB encoding or segmentation score is available.
    """
    features = np.asarray(features, dtype=np.float64)
    support = np.asarray(support, bool)
    if (features.ndim != 3 or features.shape[:2] != support.shape or features.shape[2] != len(names)
            or len(names) < 3 or support.sum() < 4 or not np.isfinite(features).all()):
        raise ValueError("Need finite aligned HWC features, at least three columns and four support pixels")
    sample = features[support]
    mean, std = sample.mean(0), sample.std(0)
    if not np.any(std > 1e-12):
        raise ValueError("PCA support contains no varying features")
    scale = np.where(std > 1e-12, std, 1.)
    standardized = (sample-mean)/scale
    model = PCA(n_components=3, svd_solver="full", whiten=False).fit(standardized)
    for row in model.components_:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    flat = features.reshape(-1, features.shape[2])
    scores = model.transform((flat-mean)/scale).reshape(*features.shape[:2], 3).astype(np.float32)
    metadata = {"feature_names": list(names), "n_features": len(names), "fit_pixels": int(support.sum()),
                "feature_mean": mean.tolist(), "feature_std_ddof0": std.tolist(), "feature_scale": scale.tolist(),
                "constant_feature_indices": np.flatnonzero(std <= 1e-12).tolist(),
                "standardization": "mean 0 and population SD 1 over coarse foreground; constant columns use scale 1",
                "components_pc_by_feature": model.components_.tolist(),
                "pca_center_after_standardization": model.mean_.tolist(),
                "explained_variance": model.explained_variance_.tolist(),
                "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
                "cumulative_explained_variance_ratio": float(model.explained_variance_ratio_.sum()),
                "singular_values": model.singular_values_.tolist(), "solver": "full", "whiten": False,
                "sign_rule": "largest absolute feature loading positive; no accuracy-based sign or order tuning",
                "fit_support": "coarse tile mask > 0 on the same ROI; no per-class weighting",
                "fine_reference_used": False, "native_shape_yx": list(features.shape[:2])}
    return scores, metadata


def pc_scores_to_rgb(scores, support):
    scores = np.asarray(scores, np.float32)
    support = np.asarray(support, bool)
    if scores.ndim != 3 or scores.shape[2] != 3 or scores.shape[:2] != support.shape or not support.any() or not np.isfinite(scores).all():
        raise ValueError("Need three finite PCs and nonempty aligned support")
    low, high = np.percentile(scores[support], [1, 99], axis=0)
    scale = np.maximum(high-low, 1e-6)
    rgb = np.rint(np.clip((scores-low)/scale, 0, 1)*255).astype(np.uint8)
    return rgb, {"mapping": {"R": "PC1", "G": "PC2", "B": "PC3"},
                 "rule": "linear P1-P99 independently per PC over coarse foreground; clip and round uint8",
                 "low": low.tolist(), "high": high.tolist(), "effective_range": scale.tolist(),
                 "gamma": 1., "minimum_range": 1e-6, "fine_reference_used": False,
                 "note": "per-PC display scaling changes relative PC amplitudes; explained variance describes unscaled scores"}


def plot_results(out, maps, results, primary_scores):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch
    from experiments.gigatime_flash.plot_overlays import bar
    he = np.asarray(Image.open(out/"image.original.png").convert("RGB"))
    mpp = results["native_mpp_xy"]
    primary = results["pca"]["pca_markers_unique"]
    ratios = primary["explained_variance_ratio"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.5), constrained_layout=True)
    images = [(he, "Original H&E"), (np.asarray(Image.open(out/"pca_markers_unique.rgb.png")), "RGB + class markers → PCA RGB (primary)"),
              (np.asarray(Image.open(out/"pca_rgb_only.rgb.png")), "RGB-only PCA control")]
    images += [(primary_scores[..., j], f"PC{j+1} → {'RGB'[j]}; variance {ratios[j]*100:.2f}%") for j in range(3)]
    for ax, (data, title) in zip(axes.flat, images):
        if data.ndim == 2:
            low, high = np.percentile(data, [1,99]); ax.imshow(data, cmap="gray", vmin=low, vmax=high)
        else: ax.imshow(data)
        ax.set_title(title, fontsize=10); bar(ax,he.shape[:2],mpp); ax.axis("off")
    fig.suptitle("Original R/G/B plus CD138, CD3 and T-bet: first three principal components\n"
                 f"Unit-variance inputs; three PCs retain {sum(ratios)*100:.2f}% of input variance. Native ≈0.5 µm/px; no image resizing.", fontsize=13)
    fig.savefig(out/"pca_rgb_and_components.png",dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13,5), constrained_layout=True)
    loadings = np.asarray(primary["components_pc_by_feature"])
    image=axes[0].imshow(loadings,cmap="RdBu_r",vmin=-1,vmax=1,aspect="auto")
    axes[0].set_xticks(range(len(primary["feature_names"])),primary["feature_names"],rotation=30,ha="right")
    axes[0].set_yticks(range(3),["PC1 / R","PC2 / G","PC3 / B"])
    for i in range(3):
        for j in range(loadings.shape[1]): axes[0].text(j,i,f"{loadings[i,j]:.3f}",ha="center",va="center",fontsize=10)
    axes[0].set_title("Primary PCA component weights (unit eigenvectors)")
    fig.colorbar(image,ax=axes[0],shrink=.8)
    for offset,(name,metadata) in enumerate(results["pca"].items()):
        axes[1].bar(np.arange(3)+(offset-1)*.24,np.asarray(metadata["explained_variance_ratio"])*100,width=.24,label=name)
    axes[1].set_xticks(range(3),["PC1","PC2","PC3"]); axes[1].set_ylabel("Explained variance (%)")
    axes[1].set_title("Variance before per-PC RGB contrast scaling"); axes[1].legend(fontsize=8)
    fig.savefig(out/"pca_loadings_and_variance.png",dpi=180); plt.close(fig)

    colors=["white","#e24a33","#56b4e9","#009e73","#cc79a7","#f0e442"]
    cmap,norm=ListedColormap(colors),BoundaryNorm(np.arange(-.5,6.5),6)
    fig,axes=plt.subplots(2,3,figsize=(15,11),constrained_layout=True)
    panels=[(None,"Original H&E"),("expert","Expert annotation"),("coarse","Tiling-mask input"),
            ("bora_he_rgb","Bora on original H&E"),("previous_pairwise_markers","Previous interface-specific markers"),
            ("pca_markers_unique","Bora on RGB + marker PCA (primary)")]
    for ax,(method,title) in zip(axes.flat,panels):
        ax.imshow(he)
        if method:
            path=out/"expert_reference.compact.tif" if method=="expert" else out/f"tile_90/{method}.tif"
            ax.imshow(np.ma.masked_equal(tifffile.imread(path),0),cmap=cmap,norm=norm,alpha=.35,interpolation="nearest")
        if method not in (None,"expert"):
            s=next(s for s in results["scores"] if s["tile_size"]==90 and s["method"]==method)
            title+=f"\nIoU {s['macro_iou']:.4f}; boundary F1 {s['internal_boundary']['f1_at_um']['4']:.4f}"
        ax.set_title(title,fontsize=10);bar(ax,he.shape[:2],mpp);ax.axis("off")
    fig.legend(handles=[Patch(color=colors[int(g)],label=n) for g,n in results["compact_class_names"].items()],loc="lower center",ncol=5,fontsize=9,frameon=False)
    fig.suptitle("PCA RGB → ordinary Bora RGB/Lab path; masks at 35% opacity\n"
                 "Same 90-pixel coarse tiles (45 µm), parameters, scoring and native frame",fontsize=13)
    fig.savefig(out/"pca_segmentation_overlays_tile_90.png",dpi=180,bbox_inches="tight");plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(15,10.5),constrained_layout=True)
    channels=results["channel_order"]
    marker_panels=[(he,"Original H&E")]+[(maps[channels.index(c)],c+" — virtual prediction") for c in ["DAPI",*results["unique_markers"]]]
    for ax,(data,title) in zip(axes.flat,marker_panels):
        if data.ndim==2:
            low,high=np.percentile(data,[1,99]);ax.imshow(data,cmap="gray",vmin=low,vmax=max(high,low+.01))
        else:ax.imshow(data)
        ax.set_title(title,fontsize=10);bar(ax,he.shape[:2],mpp);ax.axis("off")
    axes.flat[-1].axis("off")
    axes.flat[-1].text(.02,.95,"Class → selected marker\n\n"+"\n".join(f"{results['compact_class_names'][str(r['group'])]} → {r['marker']}" for r in results["class_marker_assignments"])+
                              "\n\nShared winners enter primary PCA once.\nDAPI is visual context, not a PCA input.",va="top",fontsize=11)
    fig.suptitle("Selected marker inputs and nuclear context\nNative ≈0.5 µm/px; GigaTIME CNN inference 0.25 µm/px; P1–P99 display.",fontsize=13)
    fig.savefig(out/"pca_selected_markers_and_dapi.png",dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--group-results",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); source,groups,out=args.source,args.group_results,args.output
    out.mkdir(parents=True,exist_ok=True)
    from experiments.gigatime_flash.run_pilot import CHANNELS,digest,save_json,score
    cv2.setNumThreads(2)
    previous=json.loads((source/"results.json").read_text())
    group_results=json.loads((groups/"results.json").read_text())
    map_path=source/"inference_0.250000/virtual_markers.native.npy"
    if (previous["model"]["model_family"]!="cnn" or previous["inference"]["0.250000"]["channel_order"]!=CHANNELS
            or digest(map_path)!=previous["inference"]["0.250000"]["native_maps_sha256"]
            or group_results["source_results_sha256"]!=digest(source/"results.json")):
        raise ValueError("Source CNN maps or prior comparison provenance mismatch")
    he=np.asarray(Image.open(source/"image.original.png").convert("RGB"))
    maps=np.load(map_path,mmap_mode="r")
    if tuple(maps.shape)!=(len(CHANNELS),*he.shape[:2]):raise ValueError("Marker maps differ from the native RGB frame")
    coarse90=tifffile.imread(source/"tile_90/coarse.tif")
    tiles=select_markers(np.moveaxis(maps,0,-1),coarse90,90,CHANNELS)["tiles"]
    statistics=rank_markers(tiles,CHANNELS,previous["eligible_marker_channels"],90)
    selection=group_selections(statistics)
    assignments,unique=class_marker_columns(selection)
    save_json(out/"marker_selection.json",{"assignments":assignments,"unique_markers":unique,"selection":selection,
               "rule":"existing per-group maximin pairwise AUC ranking; strongest marker for each class; no fine reference"})
    rgb_base=he.astype(np.float32)/255
    variants={"pca_markers_unique":unique,"pca_markers_per_class":[r["marker"] for r in assignments],"pca_rgb_only":[]}
    composites={};models={};primary_scores=None
    for name,markers in variants.items():
        feature_names=["H&E R","H&E G","H&E B"]+([f"{r['marker']}@class{r['group']}" for r in assignments] if name=="pca_markers_per_class" else markers)
        features=np.concatenate([rgb_base,*[np.asarray(maps[CHANNELS.index(c)],np.float32)[...,None] for c in markers]],axis=2)
        began=time.perf_counter();pcs,metadata=fit_pca_image(features,coarse90>0,feature_names)
        rgb,encoding=pc_scores_to_rgb(pcs,coarse90>0)
        np.save(out/f"{name}.scores.native.npy",pcs)
        Image.fromarray(rgb).save(out/f"{name}.rgb.png")
        tifffile.imwrite(out/f"{name}.ome.tif",rgb,photometric="rgb",metadata={"axes":"YXS","PhysicalSizeX":.5,"PhysicalSizeXUnit":"µm","PhysicalSizeY":.5,"PhysicalSizeYUnit":"µm"})
        models[name]={**metadata,"marker_columns":markers,"encoding":encoding,"seconds":time.perf_counter()-began,
                      "rgb_sha256":digest(out/f"{name}.rgb.png")}
        save_json(out/f"{name}.json",models[name]);composites[name]=rgb
        if name=="pca_markers_unique":primary_scores=pcs
        print(json.dumps({"pca":name,"features":len(feature_names),"explained_variance_ratio":metadata["explained_variance_ratio"]}),flush=True)
    config=BoundaryConfig(**previous["boundary_config"]);runs=[]
    for tile in [56,90,216]:
        folder=out/f"tile_{tile}";folder.mkdir(exist_ok=True)
        coarse=tifffile.imread(source/f"tile_{tile}/coarse.tif")
        if digest(source/f"tile_{tile}/coarse.tif")!=digest(groups/f"tile_{tile}/coarse.tif"):
            raise ValueError("Comparison coarse inputs differ")
        for old,new in [(source/f"tile_{tile}/coarse.tif","coarse"),(source/f"tile_{tile}/bora_rgb.tif","bora_he_rgb"),
                        (groups/f"tile_{tile}/pairwise_marker_rgb.tif","previous_pairwise_markers")]:shutil.copyfile(old,folder/f"{new}.tif")
        for method,rgb in composites.items():
            began=time.perf_counter();pred=smooth_tile_boundaries(rgb,coarse,tile,config=config)
            tifffile.imwrite(folder/f"{method}.tif",pred)
            runs.append({"tile_size":tile,"method":method,"seconds":time.perf_counter()-began,"output_sha256":digest(folder/f"{method}.tif")})
    # All marker selection, PCA fits, RGB encodings and predictions are fixed.
    truth=tifffile.imread(source/"expert_reference.compact.tif");scores=[]
    for tile in [56,90,216]:
        for method in ["coarse","bora_he_rgb","previous_pairwise_markers",*variants]:
            pred=tifffile.imread(out/f"tile_{tile}/{method}.tif")
            scores.append({"tile_size":tile,"method":method,**score(pred,truth,[.5,.5],previous["compact_class_names"])})
    results={"sample":previous["sample"],"image_type":previous["image_type"],"model":previous["model"],
        "native_mpp_xy":[.5,.5],"native_mpp_status":previous["native_mpp_status"],"microscope_objective_magnification":None,
        "microscope_optical_resolution_um":None,"inference":previous["inference"]["0.250000"],"channel_order":CHANNELS,
        "class_marker_assignments":assignments,"unique_markers":unique,"pca":models,"compact_class_names":previous["compact_class_names"],
        "boundary_config":asdict(config),"coarse_protocol":previous["coarse_protocol"],"runs":runs,"scores":scores,
        "primary":"pca_markers_unique","sensitivity":"pca_markers_per_class includes duplicated class winners",
        "control":"pca_rgb_only isolates changes due to the PCA/RGB encoding itself",
        "same_pca_rgb_across_all_tile_sizes":True,"fine_reference_opened_after_predictions_fixed":True,
        "fit_is_transductive_within_roi":True,"source_results_sha256":digest(source/"results.json"),
        "group_results_sha256":digest(groups/"results.json"),"source_marker_maps_sha256":digest(map_path),
        "runtime":{name:importlib.metadata.version(name) for name in ["numpy","scipy","scikit-learn","opencv-python","tifffile"]},
        "source_sha256":{name:digest(Path(__file__).resolve().parents[2]/name) for name in ["experiments/gigatime_flash/pca_rgb.py",
            "experiments/gigatime_flash/group_markers.py","experiments/gigatime_flash/top3_rgb.py","src/bora/boundary.py"]}}
    save_json(out/"results.json",results)
    flat=[{"tile":s["tile_size"],"method":s["method"],"macro_iou":s["macro_iou"],"macro_dice":s["macro_dice"],
           "boundary_f1_4um":s["internal_boundary"]["f1_at_um"]["4"],"assd_um":s["internal_boundary"]["assd_um"],
           "hd95_um":s["internal_boundary"]["hd95_um"]} for s in scores]
    write_csv(out/"scores.csv",flat)
    write_csv(out/"pca_loadings.csv",[{"variant":name,"component":j+1,"feature":feature,"weight":weight}
              for name,m in models.items() for j,row in enumerate(m["components_pc_by_feature"]) for feature,weight in zip(m["feature_names"],row)])
    for name in ["image.original.png","expert_reference.compact.tif","production_dapi_validation.json","production_channel_validation.json"]:shutil.copyfile(source/name,out/name)
    plot_results(out,maps,results,primary_scores)
    report=["# Original RGB + class markers → PCA RGB → Bora","","## Inputs and marker selection","",
        "One TIGER breast tissue ROI. Reuse the checked original GigaTIME CNN maps; no new inference or training. One marker per class is selected from central-half means of the same 169 coarse 90-pixel tiles, ranking each marker by its weakest pairwise separation from the four other classes. No fine annotation enters selection or PCA.","",
        "| Class | Selected marker | Worst-pair direction-free AUC | Worst adjusted nominal pairwise p |","|---|---|---:|---:|"]
    for row in assignments:report.append(f"| {previous['compact_class_names'][str(row['group'])]} | {row['marker']} | {row['minimum_pairwise_direction_free_auc']:.4f} | {row['worst_pair_bh_adjusted_nominal_p']:.4g} |")
    report += ["","Shared winners are included once in the **primary** PCA: original R/G/B plus "+", ".join(unique)+f" ({3+len(unique)} variables). The literal one-column-per-class sensitivity uses eight variables and repeats CD138 and CD3. RGB-only PCA is the control. These variants, component order, signs and scaling are fixed before scoring; they are not selected by accuracy.","",
        "## PCA and RGB encoding","","RGB starts as original uint8/255; marker inputs are native-aligned sigmoid activations. Each variable is centered and scaled to population SD 1 over coarse foreground pixels on this ROI. Exact full SVD keeps three PCs, without whitening or class weighting. Sign convention: largest absolute component loading is positive. PC1/PC2/PC3 map to R/G/B after independent P1–P99 linear scaling over the same coarse support, clipping and rounding to uint8. Per-PC encoding rescales relative amplitudes; variance fractions below refer to the unscaled PC scores. The same native RGB composite is reused for all three tile widths.","",
        "| Variant | Variables | PC1 variance | PC2 variance | PC3 variance | Total retained |","|---|---:|---:|---:|---:|---:|"]
    for name,m in models.items():report.append(f"| {name} | {m['n_features']} | "+" | ".join(f"{v*100:.3f}%" for v in m['explained_variance_ratio'])+f" | {m['cumulative_explained_variance_ratio']*100:.3f}% |")
    report += ["","Full input means/SDs, component weights, centering, sign rule, explained variance and RGB transforms are saved in the variant JSON files and `pca_loadings.csv`; raw native PC score arrays are saved on Chiamaka. PCA is fitted to this same ROI (transductive), not an independently trained feature extractor.","",
        "## Segmentation","","Each PCA RGB is passed as `image` to `smooth_tile_boundaries`, with no `feature_image`, using unchanged normal RGB/Lab Bora and the previous fixed configuration (2× working downsample). Fine expert labels are loaded only after all predictions have been written. Boundary F1 tolerance is 4 µm. The previous pairwise method is context and uses a different experimental wrapper; original H&E and all PCA variants share the same multiclass path.","",
        "| Tile px | Appearance / method | Macro IoU | Macro Dice | Boundary F1 | ASSD µm |","|---:|---|---:|---:|---:|---:|"]
    for s in flat:report.append(f"| {s['tile']} | {s['method']} | {s['macro_iou']:.4f} | {s['macro_dice']:.4f} | {s['boundary_f1_4um']:.4f} | {s['assd_um']:.3f} |")
    report += ["","## Resolution and scope","","Native image is a tissue ROI, not a WSI: 1224 × 1166 px at provider-reported approximately 0.5 µm/px, field 612 × 583 µm. CNN inference was at 0.25 µm/px and mapped back to the native frame. PCA changes features only; it does not resize the image. Bora's separate 2× working grid is approximately 1 µm/px. Optical resolving power, objective and numerical aperture are unknown. Every RGB OME-TIFF carries native physical pixel size. Expert masks remain unchanged; code 0 is ignored, rest is scored.","",
        "This is a controlled boundary reconstruction from expert-derived coarse labels on one ROI. Tile-level marker statistics are exploratory because neighboring tiles are correlated. Marker activations are virtual, with no measured protein truth on this TIGER image. Shared markers can distinguish two classes by different levels, and low activation can identify rest. PCA maximizes variance, not tissue separation; a useful marker signal can be lost when retaining only three PCs. No clinical, patient-level or general superiority claim follows from this pilot.","",
        "## Reproduce","","`OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:. python experiments/gigatime_flash/pca_rgb.py --source <checked-cnn-results> --group-results <group-specific-results> --output <new-output>`", "",
        "Sources: [PCA implementation and variance definition](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html); [TIGER](https://tiger.grand-challenge.org/Data/); [GigaTIME input sampling](https://github.com/prov-gigatime/GigaTIME/issues/4).",""]
    (out/"REPORT.md").write_text("\n".join(report))
    print(json.dumps({"assignments":assignments,"pca_variance":{k:v['explained_variance_ratio'] for k,v in models.items()},"scores":flat},indent=2))


if __name__=="__main__":main()
