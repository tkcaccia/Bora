#!/usr/bin/env python3
"""Select markers for each tissue group and each competing boundary pair.

Selection uses coarse tile summaries only. Pair-specific RGB images drive
the existing Bora RGB/Lab local wand; nearest-interface ownership merges
predictions without comparing costs from different synthetic color spaces.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from itertools import combinations
import json
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bora.boundary import BoundaryConfig, _interior
from bora.virtual_markers import select_markers
from experiments.gigatime_flash.top3_rgb import rank_markers, markers_to_rgb, write_csv


def group_selections(statistics):
    """Rank by the weakest pairwise separation, so one easy group cannot dominate.

    A direction-free AUC is max(AUC, 1-AUC): either low or high expression
    can distinguish a pair. No shared direction across all opponents is
    required; a middle-valued group can need two different thresholds.
    """
    groups = sorted(map(int, statistics["group_tile_counts"]))
    per_group, per_pair = {}, {}
    for group in groups:
        rows = []
        for channel in statistics["eligible_channels"]:
            pairs = [r for r in statistics["pairwise"] if r["channel"] == channel and group in (r["group_a"], r["group_b"])]
            aucs = [.5+.5*abs(r["rank_biserial_a_minus_b"]) for r in pairs]
            weakest = int(np.argmin(aucs))
            row = {"channel": channel, "minimum_pairwise_direction_free_auc": min(aucs),
                   "mean_pairwise_direction_free_auc": float(np.mean(aucs)),
                   "worst_pair_bh_adjusted_nominal_p": max(r["bh_adjusted_nominal_p_all_pairs_all_markers"] for r in pairs),
                   "hardest_opponent": next(g for g in (pairs[weakest]["group_a"], pairs[weakest]["group_b"]) if g != group)}
            rows.append(row)
        rows.sort(key=lambda r: (-r["minimum_pairwise_direction_free_auc"],
                                 -r["mean_pairwise_direction_free_auc"], r["channel"]))
        per_group[str(group)] = {"selected_channels_rgb": [r["channel"] for r in rows[:3]], "ranking": rows}
    for a, b in combinations(groups, 2):
        rows = [dict(r, direction_free_auc=.5+.5*abs(r["rank_biserial_a_minus_b"]))
                for r in statistics["pairwise"] if (r["group_a"], r["group_b"]) == (a, b)]
        rows.sort(key=lambda r: (-r["direction_free_auc"], r["nominal_p"], r["channel"]))
        per_pair[f"{a}-{b}"] = {"groups": [a, b], "selected_channels_rgb": [r["channel"] for r in rows[:3]], "ranking": rows}
    return {"per_group": per_group, "per_pair": per_pair,
            "group_selection_rule": "top three by minimum direction-free pairwise AUC across all other groups; mean AUC breaks ties",
            "pair_selection_rule": "top three by direction-free AUC within that tissue pair; no global omnibus ranking",
            "direction_free_auc_definition": "0.5 + 0.5*abs(rank-biserial correlation); derived from Mann-Whitney U",
            "scope": "descriptive selection within one ROI, not held-out predictive performance; low expression is informative",
            "fine_reference_mask_used": False}


def interface_band(labels, a, b):
    edge = np.zeros(labels.shape, bool)
    for left, right, axis in [(labels[:, :-1], labels[:, 1:], 1), (labels[:-1], labels[1:], 0)]:
        different = ((left == a) & (right == b)) | ((left == b) & (right == a))
        if axis == 1:
            edge[:, :-1] |= different
            edge[:, 1:] |= different
        else:
            edge[:-1] |= different
            edge[1:] |= different
    return edge


def refine_interfaces(labels, pair_images, tile_size, config):
    """Run the unchanged RGB wand separately at each original tissue interface.

    Original nearest-interface ownership is fixed before any refinement.
    Equal-distance junctions are protected; background stays fixed. This is
    an experimental pairwise wrapper, not the original multiclass API.
    """
    height, width = labels.shape
    size = (int(np.ceil(width/config.downsample)), int(np.ceil(height/config.downsample)))
    coarse = cv2.resize(labels, size, interpolation=cv2.INTER_NEAREST)
    tile = tile_size/config.downsample
    radius = max(2, int(round(tile*config.internal_search_fraction)))
    best = np.full(coarse.shape, np.inf, np.float32)
    owner = np.full(coarse.shape, -1, np.int16)
    tied = np.zeros(coarse.shape, bool)
    pairs = sorted(pair_images)
    active = []
    for j, (a, b) in enumerate(pairs):
        edge = interface_band(coarse, a, b)
        if not edge.any():
            continue
        distance = ndi.distance_transform_edt(~edge).astype(np.float32)
        eligible = ((coarse == a) | (coarse == b)) & (distance <= radius)
        closer = eligible & (distance < best - 1e-6)
        equal = eligible & np.isclose(distance, best, atol=1e-6, rtol=0)
        owner[closer], best[closer], tied[closer] = j, distance[closer], False
        tied[equal] = True
        active.append((j, a, b))
    answer = coarse.copy()
    for j, a, b in active:
        image = pair_images[(a, b)]
        if image.dtype != np.uint8 or image.shape != (height, width, 3):
            raise ValueError("Pair RGB images must match the native label frame")
        rgb = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        restricted = np.where((coarse == a) | (coarse == b), coarse, 0).astype(coarse.dtype)
        proposal = _interior(rgb, restricted, tile, config)
        use = (owner == j) & ~tied & ((proposal == a) | (proposal == b))
        answer[use] = proposal[use]
    answer[coarse == 0] = 0
    return cv2.resize(answer, (width, height), interpolation=cv2.INTER_NEAREST), {
        "active_pairs": [[a, b] for _, a, b in active], "protected_junction_work_pixels": int(tied.sum()),
        "working_mpp_factor": config.downsample, "nearest_interface_ownership_fixed_from_coarse": True,
        "background_fixed": True, "rgb_lab_wand_unchanged": True}


def plot(out, source, maps, selection, names, scores):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch
    from experiments.gigatime_flash.plot_overlays import bar
    channels = list(selection["channel_order"])
    groups = list(selection["per_group"])
    marker_names = selection["eligible_channels"]
    matrix = np.array([[next(r["minimum_pairwise_direction_free_auc"] for r in selection["per_group"][g]["ranking"] if r["channel"] == c)
                       for c in marker_names] for g in groups])
    fig, ax = plt.subplots(figsize=(15, 5.4), constrained_layout=True)
    shown = ax.imshow(matrix, cmap="viridis", vmin=.5, vmax=1., aspect="auto")
    ax.set_xticks(range(len(marker_names)), marker_names, rotation=45, ha="right")
    ax.set_yticks(range(len(groups)), [names[g] for g in groups])
    for i, group in enumerate(groups):
        for j, marker in enumerate(marker_names):
            star = "*" if marker in selection["per_group"][group]["selected_channels_rgb"] else ""
            ax.text(j, i, f"{matrix[i,j]:.2f}{star}", ha="center", va="center", fontsize=8,
                    color="black" if matrix[i,j] > .8 else "white")
    fig.colorbar(shown, ax=ax, label="Worst pairwise direction-free AUC (0.5 = no separation)")
    ax.set_title("Best markers for EACH tissue group: require separation from every other group\n"
                 "* top three per row; coarse 90-pixel tiles only; exploratory within one TIGER ROI", fontsize=13)
    fig.savefig(out/"markers_by_group.png", dpi=180)
    plt.close(fig)

    pairs = selection["per_pair"]
    fig, axes = plt.subplots(5, 2, figsize=(14, 15), constrained_layout=True)
    for ax, (key, row) in zip(axes.flat, pairs.items()):
        top = row["ranking"][:3]
        a, b = row["groups"]
        bars = ax.barh(range(3), [r["direction_free_auc"] for r in top], color=["#ce3434", "#258b38", "#306ac4"])
        ax.set_xlim(.5, 1.03)
        ax.set_yticks(range(3), [r["channel"] for r in top])
        ax.invert_yaxis()
        for rectangle, r in zip(bars, top):
            ax.text(.51, rectangle.get_y()+rectangle.get_height()/2,
                    f"AUC {r['direction_free_auc']:.3f}; adjusted nominal p {r['bh_adjusted_nominal_p_all_pairs_all_markers']:.2g}",
                    va="center", fontsize=8)
        ax.set_title(f"{names[str(a)]} vs {names[str(b)]}", fontsize=10)
        ax.set_xlabel("Direction-free AUC from tile means")
    fig.suptitle("Three marker channels chosen separately for EACH competing tissue pair\n"
                 "BH adjusts all 160 marker/pair tests; adjacent tiles are correlated.", fontsize=13)
    fig.savefig(out/"markers_by_interface.png", dpi=160)
    plt.close(fig)

    he = np.asarray(Image.open(source/"image.original.png").convert("RGB"))
    pair = selection["per_pair"]["1-2"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.5), constrained_layout=True)
    rgb = np.asarray(Image.open(out/"pair_rgb/1-2.png"))
    panels = [(he, "Original H&E"), (rgb, "Tumor–stroma RGB: " + ", ".join(f"{k}={v}" for k,v in zip("RGB",pair["selected_channels_rgb"]))),
              (maps[0], "DAPI (context only)")]
    panels += [(maps[channels.index(c)], c + " (tumor–stroma selected channel)") for c in pair["selected_channels_rgb"]]
    for ax, (data, title) in zip(axes.flat, panels):
        if data.ndim == 2:
            low, high = np.percentile(data, [1,99])
            ax.imshow(data, cmap="gray", vmin=low, vmax=max(high, low+.01))
        else: ax.imshow(data)
        ax.set_title(title, fontsize=10)
        bar(ax, he.shape[:2], [.5,.5])
        ax.axis("off")
    fig.suptitle("Markers relevant to the tumor–stroma interface, with the original nuclear context\n"
                 "Virtual predictions; P1–P99 display. Native ≈0.5 µm/px; CNN input 0.25 µm/px.", fontsize=13)
    fig.savefig(out/"tumor_stroma_rgb_and_markers.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10.5), constrained_layout=True)
    for ax, group in zip(axes.flat, [None, *groups]):
        if group is None:
            ax.imshow(he); title = "Original H&E"
        else:
            ax.imshow(np.asarray(Image.open(out/f"group_rgb/{group}.png")))
            row = selection["per_group"][group]
            title = names[group]+"\n"+", ".join(f"{c}={m}" for c,m in zip("RGB",row["selected_channels_rgb"]))
        ax.set_title(title, fontsize=10); bar(ax,he.shape[:2],[.5,.5]); ax.axis("off")
    fig.suptitle("Three candidate markers selected separately for each tissue group\n"
                 "Ranking requires separation from every other group; high OR low activation can distinguish a group.",fontsize=13)
    fig.savefig(out/"group_rgb_composites.png",dpi=170)
    plt.close(fig)

    colors = ["white", "#e24a33", "#56b4e9", "#009e73", "#cc79a7", "#f0e442"]
    cmap, norm = ListedColormap(colors), BoundaryNorm(np.arange(-.5,6.5),6)
    panels = [(None, "Original H&E"), ("expert", "Expert annotation"), ("coarse", "Tiling-mask input"),
              ("bora_he_rgb", "Original multiclass H&E Bora"), ("pairwise_he_rgb", "Matched pairwise H&E control"),
              ("pairwise_marker_rgb", "Bora with interface-specific marker RGB")]
    fig, axes = plt.subplots(2,3,figsize=(15,11),constrained_layout=True)
    for ax, (method,title) in zip(axes.flat,panels):
        ax.imshow(he)
        if method:
            path = out/"expert_reference.compact.tif" if method=="expert" else out/f"tile_90/{method}.tif"
            labels=tifffile.imread(path)
            ax.imshow(np.ma.masked_equal(labels,0),cmap=cmap,norm=norm,alpha=.35,interpolation="nearest")
        if method not in (None,"expert"):
            s=next(r for r in scores if r["tile_size"]==90 and r["method"]==method)
            title+=f"\nIoU {s['macro_iou']:.4f}; boundary F1 {s['internal_boundary']['f1_at_um']['4']:.4f}"
        ax.set_title(title,fontsize=10)
        bar(ax,he.shape[:2],[.5,.5]); ax.axis("off")
    fig.legend(handles=[Patch(color=colors[int(g)],label=n) for g,n in names.items()],loc="lower center",ncol=5,fontsize=9,frameon=False)
    fig.suptitle("Group-specific refinement: expert/coarse/Bora masks at 35% opacity\n"
                 "90-pixel tiles (45 µm), same native output frame; background and tied junctions protected",fontsize=13)
    fig.savefig(out/"group_specific_segmentation.png",dpi=180,bbox_inches="tight")
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args(); source,out=args.source,args.output
    out.mkdir(parents=True,exist_ok=True)
    from experiments.gigatime_flash.run_pilot import CHANNELS,digest,save_json,score
    cv2.setNumThreads(2)
    previous=json.loads((source/"results.json").read_text())
    map_path=source/"inference_0.250000/virtual_markers.native.npy"
    if digest(map_path)!=previous["inference"]["0.250000"]["native_maps_sha256"]:
        raise ValueError("Checked CNN map checksum mismatch")
    maps=np.load(map_path,mmap_mode="r")
    hwc=np.moveaxis(maps,0,-1)
    coarse90=tifffile.imread(source/"tile_90/coarse.tif")
    tiles=select_markers(hwc,coarse90,90,CHANNELS)["tiles"]
    stats=rank_markers(tiles,CHANNELS,previous["eligible_marker_channels"],90)
    selection=group_selections(stats)
    selection.update({"channel_order":CHANNELS,"eligible_channels":previous["eligible_marker_channels"],
                      "class_names":previous["compact_class_names"],"group_tile_counts":stats["group_tile_counts"],
                      "selection_tile_size_px":90,"same_selection_for_all_tile_sizes":True})
    save_json(out/"selection.json",selection)
    save_json(out/"all_marker_statistics.json",stats)
    write_csv(out/"pairwise_statistics.csv",stats["pairwise"])
    write_csv(out/"group_marker_ranking.csv",[{"group":g,**r} for g,v in selection["per_group"].items() for r in v["ranking"]])
    write_csv(out/"tile_observations.csv",[{"x":t["x"],"y":t["y"],"group":t["label"],**dict(zip(CHANNELS,t["central_mean"]))} for t in tiles])
    pair_images={}; encodings={}; group_encodings={}
    (out/"pair_rgb").mkdir(exist_ok=True); (out/"group_rgb").mkdir(exist_ok=True)
    for key,row in selection["per_pair"].items():
        indices=[CHANNELS.index(c) for c in row["selected_channels_rgb"]]
        rgb,encoding=markers_to_rgb(hwc,indices,coarse90>0)
        Image.fromarray(rgb).save(out/f"pair_rgb/{key}.png")
        pair_images[tuple(row["groups"])]=rgb; encodings[key]={"channels_rgb":row["selected_channels_rgb"],**encoding}
    for group,row in selection["per_group"].items():
        indices=[CHANNELS.index(c) for c in row["selected_channels_rgb"]]
        rgb,encoding=markers_to_rgb(hwc,indices,coarse90>0)
        Image.fromarray(rgb).save(out/f"group_rgb/{group}.png")
        group_encodings[group]={"channels_rgb":row["selected_channels_rgb"],**encoding}
    save_json(out/"pair_rgb_encoding.json",encodings)
    save_json(out/"group_rgb_encoding.json",group_encodings)
    config=BoundaryConfig(**previous["boundary_config"])
    he=np.asarray(Image.open(source/"image.original.png").convert("RGB"))
    runs=[]
    for tile in [56,90,216]:
        folder=out/f"tile_{tile}"; folder.mkdir(exist_ok=True)
        coarse=tifffile.imread(source/f"tile_{tile}/coarse.tif")
        for old,new in [("coarse","coarse"),("bora_rgb","bora_he_rgb"),("bora_markers_0.250000","previous_six")]:
            shutil.copyfile(source/f"tile_{tile}/{old}.tif",folder/f"{new}.tif")
        for images,method in [(pair_images,"pairwise_marker_rgb"),({pair:he for pair in pair_images},"pairwise_he_rgb")]:
            began=time.perf_counter()
            refined,meta=refine_interfaces(coarse,images,tile,config)
            tifffile.imwrite(folder/f"{method}.tif",refined)
            runs.append({"tile_size":tile,"method":method,"seconds":time.perf_counter()-began,**meta})
            print(json.dumps(runs[-1]),flush=True)
    # Fine annotations enter only once every prediction has been written.
    truth=tifffile.imread(source/"expert_reference.compact.tif")
    scores=[]
    for tile in [56,90,216]:
        for method in ["coarse","bora_he_rgb","previous_six","pairwise_he_rgb","pairwise_marker_rgb"]:
            pred=tifffile.imread(out/f"tile_{tile}/{method}.tif")
            scores.append({"tile_size":tile,"method":method,**score(pred,truth,[.5,.5],previous["compact_class_names"])})
    results={"sample":previous["sample"],"source":str(source),"model":previous["model"],
             "source_results_sha256":digest(source/"results.json"),"native_mpp_xy":[.5,.5],
             "image_type":previous["image_type"],"microscope_optical_resolution_um":None,
             "microscope_objective_magnification":None,"native_mpp_status":previous["native_mpp_status"],
             "sampling":previous["inference"]["0.250000"]["sampling"],"boundary_config":asdict(config),
             "runs":runs,"scores":scores,"selection":selection,"reference_used_after_predictions_fixed":True,
             "coarse_protocol":previous["coarse_protocol"],
             "method":"pair-specific top-three RGB; unchanged Bora RGB/Lab local wand; nearest original interface owns each editable pixel; tied junctions/background fixed",
             "selection_scope":"coarse tile summaries only; fine mask never used in channel selection",
             "control":"same pairwise wrapper, same ownership and parameters, original H&E RGB for every pair",
             "source_sha256":{name:digest(Path(__file__).resolve().parents[2]/name) for name in [
                "experiments/gigatime_flash/group_markers.py","experiments/gigatime_flash/top3_rgb.py","src/bora/boundary.py"]}}
    save_json(out/"results.json",results)
    write_csv(out/"scores.csv",[{"tile":s["tile_size"],"method":s["method"],"iou":s["macro_iou"],
        "dice":s["macro_dice"],"boundary_f1_4um":s["internal_boundary"]["f1_at_um"]["4"],
        "assd_um":s["internal_boundary"]["assd_um"]} for s in scores])
    for name in ["image.original.png","expert_reference.compact.tif","production_dapi_validation.json","production_channel_validation.json"]:
        shutil.copyfile(source/name,out/name)
    plot(out,source,maps,selection,previous["compact_class_names"],scores)
    report=["# Markers selected for each group and each tissue interface","",
        "One expert-annotated TIGER breast tissue ROI; controlled reconstruction. This replaces the global omnibus top-three selection for the current experiment.","",
        "## Best markers for each group","","Rank markers by the weakest direction-free pairwise AUC across the four other groups, then by their mean AUC. Thus easy separation of one group cannot dominate. AUC is derived from tile-level Mann-Whitney U and can represent either higher or lower activation. Each row shows three single-marker candidates, not a validated joint classifier.","",
        "| Tissue group | Three candidates (rank order / RGB) | Worst-pair AUCs | Worst adjusted nominal p-values |",
        "|---|---|---|---|"]
    for g,row in selection["per_group"].items():
        report.append(f"| {previous['compact_class_names'][g]} | {', '.join(row['selected_channels_rgb'])} | "+
                      ", ".join(f"{r['minimum_pairwise_direction_free_auc']:.3f}" for r in row['ranking'][:3])+" | "+
                      ", ".join(f"{r['worst_pair_bh_adjusted_nominal_p']:.3g}" for r in row['ranking'][:3])+" |")
    report += ["","## Best markers for each boundary pair","","A group can require different markers against different neighbors. The actual refinement therefore uses a separate three-marker RGB image at each interface, chosen by its pairwise AUC rather than a global statistic. Rank order maps to R/G/B; P1–P99 scaling is fixed over coarse foreground, minimum range 0.01. Per-group composites are saved for inspection but are not merged into a globally inconsistent RGB image.","",
        "| Interface | RGB markers | Direction-free AUCs | Adjusted nominal p-values |","|---|---|---|---|"]
    for row in selection["per_pair"].values():
        a,b=row["groups"]
        report.append(f"| {previous['compact_class_names'][str(a)]} vs {previous['compact_class_names'][str(b)]} | "+
                      ", ".join(row['selected_channels_rgb'])+" | "+", ".join(f"{r['direction_free_auc']:.3f}" for r in row['ranking'][:3])+" | "+
                      ", ".join(f"{r['bh_adjusted_nominal_p_all_pairs_all_markers']:.3g}" for r in row['ranking'][:3])+" |")
    report += ["","## Segmentation","","The existing Bora RGB/Lab local wand is run separately inside each two-class union, with unchanged radii, core definitions and spatial prior. Each pixel belongs to its nearest original interface; exact distance ties at junctions are protected, background is fixed. Predictions are merged using this fixed ownership. A matched H&E control uses the same wrapper, so its difference from marker RGB isolates appearance input rather than the wrapper. This experimental wrapper differs from original multiclass Bora. Fine labels enter only after all predictions are fixed.","",
        "| Tile px | Method | Macro IoU | Macro Dice | Boundary F1 @4 µm | ASSD µm |","|---:|---|---:|---:|---:|---:|"]
    for s in scores:
        report.append(f"| {s['tile_size']} | {s['method']} | {s['macro_iou']:.4f} | {s['macro_dice']:.4f} | {s['internal_boundary']['f1_at_um']['4']:.4f} | {s['internal_boundary']['assd_um']:.3f} |")
    report += ["","## Statistical and biological scope","",
        "169 coarse 90-pixel tiles: 74 invasive tumor, 37 tumor-associated stroma, 12 healthy glands, 12 inflamed stroma and 34 rest. Observations are central-half tile means. All 16 × 10 pairwise tests are BH adjusted together. These are nominal exploratory values because neighboring tiles are correlated. Group and pair rankings are not held-out accuracy estimates. High or low predicted activation can be useful, and a group does not need a uniquely overexpressed marker. The saved all-marker report also contains toroidal-shift sensitivity diagnostics for omnibus associations, not pairwise confirmation.","",
        "Native image: ROI, not full WSI; 1224 × 1166 pixels, provider-reported approximately 0.5 µm/px, 612 × 583 µm field. Original CNN inference at 0.25 µm/px is reused and aligned to native coordinates; optical resolution and objective remain unknown. Bora's fixed 2× working grid is approximately 1 µm/px. Expert mask code 0 is ignored; rest is scored. Coarse groups come from majority tiling of the expert mask, so this is controlled reconstruction. Protein channels are virtual predictions with no paired measured protein truth for this TIGER ROI. The prior reference-channel eligibility filter is unchanged. DAPI is context only, excluded from protein ranking.","",
        "## Reproduction","","`PYTHONPATH=src:. python experiments/gigatime_flash/group_markers.py --source <checked-cnn-results> --output <new-output>` with OMP/BLAS threads set to two. `selection.json`, full statistics, RGB encodings, masks and metrics contain the exact scope and parameters. All figures have 100 µm scale bars; masks use 35% opacity.","",
        "Sources: [TIGER](https://tiger.grand-challenge.org/Data/); [CNN sampling](https://github.com/prov-gigatime/GigaTIME/issues/4); [Mann-Whitney U](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html); [BH adjustment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html).",""]
    (out/"REPORT.md").write_text("\n".join(report))
    print(json.dumps({"per_group":{g:v["selected_channels_rgb"] for g,v in selection["per_group"].items()},
        "scores":[{"tile":s["tile_size"],"method":s["method"],"iou":s["macro_iou"],
                   "f1":s["internal_boundary"]["f1_at_um"]["4"]} for s in scores]},indent=2))


if __name__=="__main__":
    main()
