#!/usr/bin/env python3
"""Transparent, physically calibrated H&E overlays for a completed pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np
from PIL import Image
import tifffile


def bar(ax, shape, mpp_xy):
    h, w = shape
    x, y = w*.04, h*.95
    ax.plot([x, x+100/mpp_xy[0]], [y, y], color="black", lw=4)
    ax.text(x, y-h*.025, "100 µm", fontsize=10,
            bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none", "pad": 1})


def draw(results, data_root, tile_size=90, alpha=.35):
    if not 0 < alpha < 1:
        raise ValueError("Mask overlay opacity must be strictly between zero and one")
    r = json.loads((results/"results.json").read_text())
    rgb = np.asarray(Image.open(data_root/r["sample"]["image"]).convert("RGB"))
    reference = tifffile.imread(results/"expert_reference.compact.tif")
    mpp_xy = [r["sample"]["mpp_um"]]*2
    case = results/f"tile_{tile_size}"
    palette = ["#ffffff", "#e24a33", "#56b4e9", "#009e73", "#cc79a7", "#f0e442"]
    cmap, norm = ListedColormap(palette), BoundaryNorm(np.arange(-.5, 6.5), len(palette))
    panels = [
        ("Original H&E", None),
        ("Expert reference mask", reference),
        (f"tiling-mask: {tile_size} × {tile_size} px tiles", tifffile.imread(case/"coarse.tif")),
        ("Bora: RGB-driven refinement", tifffile.imread(case/"bora_rgb.tif")),
    ]
    for scale, inference in r["inference"].items():
        is_native = inference["sampling"]["inference_shape_yx"] == list(rgb.shape[:2])
        suffix = "" if is_native else "*"
        panels.append((f"Bora: virtual markers at {float(scale):.3f} µm/pixel{suffix}",
                       tifffile.imread(case/f"bora_markers_{scale}.tif")))
    rows = int(np.ceil(len(panels)/3))
    fig, axes = plt.subplots(rows, 3, figsize=(17, 6*rows), squeeze=False)
    for ax, (title, mask) in zip(axes.flat, panels):
        ax.imshow(rgb)
        if mask is not None:
            ax.imshow(np.ma.masked_equal(mask, 0), cmap=cmap, norm=norm,
                      interpolation="nearest", alpha=alpha)
        ax.set_title(title, fontsize=12, pad=9)
        ax.axis("off")
        bar(ax, reference.shape, mpp_xy)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    fig.legend(handles=[Patch(color=palette[int(k)], label=v)
                        for k, v in r["compact_class_names"].items()],
               loc="lower center", bbox_to_anchor=(.5, .04), ncol=3, fontsize=11)
    fig.suptitle(f"TIGER 114S tissue ROI | Native 1224 × 1166 px, approximately 0.5 µm/pixel | Mask opacity {alpha:.0%}", fontsize=14)
    fig.text(.5, .018, "All masks share the original H&E frame. *Upsampled input; no gain in optical detail. Label 0 is transparent and excluded from scoring.",
             ha="center", fontsize=10)
    fig.subplots_adjust(left=.015, right=.985, top=.92, bottom=.105, hspace=.13, wspace=.06)
    seg_name = f"overlays_tile_{tile_size}.png"
    fig.savefig(results/seg_name, dpi=200)
    plt.close(fig)
    # Include DAPI plus both globally differential and tumor/stroma channels.
    marker_names = ["DAPI", "CK_1:150", "CD4", "CD14", "Actin-D", "Transgelin"]
    files = []
    for scale, inference in r["inference"].items():
        maps = np.load(results/f"inference_{scale}/virtual_markers.native.npy", mmap_mode="r")
        order = inference["channel_order"]
        fig, axes = plt.subplots(2, 3, figsize=(17, 12))
        for ax, name in zip(axes.flat, marker_names):
            plane = np.asarray(maps[order.index(name)], np.float32)
            ax.imshow(rgb)
            ax.imshow(plane, cmap="magma", vmin=0, vmax=1,
                      alpha=np.clip(.7*plane, 0, .7), interpolation="nearest")
            ax.set_title("Virtual " + ("CK (cytokeratin)" if name == "CK_1:150" else name), fontsize=12, pad=9)
            ax.axis("off")
            bar(ax, reference.shape, mpp_xy)
        # Legend shows activation hue; actual overlay transparency is 0.7×activation.
        mappable = plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="magma")
        cbax = fig.add_axes([.31, .08, .38, .014])
        fig.colorbar(mappable, cax=cbax, orientation="horizontal", label="Predicted sigmoid activation; overlay opacity = 0.7 × activation")
        scale_label = "native input" if scale == "0.500000" else "upsampled input; Flash calibration unverified"
        fig.suptitle(f"GigaTIME-Flash virtual channels on original H&E | Inference {scale} µm/pixel ({scale_label})", fontsize=14)
        fig.text(.5, .014, "DAPI is a nuclear channel. These are model predictions, not measured immunostains. All panels use the same original frame and activation scale.",
                 ha="center", fontsize=10)
        fig.subplots_adjust(left=.015, right=.985, top=.92, bottom=.15, hspace=.13, wspace=.06)
        name = f"marker_overlays_{scale}.png"
        fig.savefig(results/name, dpi=200)
        plt.close(fig)
        files.append(name)
    manifest = {"segmentation_figure": seg_name, "marker_figures": files,
                "mask_opacity": alpha, "marker_opacity": "0.7 × predicted sigmoid activation",
                "marker_channels": marker_names, "DAPI_included": True,
                "tile_size_px": tile_size, "native_mpp_xy": mpp_xy,
                "all_panels_registered_to_original_image": True,
                "zero_mask_label_transparent": True,
                "displayed_marker_selection": "DAPI; CK/Actin/Transgelin for tissue interfaces; CD4/CD14 from global coarse-group contrasts"}
    (results/"overlay_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("/mnt/sata_ssd/Bora-benchmark"))
    p.add_argument("--tile-size", type=int, default=90)
    p.add_argument("--alpha", type=float, default=.35)
    a = p.parse_args()
    draw(a.results, a.data_root, a.tile_size, a.alpha)
