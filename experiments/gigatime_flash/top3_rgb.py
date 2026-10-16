#!/usr/bin/env python3
"""Coarse-group omnibus marker ranking, synthetic RGB, and unmodified RGB Bora.

Reuses checked CNN maps; never fits to the fine reference mask. Statistical
values describe one spatially correlated ROI, not independent patients.
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
from scipy.stats import kruskal, mannwhitneyu, rankdata, spearmanr, tiecorrect
import tifffile


def bh_adjust(pvalues):
    """Benjamini-Hochberg adjusted nominal p-values, in original order."""
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Expected a nonempty vector of finite probabilities")
    order = np.argsort(p, kind="stable")
    adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p)+1))[::-1])[::-1]
    result = np.empty_like(p)
    result[order] = np.minimum(adjusted, 1)
    return result


def rank_markers(tiles, names, eligible, tile_size):
    """Kruskal-Wallis across all coarse groups; exact grid-shift sensitivity.

    Central tile means are observations, not individual pixels. All complete
    toroidal label shifts retain group counts and spatial arrangement on the
    tile lattice. Their tail fraction is a spatial sensitivity diagnostic;
    stationarity and periodic edges are assumptions, not established here.
    """
    if len(eligible) < 3 or len(set(eligible)) != len(eligible):
        raise ValueError("Need at least three distinct eligible protein channels")
    indices = [names.index(name) for name in eligible]
    values = np.array([t["central_mean"] for t in tiles], float)[:, indices]
    labels = np.array([t["label"] for t in tiles], int)
    if not np.isfinite(values).all() or np.any(labels <= 0):
        raise ValueError("Finite tile values and positive coarse groups required")
    groups, counts = np.unique(labels, return_counts=True)
    if len(groups) < 2 or counts.min() < 5:
        raise ValueError("Each coarse group needs at least five tiles")
    coordinates = np.array([[t["y"]//tile_size, t["x"]//tile_size] for t in tiles])
    if any(t["x"] % tile_size or t["y"] % tile_size for t in tiles):
        raise ValueError("Tile coordinates must lie on the stated grid")
    shape = tuple(map(int, coordinates.max(0)+1))
    if np.prod(shape) != len(tiles) or len(np.unique(coordinates, axis=0)) != len(tiles):
        raise ValueError("Grid-shift diagnostic requires a complete rectangular tile grid")
    grid = np.zeros(shape, int)
    grid[coordinates[:, 0], coordinates[:, 1]] = labels
    ranks = rankdata(values, axis=0)
    corrections = np.array([tiecorrect(ranks[:, j]) for j in range(len(indices))])
    n = len(tiles)

    def h_values(lab):
        sums = np.stack([ranks[lab == g].sum(0) for g in groups])
        raw = 12/(n*(n+1)) * (sums*sums/counts[:, None]).sum(0) - 3*(n+1)
        return np.divide(np.maximum(raw, 0), corrections, out=np.zeros_like(raw), where=corrections > 0)

    observed = h_values(labels)
    shifted = []
    for dy in range(shape[0]):
        for dx in range(shape[1]):
            lab = np.roll(grid, (dy, dx), axis=(0, 1))[coordinates[:, 0], coordinates[:, 1]]
            shifted.append(h_values(lab))
    shifted = np.asarray(shifted)
    spatial_p = np.mean(shifted >= observed[None, :] - 1e-10, axis=0)
    rows, pairwise = [], []
    for j, (name, index) in enumerate(zip(eligible, indices)):
        samples = [values[labels == g, j] for g in groups]
        statistic, pvalue = kruskal(*samples) if corrections[j] > 0 else (0., 1.)
        if not np.isclose(statistic, observed[j], atol=1e-9):
            raise AssertionError("Grid statistic disagrees with SciPy")
        summary = {str(g): {"n": int(len(v)), "mean": float(v.mean()),
                            "median": float(np.median(v)), "q25": float(np.percentile(v, 25)),
                            "q75": float(np.percentile(v, 75))}
                   for g, v in zip(groups, samples)}
        rows.append({"channel": name, "index": index, "kruskal_h": float(statistic),
                     "nominal_p": float(pvalue), "rank_effect_h_over_n_minus_1": float(statistic/(n-1)),
                     "spatial_shift_tail_fraction": float(spatial_p[j]), "groups": summary})
        for a, b in combinations(groups, 2):
            va, vb = values[labels == a, j], values[labels == b, j]
            u, p = mannwhitneyu(va, vb, alternative="two-sided", method="asymptotic")
            pairwise.append({"channel": name, "group_a": int(a), "group_b": int(b),
                             "n_a": len(va), "n_b": len(vb), "u": float(u), "nominal_p": float(p),
                             "median_a": float(np.median(va)), "median_b": float(np.median(vb)),
                             "rank_biserial_a_minus_b": float(2*u/(len(va)*len(vb))-1)})
    for row, q, qs in zip(rows, bh_adjust([r["nominal_p"] for r in rows]), bh_adjust(spatial_p)):
        row["bh_adjusted_nominal_p"] = float(q)
        row["bh_adjusted_spatial_shift_tail_fraction"] = float(qs)
    for row, q in zip(pairwise, bh_adjust([r["nominal_p"] for r in pairwise])):
        row["bh_adjusted_nominal_p_all_pairs_all_markers"] = float(q)
    rows.sort(key=lambda r: (r["nominal_p"], r["index"]))
    selected = [r["index"] for r in rows[:3]]
    selected_values = np.array([t["central_mean"] for t in tiles])[:, selected]
    correlations = []
    for a, b in combinations(range(3), 2):
        va, vb = selected_values[:, a], selected_values[:, b]
        variable = va.std() > 0 and vb.std() > 0
        correlations.append({"marker_a": names[selected[a]], "marker_b": names[selected[b]],
            "pearson": float(np.corrcoef(va, vb)[0, 1]) if variable else None,
            "spearman": float(spearmanr(va, vb).statistic) if variable else None})
    return {"ranking": rows, "selected_channels": [r["channel"] for r in rows[:3]],
            "selected_indices": [r["index"] for r in rows[:3]], "pairwise": pairwise,
            "selected_tile_correlations": correlations,
            "group_tile_counts": {str(g): int(c) for g, c in zip(groups, counts)},
            "selection_tile_size_px": tile_size, "n_tiles": n, "eligible_channels": eligible,
            "selection_rule": "lowest omnibus Kruskal-Wallis nominal p; BH across all eligible proteins",
            "observation": "one central-half mean per coarse tile; unweighted across tiles",
            "spatial_diagnostic": {"method": "all toroidal shifts of coarse labels relative to fixed marker tile means",
                "shape_yx": list(shape), "shifts_including_identity": len(shifted),
                "minimum_tail_fraction": 1/len(shifted),
                "assumptions": "stationary periodic tile lattice; edge wrapping is artificial; sensitivity analysis only"},
            "inference_scope": "exploratory within one ROI; nominal p/BH do not account for correlated tiles; no patient-level inference",
            "fine_reference_mask_used_for_selection": False}


def markers_to_rgb(maps_hwc, indices, support, *, robust=True):
    """Rank 1/2/3 -> R/G/B, with explicit invertible-within-range linear scales."""
    maps = np.asarray(maps_hwc)
    support = np.asarray(support, bool)
    if len(indices) != 3 or len(set(indices)) != 3 or maps.ndim != 3 or maps.shape[:2] != support.shape or not support.any():
        raise ValueError("Three distinct channels and nonempty aligned coarse support required")
    selected = maps[..., indices].astype(np.float32)
    if not np.isfinite(selected).all() or selected.min() < 0 or selected.max() > 1:
        raise ValueError("Expected finite model activations within [0,1]")
    limits = np.percentile(selected[support], [1, 99], axis=0) if robust else np.array([[0.]*3, [1.]*3])
    # A nonconstant range avoids expanding an effectively constant channel.
    low, high = limits
    scale = np.maximum(high-low, .01)
    unit = np.clip((selected-low)/scale, 0, 1)
    rgb = np.rint(unit*255).astype(np.uint8)
    return rgb, {"rule": "linear P1-P99 over coarse foreground, minimum range 0.01" if robust else "fixed linear [0,1]",
                 "low": low.tolist(), "high": high.tolist(), "effective_range": scale.tolist(),
                 "channel_indices_rgb": list(indices), "gamma": 1., "dtype": "uint8",
                 "clipping_fraction_low": np.mean(selected[support] <= low, axis=0).tolist(),
                 "clipping_fraction_high": np.mean(selected[support] >= low+scale, axis=0).tolist(),
                 "fine_reference_used": False}


def write_csv(path, records):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plot_results(out, source, maps, rgb, selection, results, tiles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    from experiments.gigatime_flash.plot_overlays import bar
    he = np.asarray(Image.open(source/"image.original.png").convert("RGB"))
    names = results["compact_class_names"]
    labels = np.array([t["label"] for t in tiles])
    values = np.array([t["central_mean"] for t in tiles])
    class_colors = ["white", "#e24a33", "#56b4e9", "#009e73", "#cc79a7", "#f0e442"]
    cmap = ListedColormap(class_colors)
    norm = BoundaryNorm(np.arange(-.5, 6.5), 6)
    mpp = results["native_mpp_xy"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    rows = selection["ranking"]
    ax = axes.flat[0]
    ax.barh(np.arange(len(rows)), [-np.log10(max(r["bh_adjusted_nominal_p"], 1e-300)) for r in rows],
            color=["#a72a70" if i < 3 else "#999999" for i in range(len(rows))])
    ax.set_yticks(np.arange(len(rows)), [r["channel"] for r in rows])
    ax.invert_yaxis()
    ax.axvline(-np.log10(.05), color="black", ls=":", lw=1)
    ax.set_xlabel("−log10(BH-adjusted nominal p), 16 eligible markers")
    ax.set_title("All-group marker ranking; top three in purple")
    short = ["Invasive\ntumor", "Tumor-associated\nstroma", "Healthy\nglands", "Inflamed\nstroma", "Rest"]
    rng = np.random.default_rng(207)
    for ax, row, color in zip(list(axes.flat)[1:], rows[:3], ["#ce3434", "#258b38", "#306ac4"]):
        vals = values[:, row["index"]]
        groups = [vals[labels == g] for g in range(1, 6)]
        ax.boxplot(groups, showfliers=False, widths=.5)
        for g, v in enumerate(groups, 1):
            ax.scatter(g+rng.uniform(-.16, .16, len(v)), v, c=class_colors[g], edgecolors="black", linewidths=.3, s=17, alpha=.75)
        ax.set_xticks(range(1, 6), [f"{name}\n(n={len(v)})" for name, v in zip(short, groups)], fontsize=8)
        ax.set_ylabel("Mean predicted activation in tile centre")
        ax.set_title(f"{row['channel']}  |  H={row['kruskal_h']:.2f}\n"
                     f"nominal p={row['nominal_p']:.2e}; BH p={row['bh_adjusted_nominal_p']:.2e}\n"
                     f"spatial-shift tail fraction={row['spatial_shift_tail_fraction']:.4f}", color=color, fontsize=11)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("GigaTIME CNN: three most differentiated markers across coarse tissue groups\n"
                 "One TIGER ROI; 169 tiles, not independent patients. Spatial shifts are a sensitivity check.", fontsize=13)
    fig.savefig(out/"marker_statistics.png", dpi=180)
    plt.close(fig)

    selected = selection["selected_channels"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.5), constrained_layout=True)
    panels = [(he, "Original H&E"), (rgb, f"Synthetic RGB: R={selected[0]}, G={selected[1]}, B={selected[2]}"),
              (maps[0], "DAPI prediction (context; excluded from ranking)")]
    for name, index, color in zip(selected, selection["selected_indices"], ["R", "G", "B"]):
        panels.append((maps[index], f"{name} → {color}; displayed with P1-P99 scaling"))
    for ax, (data, title) in zip(axes.flat, panels):
        if data.ndim == 2:
            low, high = np.percentile(data, [1, 99])
            ax.imshow(data, cmap="gray", vmin=low, vmax=max(high, low+.01))
        else:
            ax.imshow(data)
        bar(ax, he.shape[:2], mpp)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("Virtual marker maps and the exact RGB image supplied to Bora\n"
                 "Native ROI: 1224 × 1166 px, ≈0.5 µm/px; CNN inference: 0.25 µm/px. Intensities are predictions.", fontsize=13)
    fig.savefig(out/"top3_rgb_and_markers.png", dpi=170)
    plt.close(fig)

    tile = 90
    methods = [(None, "Original H&E"), ("expert", "Expert annotation"), ("coarse", "Tiling-mask input"),
               ("bora_he_rgb", "Bora on H&E"), ("bora_previous_six", "Previous six-marker feature Bora"),
               ("bora_top3_rgb", "Bora on three-marker RGB (primary)")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 11), constrained_layout=True)
    for ax, (method, title) in zip(axes.flat, methods):
        ax.imshow(he)
        if method:
            path = out/"expert_reference.compact.tif" if method == "expert" else out/f"tile_{tile}"/f"{method}.tif"
            mask = tifffile.imread(path)
            ax.imshow(np.ma.masked_equal(mask, 0), cmap=cmap, norm=norm, alpha=.35, interpolation="nearest")
        if method not in (None, "expert"):
            s = next(s for s in results["scores"] if s["tile_size"] == tile and s["method"] == method)
            title += f"\nIoU {s['macro_iou']:.4f}; boundary F1 {s['internal_boundary']['f1_at_um']['4']:.4f}"
        bar(ax, he.shape[:2], mpp)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.legend(handles=[Patch(color=class_colors[int(g)], label=n) for g, n in names.items()],
               loc="lower center", ncol=5, fontsize=9, frameon=False)
    fig.suptitle("Same H&E and tissue labels; all masks at 35% opacity\n90-pixel input tiles (45 µm); fixed Bora parameters and native output frame", fontsize=13)
    fig.savefig(out/"segmentation_overlays_tile_90.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    source, out = args.source, args.output
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from bora.boundary import BoundaryConfig, smooth_tile_boundaries
    from bora.virtual_markers import select_markers
    from experiments.gigatime_flash.run_pilot import CHANNELS, digest, save_json, score
    cv2.setNumThreads(2)
    previous = json.loads((source/"results.json").read_text())
    info = previous["inference"]["0.250000"]
    if previous["model"]["model_family"] != "cnn" or info["channel_order"] != CHANNELS:
        raise ValueError("This experiment requires the checked original CNN and canonical channels")
    map_path = source/"inference_0.250000/virtual_markers.native.npy"
    if digest(map_path) != info["native_maps_sha256"]:
        raise ValueError("CNN map checksum mismatch")
    maps = np.load(map_path, mmap_mode="r")
    if list(maps.shape) != [23, previous["sample"]["height"], previous["sample"]["width"]]:
        raise ValueError("CNN maps differ from original image frame")
    coarse90 = tifffile.imread(source/"tile_90/coarse.tif")
    # Recompute observations from the checksummed maps and current coarse mask,
    # rather than trusting previously serialized tile summaries.
    tiles = select_markers(np.moveaxis(maps, 0, -1), coarse90, 90, CHANNELS)["tiles"]
    selection = rank_markers(tiles, CHANNELS, previous["eligible_marker_channels"], 90)
    save_json(out/"marker_statistics.json", selection)
    flat_stats = [{k: v for k, v in row.items() if k != "groups"} for row in selection["ranking"]]
    write_csv(out/"marker_statistics.csv", flat_stats)
    write_csv(out/"marker_pairwise_statistics.csv", selection["pairwise"])
    write_csv(out/"tile_observations.csv", [{"x": t["x"], "y": t["y"], "coarse_group": t["label"],
              **dict(zip(CHANNELS, t["central_mean"]))} for t in tiles])
    rgb, scaling = markers_to_rgb(np.moveaxis(maps, 0, -1), selection["selected_indices"], coarse90 > 0)
    raw_rgb, raw_scaling = markers_to_rgb(np.moveaxis(maps, 0, -1), selection["selected_indices"], coarse90 > 0, robust=False)
    Image.fromarray(rgb).save(out/"top3_markers.rgb.png")
    Image.fromarray(raw_rgb).save(out/"top3_markers.raw_probability.rgb.png")
    tifffile.imwrite(out/"top3_markers.ome.tif", rgb, photometric="rgb", metadata={"axes": "YXS",
                     "PhysicalSizeX": .5, "PhysicalSizeXUnit": "µm", "PhysicalSizeY": .5, "PhysicalSizeYUnit": "µm"})
    np.save(out/"top3_markers.probabilities.npy", np.asarray(maps[selection["selected_indices"]]))
    save_json(out/"rgb_encoding.json", {"mapping": dict(zip("RGB", selection["selected_channels"])),
              "primary": scaling, "fixed_range_sensitivity": raw_scaling,
              "native_shape_yx": list(rgb.shape[:2]), "native_mpp_xy": [.5, .5],
              "synthetic_image_not_he": True})
    config = BoundaryConfig(**previous["boundary_config"])
    runs = []
    # No fine expert mask is opened before all new predictions are fixed.
    for tile in [56, 90, 216]:
        folder = out/f"tile_{tile}"
        folder.mkdir(exist_ok=True)
        coarse = tifffile.imread(source/f"tile_{tile}/coarse.tif")
        for old, new in [("coarse", "coarse"), ("bora_rgb", "bora_he_rgb"),
                         ("bora_markers_0.250000", "bora_previous_six")]:
            shutil.copyfile(source/f"tile_{tile}/{old}.tif", folder/f"{new}.tif")
        for image, method in [(rgb, "bora_top3_rgb"), (raw_rgb, "bora_top3_raw_probability_rgb")]:
            began = time.perf_counter()
            # The ordinary RGB/Lab Bora path, without feature_image.
            pred = smooth_tile_boundaries(image, coarse, tile, config=config)
            tifffile.imwrite(folder/f"{method}.tif", pred)
            runs.append({"tile_size": tile, "method": method, "seconds": time.perf_counter()-began,
                         "coarse_sha256": digest(folder/"coarse.tif"), "output_sha256": digest(folder/f"{method}.tif")})
    truth = tifffile.imread(source/"expert_reference.compact.tif")
    scores = []
    for tile in [56, 90, 216]:
        for method in ["coarse", "bora_he_rgb", "bora_previous_six", "bora_top3_rgb", "bora_top3_raw_probability_rgb"]:
            pred = tifffile.imread(out/f"tile_{tile}/{method}.tif")
            scores.append({"tile_size": tile, "method": method, **score(pred, truth, [.5, .5], previous["compact_class_names"])})
    results = {"source_results": str(source), "source_results_sha256": digest(source/"results.json"),
               "sample": previous["sample"], "image_type": previous["image_type"], "model": previous["model"],
               "inference": info, "native_mpp_xy": [.5, .5], "native_mpp_status": previous["native_mpp_status"],
               "microscope_objective_magnification": None, "microscope_optical_resolution_um": None,
               "compact_class_names": previous["compact_class_names"], "boundary_config": asdict(config),
               "bora_input": "synthetic 8-bit RGB -> ordinary RGB/Lab path; feature_image omitted",
               "selected_channels_rgb": selection["selected_channels"], "selection_tile_size_px": 90,
               "selection_fixed_across_tile_sizes": True, "rgb_encoding": scaling,
               "sensitivity_encoding": raw_scaling, "runs": runs, "scores": scores,
               "coarse_protocol": previous["coarse_protocol"], "reference_mask_used_after_predictions_fixed": True,
               "limitations": ["one ROI and correlated coarse tiles, exploratory statistics only",
                 "all tile sizes are degradations of one expert reference, not independent cases",
                 "virtual markers have no measured protein truth in this TIGER ROI",
                 "RGB color assignment is rank order and affects Lab distance; no truth-based color/permutation tuning",
                 "P1-P99 primary and fixed [0,1] sensitivity were specified before scoring"],
               "source_sha256": {name: digest(Path(__file__).resolve().parents[2]/name) for name in [
                   "experiments/gigatime_flash/top3_rgb.py", "src/bora/boundary.py", "experiments/gigatime_flash/run_pilot.py"]}}
    save_json(out/"results.json", results)
    write_csv(out/"scores.csv", [{"tile_size": s["tile_size"], "method": s["method"],
        "accuracy": s["pixel_accuracy"], "macro_iou": s["macro_iou"], "macro_dice": s["macro_dice"],
        "boundary_f1_4um": s["internal_boundary"]["f1_at_um"]["4"],
        "assd_um": s["internal_boundary"]["assd_um"], "hd95_um": s["internal_boundary"]["hd95_um"]} for s in scores])
    for name in ["image.original.png", "expert_reference.compact.tif", "production_channel_validation.json", "production_dapi_validation.json"]:
        shutil.copyfile(source/name, out/name)
    plot_results(out, source, maps, rgb, selection, results, tiles)
    report = ["# Top-three virtual-marker RGB experiment", "", "Exploratory controlled reconstruction of one expert-annotated TIGER breast tissue ROI.", "",
        "## Marker statistics", "", "Selection: central-half tile means on the 90-pixel coarse grid, all five groups; 169 tiles. No fine reference used for selection.", "",
        "| RGB | Marker | Kruskal H | Nominal p | BH-adjusted nominal p | Spatial-shift tail fraction |", "|---|---|---:|---:|---:|---:|"]
    for color, row in zip("RGB", selection["ranking"][:3]):
        report.append(f"| {color} | {row['channel']} | {row['kruskal_h']:.4f} | {row['nominal_p']:.4g} | {row['bh_adjusted_nominal_p']:.4g} | {row['spatial_shift_tail_fraction']:.4f} |")
    report += ["", "BH adjusts across all 16 eligible proteins. Nominal tests assume independent observations, which adjacent tiles do not satisfy. All 169 toroidal label shifts are a spatial sensitivity check with artificial wraparound, not patient-level confirmation. Pairwise tests across all 160 marker/group pairs are separately BH adjusted in the CSV.", "",
        "Primary RGB scaling: independent linear P1–P99 over coarse foreground, minimum activation range 0.01, clipping and rounding to uint8. Rank 1/2/3 goes to R/G/B. The same image and default Bora configuration are used for all tile sizes. Fixed [0,1] scaling is reported as a sensitivity check; no output-score-based selection of marker, color, scaling or parameters.", "",
        "## Segmentation", "", "All values on the unchanged native reference; ignore code 0 only. Boundary F1 tolerance 4 µm.", "",
        "| Tile px | Method | Macro IoU | Macro Dice | Boundary F1 | ASSD µm |", "|---:|---|---:|---:|---:|---:|"]
    for s in scores:
        report.append(f"| {s['tile_size']} | {s['method']} | {s['macro_iou']:.4f} | {s['macro_dice']:.4f} | {s['internal_boundary']['f1_at_um']['4']:.4f} | {s['internal_boundary']['assd_um']:.3f} |")
    report += ["", "## Interpretation", "", "At 90-pixel tiles, primary three-marker RGB improves boundary F1 from 0.2037 to 0.2161 versus H&E Bora but reduces macro IoU from 0.7476 to 0.7425. It also trails the previous six-feature approach at this tile width. At 216 pixels it improves boundary F1 while lowering IoU; at 56 pixels it is worse than H&E Bora on both. This does not establish an overall segmentation benefit.", "",
        "The top three have strongly redundant tile signals. Their high omnibus significance is driven by differences involving inflamed stroma and rest, and does not establish separation of invasive tumor from tumor-associated stroma. Selecting markers for a particular adjacent tissue pair, with a redundancy constraint, is a future experiment rather than a result of this run.", "",
        "| Selected marker pair | Tile Pearson r | Tile Spearman rho |", "|---|---:|---:|"]
    for row in selection["selected_tile_correlations"]:
        report.append(f"| {row['marker_a']} / {row['marker_b']} | {row['pearson']:.4f} | {row['spearman']:.4f} |")
    report += ["", "| Marker | Tumor vs tumor-associated stroma nominal p | BH-adjusted p across all 160 pairs |", "|---|---:|---:|"]
    for row in selection["pairwise"]:
        if row["channel"] in selection["selected_channels"] and (row["group_a"], row["group_b"]) == (1, 2):
            report.append(f"| {row['channel']} | {row['nominal_p']:.4f} | {row['bh_adjusted_nominal_p_all_pairs_all_markers']:.4f} |")
    report += ["", "## Resolution and annotation", "", "Native 1224 × 1166 px tissue ROI (not WSI), approximately 0.5 µm/px; field of view 612 × 583 µm. GigaTIME CNN inference uses 0.25 µm/px and returns maps to native coordinates. Upsampling adds no optical information. Objective, numerical aperture and optical resolving power are unknown. Bora runs at its fixed 2× downsample (approximately 1 µm/working pixel). Synthetic OME-TIFF carries native physical pixel size; JSON records channel identities and intensity transforms.", "",
        "Labels: 1 invasive tumor, 2 tumor-associated stroma, 3 healthy glands, 4 inflamed stroma, 5 rest, 0 ignored. Expert masks are manual; coarse tiling-mask labels were derived from them, so this tests boundary reconstruction rather than independent classification. Protein maps are virtual predictions, not measured stains. DAPI is shown for context and excluded from protein ranking.", "",
        "## Reproduction and sources", "", "Run `PYTHONPATH=src python experiments/gigatime_flash/top3_rgb.py --source <checked-cnn-results> --output <new-output>` with two BLAS/OpenMP threads. Input/model/map hashes, parameters, all marker statistics, group summaries and full per-class segmentation scores are saved in JSON/CSV. Fine labels are opened only after predictions are fixed.", "",
        "Sources: [TIGER](https://tiger.grand-challenge.org/Data/); [GigaTIME sampling clarification](https://github.com/prov-gigatime/GigaTIME/issues/4); [Kruskal-Wallis](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html); [BH procedure](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html).", ""]
    (out/"REPORT.md").write_text("\n".join(report))
    print(json.dumps({"selected_rgb": selection["selected_channels"], "top_statistics": flat_stats[:3],
                      "scores": [{"tile": s["tile_size"], "method": s["method"], "iou": s["macro_iou"],
                                  "boundary_f1": s["internal_boundary"]["f1_at_um"]["4"]} for s in scores]}, indent=2))


if __name__ == "__main__":
    main()
