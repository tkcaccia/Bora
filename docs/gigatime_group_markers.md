# Markers selected for each group and each tissue interface

One expert-annotated TIGER breast tissue ROI; controlled reconstruction. This replaces the global omnibus top-three selection for the current experiment.

## Best markers for each group

Rank markers by the weakest direction-free pairwise AUC across the four other groups, then by their mean AUC. Thus easy separation of one group cannot dominate. AUC is derived from tile-level Mann-Whitney U and can represent either higher or lower activation. Each row shows three single-marker candidates, not a validated joint classifier.

| Tissue group | Three candidates (rank order / RGB) | Worst-pair AUCs | Worst adjusted nominal p-values |
|---|---|---|---|
| invasive tumor | CD138, Caspase3-D, PD-L1 | 0.699, 0.654, 0.621 | 0.000981, 0.0345, 0.048 |
| tumor-associated stroma | CD138, CK, Caspase3-D | 0.699, 0.669, 0.654 | 0.000981, 0.0938, 0.011 |
| healthy glands | CD3, CD4, PHH3-B | 0.785, 0.757, 0.750 | 0.0249, 0.0432, 0.0482 |
| inflamed stroma | CD3, CD14, CD4 | 0.785, 0.771, 0.757 | 0.0249, 0.033, 0.0432 |
| rest | T-bet, Ki67, CD3 | 0.915, 0.913, 0.912 | 2.83e-06, 4.64e-06, 2.98e-06 |

## Best markers for each boundary pair

A group can require different markers against different neighbors. The actual refinement therefore uses a separate three-marker RGB image at each interface, chosen by its pairwise AUC rather than a global statistic. Rank order maps to R/G/B; P1–P99 scaling is fixed over coarse foreground, minimum range 0.01. Per-group composites are saved for inspection but are not merged into a globally inconsistent RGB image.

| Interface | RGB markers | Direction-free AUCs | Adjusted nominal p-values |
|---|---|---|---|
| invasive tumor vs tumor-associated stroma | CK, CD138, Caspase3-D | 0.762, 0.699, 0.654 | 1.67e-05, 0.000981, 0.011 |
| invasive tumor vs healthy glands | CD3, CD4, CD8 | 0.842, 0.832, 0.832 | 0.000279, 0.000417, 0.000417 |
| invasive tumor vs inflamed stroma | CD3, CD4, PHH3-B | 0.993, 0.953, 0.938 | 2.56e-07, 2.14e-06, 3.57e-06 |
| invasive tumor vs rest | CD3, CD4, PHH3-B | 0.962, 0.953, 0.950 | 2.48e-12, 3.51e-12, 3.51e-12 |
| tumor-associated stroma vs healthy glands | CD138, PD-1, CD8 | 0.887, 0.860, 0.854 | 0.000126, 0.000367, 0.000467 |
| tumor-associated stroma vs inflamed stroma | CD3, CD4, PHH3-B | 0.966, 0.953, 0.941 | 4.21e-06, 7.5e-06, 1.25e-05 |
| tumor-associated stroma vs rest | T-bet, Ki67, CD3 | 0.915, 0.913, 0.912 | 1.83e-08, 1.99e-08, 2.17e-08 |
| healthy glands vs inflamed stroma | CD3, CD14, CK | 0.785, 0.771, 0.771 | 0.0249, 0.033, 0.033 |
| healthy glands vs rest | CD11c, CD4, CD16 | 0.988, 0.985, 0.983 | 2.49e-06, 2.71e-06, 2.83e-06 |
| inflamed stroma vs rest | CD14, CD3, CD4 | 1.000, 1.000, 1.000 | 1.54e-06, 1.54e-06, 1.54e-06 |

## Segmentation

The existing Bora RGB/Lab local wand is run separately inside each two-class union, with unchanged radii, core definitions and spatial prior. Each pixel belongs to its nearest original interface; exact distance ties at junctions are protected, background is fixed. Predictions are merged using this fixed ownership. A matched H&E control uses the same wrapper, so its difference from marker RGB isolates appearance input rather than the wrapper. This experimental wrapper differs from original multiclass Bora. Fine labels enter only after all predictions are fixed.

| Tile px | Method | Macro IoU | Macro Dice | Boundary F1 @4 µm | ASSD µm |
|---:|---|---:|---:|---:|---:|
| 56 | coarse | 0.7819 | 0.8742 | 0.1940 | 12.777 |
| 56 | bora_he_rgb | 0.8150 | 0.8954 | 0.3042 | 11.372 |
| 56 | previous_six | 0.7926 | 0.8810 | 0.2512 | 11.823 |
| 56 | pairwise_he_rgb | 0.8139 | 0.8947 | 0.3048 | 11.371 |
| 56 | pairwise_marker_rgb | 0.7916 | 0.8804 | 0.2432 | 12.152 |
| 90 | coarse | 0.7220 | 0.8341 | 0.1463 | 16.314 |
| 90 | bora_he_rgb | 0.7476 | 0.8515 | 0.2037 | 15.856 |
| 90 | previous_six | 0.7448 | 0.8493 | 0.2370 | 13.757 |
| 90 | pairwise_he_rgb | 0.7464 | 0.8507 | 0.1962 | 15.838 |
| 90 | pairwise_marker_rgb | 0.7539 | 0.8553 | 0.2505 | 13.757 |
| 216 | coarse | 0.5184 | 0.6486 | 0.1555 | 24.277 |
| 216 | bora_he_rgb | 0.5515 | 0.6845 | 0.1794 | 19.595 |
| 216 | previous_six | 0.5547 | 0.6890 | 0.2096 | 18.705 |
| 216 | pairwise_he_rgb | 0.5488 | 0.6806 | 0.1829 | 19.335 |
| 216 | pairwise_marker_rgb | 0.5464 | 0.6792 | 0.2122 | 19.037 |

## Statistical and biological scope

At the primary 90-pixel grid, interface-specific marker RGB improves macro
IoU from **0.7476 to 0.7539** and boundary F1 from **0.2037 to 0.2505** versus
the original H&E Bora result. It also improves both measures versus the
matched pairwise H&E control (IoU 0.7464; boundary F1 0.1962). This benefit is
not consistent across scales: H&E is better at 56 pixels; at 216 pixels the
marker version improves boundary F1 while slightly reducing overlap.

CD3/CD4 features associated with gland-labelled tiles are empirical virtual
signals in this ROI; they must not be described as gland-epithelium-specific
biomarkers. Tile mixtures, nearby immune cells and prediction errors can all
contribute. Likewise, rest is distinguished largely by **lower** activation,
not uniquely positive expression. CK's weaker stroma-versus-inflamed-stroma
contrast does not pass the nominal adjusted 0.05 threshold; its strong
tumor–stroma contrast is the reason it is useful for that particular interface.

169 coarse 90-pixel tiles: 74 invasive tumor, 37 tumor-associated stroma, 12 healthy glands, 12 inflamed stroma and 34 rest. Observations are central-half tile means. All 16 × 10 pairwise tests are BH adjusted together. These are nominal exploratory values because neighboring tiles are correlated. Group and pair rankings are not held-out accuracy estimates. High or low predicted activation can be useful, and a group does not need a uniquely overexpressed marker. The saved all-marker report also contains toroidal-shift sensitivity diagnostics for omnibus associations, not pairwise confirmation.

Native image: ROI, not full WSI; 1224 × 1166 pixels, provider-reported approximately 0.5 µm/px, 612 × 583 µm field. Original CNN inference at 0.25 µm/px is reused and aligned to native coordinates; optical resolution and objective remain unknown. Bora's fixed 2× working grid is approximately 1 µm/px. Expert mask code 0 is ignored; rest is scored. Coarse groups come from majority tiling of the expert mask, so this is controlled reconstruction. Protein channels are virtual predictions with no paired measured protein truth for this TIGER ROI. The prior reference-channel eligibility filter is unchanged. DAPI is context only, excluded from protein ranking.

## Reproduction

`PYTHONPATH=src:. python experiments/gigatime_flash/group_markers.py --source <checked-cnn-results> --output <new-output>` with OMP/BLAS threads set to two. `selection.json`, full statistics, RGB encodings, masks and metrics contain the exact scope and parameters. All figures have 100 µm scale bars; masks use 35% opacity.

Sources: [TIGER](https://tiger.grand-challenge.org/Data/); [CNN sampling](https://github.com/prov-gigatime/GigaTIME/issues/4); [Mann-Whitney U](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html); [BH adjustment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html).

## Saved results and verification

- Chiamaka: `/mnt/sata_ssd/Bora-benchmark/results/gigatime-group-rgb-2026-10-16/`.
- Local review: `/Users/stefano/Documents/ChatGPT/Bora/benchmark/results/gigatime-group-rgb-2026-10-16/`.
- [Reproduction guide and scripts](../experiments/gigatime_flash/README.md).

The per-group candidate RGBs, all ten interface RGBs, their channel/intensity
metadata, original image, expert/coarse/refined masks, full statistical
rankings, raw tile observations and segmentation scores are saved. Five
figures show per-group coverage, per-interface marker statistics, per-group
composites, tumor–stroma markers with DAPI and transparent segmentation masks.
Twenty-four focused tests pass. Source/map checksums, native-frame shapes,
score recomputation and final figures are checked separately. Raw data and
generated rasters remain outside Git.

`all_markers/` additionally shows all 23 CNN outputs, including DAPI and the
TRITC/Cy5 background channels. `all_markers.contrast.png` uses a separate
P1–P99 display range for each channel (minimum range 0.01); brightness cannot
be compared across these panels. `all_markers.fixed_scale.png` uses 0–1 for
every panel. Twenty-three native-resolution PNGs and `manifest.json` record
the exact channel identity, scale and original physical frame. CD20, Tryptase,
Actin-D and Transgelin are displayed but labelled below the reference screen.
