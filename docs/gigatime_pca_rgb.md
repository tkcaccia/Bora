# Original RGB + class markers → PCA RGB → Bora

## Inputs and marker selection

One TIGER breast tissue ROI. Reuse the checked original GigaTIME CNN maps; no new inference or training. One marker per class is selected from central-half means of the same 169 coarse 90-pixel tiles, ranking each marker by its weakest pairwise separation from the four other classes. No fine annotation enters selection or PCA.

| Class | Selected marker | Worst-pair direction-free AUC | Worst adjusted nominal pairwise p |
|---|---|---:|---:|
| invasive tumor | CD138 | 0.6994 | 0.0009812 |
| tumor-associated stroma | CD138 | 0.6994 | 0.0009812 |
| healthy glands | CD3 | 0.7847 | 0.02491 |
| inflamed stroma | CD3 | 0.7847 | 0.02491 |
| rest | T-bet | 0.9149 | 2.833e-06 |

Shared winners are included once in the **primary** PCA: original R/G/B plus CD138, CD3, T-bet (6 variables). The literal one-column-per-class sensitivity uses eight variables and repeats CD138 and CD3. RGB-only PCA is the control. These variants, component order, signs and scaling are fixed before scoring; they are not selected by accuracy.

## PCA and RGB encoding

RGB starts as original uint8/255; marker inputs are native-aligned sigmoid activations. Each variable is centered and scaled to population SD 1 over coarse foreground pixels on this ROI. Exact full SVD keeps three PCs, without whitening or class weighting. Sign convention: largest absolute component loading is positive. PC1/PC2/PC3 map to R/G/B after independent P1–P99 linear scaling over the same coarse support, clipping and rounding to uint8. Per-PC encoding rescales relative amplitudes; variance fractions below refer to the unscaled PC scores. The same native RGB composite is reused for all three tile widths.

| Variant | Variables | PC1 variance | PC2 variance | PC3 variance | Total retained |
|---|---:|---:|---:|---:|---:|
| pca_markers_unique | 6 | 56.691% | 37.511% | 2.577% | 96.779% |
| pca_markers_per_class | 8 | 60.798% | 32.712% | 3.818% | 97.328% |
| pca_rgb_only | 3 | 96.600% | 2.569% | 0.831% | 100.000% |

Full input means/SDs, component weights, centering, sign rule, explained variance and RGB transforms are saved in the variant JSON files and `pca_loadings.csv`; raw native PC score arrays are saved on Chiamaka. PCA is fitted to this same ROI (transductive), not an independently trained feature extractor.

## Segmentation

Each PCA RGB is passed as `image` to `smooth_tile_boundaries`, with no `feature_image`, using unchanged normal RGB/Lab Bora and the previous fixed configuration (2× working downsample). Fine expert labels are loaded only after all predictions have been written. Boundary F1 tolerance is 4 µm. The previous pairwise method is context and uses a different experimental wrapper; original H&E and all PCA variants share the same multiclass path.

| Tile px | Appearance / method | Macro IoU | Macro Dice | Boundary F1 | ASSD µm |
|---:|---|---:|---:|---:|---:|
| 56 | coarse | 0.7819 | 0.8742 | 0.1940 | 12.777 |
| 56 | bora_he_rgb | 0.8150 | 0.8954 | 0.3042 | 11.372 |
| 56 | previous_pairwise_markers | 0.7916 | 0.8804 | 0.2432 | 12.152 |
| 56 | pca_markers_unique | 0.8011 | 0.8864 | 0.2644 | 11.872 |
| 56 | pca_markers_per_class | 0.7991 | 0.8849 | 0.2607 | 11.580 |
| 56 | pca_rgb_only | 0.8012 | 0.8867 | 0.2789 | 11.876 |
| 90 | coarse | 0.7220 | 0.8341 | 0.1463 | 16.314 |
| 90 | bora_he_rgb | 0.7476 | 0.8515 | 0.2037 | 15.856 |
| 90 | previous_pairwise_markers | 0.7539 | 0.8553 | 0.2505 | 13.757 |
| 90 | pca_markers_unique | 0.7483 | 0.8517 | 0.2267 | 15.018 |
| 90 | pca_markers_per_class | 0.7456 | 0.8497 | 0.2279 | 15.458 |
| 90 | pca_rgb_only | 0.7335 | 0.8416 | 0.2127 | 15.787 |
| 216 | coarse | 0.5184 | 0.6486 | 0.1555 | 24.277 |
| 216 | bora_he_rgb | 0.5515 | 0.6845 | 0.1794 | 19.595 |
| 216 | previous_pairwise_markers | 0.5464 | 0.6792 | 0.2122 | 19.037 |
| 216 | pca_markers_unique | 0.5550 | 0.6900 | 0.1494 | 19.949 |
| 216 | pca_markers_per_class | 0.5469 | 0.6829 | 0.1408 | 19.658 |
| 216 | pca_rgb_only | 0.5488 | 0.6819 | 0.1860 | 18.661 |

## Resolution and scope

At the 90-pixel grid, primary PCA + markers gives macro IoU **0.7483** and
boundary F1 **0.2267**, versus **0.7476 / 0.2037** for original H&E Bora.
The marker-enhanced PCA is also better than the RGB-only PCA control
(**0.7335 / 0.2127**), but remains below the previous interface-specific
marker method (**0.7539 / 0.2505**). The literal eight-column variant is
**0.7456 / 0.2279**, so duplicating shared winners does not improve overlap.

The effect is not consistent across scales. At 56 pixels primary PCA loses
both overlap and boundary F1 against H&E. At 216 pixels it improves macro
IoU slightly (**0.5550 vs 0.5515**) but lowers boundary F1 (**0.1494 vs
0.1794**), including a lower F1 than the coarse mask at that tolerance.
Retaining most input variance therefore does not guarantee better tissue
boundaries. These are descriptive one-ROI comparisons.

Native image is a tissue ROI, not a WSI: 1224 × 1166 px at provider-reported approximately 0.5 µm/px, field 612 × 583 µm. CNN inference was at 0.25 µm/px and mapped back to the native frame. PCA changes features only; it does not resize the image. Bora's separate 2× working grid is approximately 1 µm/px. Optical resolving power, objective and numerical aperture are unknown. Every RGB OME-TIFF carries native physical pixel size. Expert masks remain unchanged; code 0 is ignored, rest is scored.

This is a controlled boundary reconstruction from expert-derived coarse labels on one ROI. Tile-level marker statistics are exploratory because neighboring tiles are correlated. Marker activations are virtual, with no measured protein truth on this TIGER image. Shared markers can distinguish two classes by different levels, and low activation can identify rest. PCA maximizes variance, not tissue separation; a useful marker signal can be lost when retaining only three PCs. No clinical, patient-level or general superiority claim follows from this pilot.

## Reproduce

`OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:. python experiments/gigatime_flash/pca_rgb.py --source <checked-cnn-results> --group-results <group-specific-results> --output <new-output>`

Sources: [PCA implementation and variance definition](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html); [TIGER](https://tiger.grand-challenge.org/Data/); [GigaTIME input sampling](https://github.com/prov-gigatime/GigaTIME/issues/4).

## Saved outputs and checks

- Chiamaka: `/mnt/sata_ssd/Bora-benchmark/results/gigatime-pca-rgb-2026-10-16/`.
- Local review: `/Users/stefano/Documents/ChatGPT/Bora/benchmark/results/gigatime-pca-rgb-2026-10-16/`.
- [Full reproduction guide](../experiments/gigatime_flash/README.md).

`pca_rgb_and_components.png` shows the primary input, RGB-only control and
individual PCs; `pca_loadings_and_variance.png` shows exact component weights
and variance fractions. `pca_segmentation_overlays_tile_90.png` shows expert,
tiling-mask and refined masks over original H&E at 35% opacity.
`pca_selected_markers_and_dapi.png` includes DAPI for nuclear context; DAPI
is not a PCA feature. Native RGB OME-TIFFs, per-variant JSON and all masks and
score tables are available locally. Dense floating PC arrays stay on Chiamaka.

Thirty-two focused tests pass. Independent audits check the saved transform,
source hashes, RGB encoding, native frame, OME physical pixel size, all 18
macro IoU rows and the primary 90-pixel boundary F1. Generated images, masks
and numeric score arrays remain outside Git.
