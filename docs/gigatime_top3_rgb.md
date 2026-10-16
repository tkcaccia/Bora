# Top-three virtual-marker RGB experiment

Exploratory controlled reconstruction of one expert-annotated TIGER breast tissue ROI.

## Marker statistics

Selection: central-half tile means on the 90-pixel coarse grid, all five groups; 169 tiles. No fine reference used for selection.

| RGB | Marker | Kruskal H | Nominal p | BH-adjusted nominal p | Spatial-shift tail fraction |
|---|---|---:|---:|---:|---:|
| R | CD3 | 95.5996 | 8.496e-20 | 1.359e-18 | 0.0059 |
| G | CD4 | 91.7688 | 5.542e-19 | 4.434e-18 | 0.0059 |
| B | PHH3-B | 90.5157 | 1.023e-18 | 5.457e-18 | 0.0059 |

BH adjusts across all 16 eligible proteins. Nominal tests assume independent observations, which adjacent tiles do not satisfy. All 169 toroidal label shifts are a spatial sensitivity check with artificial wraparound, not patient-level confirmation. Pairwise tests across all 160 marker/group pairs are separately BH adjusted in the CSV.

Primary RGB scaling: independent linear P1–P99 over coarse foreground, minimum activation range 0.01, clipping and rounding to uint8. Rank 1/2/3 goes to R/G/B. The same image and default Bora configuration are used for all tile sizes. Fixed [0,1] scaling is reported as a sensitivity check; no output-score-based selection of marker, color, scaling or parameters.

## Segmentation

All values on the unchanged native reference; ignore code 0 only. Boundary F1 tolerance 4 µm.

| Tile px | Method | Macro IoU | Macro Dice | Boundary F1 | ASSD µm |
|---:|---|---:|---:|---:|---:|
| 56 | coarse | 0.7819 | 0.8742 | 0.1940 | 12.777 |
| 56 | bora_he_rgb | 0.8150 | 0.8954 | 0.3042 | 11.372 |
| 56 | bora_previous_six | 0.7926 | 0.8810 | 0.2512 | 11.823 |
| 56 | bora_top3_rgb | 0.7843 | 0.8757 | 0.2267 | 12.123 |
| 56 | bora_top3_raw_probability_rgb | 0.7811 | 0.8734 | 0.2359 | 12.017 |
| 90 | coarse | 0.7220 | 0.8341 | 0.1463 | 16.314 |
| 90 | bora_he_rgb | 0.7476 | 0.8515 | 0.2037 | 15.856 |
| 90 | bora_previous_six | 0.7448 | 0.8493 | 0.2370 | 13.757 |
| 90 | bora_top3_rgb | 0.7425 | 0.8478 | 0.2161 | 14.244 |
| 90 | bora_top3_raw_probability_rgb | 0.7410 | 0.8466 | 0.2343 | 13.879 |
| 216 | coarse | 0.5184 | 0.6486 | 0.1555 | 24.277 |
| 216 | bora_he_rgb | 0.5515 | 0.6845 | 0.1794 | 19.595 |
| 216 | bora_previous_six | 0.5547 | 0.6890 | 0.2096 | 18.705 |
| 216 | bora_top3_rgb | 0.5449 | 0.6778 | 0.2107 | 19.269 |
| 216 | bora_top3_raw_probability_rgb | 0.5443 | 0.6782 | 0.2059 | 19.090 |

## Interpretation

At 90-pixel tiles, primary three-marker RGB improves boundary F1 from 0.2037 to 0.2161 versus H&E Bora but reduces macro IoU from 0.7476 to 0.7425. It also trails the previous six-feature approach at this tile width. At 216 pixels it improves boundary F1 while lowering IoU; at 56 pixels it is worse than H&E Bora on both. This does not establish an overall segmentation benefit.

The top three have strongly redundant tile signals. Their high omnibus significance is driven by differences involving inflamed stroma and rest, and does not establish separation of invasive tumor from tumor-associated stroma. Selecting markers for a particular adjacent tissue pair, with a redundancy constraint, is a future experiment rather than a result of this run.

| Selected marker pair | Tile Pearson r | Tile Spearman rho |
|---|---:|---:|
| CD3 / CD4 | 0.9939 | 0.9754 |
| CD3 / PHH3-B | 0.9861 | 0.9409 |
| CD4 / PHH3-B | 0.9915 | 0.9801 |

| Marker | Tumor vs tumor-associated stroma nominal p | BH-adjusted p across all 160 pairs |
|---|---:|---:|
| CD4 | 0.1341 | 0.1450 |
| CD3 | 0.1776 | 0.1870 |
| PHH3-B | 0.0711 | 0.0813 |

## Resolution and annotation

Native 1224 × 1166 px tissue ROI (not WSI), approximately 0.5 µm/px; field of view 612 × 583 µm. GigaTIME CNN inference uses 0.25 µm/px and returns maps to native coordinates. Upsampling adds no optical information. Objective, numerical aperture and optical resolving power are unknown. Bora runs at its fixed 2× downsample (approximately 1 µm/working pixel). Synthetic OME-TIFF carries native physical pixel size; JSON records channel identities and intensity transforms.

Labels: 1 invasive tumor, 2 tumor-associated stroma, 3 healthy glands, 4 inflamed stroma, 5 rest, 0 ignored. Expert masks are manual; coarse tiling-mask labels were derived from them, so this tests boundary reconstruction rather than independent classification. Protein maps are virtual predictions, not measured stains. DAPI is shown for context and excluded from protein ranking.

## Reproduction and sources

Run `PYTHONPATH=src python experiments/gigatime_flash/top3_rgb.py --source <checked-cnn-results> --output <new-output>` with two BLAS/OpenMP threads. Input/model/map hashes, parameters, all marker statistics, group summaries and full per-class segmentation scores are saved in JSON/CSV. Fine labels are opened only after predictions are fixed.

Sources: [TIGER](https://tiger.grand-challenge.org/Data/); [GigaTIME sampling clarification](https://github.com/prov-gigatime/GigaTIME/issues/4); [Kruskal-Wallis](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html); [BH procedure](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html).
