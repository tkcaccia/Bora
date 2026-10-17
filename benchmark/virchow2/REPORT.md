# Twenty Virchow2 experiments for Bora

Completed on Chiamaka, 16 October 2026. Twenty distinct interventions were run on the same TIGER ROI; this is not a decoder grid.

Best macro IoU: **09 Scale-persistent evidence**, 0.7673; H&E Bora control 0.7476; coarse mask 0.7220.

## Controls and class tradeoffs

| Control | Macro IoU | Weakest class IoU | Boundary F1, 8 µm | HD95 µm |
|---|---:|---:|---:|---:|
| Coarse tiles | 0.7220 | 0.5828 | 0.3220 | 45.65 |
| H&E Bora | 0.7476 | 0.6026 | 0.3749 | 51.44 |
| Plain selected-feature RGB Bora | 0.7529 | 0.5790 | 0.4623 | 51.94 |
| H&E barrier wrapper | 0.7249 | 0.5765 | 0.3802 | 47.50 |

Best average method 09 improves macro IoU by 1.97 percentage points over H&E Bora and 1.44 points over the plain selected-feature RGB control. This is **not an improvement in every class**.

| Class | H&E Bora IoU | Best average method IoU | Change, percentage points |
|---|---:|---:|---:|
| invasive tumor | 0.8862 | 0.9281 | +4.19 |
| tumor-associated stroma | 0.6975 | 0.7852 | +8.77 |
| healthy glands | 0.6026 | 0.5992 | -0.34 |
| inflamed stroma | 0.7042 | 0.6687 | -3.55 |
| rest | 0.8475 | 0.8553 | +0.78 |

The best average method gains mainly on tumor and stroma, while healthy-gland and inflamed-stroma IoU decrease. It also has a lower class-agnostic boundary F1 than some other ideas. Selection among methods therefore depends on the tissue classes and boundary outcomes of interest. One ROI does not establish a reliable winner or statistical significance. The best-of-twenty method is selected on this same evaluation image, not on an independent model-selection cohort.

Provider native-to-compact mask mapping: 1→1 tumor, 2→2 stroma, 4→3 healthy glands, 6→4 inflamed stroma, 7→5 rest. Code 0 is ignored. The expert mask is remapped, not newly annotated.

## Goal failures and classifier sensitivity

Idea 01 reduced the mean window feature-fitting loss from 4.03 to 2.14, but the stitched output remained noise-like. It is a failed tissue reconstruction at the tested budget, not a recovered slide.

Idea 02 preserved approximately 98.6% of the stitched-grid classifier decisions, but its soft mask averaged 0.861 and no pixel fell below a mask weight of 0.5. It did not identify a small retained evidence region. Do not interpret the contrast-scaled heatmap as successful sparse localization.

Idea 03 changed most stitched-grid classifier decisions with a mean RGB change of about 8/255; about 57% reached the intended runner-up class. These diagnostics include the padded border tokens. The almost unchanged H&E appearance and negligible Bora benefit illustrate classifier sensitivity, not a biological tissue transformation.

## Scope and physical resolution

One breast tissue ROI, 1224 × 1166 pixels, provider-reported approximately 0.5 µm/px, field 612 × 583 µm. Not a full WSI. Optical resolving power, scanner and objective are unknown. Virchow2 windows: 224 pixels/112 µm; spatial token: 14 pixels/7 µm. Coarser-scale windows cover approximately 224 and 448 µm. No resampling improves optical resolution.

Five scored classes: invasive tumor, tumor-associated stroma, healthy glands, inflamed stroma, rest. Expert code 0 is ignored. Rest is scored.

## Fair comparison and limits

All predictions and their SHA-256 hashes were frozen before this pipeline opened the fine reference. Coarse labels are majority tiles of expert annotations, so this is controlled boundary reconstruction, not an independent tissue-classification benchmark. Model selection uses coarse tile interiors only. No fine-mask optimization or tuning was performed. Single ROI; no patient-level inference or significance claim.

H&E control and ordinary input substitutions use Bora's internal balanced local wand with a 2× working grid (~1 µm/px), tile width 90 native pixels, fixed settings and fixed coarse exterior. Ideas 06, 13 and 17 explicitly change the interface wrapper/decision rule. Idea 13 also has an H&E-gradient watershed control. This is an exploratory comparison, not a claim that every method runs an identical algorithm.

Class features were selected from 169 coarse tile interiors: 74 tumor, 37 stroma, 12 healthy glands, 12 inflamed stroma, 34 rest. Separate initial spatial-group validation refits selection inside each fold; overlapping encoder context and one ROI limit independence. Reported fine-mask scores evaluate refinement on this ROI, not generalization. Latent dimension IDs are not biomarkers.

## Results

| ID | Idea | Macro IoU | Δ H&E | Weakest IoU | Boundary F1, 8 µm | HD95 µm |
|---|---|---:|---:|---:|---:|---:|
| 01 | Selective feature inversion | 0.7244 | -0.0232 | 0.5863 | 0.3411 | 46.51 |
| 02 | Minimal visual evidence | 0.7477 | +0.0001 | 0.5988 | 0.3850 | 50.22 |
| 03 | Nearest-class counterfactual | 0.7470 | -0.0006 | 0.5990 | 0.3819 | 51.28 |
| 04 | Class-specific concepts | 0.7568 | +0.0092 | 0.5938 | 0.4301 | 49.99 |
| 05 | Worst-class colour geometry | 0.7657 | +0.0181 | 0.5676 | 0.4632 | 49.16 |
| 06 | Boundary-specific experts | 0.7499 | +0.0023 | 0.5601 | 0.4442 | 49.30 |
| 07 | Colour-explained subspace removal | 0.7420 | -0.0056 | 0.5645 | 0.4578 | 51.54 |
| 08 | Stain-perturbation invariance | 0.7593 | +0.0117 | 0.5852 | 0.4455 | 50.60 |
| 09 | Scale-persistent evidence | 0.7673 | +0.0197 | 0.5992 | 0.4474 | 50.01 |
| 10 | Context chooses, detail places | 0.7602 | +0.0126 | 0.5813 | 0.4617 | 51.29 |
| 11 | Context-disagreement gating | 0.7571 | +0.0095 | 0.6013 | 0.4189 | 49.00 |
| 12 | Geodesic evidence transport | 0.7487 | +0.0011 | 0.5727 | 0.4164 | 49.52 |
| 13 | Semantic boundary barriers | 0.7301 | -0.0175 | 0.5786 | 0.3405 | 46.52 |
| 14 | Native-image superpixels | 0.7557 | +0.0081 | 0.5830 | 0.4545 | 49.95 |
| 15 | Object relationships | 0.7552 | +0.0076 | 0.5768 | 0.4606 | 51.71 |
| 16 | Soft tissue unmixing | 0.7555 | +0.0079 | 0.5669 | 0.4417 | 51.34 |
| 17 | Topology-aware refinement | 0.7593 | +0.0117 | 0.5783 | 0.4695 | 51.44 |
| 18 | Feature-matched tissue quilting | 0.7383 | -0.0093 | 0.5811 | 0.3868 | 49.78 |
| 19 | Predictable latent structure | 0.7585 | +0.0109 | 0.5880 | 0.4505 | 50.98 |
| 20 | Feature-guided wavelet detail | 0.7479 | +0.0003 | 0.6035 | 0.3760 | 51.04 |

Boundary F1/HD95 here compare internal boundary locations without class identity; always interpret them alongside class-specific IoU/Dice. Distances use 0.5 µm/pixel.

## Reconstruction and evidence failures

Idea 01 performed genuine frozen-encoder pixel optimization from two noise starts, 100 overlapping windows per start, 32 steps/window. It remains noise-like; the available optimization budget did not recover useful histology. This failure does not prove that a better inverse or a learned image prior cannot work.

Idea 02 uses a soft retained-image mask, not a hard gland/tumor annotation. Its first per-location preservation objective was overly restrictive/unstable; an explicitly logged second preset preserves aggregate evidence for each present class, retains the best iterate, and uses a lower learning rate. Neither claims a globally minimal solution. Original image pixels supply all retained anatomy.

Idea 03 is an intentional counterfactual classifier intervention. Window-level feature changes do not imply biologically valid tissue changes, and overlap stitching can reduce the counterfactual effect. Both window metrics and re-encoded stitched-image metrics are saved.

Idea 18 records every donor patch, disjoint target/donor encoder-context boxes, and exact donor coordinates for every output pixel. Donors must lie wholly inside the native image. The mosaic is retrieval, not generation.

Feature RGB panels show model evidence or structure. H&E detail in 02, 03, 11, 14, 15, 18 and 20 is explicitly supplied by the original image. Candidate nuclei/lumens in 15 are unvalidated image-derived objects, not measured DAPI or manual annotations. Mixture weights in 16 are model assignments, not measured tissue fractions.

## Files and reproducibility

Open `gallery.html` for adjustable 35%-default overlays of expert, coarse and refined masks, all 20 representations, full-resolution files, fixed 224-pixel crops and diagnostics. `scores.csv` and `results.json` contain every result; numbered folders contain actual input PNG, output TIFF and method details. `PREDICTIONS_FROZEN.json` records pre-evaluation hashes.

Code: `experiments/virchow2_inverse/`. Model: [paige-ai/Virchow2](https://huggingface.co/paige-ai/Virchow2), pinned revision `3158645804b69e3f3bc4439d4116edddf0840a72`. Weights, data, outputs and intermediate arrays are not committed to Git.
