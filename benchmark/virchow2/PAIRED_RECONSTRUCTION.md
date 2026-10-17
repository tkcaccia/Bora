# Virchow2 paired reconstruction proof of concept

## Question and outcome

Can the final spatial tokens regenerate H&E, and can a bounded edit improve architectural contrast without changing anatomy? Both outputs use the same learned RGB decoder. This is a separate paired experiment alongside the twenty earlier ideas.

Full reconstruction passed the predeclared fidelity screen: **False**. Selective preprocessing supported by all exploratory criteria: **False**.

A failed screen means this decoder and training budget did not establish a faithful preprocessing method. It does not prove that the tokens lack the missing information, or that another decoder could not recover it.

## Paper and implementation

The attached paper, *Computationally Efficient Pathology Segmentation using Knowledge Distillation from Foundation Models* (Lv et al., arXiv:2609.03947v1, 3 September 2026), learns a segmentation decoder over frozen or LoRA-adapted Virchow2 spatial tokens (§4.2.1, Fig.2). It does not demonstrate RGB reconstruction. Our RGB decoder and latent-edit experiment are new exploratory adaptations, not reproduction of a published inverse.

The frozen, SHA-verified paige-ai/Virchow2 checkpoint supplies 256 final spatial tokens (16×16), each with all 1280 channels. CLS and four registers are excluded. Each 14 px token covers7µm. The decoder projects 1280→256, applies three residual convolution blocks, projects to 588 channels, and pixel-shuffles 14× to 224×224 RGB. There are no source RGB pixels, raw-pixel residuals, image coordinates or reference masks in decoder inputs. No extra generative model is used.

Training uses RGB L1 +0.2 gradient L1 +0.15(1-local SSIM), AdamW, train-only token normalization and validation checkpoint selection. Encoder weights are frozen. No target-case fine-tuning, target-RGB inversion optimization or test-set checkpoint selection is performed.

Retained epoch **98** of **100**, checkpoint SHA256 `54ee0519d721d1d987bbb8f3ab4b58acfcdc5c545d0326f53d3e54ae4e0a2b34`. See TRAIN_PROTOCOL.json and training_history.json.

## Data and physical scale

TIGER ROI release: https://zenodo.org/records/6014422. The provider reports images and annotations at **0.5µm/pixel**. These are ROI images, not complete WSIs. Microscopy optical resolving power and objective magnification are unreported; sampling must not be interpreted as either. The selected ROI is 1224×1166px (612×583µm); model crops 224×224px (112×112µm), with no resolution change. TIGER's BCSS-derived ROI subset here uses its released 0.5µm/px sampling, rather than an independent original BCSS mirror's sampling.

| Decoder split | Native224px tiles | Source-case proxies | Source ROIs |
|---|---:|---:|---:|
| train | 3920 | 152 | 241 |
| validation | 216 | 21 | 36 |
| test | 216 | 21 | 40 |

All source identifiers beginning 114, including 114S and 114B, are excluded from every decoder split. TCGA cases use the 12-character patient identifier; other source cases use the conservative leading numeric or TC_Sxx_Pxxxxxx prefix. These are case/patient proxies, not independently verified patient mappings. Related or overlapping ROIs remain within one split. See corpus.json for every crop coordinate, split and image, CACHE_READY.json for source hashes, and DATASET_SCOPE.json for exact scope. No masks are needed for decoder training. Virchow2 pretraining overlap with public source cases cannot be excluded.

## Same-tile paired intervention

Focal direct window index36, native top-left(x,y)=(672,336). Selected by maximum coarse-class entropy among complete, unpadded windows with at least three coarse classes. The same native224px image is used for both reconstructions; this selection does not read fine annotations.

The full control changes no token features. The primary selective control amplifies an equal-class mean contrast subspace by 1.25×, learned from the existing 64-dimensional balanced-MRMR pool using only 169 coarse 90 px tile observations. The contrast basis has rank ≤ 4 for five classes. All 1,216 unselected coordinates remain exactly unchanged, and all 1,280 channels remain present. Before capping, the edit lies in the class-contrast subspace; the elementwise cap can introduce a bounded orthogonal component within the 64 selected coordinates. Each edited component is clipped to ±0.5 training standard deviations and the total edit to 5% of the token's normalized norm. Fixed0.75× suppression and 1.5× amplification are sensitivity controls, not new members of the earlier 20-idea study. The decoder is identical and frozen for all four outputs. Latent coordinates are not protein markers.

The full ROI is decoded from the same 100 original native windows and Hann-weighted overlap-added, then cropped to its native frame. Direct focal-tile results are saved separately so overlap averaging does not conceal single-tile fidelity. No source image detail is blended into any reconstruction.

## Reconstruction and anatomy checks

The full fidelity screen was frozen before scoring: RGB SSIM ≥ 0.8 on both the full ROI and direct focal tile, candidate-nucleus matching F1 ≥ 0.8, median matched displacement ≤ 2µm. These exploratory thresholds are not clinical validation criteria. Nuclei are automatic hematoxylin-based candidates, with a threshold fixed from original H&E and one-to-one matching within 2µm. They are not expert nucleus masks and their errors can distort the comparison. Hematoxylin-gradient correlation provides an additional structural check.

| Reconstruction | ROI SSIM | ROI PSNR dB | H-gradient correlation | Nucleus proxy F1 | Direct-tile SSIM |
|---|---:|---:|---:|---:|---:|
| full | 0.7325 | 24.38 | 0.8614 | 0.7288 | 0.7728 |
| selective | 0.7308 | 24.36 | 0.8605 | 0.7267 | 0.7712 |
| suppression | 0.7319 | 24.33 | 0.8603 | 0.7207 | 0.7723 |
| stronger | 0.7304 | 24.33 | 0.8600 | 0.7297 | 0.7697 |

On **216 held-out test tiles from21 case proxies**, case-macro mean SSIM is **0.7433**, PSNR **24.34dB**. Case medians and quartiles are in results.json. The gallery shows the worst,25th,50th,75th and best SSIM test examples, with exact indices, rather than only attractive examples.

## Architectural contrast and Bora

Contrast is measured in decoded Lab pixels as class mean separation divided by pooled within-class variance, using the same central-half coarse tiles. The minimum over all ten class pairs tests the weakest distinction. This is a descriptive contrast proxy, not statistical significance or biological identity. The same fixed internal Bora wand configuration refines the same coarse labels for each input; exterior coarse background is held fixed.

| Input | Worst-pair Lab contrast | Mean-pair Lab contrast |
|---|---:|---:|
| original | 0.0118 | 0.4290 |
| full | 0.0415 | 0.4149 |
| selective | 0.0368 | 0.3855 |
| suppression | 0.0483 | 0.4459 |
| stronger | 0.0387 | 0.3926 |

| Bora input/control | Macro IoU | Weakest class IoU | Boundary F1 (8µm) |
|---|---:|---:|---:|
| Coarse tiling mask | 0.7220 | 0.5828 | 0.3220 |
| Original H&E + Bora | 0.7476 | 0.6026 | 0.3749 |
| Full reconstruction + Bora | 0.7445 | 0.6020 | 0.3726 |
| Selective 1.25× + Bora | 0.7446 | 0.6022 | 0.3722 |
| Suppression 0.75× + Bora | 0.7443 | 0.6013 | 0.3718 |
| Selective 1.5× + Bora | 0.7447 | 0.6022 | 0.3720 |

| Class | Original H&E Bora | Full reconstruction Bora | Selective Bora |
|---|---:|---:|---:|
| Invasive tumor | 0.8862 | 0.8825 | 0.8829 |
| Tumor-associated stroma | 0.6975 | 0.6886 | 0.6890 |
| Healthy glands | 0.6026 | 0.6020 | 0.6022 |
| Inflamed stroma | 0.7042 | 0.7083 | 0.7080 |
| Rest | 0.8475 | 0.8411 | 0.8409 |

The fine expert tissue reference was opened only after all reconstructions and Bora outputs were frozen and SHA-verified. These are expert tissue annotations; code 0 is ignored. Coarse90px tiles were originally derived from these annotations, so this is a controlled boundary-refinement experiment, not an independently predicted coarse-label benchmark. This ROI has also been examined in the previous20 experiments. There are no independent test cases for the selective edit or segmentation claim, no patient-level significance tests, and no claim of universal improvement.

## Decision

- full_reconstruction_passed_fidelity_screen: **False**
- selective_retains_full_nuclei_proxy: **True**
- selective_improves_worst_class_colour_contrast: **False**
- selective_bora_beats_original_he: **False**
- preprocessing_proof_of_concept_supported: **False**

Only if full fidelity, selective anatomy retention, weakest-pair contrast and Bora improvement all pass should this be carried forward as a candidate preprocessing method. Even then it requires new cases, independently supplied coarse labels, expert morphology review and externally tested gains. Otherwise retain original H&E for Bora and treat the paired output as a diagnostic reconstruction study.

## Files and earlier experiments

- gallery.html: paired images, editable mask opacity, all fixed controls, held-out examples.
- paired_tile.png: same native224px tile, original/full/selective.
- paired_roi.png: full native tissue field, original/full/selective.
- segmentation.png: original expert/coarse masks and Bora overlays at 35% opacity.
- full.png, selective.png, suppression.png, stronger.png: native-size outputs.
- bora.*.tif: compact labels1tumor,2stroma,3healthy glands,4inflamed stroma,5rest;0ignore.
- PROTOCOL.json, PREDICTIONS_FROZEN.json, results.json: settings, frozen hashes and complete metrics.
- Remote decoder.pt, tokens.npy and rgb.npy: model and numerical training cache; not committed to Git.
- Earlier20 experiments: ../virchow2-inverse-2026-10-16/gallery.html and REPORT.md; outputs remain unchanged.

Code: experiments/virchow2_paired/. Reproduction commands and environment are documented in that folder's README.md.

## Additional full-token consistency diagnostic

After the primary experiment, both focal reconstructions were optimized for 400 steps against all 1280 target spatial channels, initialized solely from their respective decoder images. Both use the same FP32 forward/gradient computation, learning-rate schedule and weak TV/initialization regularizers. No original RGB pixels or fine masks enter the optimization. This is a single-tile diagnostic, not another full-ROI Bora experiment and not a replacement for the primary frozen comparison.

Two initial unscaled FP16 optimization pilots were unstable; their protocols, images and logs are preserved under diagnostics/. The final diagnostic uses FP32 to avoid small input-gradient underflow. The change of precision was motivated by feature-objective instability, not fine-mask scoring or a search over visual attractiveness. The actual-model finite-difference audit and FP16/FP32 gradient comparison are saved in GRADIENT_AUDIT.json.

| Focal reconstruction | RGB SSIM | PSNR dB | H-gradient correlation | Nucleus proxy F1 |
|---|---:|---:|---:|---:|
| decoder.full | 0.7728 | 23.70 | 0.8581 | 0.8018 |
| consistency.full | 0.7766 | 23.82 | 0.8732 | 0.7783 |
| decoder.selective | 0.7712 | 23.62 | 0.8571 | 0.7793 |
| consistency.selective | 0.7749 | 23.74 | 0.8720 | 0.7763 |

full: normalized token MSE 0.77778 initially, 0.02889 at the final iterate; best objective 0.02896 at step399. Optimization uses only token loss to retain an iterate.

selective: normalized token MSE 0.77004 initially, 0.03020 at the final iterate; best objective 0.03027 at step399. Optimization uses only token loss to retain an iterate.

Closer token matching with little pixel-level improvement would show that latent agreement alone is not a fidelity certificate. It would not establish that accurate inversion is impossible. See paired_tile_consistency.png, CONSISTENCY_PROTOCOL.json and consistency_results.json.
