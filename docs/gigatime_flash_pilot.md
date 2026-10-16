# GigaTIME-Flash-driven Bora: annotated TIGER pilot, 16 October 2026

**Rejected pilot: DAPI target-image QC failed.** This report is retained for
traceability; its virtual-marker results must not support claims about Bora's
performance. The inferred input scale below also omitted the authors' 556→512
resize. The replacement uses the original CNN at 0.25 µm/pixel and includes
paired-reference checks: [current report](gigatime_cnn_validation.md).

**Finding:** virtual-marker refinement improves over the coarse tile masks but
does not consistently outperform RGB-driven adaptive Bora. A boundary-distance
improvement occurs at tile width 90; overlap and boundary F1 generally favour
the RGB baseline. This single-ROI result does not justify replacing the RGB
method. Code, exact resolution handling and reproducibility instructions are
in [the experiment guide](../experiments/gigatime_flash/README.md).

The image is the non-TCGA TIGER tissue ROI
`114S_[33188, 13944, 34412, 15110]`, not a full WSI. Expert semantic labels
cover invasive tumor, tumor-associated stroma, healthy glands, inflamed stroma,
and `rest`. Unannotated/excluded label 0 is omitted from scoring.
The coarse masks were generated from the reference by `tiling-mask-cpp`.
Results therefore address controlled boundary reconstruction, with known
coarse-label information, rather than end-to-end unsupervised segmentation.

## Physical scale

Native width × height is 1224 × 1166 pixels at a **provider-reported**
approximately 0.5 µm/pixel (field of view approximately 612 × 583 µm).
Microscope objective, numerical aperture and optical resolving power remain
unknown. [TIGER release](https://zenodo.org/records/6014422).

Two fixed inference conditions were compared: native sampling and upsampling
to the approximately 0.230 µm/pixel registered-grid spacing inferred from the
original GigaTIME methods. Flash-specific H&E physical calibration is
**unverified**. Upsampling supplies no additional optical detail. The
upsampled image is 2658 × 2532 pixels at effective x/y spacings of
0.2302483/0.2302528 µm/pixel. Dense predictions return to the original frame;
reference masks are unchanged. The model input/output windows are 256 × 256
pixels, not a microscopy-resolution specification.
[Original GigaTIME](https://doi.org/10.1016/j.cell.2025.11.016),
[Flash model](https://huggingface.co/prov-gigatime/gigatime-flash).

## All comparisons

The same default adaptive-wand parameters were used throughout. No fine-mask
tuning or selection of a best sampling condition was performed. Boundary F1
below uses a 4 µm tolerance; lower ASSD is better. The three tile widths are
three degradations of the **same image**, not three independent cases.

| Tile width (px) | Appearance | Macro IoU | Macro Dice | Boundary F1 | ASSD (µm) |
|---:|---|---:|---:|---:|---:|
| 56 | Coarse input | 0.7819 | 0.8742 | 0.1940 | 12.7766 |
| 56 | RGB Bora | 0.8150 | 0.8954 | 0.3042 | 11.3719 |
| 56 | Virtual markers, native 0.5 µm/px | 0.7903 | 0.8780 | 0.2654 | 11.3453 |
| 56 | Virtual markers, exploratory ≈0.230 µm/px | 0.7986 | 0.8844 | 0.2707 | 11.2089 |
| 90 | Coarse input | 0.7220 | 0.8341 | 0.1463 | 16.3139 |
| 90 | RGB Bora | 0.7476 | 0.8515 | 0.2037 | 15.8557 |
| 90 | Virtual markers, native 0.5 µm/px | 0.7327 | 0.8392 | 0.1816 | 13.6581 |
| 90 | Virtual markers, exploratory ≈0.230 µm/px | 0.7344 | 0.8416 | 0.1949 | 14.2868 |
| 216 | Coarse input | 0.5184 | 0.6486 | 0.1555 | 24.2772 |
| 216 | RGB Bora | 0.5515 | 0.6845 | 0.1794 | 19.5947 |
| 216 | Virtual markers, native 0.5 µm/px | 0.5522 | 0.6797 | 0.1510 | 20.4111 |
| 216 | Virtual markers, exploratory ≈0.230 µm/px | 0.5469 | 0.6727 | 0.1527 | 20.7126 |

## Differential channels

Selection used one central-half mean per coarse tile, at least three tiles
per class, and the largest absolute pairwise Hedges g for each channel.
The nuclear DAPI channel and the two background channels were excluded from
protein selection. No fine reference masks entered selection or refinement;
expert labels entered scoring after every prediction was fixed. The coarse
groups themselves are reference-derived, as stated above.

At width 90, the native-scale global top six were **CD4, CD3, CD11c, PD-1,
PD-L1 and CD14**. At the exploratory finer sampling they were **Caspase3-D,
CD14, Tryptase, CD16, CD4 and PD-1**. The ranking changes with sampling scale.
The class tile counts were tumor 74, stroma 37, glands 12, inflamed stroma 12,
and rest 34. Effect sizes are descriptive; these spatially correlated tiles
are not independent biological replicates.

The strongest tumor–stroma contrast was **CK** at both scales. At native
sampling, the mean predicted CK activation was 0.6756 in tumor tiles versus
0.0821 in stromal tiles (Hedges g 3.0004); at the exploratory scale it was
0.5507 versus 0.0689 (g 2.2743). These are predicted activations, not measured
protein concentrations. An important limitation is that global maximum-effect
ranking can choose several redundant immune-associated channels and omit a
channel such as CK that distinguishes a particular interface. Pair-specific
selection and an RGB/marker combination are reasonable subsequent experiments;
they were not fitted retrospectively to this expert mask.

## Requested visual review

The overlay figures place **every mask on the original H&E** at 35% opacity:
expert reference, `tiling-mask` coarse input, RGB Bora and both marker-driven
Bora outputs. All panels share coordinates, colors and 100 µm scale bars.
Reference label 0 is transparent. Separate marker overlays include **DAPI,
CK, CD4, CD14, Actin-D and Transgelin**, at both sampling conditions. Marker
opacity is `0.7 × activation` on a shared 0–1 scale. DAPI is included for
structural review despite being excluded from protein selection.

Files in the result folder:

- `overlays_tile_90.png`: requested semitransparent mask comparison.
- `marker_overlays_0.500000.png`, `marker_overlays_0.230216.png`: six virtual
  channels, including DAPI, overlaid on the unchanged original H&E.
- `scores.csv`, `results.json`: all scores, per-class metrics and provenance.
- `tile_*/marker_selection_*.json`, `marker_contrasts_*.csv`: channel rankings,
  group means, effects and individual coarse-tile observations.
- `tile_*/*.tif`: exact coarse and refined label rasters; compact-to-original
  class IDs are recorded in `results.json`.
- `inference_*/virtual_markers.native.npy` and `.model_grid.npy`: full 23-channel
  predictions, on Chiamaka only; `inference.json` documents their grids.
- `overlay_manifest.json`: transparency, channel names and alignment details.

Chiamaka:
`/mnt/sata_ssd/Bora-benchmark/results/gigatime-flash-pilot-2026-10-16/`.
Local figures, masks and metadata:
`/Users/stefano/Documents/ChatGPT/Bora/benchmark/results/gigatime-flash-pilot-2026-10-16/`.

The pinned Flash checkpoint loaded all 254 tensors strictly, including patch
convolution adapters (23,806,559 parameters). On the RTX 5060 Ti, 81 native
windows took 2.42 s and 380 upsampled windows 6.95 s, including assembly and
writing within the inference routine; checkpoint loading and downstream Bora
are outside those timings. Peak PyTorch GPU allocation was approximately
188 MiB, excluding CUDA context and other processes. Batch size was four.
The isolated environment did not alter other projects' environments.

Validation: 13 tests passed, covering existing RGB behavior, feature-driven
boundary movement, frame/finite-value validation, exclusion of non-protein
channels, coarse-tile counts and physical field-of-view preservation. The
saved overlay figures were inspected visually. No manuscript-level performance
claim is supported by this one-image pilot.
