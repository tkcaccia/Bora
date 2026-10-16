# Reference-checked GigaTIME CNN refinement

The current workflow uses **the original GigaTIME CNN**, with unmodified RGB
and input sampling of **0.25 µm/pixel**. The earlier Flash pilot is rejected
for target-image DAPI quality. This directory retains its historical name so
that all diagnostic scripts and previous runs remain traceable.

See [the CNN validation and results report](../../docs/gigatime_cnn_validation.md).
The current [group-specific RGB experiment](../../docs/gigatime_group_markers.md)
selects three candidates for each group by its weakest separation from the
other groups. Refinement uses three markers chosen separately for each tissue
pair, with a matched H&E interface control. The earlier
[global three-marker trial](../../docs/gigatime_top3_rgb.md) finds CD3, CD4 and
PHH3-B, but is dominated by inflamed-stroma/rest contrasts and does not improve
overlap over H&E Bora.
The subsequent [RGB + class-marker PCA experiment](../../docs/gigatime_pca_rgb.md)
preserves original R/G/B as input variables, adds the strongest marker per
class, and passes the first three PCs as RGB to normal multiclass Bora.
The [old Flash report](../../docs/gigatime_flash_pilot.md) is an archive of a
failed pilot, not supporting evidence for the method.

## What is checked

- Exact released CNN weights, loaded strictly; channel order is fixed to the
  [author-confirmed order](https://github.com/prov-gigatime/GigaTIME/issues/8).
- Author-confirmed preprocessing: a registered 128 µm field contains 556 pixels,
  resized to 512 pixels, then processed in 256-pixel windows. The model sampling
  is therefore **128/512 = 0.25 µm/pixel**, rather than the registration-grid
  spacing of 128/556. [Author clarification](https://github.com/prov-gigatime/GigaTIME/issues/4).
- The exact production overlap-add workflow is evaluated on 45 released paired
  H&E/mIF examples, after five separate patches were used for diagnostic work.
- All channels are screened against the paired reference masks. An exploratory
  feature-eligibility rule requires mean 8-pixel-window Pearson correlation
  at least 0.30 in at least 20 variable reference patches. DAPI and the two
  background channels are excluded from protein-feature selection. CD20,
  Tryptase, Actin-D and Transgelin fail this screen and cannot drive this run.
- Cached predictions are reused only if the original RGB, model checkpoint,
  preprocessing code, architecture, configuration and sampling fingerprint
  match. Changing an inference implementation cannot silently reuse old maps.

These establish reference-data performance and reproducible implementation.
They do not establish measured protein expression or cell-level accuracy on
TIGER, which has expert tissue masks but no paired immunofluorescence.
Reference patches also lack a released patient-level mapping for this audit;
45 patches must not be described as 45 independent patients. No model is
retrained and no tissue-reference masks are used to choose marker channels.

## Scale and frame

TIGER 114S is a **tissue ROI**, 1224 × 1166 native pixels at provider-reported
approximately 0.5 µm/pixel. Its field of view is approximately 612 × 583 µm.
The microscope objective, numerical aperture and optical resolving power are
unknown. Upsampling to 2448 × 2332 pixels at 0.25 µm/pixel cannot recover lost
optical detail. Native image dimensions, optical uncertainty, model sampling,
256 × 256 input/output dimensions, and Bora's separate 2× working downsample
are recorded independently.

Inference uses RGB/ImageNet normalization, 256-pixel windows, 128-pixel stride,
probability-space Hann overlap-add, and full edge coverage. Output maps return
bilinearly to the original pixel-centre frame. Expert masks are never
interpolated. Saved dense maps are float16 `C,Y,X` arrays, in the documented
23-channel order. They are virtual sigmoid activations, not measured stains.

The coarse masks are the existing `tiling-mask-cpp` tile-majority reconstructions
at widths 56, 90 and 216, with 0% injected border error. They were generated
from the expert mask: this is a **controlled reconstruction experiment**, not
independent tissue classification. One central-half mean per original tile is
used for marker contrasts. At least three tiles per group are required. The
six eligible channels with largest absolute pairwise Hedges g are selected,
then median-centred, scaled by `max(IQR, 0.05)` and clipped to ±10. These
features drive the same local wand and spatial prior used by the RGB baseline.
Fine tissue masks enter only after predictions are fixed, for scoring.

## Files

- `download_reference.py`: checksummed acquisition of the authors' 50 examples.
- `model.py`: Flash architecture, retained for diagnostics, with all released
  LoRA tensors represented; `original_cnn.py`: official CNN architecture.
- `audit_dapi.py`: explicit architecture/preprocessing hypotheses on five
  calibration patches; no diagnostic forward variant is used in production.
- `audit_transfer.py`, `audit_stains.py`, `audit_scale.py`, `audit_highres.py`:
  model, stain and scale checks; all outcomes are retained, including failures.
- `confirm_dapi.py`, `validate_channels.py`: reference checks with the official
  disjoint-window control, for comparison between models.
- `validate_production.py`: checks the actual overlap-add pipeline and saved
  float16 maps, and writes the reports required by the CNN refinement runner.
- `run_pilot.py`: reference-gated CNN inference, coarse-only marker selection,
  Bora comparison and native-frame expert-mask scoring.
- `plot_overlays.py`, `plot_dapi_qa.py`: mask overlays at 35% opacity, virtual
  DAPI/CK/CD3/CD4/CD68/Ki67 overlays, fixed nuclear zoom and reference examples.
- `top3_rgb.py`: Kruskal-Wallis ranking across five coarse tissue groups, BH
  adjustment across 16 eligible proteins, all 169 toroidal grid shifts as a
  spatial sensitivity check, and pairwise contrasts with separate adjustment
  across all 160 marker/group pairs. It records selected-channel redundancy,
  makes an explicit rank-ordered RGB image, runs unmodified RGB/Lab Bora, and
  writes statistics, physical metadata, predictions and 35% opacity overlays.
- `group_markers.py`: group-specific ranking by worst pairwise direction-free
  AUC across all opponents; interface-specific RGB selection, the unchanged
  Bora RGB/Lab local wand and a matched pairwise H&E control. Fixed nearest
  original-interface ownership merges predictions; tied junctions and
  background stay fixed. This is an experimental wrapper rather than a change
  to the original multiclass RGB API.
- `plot_all_markers.py`: original H&E plus all 23 CNN channels in canonical
  order, with separate shared 0–1 and per-channel P1–P99 display atlases,
  native-resolution PNGs and a channel/scaling manifest. It labels DAPI,
  background channels and proteins below the reference screen explicitly.
- `pca_rgb.py`: standardized original RGB plus class-marker PCA, exact full
  SVD, fixed PC signs/order, P1–P99 conversion to RGB and unchanged multiclass
  Bora. It records all means/scales/component weights and compares deduplicated
  class winners, literal one-column-per-class and RGB-only PCA controls.

## Reproduce on Chiamaka

Use the isolated Python environment under
`/mnt/sata_ssd/Bora-benchmark/experiments/gigatime-flash-2026-10-16/.venv/`.
The exact runtime versions and source hashes are recorded in `results.json`.
For another environment, install Bora, CUDA-compatible PyTorch/torchvision,
`timm==1.0.22`, `safetensors==0.5.3`, `opencv-python==4.13.0.92`,
`scikit-learn` and `matplotlib`. No other project's environment is modified.

Download approved model files using the Hugging Face CLI:

```bash
hf download prov-gigatime/GigaTIME \
  --revision a6556030bdb8071857559e12b921620aa1bbac5b \
  --include model.pth config.json LICENSE
python experiments/gigatime_flash/download_reference.py
```

The CNN weight SHA-256 is
`d200308302c7fefd13b259c62de546f2c859c28d1079e9f67dc8808915d95d0c`.
Existing approved access was used; no token is saved in code or reports.
Set the following paths to the downloaded snapshot and output locations:

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
MODEL=/path/to/pinned/snapshot/model.pth
CONFIG=/path/to/pinned/snapshot/config.json
DATA=/mnt/sata_ssd/Bora-benchmark/raw/gigatime-official-test/extracted/sample_test_data/data
QA=/mnt/sata_ssd/Bora-benchmark/results/gigatime-flash-audit-2026-10-16
OUT=/mnt/sata_ssd/Bora-benchmark/results/gigatime-cnn-pilot-2026-10-16
PYTHONPATH=src python experiments/gigatime_flash/validate_production.py \
  --data "$DATA" --weights "$MODEL" --output "$QA/production"
PYTHONPATH=src python experiments/gigatime_flash/run_pilot.py \
  --model-family cnn --weights "$MODEL" --config "$CONFIG" --output "$OUT" \
  --dapi-validation-report "$QA/production/production_dapi_validation.json" \
  --channel-validation-report "$QA/production/production_channel_validation.json"
PYTHONPATH=src python experiments/gigatime_flash/plot_dapi_qa.py \
  --results "$OUT" --weights "$MODEL"
PYTHONPATH=src pytest tests/test_boundary.py tests/test_virtual_markers.py tests/test_gigatime_lora.py -q
```

After the checked CNN run, reproduce the three-marker RGB follow-up:

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PYTHONPATH=src python experiments/gigatime_flash/top3_rgb.py \
  --source /mnt/sata_ssd/Bora-benchmark/results/gigatime-cnn-pilot-2026-10-16 \
  --output /mnt/sata_ssd/Bora-benchmark/results/gigatime-top3-rgb-2026-10-16
PYTHONPATH=src:. pytest tests/test_top3_rgb.py tests/test_boundary.py \
  tests/test_virtual_markers.py tests/test_gigatime_lora.py -q
```

The primary selection grid is the previous 90-pixel display grid, where each
group has at least 12 tiles. One central-half activation mean per tile is the
observation. Select the lowest omnibus nominal p-values after the independent
reference-channel eligibility screen; report BH-adjusted values for the full
16-marker family. The top three become R/G/B in that order. Primary encoding
linearly maps P1–P99 over coarse foreground to 0–255, with minimum activation
range 0.01. Fixed [0,1] encoding is a prespecified sensitivity comparison.
The same composite is reused for tile widths 56, 90 and 216. No marker,
encoding, colour permutation or parameter is chosen by fine-mask accuracy.

The statistics are exploratory within one ROI. Nominal tests assume
independent observations; neighbouring tiles are correlated. Toroidal shifts
retain coarse group sizes and lattice patterns, but rely on stationarity and
artificial periodic edges. Neither diagnostic establishes patient-level
significance or measured target-sample protein expression. The three selected
signals are highly redundant and do not significantly separate invasive tumor
from tumor-associated stroma in their nominal pairwise tests.

Reproduce the current group-specific experiment after the checked CNN run:

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PYTHONPATH=src:. python experiments/gigatime_flash/group_markers.py \
  --source /mnt/sata_ssd/Bora-benchmark/results/gigatime-cnn-pilot-2026-10-16 \
  --output /mnt/sata_ssd/Bora-benchmark/results/gigatime-group-rgb-2026-10-16
PYTHONPATH=src:. python experiments/gigatime_flash/plot_all_markers.py \
  --source /mnt/sata_ssd/Bora-benchmark/results/gigatime-cnn-pilot-2026-10-16 \
  --output /mnt/sata_ssd/Bora-benchmark/results/gigatime-group-rgb-2026-10-16/all_markers
PYTHONPATH=src:. pytest tests/test_group_markers.py tests/test_top3_rgb.py \
  tests/test_boundary.py tests/test_virtual_markers.py tests/test_gigatime_lora.py -q
```

For group-specific ranking, each candidate is scored by the **minimum** of
its four direction-free pairwise AUCs; mean pairwise AUC breaks ties. An easy
contrast against one group cannot compensate for failure against another.
These are descriptive single-marker separations from the same coarse tiles,
not held-out estimates of a joint three-marker classifier. Higher or lower
activation can be informative; no biological tissue-marker specificity is
established on TIGER. The full ranking includes weak candidates as such.

Each boundary has its own three-marker RGB, ranked by separation of that pair.
For tumor–stroma this selects CK, CD138 and Caspase3-D. All 160 marker/pair
tests are adjusted together. The normal Bora RGB/Lab interior wand runs in each
two-class union with fixed parameters. Each editable pixel is assigned to its
nearest original interface, and exact distance ties are protected. Costs from
different RGB spaces are never compared. The original background stays fixed;
the matched H&E control uses precisely the same wrapper and ownership rule.

The runner also expects `prepared/tiger.jsonl` and the existing benchmark's
`results/bora-pilot-independent-2026-10-16/tiger/sample.json` and
`error_0/tile_{56,90,216}/coarse.tif`. Acquisition and original coarse-generation
code are maintained in [Bora-extra](https://github.com/tkcaccia/Bora-extra).
`plot_dapi_qa.py` uses the retained diagnostic confirmation list and rejected
Flash outputs for its fixed before/after comparison. The reference-check
figures are not selected by score.

## Saved artifacts

Chiamaka: `/mnt/sata_ssd/Bora-benchmark/results/gigatime-cnn-pilot-2026-10-16/`.
Local review copy:
`/Users/stefano/Documents/ChatGPT/Bora/benchmark/results/gigatime-cnn-pilot-2026-10-16/`.
Dense marker maps remain on Chiamaka. The figures, source image, expert mask,
coarse/refined rasters, scores, reference-check reports and provenance are
available in the review copy. Failed Flash outputs and all diagnostic
alternatives remain in their separate original folders. Raw imagery,
checkpoints, dense maps and generated label rasters remain outside Git.

Three-marker RGB outputs use the corresponding `gigatime-top3-rgb-2026-10-16`
result directory on Chiamaka and locally. `marker_statistics.json`/`.csv`
contains the full ranking; `tile_observations.csv` contains the analysis
observations; `rgb_encoding.json` maps R/G/B to marker names and scaling.
`top3_markers.rgb.png` is the exact primary Bora input. Its OME-TIFF companion
records native physical pixel size; original probability planes remain on
Chiamaka. `marker_statistics.png`, `top3_rgb_and_markers.png` and
`segmentation_overlays_tile_90.png` show the marker ranking, DAPI/selected
maps and transparent expert/coarse/Bora comparisons respectively.

Current group-specific outputs are in `gigatime-group-rgb-2026-10-16` under
both result roots. `selection.json`, `group_marker_ranking.csv` and
`pairwise_statistics.csv` document all markers and contrasts. `group_rgb/`
contains the five group-specific candidate composites; `pair_rgb/` contains
the ten interface composites actually used for refinement. Both encoding JSON
files record channel names and intensity transforms. `results.json` records
the native physical frame, fixed configuration, source hashes and full scores.
Figures include `markers_by_group.png`, `markers_by_interface.png`,
`group_rgb_composites.png`, `tumor_stroma_rgb_and_markers.png` (including DAPI)
and `group_specific_segmentation.png` (35% opacity masks). Twenty-four focused
tests cover rank adjustment, group-specific selection, constant/sparse cases,
channel/frame preservation, unchanged two-class RGB wand behavior, protected
junctions, order-independent merging and prior model/feature checks.
`all_markers/` contains 23 native-resolution display PNGs, two complete
atlases and `manifest.json`; individually stretched intensities cannot be
compared quantitatively across channels. Use the shared-scale atlas when
comparing predicted activation values.

## Original RGB plus class-marker PCA

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PYTHONPATH=src:. python experiments/gigatime_flash/pca_rgb.py \
  --source /mnt/sata_ssd/Bora-benchmark/results/gigatime-cnn-pilot-2026-10-16 \
  --group-results /mnt/sata_ssd/Bora-benchmark/results/gigatime-group-rgb-2026-10-16 \
  --output /mnt/sata_ssd/Bora-benchmark/results/gigatime-pca-rgb-2026-10-16
PYTHONPATH=src:. pytest tests/test_pca_rgb.py tests/test_group_markers.py \
  tests/test_top3_rgb.py tests/test_boundary.py tests/test_virtual_markers.py \
  tests/test_gigatime_lora.py -q
```

Class winners are CD138 for invasive tumor and tumor-associated stroma, CD3
for healthy glands and inflamed stroma, and T-bet for rest. These are empirical
virtual features from the current ROI; their levels can distinguish classes
without unique positive expression. The primary PCA uses each shared channel
once: R/G/B, CD138, CD3, T-bet. A literal eight-column version retains repeated
CD138/CD3 class winners as a sensitivity check; RGB-only PCA controls for the
color projection itself. All three variants are fixed before scoring.

PCA uses every coarse-foreground pixel of the same ROI after feature-wise
mean/SD scaling, exact full SVD, no whitening and no class weighting. The
largest absolute loading of each component is made positive. PC1/2/3 become
R/G/B using fixed per-PC P1–P99 scaling over coarse support. This encoding
changes relative component amplitudes, so explained variance is reported
before encoding. No component/sign/color permutation is selected using
accuracy. A single composite is reused at tile widths 56, 90 and 216.

The `gigatime-pca-rgb-2026-10-16` directory under both result roots contains
variant RGB PNGs/OME-TIFFs, exact feature/PCA/encoding parameters, raw scores
and source hashes, class assignments, segmentation masks/metrics and four
figures. Dense native PC-score arrays remain on Chiamaka. The primary three
PCs retain 96.78% of standardized variance. At 90-pixel tiles, IoU and boundary
F1 improve slightly versus original H&E Bora, but the previous pair-specific
approach is stronger. Other tile sizes are mixed; see the full report.

Thirty-two focused tests pass, including feature-unit invariance, exclusion
of pixels outside coarse support, exact saved-transform reconstruction,
deterministic PC signs, shared-marker deduplication and RGB channel order.
