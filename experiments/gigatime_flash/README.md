# Reference-checked GigaTIME CNN refinement

The current workflow uses **the original GigaTIME CNN**, with unmodified RGB
and input sampling of **0.25 µm/pixel**. The earlier Flash pilot is rejected
for target-image DAPI quality. This directory retains its historical name so
that all diagnostic scripts and previous runs remain traceable.

See [the CNN validation and results report](../../docs/gigatime_cnn_validation.md).
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
