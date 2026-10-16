# GigaTIME-Flash marker-driven Bora pilot

This experiment uses virtual mIF predictions from the **original H&E RGB** to
replace RGB appearance channels in Bora's adaptive local wand. It compares a
fixed RGB baseline with marker-driven refinement of the same coarse masks.
It does not change Bora's CLI defaults. Results from one annotated tissue ROI
are exploratory; see [the pilot report](../../docs/gigatime_flash_pilot.md).

## Files and locations

- `model.py`: architecture adapted from the official Apache-2.0 notebook,
  with the released patch-projection convolution's LoRA adapters included.
- `run_pilot.py`: strict checkpoint loading, resolution-aware window inference,
  marker selection, refinement, and native-frame expert-mask evaluation.
- `plot_overlays.py`: semitransparent reference, tiling-mask, and Bora overlays
  on the original image, plus DAPI and five other virtual-channel overlays.
- `src/bora/virtual_markers.py`: sampling plans and coarse-label-only contrasts.
- Chiamaka result root:
  `/mnt/sata_ssd/Bora-benchmark/results/gigatime-flash-pilot-2026-10-16/`.
- Local review copy:
  `/Users/stefano/Documents/ChatGPT/Bora/benchmark/results/gigatime-flash-pilot-2026-10-16/`.
  Dense virtual maps remain on Chiamaka.

All datasets, checkpoints and image-derived outputs remain outside Git.
Benchmark acquisition code and manifests are maintained in
[Bora-extra](https://github.com/tkcaccia/Bora-extra).

## Resolution contract

Distinguish **optical resolving power**, objective magnification, **physical
sampling** (µm per pixel), and array dimensions. Magnification alone does not
establish sampling, and resampling cannot restore optical detail.

The selected TIGER 114S ROI is 1224 × 1166 pixels. The provider reports sampling
of 0.5 µm/pixel, giving a field of view of approximately 612 × 583 µm.
The PNG does not establish scanner, objective magnification or numerical
aperture; these are recorded as unknown. [TIGER data](https://tiger.grand-challenge.org/Data/),
[ROI release](https://zenodo.org/records/6014422).

The original GigaTIME methods describe registered 128 × 128 µm mIF patches
containing 556 × 556 pixels, with 256 × 256 training subregions. The implied
registered-grid spacing is `128/556 ≈ 0.230216 µm/pixel`. Flash reports using
the same training data, but its model card does **not** specify a separate H&E
physical calibration. This inferred scale is therefore an **exploratory
reference**, not a confirmed Flash requirement. The 20× tiling described for
GigaPath slide classification is not a GigaTIME-Flash calibration.
[Original methods](https://doi.org/10.1016/j.cell.2025.11.016),
[Flash §3.1](https://arxiv.org/html/2607.18218v1).

Both native 0.5 µm/pixel and exploratory ≈0.230 µm/pixel inference are retained.
For the latter, integer dimensions are 2658 × 2532 and the effective spacing
is x=0.2302483, y=0.2302528 µm/pixel. Field of view is preserved. RGB is
upsampled bicubically; predicted maps are returned bilinearly to the original
pixel-centre frame. The expert mask is never interpolated. Bora's fixed 2×
working downsample is separately recorded (approximately 1 µm/working pixel).

The model consumes 256 × 256 RGB windows and outputs 23 × 256 × 256 maps.
Its transformer patch is 16 × 16 pixels, giving a 16 × 16 token grid; dense
output pixels do not establish optical or cellular accuracy. At native
sampling a window spans 128 µm per side; at the exploratory scale it spans
approximately 58.94 µm. Window stride is 128 pixels, overlapping probabilities
are Hann blended, and partial image edges are retained.

## Selection and comparison

The input coarse masks are existing **reference-derived tile-majority masks**
from `tiling-mask-cpp`, at tile widths 56, 90 and 216, with 0% injected border
error. This is controlled boundary reconstruction, not an independently
predicted tissue-classification benchmark. The sample inventory and tiler hash
are copied into the run metadata. Coarse arrays are copied unchanged.

Within each original tile, the central half supplies one mean per channel.
Groups need at least three tiles. Channels are ranked by their largest
absolute pairwise Hedges g, using a 0.01 activation-SD floor. The top six are
selected independently for each coarse grid and sampling condition. DAPI
(nuclear stain), TRITC and Cy5 (background) are excluded from protein-channel
selection. DAPI is included in the requested visualizations. Effects are
descriptive: spatial correlation prevents treating these tile observations as
independent biological replicates.

Selected activations are median-centred and divided by `max(IQR, 0.05)` over
coarse foreground, clipped to ±10. The existing balanced local wand uses these
channels for internal competition, with a unit variance floor. RGB baseline
uses its unchanged Lab path. Both use the same `BoundaryConfig()` defaults,
local spatial prior, protected seeds and H&E exterior stage. The run does not
train a segmentation model, retrain Flash or tune against the expert mask.

Only after all predictions are fixed does the runner read fine reference
labels for evaluation. Class IDs are compacted with an explicit reversible
map; original 0 is excluded, and original 7 (`rest`) remains a scored class.
IoU/Dice and accuracy are scored in the native frame. Internal-boundary F1
uses 4, 8 and 16 µm tolerances; ASSD and HD95 use physical x/y sampling.
No protein-stain ground truth is available in this TIGER case.

## Reproduce

Use an isolated environment with a compatible CUDA PyTorch installation:

```bash
python -m pip install -e '.[test]'
python -m pip install timm==1.0.22 safetensors==0.5.3 opencv-python==4.13.0.92 matplotlib
hf auth login
hf download prov-gigatime/gigatime-flash \
  --revision 47a5ee847d77cbe0e75f2f67f0c95d45e9bd7ca2 \
  --include model.pth config.json LICENSE
```

Access must already be approved by the checkpoint owner. Do not put a token
in scripts, reports or shell command arguments. This run used existing access
and did not save a token on Chiamaka. The exact weights are SHA-256
`ac93f0ea65df2cc6de388415ec5f5c074feab4754aabab450216ef69fe5dd411`.
All 254 state tensors load strictly (23,806,559 parameters); missing or
unexpected tensors abort the run. The official notebook fallback drops the
patch convolution adapters; this implementation retains them.

Set the two model paths to the pinned local snapshot, then run from Bora:

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PYTHONPATH=src python experiments/gigatime_flash/run_pilot.py \
  --data-root /mnt/sata_ssd/Bora-benchmark \
  --output /mnt/sata_ssd/Bora-benchmark/results/gigatime-flash-pilot-2026-10-16 \
  --weights /path/to/pinned/snapshot/model.pth \
  --config /path/to/pinned/snapshot/config.json
python experiments/gigatime_flash/plot_overlays.py \
  --results /mnt/sata_ssd/Bora-benchmark/results/gigatime-flash-pilot-2026-10-16 \
  --tile-size 90 --alpha 0.35
PYTHONPATH=src pytest tests/test_boundary.py tests/test_virtual_markers.py -q
```

The runner expects the prepared TIGER inventory and the existing pilot's
`results/bora-pilot-independent-2026-10-16/tiger/sample.json` plus
`error_0/tile_{56,90,216}/coarse.tif`. The coarse-generation implementation is
[`Bora-extra/benchmark/run_bora_pilot.py`](https://github.com/tkcaccia/Bora-extra/blob/main/benchmark/run_bora_pilot.py).
Use a fresh output directory for a different model, image or inference policy.

Results include checksums, environment versions, physical sampling and
dimensions, all marker contrasts and tile observations, refined masks,
per-class scores, timings, dense maps, and an overlay manifest. Model-grid and
native maps are `C,Y,X` float16 NumPy arrays, in `config.json` channel order.
They are virtual sigmoid activations, not calibrated expression measurements.
The code snapshot used for the run is retained on Chiamaka under
`experiments/gigatime-flash-2026-10-16/code/`; source hashes are in `results.json`.
