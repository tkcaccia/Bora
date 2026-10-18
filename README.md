# Bora

Bora converts a coarse tissue-group map into a refined, named tissue mask and
compact GeoJSON. There is one supported pipeline:

**Virchow2 spatial tokens → fixed Virchow2 RGB evidence → Bora-AF → Neighbor context → GeoJSON.**

Neighbor context pools all 1280 token channels over each complete anonymous group, including
disconnected pieces. It concatenates that mean with its difference from
neighbouring group means, weighted by shared boundary length in micrometres.
One native CPU fastPLS SIMPLS-LDA head predicts one tissue identity per group.
The audited 25-class model uses 24 components; smaller training vocabularies
are capped at their class-discriminant dimension. No classifier selector or
automatic algorithm search is installed.

Benchmark acquisition, historical research, figures and the public viewer live
in [Bora-Analysis](https://github.com/tkcaccia/Bora-Analysis).
[Open the benchmark viewer](https://tkcaccia.github.io/Bora-Analysis/).

## Install

Use Python 3.10–3.13 and a C++17 compiler. Linux also needs BLAS/LAPACK development
libraries (`libopenblas-dev liblapack-dev` on Ubuntu); macOS uses Accelerate.
The MIT-licensed fastPLS C++ core and Python bindings are copied into Bora at
their audited revisions, with licenses and source hashes in `vendor/`. No
separate fastPLS installation or Git dependency is required. Scientific
dependencies are pinned to the benchmark runtime.

```bash
git clone https://github.com/tkcaccia/Bora.git
cd Bora
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
pytest -q
bora --help
```

## Inputs and calibration

Bora consumes encoder features; it does not download a model or launch GPU/HPC
work. Supply the full, **unselected 1280-channel Virchow2 spatial-token grid**
from the declared checkpoint. The encoder must cover all image tissue,
independently of annotation coverage. Never infer microscopy resolution from
pixel dimensions or a magnification label alone.

Each `case.json` declares an aligned H&E working image, categorical coarse mask,
independent tissue support, features and actual per-axis pixel spacing. Working
sampling must be approximately 0.5 µm/px. Encoder windows are 224 pixels and
spatial tokens have a 14-pixel stride; padded edge tokens must keep their original
coordinates. At least 16 substantial 90-pixel coarse tiles and two observed
groups are required. Every group must have an eligible tile observation.

```json
{
  "image": "he.png",
  "coarse": "coarse.tif",
  "tissue": "tissue.tif",
  "tokens": "base.npy",
  "mpp_xy": [0.5, 0.5],
  "native_mpp_xy": [0.25, 0.25],
  "native_tissue": "tissue.native.tif",
  "evidence_rgb": "rgb9-plus-19.png",
  "encoder_revision": "3158645804b69e3f3bc4439d4116edddf0840a72",
  "encoder_sha256": "8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b"
}
```

Paths are relative to the JSON file. A path can instead be an object containing
`path` and `sha256`; a mismatch fails before inference. All consumed input hashes
are recorded. Native and working dimensions must describe the same physical
field. Without a different output grid, omit `native_tissue` and
`native_mpp_xy`. Pixel spacing does not establish optical resolving power.

The example reuses a trusted, checksum-recorded RGB9+19 cache. For direct RGB
conversion, omit `evidence_rgb` and supply a `palette` array indexed by coarse
ID minus one, plus `variants.scale_1` and `variants.scale_2`. Each variant has
`features`, `shape_yx`, and actual `mpp_xy` for approximately 1 and 2 µm/px.
Variant arrays may contain all 1280 channels, or the exact 64 selected channels
with their `selected_dimensions` indices. A mismatched selection fails. The
conversion averages #9 and #19 class probabilities before RGB quantization;
the formulas and their within-image fits are unchanged.

## Prepare, train and refine

Prepare geometry/descriptors for each annotated training field:

```bash
bora prepare training-case.json --output prepared/training-case
```

The output contains `groups.ome.tif`, `descriptors.npz`, and provenance. Assign
one expert tissue target to each original anonymous group ID in a training
manifest. These are group targets, not independently classified fragments.

```json
{
  "format": "bora-training-v1",
  "classes": {"1": "tumor", "2": "stroma"},
  "cases": [{
    "descriptors": "prepared/training-case/descriptors.npz",
    "source_group": "patient-A",
    "split": "train",
    "labels": {"1": 1, "2": 2}
  }]
}
```

Use sufficient independent training groups for the intended vocabulary. Keep
patients/institutions used for final evaluation out of training and model
selection. Class names must match represented training targets; a missing class
cannot be invented by the classifier.

```bash
bora train training.json --output neighbor-context.npz
bora refine target-case.json --model neighbor-context.npz --output results/target
```

The inference command accepts no fine annotation or reference-mask path.
Outputs include the converted RGB, anonymous groups, classified mask before
export, final `mask.ome.tif`, `mask.geojson`, and `report.json`. The final TIFF
is rasterized from the actual final GeoJSON. New output paths are required;
completed results are not silently overwritten.

fastPLS 0.3 does not expose portable fitted-state serialization. The safe NPZ
artifact therefore retains training descriptors and targets and refits the
deterministic native head once on loading. It contains no pickle. Loading is
a separate startup cost; reuse the loaded Python object for a batch. Training
features in the artifact remain subject to the source data's sharing terms.

## Geometry and timing

Bora-AF lets groups compete throughout supported tissue, without fixed boundary
margins or hard seeds. Glass is removed after refinement. The published native
coverage converter is retained unchanged (accurate profile: initial component
filter 500 native pixels, simplification 2 pixels). This initial filter is
separate from the final cleanup rule: polygon parts and enclosed holes strictly
below **20 native pixels of area** are removed/filled; exactly 20 is retained
by that cleanup. Isolated tiny specks become background.

Shared boundaries are fitted once as straight/diagonal arcs, using at most
3 µm tolerance, at least 97% per-class agreement with the preceding predicted
geometry and at most 1.5% tissue-area change. Guards use predictions, never
expert masks. Coordinates are local native image pixel edges, with per-axis
spacing in metadata. The vectorization is intentionally approximate.

Reports separate conversion, refinement, pooling/context, classifier, model
startup and export costs. The encoder is upstream and excluded. The existing
116-image benchmark is a development evaluation with adaptively reused data;
it does not establish untouched-test accuracy or complete-WSI throughput.

## Scope and migration

Version 0.2 replaces the previous experimental CLI. Watershed, MedSAM,
PathSegmentor, earlier evidence solvers, alternative classifiers and evolution
launchers are removed from the application. Historical source/evidence is
preserved in Bora-Analysis and Git history. No background research cycle is
started by installation, import or command execution.

This implementation processes aligned fields in memory. Full-slide tiling,
cross-field identity reconciliation and the proposed source-WSI transfer study
require separate validation; native-coordinate output alone is not proof of
whole-slide scalability.

## License

The vendored fastPLS core and bindings retain their MIT license, including
Stefano Cacciatore's copyright notice. No new overall project license is
selected by this release. Upstream libraries,
datasets and reference materials retain their respective licenses.
