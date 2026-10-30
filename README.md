# Bora

Bora takes an H&E image, anonymous tile-cluster labels, calibrated Virchow2
spatial features and a user-trained fastPLS tissue classifier. It returns a
refined tissue mask and compact GeoJSON. When a relevant tissue model is
supplied, each complete segment also receives a predicted tissue name.

The frozen whole-slide route, released in **0.2.2**, is:

**Tile groups → one slide-wide RGB9+19 response fit → all group responses + H&E refinement → one global mask → compact diagonal GeoJSON.**

This is the numerical route used for the approved colorectal whole-slide
result. All response channels enter refinement directly; the RGB image is
only a visualization. A diagnostic field is cropped from the completed slide,
without fitting, refinement or area cleanup in that field. Components and
enclosed background holes strictly smaller than **400 µm²** are removed or
filled globally. Contours are smoothed at 1.5 µm and shared boundaries are
simplified with physical, per-class agreement guards.

Start with the [frozen whole-slide workflow](docs/FROZEN_COLORECTAL_WSI.md).
For a reviewed tissue boundary supplied as an input, see the
[tissue-area GeoJSON manifest](docs/tissue-geojson-input.md).
`bora wsi-fit`, `bora wsi-project`, `bora wsi` and `bora wsi-crop` expose this
route. Its [reference profile](src/bora/frozen_colorectal.json) records the
settings, source hashes and accepted output identities. Input groups and
features are supplied by the user; slide-specific weights are fitted once.

Tissue naming is a separate step. Neighbor context pools all 1280 token
channels over each complete anonymous group, including disconnected pieces.
It concatenates that mean with its difference from neighboring group means,
weighted by shared boundary length in micrometres. One native CPU fastPLS
SIMPLS-LDA head predicts one tissue identity per group. The audited 25-class
model uses 24 components; smaller training vocabularies are capped at their
class-discriminant dimension. There is no classifier selector. The approved
colorectal preview contains anonymous groups, without biological tissue names.
See [classifying accepted regions](docs/REGION_IDENTITY.md) to add names using
an independently trained tissue model while preserving the contours.

The older three-group PCA32 commands and field-level `bora refine` remain
compatibility APIs. They are not the frozen colorectal whole-slide method.

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
# Install libvips first: Ubuntu: apt install libvips-dev; macOS: brew install vips
# macOS Framework Python may also need:
# export DYLD_FALLBACK_LIBRARY_PATH="$(brew --prefix)/lib"
python -m pip install -e '.[test,wsi]'
pytest -q
bora --help
```

## Field-level compatibility workflow

The following case format and `prepare`/`refine` examples describe the older
field API. For the selected full-slide method use the workflow linked above.

### Inputs and calibration

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

`coarse` may be an expanded categorical TIFF, or supply a two-dimensional
`tile_labels` NumPy array and `tile_size_yx` (for example `[90, 90]`) instead.
The tile grid is expanded from the top-left image edge and cropped at the
right/bottom edges. Cluster IDs are anonymous integers in 1..255; zero is
unassigned. These IDs are **not** tissue class IDs. See the
[complete training and inference guide](docs/WORKFLOW.md).

## Prepare, train and refine

Prepare geometry/descriptors for each annotated training field. Its case file
contains the image, tile clusters and Virchow2 tokens, **not** the expert mask:

```bash
bora prepare training-case.json --output prepared/training-case
```

The output contains `groups.ome.tif`, `descriptors.npz`, and provenance. The
training manifest can supply an aligned expert integer mask or GeoJSON for each
prepared image. Bora determines one tissue target for each sufficiently covered,
unambiguous whole group. The training annotation is never accepted by `refine`.

```json
{
  "format": "bora-training-v1",
  "classes": {"1": "tumor", "2": "stroma"},
  "cases": [{
    "descriptors": "prepared/training-case/descriptors.npz",
    "groups": "prepared/training-case/groups.ome.tif",
    "annotation": "training-case/expert.tif",
    "annotation_values": {"10": 1, "20": 2},
    "source_group": "patient-A",
    "split": "train"
  }]
}
```

The mapping is from **expert mask value** to **model class ID**; here 10 means
tumor and 20 means stroma. Zero is ignored unless explicitly mapped to a class.
For GeoJSON, use `annotation_field` (for example `classification.name`) and
declare `annotation_coordinate_space: native_image_pixel_edges`. Regions with
less than 50% annotated coverage or less than 80% dominant-class purity are
excluded by default; both thresholds can be set per case. An existing manual
`labels` mapping remains available when its whole-group targets have already
been independently adjudicated. See the guide for both complete examples.

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
Version 0.2.1 vendors the upstream SIMPLS–LDA score-moment correction and
matrix allocation guard. Model artifacts made with the earlier fastPLS core
are rejected because loading refits the native head; retrain or explicitly
refit from the saved training descriptors and labels with this version. The
reproducibility check found identical tissue predictions on all 116 saved
evaluation fields (399 whole groups), so the published masks and scores do not
change for that cohort. See [the correction audit](docs/FASTPLS_CORRECTION.md).

## Field-level geometry and timing

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

## Whole-slide input and migration

Use the [frozen whole-slide workflow](docs/FROZEN_COLORECTAL_WSI.md) and the
`wsi-fit → wsi-project → wsi → wsi-crop` commands for new WSI analyses.
The optional `wsi` dependency supplies pyvips; the host needs libvips.
Upstream image calibration and feature extraction must retain their pixel
spacing, source identities and channel selection. The package never launches
an encoder or GPU job.

The older `bora.wsi.WholeSlide` reader, windowed export, field API and
three-group PCA32 commands remain available for reproducing previous results.
Their field settings do not override the frozen full-slide settings above.
Bora 0.2.2 preserves the accepted numerical kernels and adds portable manifests,
source hashes, global-output receipts and diagnostic crop verification.

No background search or benchmark campaign starts during installation, import
or execution. Historical studies stay outside the application. The full-slide
freeze covers anonymous group segmentation; biological tissue identities
require a relevant independently trained Neighbor context fastPLS model.

## License

The vendored fastPLS core and bindings retain their MIT license, including
Stefano Cacciatore's copyright notice. No new overall project license is
selected by this release. Upstream libraries,
datasets and reference materials retain their respective licenses.
