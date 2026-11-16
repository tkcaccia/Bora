# Bora for R 0.2.0

The [reference and acceptance manual](REFERENCE_MANUAL.md) explains the five
user-facing functions, their inputs and outputs, the completed software checks,
and the limits of those checks. The [0.1.7 PDF edition](../docs/reference/Bora_R_0.1.7_reference_and_acceptance_manual.pdf)
is retained as historical documentation; use the current Markdown manual for
the explicit-spacing API.

This MIT R package is released alongside the Bora Python backend.
`Bora.preparation`, `Bora.training`, and the
supervised branch of `Bora.prediction` have synthetic end-to-end pilots.
`Bora.spatial()` adds a same-section expression-guided route for registered
Seurat, Giotto and SpatialExperiment spot groups. It returns a categorical
TIFF, source-image-pixel GeoJSON, binary group images and a preview. Group
names come from the supplied expression object; this route fits to those
spots and does not claim expert tissue labels or cell-level expression.
The [Benchmark 4 R example](examples/benchmark4_spatial.R) accepts an
already clustered spatial object. The Bora-Analysis Benchmark 4 tutorial also
shows how its expression-clustering front ends feed this function.
The joint-image KODAMA branch of `Bora.prediction` also passed a fabricated
two-image, two-resolution end-to-end pilot. One BCSS H&E region has passed
source-registration and frozen supervised-head regressions. A separate frozen
BCSS training image has passed real-image R fitting and prediction with an
independent source-mask, coefficient and vector audit. The same BCSS training
image passed masked three-scale Virchow2 extraction through the installed R
function: all three full-1,280-channel arrays were byte-identical to frozen
historical caches. One public ovarian whole slide passed installed-R
prediction regression: raw and smoothed score arrays matched the frozen
Python hashes, and native-coordinate GeoJSON roundtripped exactly to the
final raster. That prediction took 2 min 49 s on CPU with 3.15 GiB peak host
memory after feature extraction. The R source package builds and checks on
Linux, macOS and Windows in CI; the Python CPU test matrix passes on Linux
and macOS. These are selected technical regressions; multi-platform GPU
feature extraction and broad biological validation remain pending.

The [R installation and one-result guide](REPRODUCE_ONE_RESULT.md) records
the exact real training-image input schema, commands and output hashes.

The R 0.2.0 package calls the matching Bora Python backend through
`reticulate`. Virchow2 weights are separately licensed and are never bundled
with Bora. A user-managed Python environment with Bora, PyTorch, timm and the
licensed Virchow2 checkpoint is required for feature extraction. Set
`RETICULATE_PYTHON` before starting R and `BORA_VIRCHOW2_MODEL` to the model
directory. Install the pinned source release and its matching Python backend:

```bash
git clone https://github.com/tkcaccia/Bora.git
cd Bora
python3 -m venv .venv
.venv/bin/python -m pip install '.[wsi,model-download]'
mkdir -p .R-library
R CMD INSTALL --library="$PWD/.R-library" r
export R_LIBS_USER="$PWD/.R-library"
export RETICULATE_PYTHON="$PWD/.venv/bin/python"
```

Install compatible PyTorch and timm separately for Virchow2 extraction.
Each user must [accept Virchow2's gated terms](https://huggingface.co/paige-ai/Virchow2)
with their own institutional Hugging Face account. After authenticating with
`hf auth login` or setting `HF_TOKEN` in the Python environment, download it:

```r
library(Bora)
model <- Bora.download_virchow2("/path/to/models/Virchow2-v1")
Sys.setenv(BORA_VIRCHOW2_MODEL = model$model_dir)
# Bora.preparation(..., model_dir = model$model_dir)
```

The helper downloads only `config.json` and `model.safetensors` into a new
directory, records the exact Hub commit and both SHA256 values in
`DOWNLOAD_COMPLETE.json`, and leaves any incomplete transfer outside the
usable model directory. It does not bundle weights or bypass the model gate.
The [one-result guide](REPRODUCE_ONE_RESULT.md) is a frozen 0.1.4 replay, not
a current eight-class benchmark reproduction. It
provides exact commands, source/model hashes and expected output hashes.

For predictions and spatial expression workflows, pass measured X/Y source
pixel spacing explicitly, for example `source_mpp_xy = c(0.5, 0.5)` only when
both values were verified from that image's metadata. The former implicit
0.5 µm default was removed in 0.2.0 to avoid silent physical-coordinate errors.

`Bora.training` reads exact-size class raster masks or pixel-coordinate class
GeoJSON. Class IDs must be present in the annotation or supplied with
`class_labels` as a list of named integer mappings (one per image); zero is
unscored. It samples the prepared full 1,280-channel factor-1 array
at token centres, selects classwise contrast channels and fits a
linear ridge head. This general R route is separate from the frozen historical
eight-class checkpoint until full regression is complete.

The real training-image pilot used four expert-derived broad classes from one
BCSS training case. Its 821 sampled finite tokens yielded 55 selected channels;
an independent audit reproduced the model coefficients exactly. The resulting
preview is a technical end-to-end check on a training image, not a held-out
accuracy or generalization result.

For benchmark runs, supply an independently prepared tissue mask for every
image. Without a mask, preparation intentionally encodes the full H&E field,
including empty glass slide background outside the specimen. Class annotations used by training are separate from these
tissue-support masks.

### Reuse verified full-channel features

`Bora.preparation(..., precomputed_manifest = "verified-grids.json")` can adopt
previously extracted Virchow2 grids without copying them or starting a GPU
worker. The JSON lists the input images in the same order as `images`, and
binds their SHA256 values, the Virchow2 checkpoint and configuration hashes,
and each factor-1/2/4 float16 `H × W × 1280` NumPy array. Each scale needs a
separately verified source receipt whose hash is in the manifest. This is a
reuse route for audited caches, not a way to certify an unaudited array.

The manifest format is `bora-precomputed-features-v1`. Its top level has
`model_checkpoint_sha256`, `model_config_sha256`, and `images`. Each image
entry has absolute `image_path`, `image_sha256`, `encoder_selection`
(`full-field` or `supplied-tissue`), optional `encoder_tissue_sha256` for the
latter, and a `scales` object keyed by factor. Each scale has absolute `path`,
`sha256`, `source_receipt_path`, `source_receipt_sha256`, and `registration`:

```json
{"axis_order":"yxc","origin_source_px_yx":[0,0],
 "token_stride_source_px_yx":[14,14],
 "padding":"right-bottom-to-multiple-896"}
```

The stride is `[28,28]` at factor 2 and `[56,56]` at factor 4. Every source
receipt uses `bora-precomputed-scale-receipt-v1`, status `VERIFIED`, and repeats
the source image hash, model hashes, factor, array hash, `shape_yxc`,
registration, encoder selection, and encoder tissue hash (or JSON `null`).
Adoption checks all array and receipt bytes and shapes before creating the
output directory. A `full-field` cache may be paired with a new tissue mask:
the manifest records that the encoder once covered empty slide background, while the supplied
mask now controls Bora tile admission and final support. A `supplied-tissue`
cache requires the exact same mask hash. Subsequent `Bora.training` and
`Bora.prediction` consume the standard preparation manifest.

For a reviewed tissue-area GeoJSON, the Python package function
`bora.r_tissue_support.materialize_reviewed_tissue_support` makes an aligned,
tiled TIFF and an exact-pixel/source-hash receipt. The Bora-Analysis benchmark
wrapper calls that function; the package owns the rasterization and tile-count
checks. This procedure does not infer the tissue boundary or certify the
manual review.

Supervised prediction projects class scores from the full-channel factor-1
cache, saves native and common-support-clipped masks separately, applies
measured-area and contour cleanup, and writes class-specific binary TIFFs,
the multiclass TIFF, source-pixel-coordinate GeoJSON, and a preview. The
physical pixel spacing must be supplied for each calibrated image. A read-only
adapter of the frozen 203-channel eight-class ridge head reproduced one
historical BCSS display mask exactly (453,888 of 453,888 pixels), and R and
Python projections produced identical score-array hashes. This is one
inspected-development image, with no expert reference opened; it does not
establish broad numerical equivalence. The separate ovarian WSI check
established score-array identity and exact output geometry, but did not use
a whole-slide expert reference. Its historical public display used a
four-class class-presence gate; this regression projected all eight classes,
so its final raster is not expected to match that display.

With `predictor = NULL`, prediction pools every strictly admitted 224-pixel
tile from the prepared images into one source-bound KODAMA 0.99.7 fit. A
supplied tissue mask admits a tile only when more than 90% of its pixels are
supported; artificial padding is background. Without a mask, the entire
source image, including empty slide background, remains eligible under the same padding rule.
The route retains each row's image and tile identity, creates one graph
solution for each requested `resolutions` value, selects channels from the
resulting anonymous groups, and projects them to token and pixel masks.
One-group solutions stop with an explicit error because they contain no
internal tissue boundary. Group IDs are **not tissue class names**.

KODAMA 0.99.7, RANN, and igraph are needed only for the joint-image route.
The current implementation has a configurable `max_joint_tiles` admission
limit (default 10,000) and never silently downsamples the joint fit. Its
performance on large whole slides remains to be profiled. The fabricated
pilot is an interface and geometry check, not evidence of biological
segmentation quality or historical numerical equivalence.

**Preview performance.** In a source-only profile of one approved ovarian
whole slide, 137.6 of 172.2 seconds (79.9%) of CPU prediction time went to
creating the 1,024-pixel H&E preview, mostly decoding a single-page TIFF in
libvips. The mask and GeoJSON remained byte- or pixel-identical to the frozen
regression. The joint-image route now caches that source RGB thumbnail once
per image and reuses it across requested clustering resolutions. On the same
slide, the first preview took 150.61 seconds and reuse took 0.25 seconds;
both preview files had the historical SHA256. This measures preview reuse,
not an overall prediction speedup or KODAMA fit time. The core interpolation
and geometry work already runs in compiled OpenCV, rasterio and GEOS paths;
we have not replaced it with an unprofiled C++ rewrite.

For a faster preview on a large tiled TIFF, pass
`preview_backend = "gdal-lanczos"` to `Bora.prediction`. The default
`"vips-lanczos3"` keeps historical preview PNG bytes reproducible. On the
same ovarian slide, a warmed GDAL thumbnail took 1.37 seconds versus
120.82 seconds for the libvips path. A complete source-only replay with
the GDAL option took 29.96 seconds; raw and smoothed scores, native and final
mask TIFFs, eight binary TIFFs, common support and GeoJSON remained
byte-identical to the frozen result. The preview RGB changed slightly
(mean absolute channel difference 0.978/255). This is one slide profile,
not a cross-format speed guarantee.

A source-only BCSS H&E region regression reused an independently verified
full-1,280-channel historical cache and a separately computed H&E tissue
support. Forty-eight strict tiles yielded two and five anonymous groups at
resolutions 1.0 and 1.5. The binary TIFFs, final raster, and source-coordinate
GeoJSON passed exact pixel equality after a one-pixel support-hole mismatch
was found and fixed in the developing vector export. This checks one real
region's source registration and output geometry. It does not establish class
accuracy, historical KODAMA–Bora equivalence, or whole-slide scalability.
