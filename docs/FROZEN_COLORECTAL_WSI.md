# Frozen colorectal whole-slide workflow — Bora 0.2.2

This release preserves the numerical method used for the accepted colorectal
whole-slide result on 28 October 2026. The same method was applied to pancreas
and brain slides. The public package contains portable CPU code, without
private data, workstation paths, model downloads or GPU launchers.

The source and reference output hashes are in
[`frozen_colorectal.json`](../src/bora/frozen_colorectal.json). The reference
colorectal analysis covers the entire tissue above the excluded scanner bar.
Its calibrated raster is 22,398 × 41,209 pixels. The original WSI coordinates
are preserved in the exported GeoJSON. This is an anonymous-group segmentation
result. Biological tissue naming was not applied in that preview.

## Inputs

Prepare a uint8 RGB H&E image at approximately 0.5 µm/pixel and record the
actual spacing on both axes. Use the original spacing and dimensions for
native-coordinate export. Resampling does not increase optical resolution.

The reference clustering uses 11,469 tissue-containing, standard-grid
224 × 224 working-pixel tiles. Full 1,280-channel pooled Virchow2 vectors were
standardized and reduced to 50 principal components before KODAMA. KODAMA
0.99.7 was run with M=100 and four cores. A Leiden graph with k=15 and
resolution 0.46 produced the frozen 16 colorectal groups. These are input
labels, not expert tissue names; another slide supplies its own grouping.
The 224-pixel spatial encoder windows use stride 112 to blend responses.
Tiles containing some tissue remain eligible; pure glass has no response.

Supply spatial features at the three calibrated scales, with factors 1, 2
and 4 relative to the working raster. Each standard tile spans 16 × 16 base
tokens. The categorical group grid is padded to multiples of four tiles;
zero denotes pure background. The three spatial arrays contain the **same
64 selected channels** of the original 1,280, in the declared order. The
channel-selection indices must be preserved with their source provenance.
Do not independently select channels in each diagnostic field. The full
unselected 1,280 channels remain necessary for later tissue classification.

## Fit and project once across the slide

An input manifest contains predictors only. Paths are relative to the
manifest; any file reference may instead be `{ "path": "...", "sha256":
"..." }` to require an expected checksum. `selected_dimensions` must contain
the actual 64 unique channel indices; replace the abbreviated list below.

```json
{
  "format": "bora-wsi-response-input-v1",
  "source_frame_id": "slide-A/calibrated-analysis-frame",
  "image": "he-0p5um.tif",
  "shape_yx": [22398, 41209],
  "group_grid": "groups.npy",
  "selected_dimensions": [0, 1, 2],
  "features": {
    "1": "selected-spatial-factor1.npy",
    "2": "selected-spatial-factor2.npy",
    "4": "selected-spatial-factor4.npy"
  }
}
```

For a group grid of R × C tiles, the arrays have shapes
`(16R/factor, 16C/factor, 64)`. Eligible tissue tokens must be finite.
The package validates geometry, group IDs and consumed input hashes.

```bash
bora wsi-fit responses-input.json --output slide-fit
bora wsi-project responses-input.json --fit slide-fit --output slide-responses
```

The accepted RGB9+19 mathematics is retained: multiscale persistence selects
32 positions within the 64 channels; balanced logistic response heads use
C=0.3. Neighbor reconstruction uses four spatially held-out quadrant fits
with ridge alpha=100. Their averaged prediction is passed through the second
response head, and the two probability fields are averaged. The `bora` launcher
initializes four BLAS threads before importing numeric
libraries, matching the reference run. When importing the Python functions
directly, set `OPENBLAS_NUM_THREADS=4` and `OMP_NUM_THREADS=4` before starting
Python; changing a smaller initialized thread pool at runtime can crash some
OpenBLAS builds. The numeric model is saved as NPZ without pickle. These response heads reproduce
anonymous groups; they are separate from the fastPLS tissue classifier.

Projection writes the full token probability field, an RGB visualization and
`GLOBAL_PROJECTION_COMPLETE.json`. The receipt binds one slide-wide model,
image, group grid and feature selection. RGB colors do not enter the solver;
all group probabilities do. No head is fitted for a zoom window.

## Refine one complete analysis frame

Create the segmentation manifest after projection. This example illustrates
an untrimmed image at identical working/native resolution; use the actual
original WSI metadata when they differ. For a scanner-bar exclusion, include
`analysis_native_bounds_xyxy` in original WSI coordinates. The calibrated
image must represent exactly that physical rectangle.

```json
{
  "format": "bora-wsi-colorectal-v1",
  "source_frame_id": "slide-A/calibrated-analysis-frame",
  "image": "he-0p5um.tif",
  "shape_yx": [22398, 41209],
  "group_grid": "groups.npy",
  "probabilities": "slide-responses/group-probabilities-token-grid.f16.npy",
  "response_receipt": "slide-responses/GLOBAL_PROJECTION_COMPLETE.json",
  "working_mpp_xy": [0.5, 0.5],
  "native_mpp_xy": [0.5, 0.5],
  "native_shape_yx": [22398, 41209]
}
```

```bash
bora wsi slide.json --output result --memory-gib 12 --minimum-free-gib 80
bora wsi-crop result --box 1200 2400 3072 3072 --output diagnostic-01
```

The frozen solver uses H&E optical-density conductance (sigma=0.18,
floor=0.02), a weak tile prior of 0.10, diffusion strength 3 and 16 iterations
on a half-resolution analysis grid. Cores of 768 analysis pixels use a
24-pixel halo. Several seams are independently recomputed and must agree
exactly. Source support is calculated globally: tissue islands below
100 µm² are removed and enclosed support holes below 900 µm² are filled.

The mask is smoothed with a 1.5 µm Gaussian one-hot competition. Components
and enclosed background holes **strictly below 400 µm²** are then merged or
filled on the whole slide; exactly 400 µm² is retained. The planar coverage
converter and shared-boundary RDP fit produce diagonal contours. Chord
tolerance is at most 5 µm, guarded by per-class predicted-shape IoU ≥0.95 and
tissue-area change ≤2.5%. These are export-fidelity guards, not accuracy
against expert labels. Raster and continuous geometry audits must pass.

The method streams local diffusion and rasterization, but connected-component
audits still need full-frame integer storage. The memory preflight is an
estimate, not an operating-system memory limit; allow additional headroom
for geometry. Output paths must be fresh. Failures preserve `FAILED.json`
and partial evidence rather than producing a completion receipt.

## Outputs and diagnostics

- `BORA_FULL_SLIDE.mask-0p5um.tif`: one final global raster, derived from the
  final working-coordinate geometry.
- `BORA_FULL_SLIDE.working.geojson`: calibrated working-pixel contours.
- `BORA_FULL_SLIDE.wsi.geojson`: original WSI-pixel contours and physical-area
  metadata. A guarded native-coordinate repair removes numerical remnants.
- `FULL_SLIDE_COMPLETE.json`: input/output identities, settings and audits.
- Each crop contains its exact raster window, clipped native GeoJSON and
  receipt stating that neither refinement nor cleanup ran in that crop.

Cropping can create a fragment below 400 µm² where a large global polygon
crosses the diagnostic edge. Such a fragment is retained: it is not a small
global tissue region. Removing it would make the zoom disagree with the WSI.
Display the final GeoJSON boundary as black lines over the colored mask and
H&E; preserve the separate grayscale responses and full-slide RGB.

For tissue names, use the sole Neighbor context fastPLS classifier described
in [REGION_IDENTITY.md](REGION_IDENTITY.md), with independent training labels
and full-channel descriptors. The anonymous segmentation freeze does not
establish semantic accuracy or an independent test: the colorectal slide was
inspected during development.
