# KODAMA three-group Bora route

This CPU route maps a **registered, global** Virchow2 PCA32 token field to three
anonymous KODAMA groups, refines boundaries against native H&E, and publishes
one full-slide tissue annotation. It does not include a pretrained example-slide model
or infer tumor/stroma names. Biological identities require independent labels
and a holdout evaluation. The existing fastPLS Bora-AF `bora refine` command
remains available for its trained tissue-classification workflow.

## 1. Fit a numeric head

Make a `.npz` containing `features: float32[N,32]` and integer `labels: [N]`
with IDs 1, 2, 3. All samples must share one documented PCA projection and
registered Virchow2 token frame. Put the file hash, the training annotation
hash, and the PCA projection hash in `fit.json`:

```json
{
  "format": "bora-kodama-three-group-fit-v1",
  "source_frame_id": "training-slide/registered-token-grid",
  "source_image_sha256": "64 lowercase hexadecimal characters for native H&E",
  "training_annotation_sha256": "64 lowercase hexadecimal characters",
  "pca_projection_sha256": "64 lowercase hexadecimal characters",
  "class_ids": [1, 2, 3],
  "samples": {"path": "samples.npz", "sha256": "64 lowercase hexadecimal characters"},
  "regularization_c": 0.1,
  "training_prior": [0.3, 0.4, 0.3]
}
```

```bash
bora kodama-fit fit.json --output model
```

`model/numeric_head.npz` contains only fitted numeric arrays and the explicit
prior. `model/FIT_COMPLETE.json` binds model, samples, annotation, PCA
projection, source image, and source frame by SHA-256. The optional prior in the manifest
must describe the intended training distribution; if omitted, Bora records
the observed class frequency of the supplied samples.

## 2. Score the complete token lattice

Supply a `float32[Y,X,32]` global PCA lattice and a `bool[Y,X]` tissue-support
array as `.npy` files. They must share exact token coordinates and the same PCA
projection as the fitted head. Hash-bound `score.json` declares this assertion:

```json
{
  "format": "bora-kodama-three-group-score-v1",
  "registration_method": "same_global_token_grid",
  "source_frame_id": "training-slide/registered-token-grid",
  "source_image_sha256": "same native H&E SHA-256 as fit.json",
  "pca_projection_sha256": "same SHA-256 as fit.json",
  "field": {"path": "pca32.npy", "sha256": "64 lowercase hexadecimal characters"},
  "support": {"path": "token-support.npy", "sha256": "64 lowercase hexadecimal characters"}
}
```

```bash
bora kodama-score score.json --model model --output score \
  --sigma-tokens 2.9 --prior-exponent 0.78
```

The score manifest must identify the same source image and frame as the fitted
head. A head fitted on one slide cannot silently score another slide. The output
is `score/three_group_probability.npy` and a
`score/SCORE_COMPLETE.json` binding inputs, model, parameters, output, and
source frame. Pooling crosses encoder-patch seams while excluding background.
The scorer streams bounded token tiles into a memory-mapped `.npy`; each tile
uses the complete finite Gaussian halo, so output agrees with whole-field
scoring across tile and encoder seams. The default core is 256×256 tokens;
the public Python API accepts `tile_size` in 32..2048. Disk admission reserves
the output size plus 1 GiB.
The sigma/prior values above are an example, **not** universal
calibration. Choose and validate values on the new tissue's training and
holdout regions before publication.

## 3. Refine native H&E and assemble once

The selected route consumes the **one saved global probability field** from
step 2, the original calibrated H&E OME-TIFF, and a binary tissue-support TIFF.
The support may have a different raster size; absolute pixel centres are used
when it is sampled in the native image frame. Put the actual hashes and
micrometres-per-pixel values in `native.json`:

```json
{
  "format": "bora-kodama-native-v1",
  "source_frame_id": "slide/native-frame",
  "working_shape_yx": [224, 336],
  "working_mpp_xy": [0.5, 0.5],
  "native_mpp_xy": [0.25, 0.25],
  "score": {"path": "score/three_group_probability.npy", "sha256": "64 lowercase hexadecimal characters"},
  "score_receipt": {"path": "score/SCORE_COMPLETE.json", "sha256": "64 lowercase hexadecimal characters"},
  "native_he": {"path": "image/native-he.ome.tif", "sha256": "64 lowercase hexadecimal characters"},
  "tissue_support": {"path": "image/tissue-support.tif", "sha256": "64 lowercase hexadecimal characters"},
  "core_size": 256,
  "halo_pixels": 72,
  "boundary_spec": {
    "format": "bora-local-he-boundary-v1",
    "uncertainty_margin": 0.3,
    "maximum_distance_um": 12.0,
    "he_blur_um": 0.5,
    "he_edge_sigma": 0.12,
    "smoothness": 1.5,
    "iterations": 4
  }
}
```

```bash
bora kodama-native native.json --output native-mask
```

The native score is interpolated from **absolute** token and pixel centres,
so changing a processing window does not shift its coordinates. H&E guides
only uncertain internal boundaries. No class is fitted separately in a
window. A complete run writes `semantic-mask.uint8.dat` and
`ASSEMBLY_COMPLETE.json`; an interrupted run has no completion receipt. The
halo must cover the specified physical movement and H&E blur. Declared native
spacing must match the OME pixel sizes. Input hashes
are checked again before the mask is marked complete.

## 4. Publish exact and editable contours

```json
{"1": "KODAMA Group 1", "2": "KODAMA Group 2", "3": "KODAMA Group 3"}
```

Save that as `group-names.json` and run:

```bash
bora kodama-contours native-mask native-he.tif group-names.json \
  --output contours --minimum-area-um2 1000 --minimum-free-gib 44
```

The free-space reserve is an execution policy; 1 GiB is the CLI default and
the example requests a 44 GiB reserve. Bora globally polygonizes the single
full-slide mask, removes components and holes **strictly below** the specified
physical area, and preserves an exact cleaned mask and GeoJSON. Guarded
coverage simplification then selects the largest passing whole-boundary
tolerance, producing diagonal contours without independently fitted window
cuts. Acceptance checks include component identity/count, valid nonoverlapping
geometry, minimum class IoU 0.995, changed-pixel displacement at most 5 µm,
and separate tissue-loss and tissue-gain limits of 0.05% each (0.1% total).

`contours/COMPLETE.json` marks success and points to
`global_vectors/exact/mask.uint8.dat`, `global_vectors/exact/mask.geojson`,
and `tissue_annotations/BORA_TISSUE_ANNOTATIONS.geojson`. Import only that
last GeoJSON as ordinary editable tissue annotations in wsiTools; retain the
exact mask as the pixel-level reproducibility reference. If a guard fails,
there is no completion receipt. The choice of area floor and simplification tolerance should be validated on
the target tissue and cannot establish biological accuracy by itself.

## 5. Add trained tissue identities without changing contours

The anonymous groups can subsequently receive tissue names from the frozen
Neighbor-context PLS model with `bora classify-regions`. This uses the original
1280-channel token field, not the PCA32 numeric group head. All pieces of one
source group share one tissue prediction, and every accepted geometry stays
unchanged. See [the registration, outputs and transfer limitations](REGION_IDENTITY.md).
