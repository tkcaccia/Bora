# Bora for R 0.1.6: reference and acceptance manual

Bora provides four user functions. `Bora.preparation()` extracts reusable
Virchow2 features; `Bora.training()` fits a class predictor from registered
expert masks; `Bora.prediction()` produces class or anonymous-group maps; and
`Bora.spatial()` projects expression-derived spot groups onto an H&E image.
The R package uses the matching Bora Python backend 0.3.5 through `reticulate`.
Bora code is MIT licensed. Virchow2 weights and image datasets retain their
own terms and are not included.

## Set up

Install the R package and Python backend from tag `bora-r-v0.1.6`. Supply a
licensed Virchow2 checkpoint yourself. Set `RETICULATE_PYTHON` to the Python
interpreter containing Bora, PyTorch, timm and rasterio, and set
`BORA_VIRCHOW2_MODEL` to the model directory before starting R. Read
`REPRODUCE_ONE_RESULT.md` for a complete frozen training-image replay.

Images must be calibrated RGB H&E with known physical X/Y pixel spacing.
Masks and GeoJSON must be registered to the exact source image. Keep distinct
directories for inputs, features, predictors and predictions. Use a new output
directory for each run; retain the JSON SHA256 receipts.

```r
library(Bora)
prepared <- Bora.preparation(
  images = c("roi-a.tif", "roi-b.tif"),
  tissue_masks = c("roi-a-tissue.tif", "roi-b-tissue.geojson"),
  feature_dir = "features-v1", model_dir = Sys.getenv("BORA_VIRCHOW2_MODEL"),
  scales = c(1L, 2L, 4L), max_seconds = 3600L,
  minimum_free_gib = 100)
head <- Bora.training(
  images = c("roi-a.tif", "roi-b.tif"),
  annotations = c("roi-a-classes.tif", "roi-b-classes.geojson"),
  feature_dir = prepared, output_dir = "predictor-v1")
result <- Bora.prediction(
  preparation = prepared, predictor = head, output_dir = "prediction-v1",
  source_mpp_xy = c(0.5, 0.5), output_mpp_xy = c(4, 4),
  minimum_free_gib = 100)
```

Here `L` means an R integer. `3600L` is 3,600 seconds, the default maximum
for **each image-scale encoder worker**. `c(1L, 2L, 4L)` requests the finest
token grid and two coarser image scales. It is not a list of tissue classes.
The default free-space reserve is only 1 GiB; set a larger local reserve when
running whole slides. Chiamaka validation uses 100 GiB.

### Preparation and background

A tissue mask controls **encoder coverage**. It is separate from the expert
class annotations used for fitting. Without a mask, Bora encodes the entire
image field, including the empty transparent glass slide background visible
outside tissue. For benchmark work, provide a reviewed, precomputed tissue
mask. The factor-1/2/4 caches contain all 1,280 Virchow2 channels at every
generated token, even if a later predictor selects fewer channels. A verified
full-field cache can be adopted with a new tissue mask; its receipt still
discloses that the encoder originally covered background.

### Supervised training and prediction

Raster annotations hold class IDs; GeoJSON features may hold `class_id`.
Optional `class_labels` maps source labels to positive IDs from 1 to 254.
Zero is unscored. A binary tissue-support mask does not provide semantic
class training. Bora samples finite factor-1 token centres from annotated
training images, selects contrast channels and fits a ridge head. Select
channels, hyperparameters and thresholds using training-only folds. Do not
use evaluation masks in fitting or model selection. A class absent from all
training annotations cannot be learned.

With a predictor, `Bora.prediction()` projects the head to each prepared
image. It writes native and common-support masks, cleaned multiclass TIFF,
one 0/255 binary TIFF per predicted class, source-image-pixel GeoJSON and a
preview. Binary images and vectors are derived from the same final multiclass
mask. Record measured `source_mpp_xy`; the vector is in source pixel-edge
coordinates, not output-grid pixels. A different preview backend may change
the preview RGB but must leave mask and vector pixels unchanged.

### Joint anonymous groups

With `predictor = NULL`, Bora pools strictly admitted tissue tiles across
**all** images in the preparation, runs pinned KODAMA 0.99.7 once, and returns
each requested `resolutions` solution to each original image grid. A supplied
tissue mask admits a 224-pixel tile only if more than 90% of its area is tissue;
padding counts as background. `max_joint_tiles` is an admission limit; the
function does not silently sample tiles to meet it. Group numbers are
anonymous partitions and are not tissue diagnoses.

### Spatial expression-guided route

`Bora.spatial(image, object, output_dir, ...)` accepts a Seurat, Giotto,
SpatialExperiment or spot-table object. It reads spot IDs, cluster labels and
X/Y image coordinates, then fits the group names to the corresponding
Virchow2 tokens on the **same section**. Supply full-resolution image pixel
coordinates. Use `coordinate_scale_xy` only for a known coordinate conversion.
The function rejects duplicate IDs, missing/nonfinite coordinates, points
outside the image and different groups at one token. It returns binary group
images, a multiclass TIFF, source-pixel GeoJSON, preview and a spot receipt.
The output describes morphology associated with input expression groups; it
is not a cell segmentation, independent tissue annotation or estimate of
individual-cell expression.

## Verification performed on Chiamaka, 16 November 2026

The following are software and geometry checks, not new biological accuracy
estimates. Evidence and source hashes are under
`migration/bora-r-acceptance-2026-11-16-v1/` in the Bora workspace.

| Route | Check and acceptance result |
| --- | --- |
| Installation | R 0.1.6 source package built, installed in a clean local library, and `R CMD check --no-manual --no-vignettes` ended `Status: OK`. |
| R and Python contracts | R package tests passed, including invalid scales, class mappings, resource limits, duplicate spots and coordinates. The 24 focused Python R-API tests passed, including source-bound cache rejection and GPU admission/cleanup gates. |
| Real Virchow2 preparation | On a frozen 448 x 448 synthetic H&E fixture, a tissue mask admitted 1 source tile and 6 finest-scale encoder windows; no mask admitted all 4 source tiles and 24 windows. All three arrays were SHA256- and row-hash verified as float16 with 1,280 channels. Both services ended with no owned process or GPU context. |
| Training formats | Equivalent raster and GeoJSON annotations labelled 128 finite token centres per class. Selected channels, normalization and ridge coefficients matched exactly; the model file hashes differ only because the source annotation identity is recorded. |
| Supervised output | Both annotation formats produced the same 56 x 56 mask pixels. Binary TIFFs equalled each class in the multiclass mask, and independent rasterization of the source-pixel GeoJSON equalled the final TIFF. |
| Real training-region replay | Installed R 0.1.6 fitted one approved BCSS **training** region from its SHA256-verified full-channel cache. Every predictor array, final mask TIFF, preview PNG and source GeoJSON matched the frozen prior result exactly; independent vector rasterization and four binary-class images matched the 224 x 224 mask. This is a regression on a training image, not validation accuracy. |
| Spatial output | Eight spots in two expression groups gave the same 28 x 28 mask after input row permutation and after halving coordinates with `coordinate_scale_xy = c(2, 2)`. Binary TIFFs and GeoJSON roundtripped exactly. Duplicate IDs, out-of-bounds/nonfinite positions and one-token conflicting groups were rejected before output. |
| Joint KODAMA | Two distinct-patient BCSS **training** images, 83 strict tiles and two resolutions completed with pinned KODAMA 0.99.7. Four masks and GeoJSON files matched the frozen prior run byte-for-byte. This is a source-registration regression, not test accuracy. |
| Storage and lifecycle | An impossible free-space reserve stopped preparation before any GPU worker. A simulated occupied GPU blocked worker start; an unavailable GPU inspection kept cleanup in `UNKNOWN_OR_FAILED` state. |

## Scientific and operational limits

- These acceptance checks establish the software contracts and output
  geometry. They do not establish new held-out class accuracy or patient- and
  centre-disjoint generalization. Report class, boundary and region metrics
  with completion on a separately frozen expert cohort for those claims.
- Some repository references leave background or tissue types unannotated.
  Do not convert absent annotations into negative labels or combine weak
  proxy labels with expert masks without disclosure.
- Real Virchow2 extraction requires the licensed checkpoint and an admitted
  CUDA GPU. The CPU training and vector stages still need enough host memory
  and disk for their specific image grid. Recheck resource admission per WSI.
- The joint KODAMA route is bounded by `max_joint_tiles` and has not been
  qualified as a full-cohort whole-slide accuracy estimator. Anonymous groups
  require separate expert mapping before semantic scores can be reported.
- CAMELYON16 technical results remain local pending display-rights
  clarification; the acceptance archive does not redistribute source slides.
