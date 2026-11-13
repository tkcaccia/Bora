# Supervised Bora commands

For one checksum-pinned BCSS ROI command and its accepted pixel hash, see
[reproduce one result](reproduce-one-result.md).

The installable `bora` program now keeps the KODAMA/fastPLS commands and adds
`fit`, `predict`, `evaluate`, and `verify` for the frozen eight-class supervised
Virchow2 ridge method. These are CPU commands. Supply complete, unmodified
Virchow2 grids with **1,280 channels**; Bora selects channels internally.

The model is a portable `.npz` containing only arrays, not Python pickle. The
fixed training recipe standardizes all training channels, takes the stable
union of the 32 largest one-versus-rest mean differences per class, and fits a
weighted multiclass ridge with alpha 10. It requires all eight retained tissue
classes in the training sample. The corrected PEsO variant uses a separate
unweighted, 64-channel binary ridge trained on classes 2 (weak proxy stroma)
and 4 (expert epithelium), with a fixed zero threshold. Set
`"method": "peso-binary-ridge64"` in its training manifest. Model selection
and cohort splits must be frozen *before* `fit`; these commands do not tune on
evaluation cases.

The historical global head was chosen on development-era training folds later
found to contain cross-dataset TCGA patient-ID overlap. The software command
reproduces that head but does not certify patient-disjoint model selection or
external performance. A repaired-fold selection analysis is maintained
separately; a changed training decision requires a new versioned model and
benchmark receipt.

## Fit

`bora fit training.json --output model.npz`

```json
{
  "format": "bora-supervised-fit-v1",
  "split": "train",
  "source_full_channels": 1280,
  "features": {"path": "training-features.npy", "sha256": "<64 hex>"},
  "labels": {"path": "training-labels.npy", "sha256": "<64 hex>"}
}
```

`training-features.npy` is an `N × 1280` array, and `training-labels.npy` has
values 1–8 with optional zero ignored. To map a frozen older ontology, include
`"old_to_new": {"1": 1, "2": 3, "24": null, ...}`. The output receipt is
`model.npz.json`. A historical `joblib` artifact is never loaded implicitly.

## Predict

`bora predict image.json --output prediction-dir`

```json
{
  "format": "bora-supervised-predict-v1",
  "model": {"path": "model.npz", "sha256": "<64 hex>"},
  "features": {"path": "full-virchow2-grid.npy", "sha256": "<64 hex>"},
  "roi": {
    "working_size_wh": [896, 896],
    "display_size_wh": [896, 896],
    "allowed_classes": [1, 2, 3, 4, 5, 6, 7, 8],
    "mpp_xy": [0.5, 0.5],
    "polish": true
  }
}
```

The feature grid is `H × W × 1280` `float16` or `float32`, with each token
corresponding to a 14 × 14 pixel region. Prediction writes class-first raw
and Gaussian-smoothed score grids, and for ROIs writes a pixel-level class-mask
PNG. Nearest finite selected tokens fill nonfinite gaps without changing the
source cache. The ROI contour rule smooths labels over 1.5 µm and reassigns
8-connected islands below 400 µm². For the corrected PEsO head, supply
`"allowed_classes": [2, 4]` and a `"support_mask"` path/SHA object inside
`roi`; the stroma target is an H&E-derived weak proxy, not pathologist
annotation. Without `roi`, `predict` writes bounded-row score
grids. To finalize a WSI, add a `wsi` object:

```json
{
  "calibrated_shape_yx": [22398, 41209],
  "source_frame_id": "slide-id:calibrated_0p5um",
  "source_image": {"path": "calibrated.tif", "sha256": "<64 hex>"},
  "support_geojson": {"path": "reviewed-bora-support.geojson", "sha256": "<64 hex>"},
  "ontology": {"path": "eight-class-ontology.json", "sha256": "<64 hex>"},
  "class_presence_receipt": {"path": "CLASS_PRESENCE_FROZEN.json", "sha256": "<64 hex>"},
  "minimum_predicted_tissue_fraction": 0.02
}
```

The complete-slide path applies reviewed Bora tissue support, the
training-frozen 2% class-presence rule, bicubic pixel projection, 400 µm²
precontour cleanup, 6 µm contour smoothing, and 2,000 µm² final raster and
vector cleanup. It checks every positive island and enclosed hole, then writes
a full-resolution TIFF and native-coordinate GeoJSON. This is the same
physical-scale finalization policy used by the current KODAMA–Bora whole-slide
route; the class scores and group generation remain different. The supplied
support must be in original native pixel edges and include a calibrated-to-
native transform. Crops are display-only and never fitted independently.

Python users can run the complete supervised slide pipeline as one function:

```python
from pathlib import Path
from bora import segment_full_slide

result = segment_full_slide(Path("image.json"), Path("output/slide-001"))
print(result["mask_tiff"])
print(result["native_geojson"])
```

`image.json` is the `bora-supervised-predict-v1` manifest above with its `wsi`
section, a SHA256-pinned eight-class ridge model, and a SHA256-pinned full
1,280-channel feature grid; omit `roi`. This CPU function calls the same path
as `bora predict image.json --output output/slide-001` and verifies the final
vector against the TIFF and reviewed support. It refuses an existing output
directory and preserves partial files on failure. The output folder also
contains `full-slide/FULL_SLIDE_VERIFIED.json`, class selection, cleanup and
contour receipts. The installed package matched one frozen public ovarian
slide in every full-resolution mask pixel and native polygon feature; that
regression does not establish accuracy or equivalence on all slides.

## Evaluate and verify

`bora evaluate cases.json --output scores.json` accepts a manifest containing
`"format": "bora-supervised-evaluate-v1"` and a nonempty `cases` list. Each
row has `id`, `prediction` and `target` objects with paths and SHA256 values,
plus measured `spacing_yx_um`. The two masks must already share the same
registered frame; the scorer never resizes or remaps them. Zero in the target
is ignored; 1–8 are scored. The result reports per-present-class IoU/Dice,
case macro scores, scored-pixel coverage, internal-interface boundary F1 at
5 µm, symmetric boundary HD95 and symmetric region covering independent of
class names. These endpoints remain separate. The scorer rejects windows over
24 million pixels; whole slides require prespecified annotated windows. Mark
`is_independent_test` true only for a genuinely untouched test cohort.

`bora verify model.npz`, `bora verify prediction-dir`, and
`bora verify scores.json` validate formats, hashes, array dimensions and, for
evaluations, recompute each case's metrics. Use a fresh versioned output path
for every run. Existing output folders and partial files are not overwritten.

The KODAMA route retains its established `kodama-fit`, `kodama-score`,
`kodama-native`, `prepare`, `train`, `refine`, and whole-slide commands. Frozen
historical scripts and receipts are preserved; new commands are compared to
those outputs before any entry point is retired.
