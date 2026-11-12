# Bora validation code freeze — 12 November 2026

This release fixes the **code candidate** for a new external validation.
The Git tag identifies source bytes; `VALIDATION_FREEZE_2026-11-12.json`
records SHA256 hashes for every source file in the release. Provider images,
expert masks, Virchow2 weights/features, and fitted project data remain in
their governed stores. The scripts require checksum-bound local manifests.

## Two distinct inference routes

* **KODAMA–Bora:** calibrated H&E and a reusable cache of all 1,280 Virchow2
  spatial channels feed image-only clustering and Bora group projection.
  KODAMA uses a PCA representation of sampled tiles; Bora selects response
  channels internally from the full-channel source. The group
  number and geometry are anonymous. A separate, training-fitted fastPLS
  model predicts a tissue name for each group without changing its number,
  color or boundaries. See `src/bora/` and `methods/kodama_bora/`.
* **Supervised Bora:** a classwise training-only selector retains 203 of the
  1,280 channels and fits an eight-class ridge classifier on 397 expert-masked
  training images. `methods/benchmark1/ridge/` evaluates ROI outputs;
  `methods/benchmark2/supervised_ridge/` projects the same frozen head to
  full slides, applies the training-selected 2% class-presence floor, 6 µm
  contour smoothing and 2,000 µm² physical-area cleanup, and exports TIFF
  plus native-coordinate GeoJSON.

The retained 87 Benchmark 1 images were inspected during development. Their
0.701618 mean case macro IoU is a **development** result, not an independent
test. The 13 public Benchmark 2 whole-slide displays have no independent
whole-slide expert masks or accuracy estimate. The fresh eight-class
KODAMA/group-majority refit failed its prespecified rare-class viability
gate and is not a fitted final model. Earlier KODAMA group outputs and the
separate fastPLS tissue names remain distinct from the supervised ridge head.

## Prospective evaluation contract

Benchmark 1 is ROI-only; Benchmark 2 is WSI-only. Before any external
reference annotations are opened, freeze the cohort/patient identities,
source overlap audit, ontology mapping, applicable method, support/background
rule, completion definition, failure penalty, scorer and runtime environment.
Report all-selected-case completion and accuracy as well as conditional
accuracy. A dataset-specific adaptation is a separate experiment with its
own patient-disjoint train/validation/test partition; it cannot replace the
zero-shot external result. PM016 and the withdrawn BEETLE source are outside
this validation release.
