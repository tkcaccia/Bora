# Bora for R (development)

This R package is being built alongside the MIT Bora Python package. It is
not yet a released interface. `Bora.preparation`, `Bora.training`, and the
supervised branch of `Bora.prediction` have synthetic end-to-end pilots.
The joint-image KODAMA branch of `Bora.prediction` also passed a fabricated
two-image, two-resolution end-to-end pilot. One BCSS H&E region has passed
source-registration and frozen supervised-head regressions. A separate frozen
BCSS training image has passed real-image R fitting and prediction with an
independent source-mask, coefficient and vector audit. Full-slide resource
qualification and a broader reproducibility audit remain pending.

The R package calls the version-matched Bora Python backend through
`reticulate`. Virchow2 weights are separately licensed and are never bundled
with Bora. A user-managed Python environment with Bora, PyTorch, timm and the
licensed Virchow2 checkpoint is required for feature extraction. Set
`RETICULATE_PYTHON` before starting R and `BORA_VIRCHOW2_MODEL` to the model
directory. The production install command and result reproduction guide will
be published after all three end-to-end routes pass regression.

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
including glass. Class annotations used by training are separate from these
tissue-support masks.

Supervised prediction projects class scores from the full-channel factor-1
cache, saves native and common-support-clipped masks separately, applies
measured-area and contour cleanup, and writes class-specific binary TIFFs,
the multiclass TIFF, source-pixel-coordinate GeoJSON, and a preview. The
physical pixel spacing must be supplied for each calibrated image. A read-only
adapter of the frozen 203-channel eight-class ridge head reproduced one
historical BCSS display mask exactly (453,888 of 453,888 pixels), and R and
Python projections produced identical score-array hashes. This is one
inspected-development image, with no expert reference opened; it does not
establish whole-slide memory or broad numerical equivalence.

With `predictor = NULL`, prediction pools every strictly admitted 224-pixel
tile from the prepared images into one source-bound KODAMA 0.99.7 fit. A
supplied tissue mask admits a tile only when more than 90% of its pixels are
supported; artificial padding is background. Without a mask, the entire
source image, including glass, remains eligible under the same padding rule.
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

A source-only BCSS H&E region regression reused an independently verified
full-1,280-channel historical cache and a separately computed H&E tissue
support. Forty-eight strict tiles yielded two and five anonymous groups at
resolutions 1.0 and 1.5. The binary TIFFs, final raster, and source-coordinate
GeoJSON passed exact pixel equality after a one-pixel support-hole mismatch
was found and fixed in the developing vector export. This checks one real
region's source registration and output geometry. It does not establish class
accuracy, historical KODAMA–Bora equivalence, or whole-slide scalability.
