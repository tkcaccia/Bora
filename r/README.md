# Bora for R (development)

This R package is being built alongside the MIT Bora Python package. It is
not yet a released interface. `Bora.preparation`, `Bora.training`, and the
supervised branch of `Bora.prediction` have synthetic end-to-end pilots.
The KODAMA branch of `Bora.prediction` is still under development. None of
these routes has yet passed the required real-image regression for release.

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

For benchmark runs, supply an independently prepared tissue mask for every
image. Without a mask, preparation intentionally encodes the full H&E field,
including glass. Class annotations used by training are separate from these
tissue-support masks.

Supervised prediction projects class scores from the full-channel factor-1
cache, saves native and common-support-clipped masks separately, applies
measured-area and contour cleanup, and writes class-specific binary TIFFs,
the multiclass TIFF, source-pixel-coordinate GeoJSON, and a preview. The
physical pixel spacing must be supplied for each calibrated image. Synthetic
output equivalence does not establish whole-slide memory or numerical
regression against the historical eight-class release.
