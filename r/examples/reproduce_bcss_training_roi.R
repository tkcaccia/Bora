#!/usr/bin/env Rscript
# Technical reproduction on one previously used BCSS training ROI.
# Inputs are supplied separately under their source-data terms.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5L) {
  stop("Usage: Rscript reproduce_bcss_training_roi.R IMAGE SUPPORT CLASS_MASK MODEL_DIR NEW_OUTPUT_DIR")
}
if (file.exists(args[[5L]])) stop("NEW_OUTPUT_DIR already exists")
dir.create(args[[5L]], recursive = TRUE)
library(Bora)
prep <- Bora.preparation(
  images = args[[1L]], tissue_masks = args[[2L]],
  feature_dir = file.path(args[[5L]], "features"),
  model_dir = args[[4L]], scales = c(1L, 2L, 4L),
  minimum_free_gib = 100)
model <- Bora.training(
  images = args[[1L]], annotations = args[[3L]],
  feature_dir = prep, output_dir = file.path(args[[5L]], "model"),
  max_tokens_per_class = 256L, channels_per_class = 16L,
  alpha = 10)
prediction <- Bora.prediction(
  preparation = prep, predictor = model,
  output_dir = file.path(args[[5L]], "prediction"),
  source_mpp_xy = c(0.5, 0.5), output_mpp_xy = c(4, 4),
  minimum_area_um2 = 2000, contour_sigma_um = 6,
  minimum_free_gib = 100, max_new_gib = 4)
cat("Preparation: ", prep$manifest, "\n", sep = "")
cat("Model: ", model$manifest, "\n", sep = "")
cat("Prediction: ", prediction$manifest, "\n", sep = "")
