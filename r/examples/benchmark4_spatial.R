#!/usr/bin/env Rscript
# Use a registered Seurat, Giotto or SpatialExperiment object (or a spot table)
# and one full-resolution H&E image to generate a new expression-guided mask.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 7L)
  stop("Usage: Rscript benchmark4_spatial.R IMAGE OBJECT.rds TISSUE_MASK MODEL_DIR OUTPUT_DIR MPP_X MPP_Y")
image <- normalizePath(args[[1L]], mustWork = TRUE)
object <- readRDS(normalizePath(args[[2L]], mustWork = TRUE))
support <- normalizePath(args[[3L]], mustWork = TRUE)
model <- normalizePath(args[[4L]], mustWork = TRUE)
out <- normalizePath(args[[5L]], mustWork = FALSE)
mpp <- as.numeric(args[6:7])
if (anyNA(mpp) || any(!is.finite(mpp)) || any(mpp <= 0))
  stop("Measured positive X/Y image spacing is required")
result <- Bora::Bora.spatial(
  image = image, object = object,
  cluster_column = "KODAMA_Louvain",
  tissue_mask = support, model_dir = model,
  source_mpp_xy = mpp, output_dir = out)
cat("Mask:", result$mask, "\nGeoJSON:", result$geojson,
    "\nReceipt:", result$manifest, "\n")
