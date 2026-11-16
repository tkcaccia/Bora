## ----contract-----------------------------------------------------------------
if (requireNamespace("Bora", quietly = TRUE)) {
  stopifnot(utils::packageVersion("Bora") >= "0.2.0")
  missing_spacing <- tryCatch(
    Bora::Bora.prediction(tempdir(), output_dir = tempfile("bora-vignette-")),
    error = conditionMessage
  )
  stopifnot(grepl("Measured spacing source_mpp_xy", missing_spacing,
                fixed = TRUE))
}


## ----prepare, eval=FALSE------------------------------------------------------
# image <- "/data/roi-a.tif"
# tissue <- "/data/roi-a-tissue.geojson"
# model <- "/models/Virchow2-v1"
# prepared <- Bora.preparation(
#   images = image, tissue_masks = tissue,
#   feature_dir = "/results/roi-a-features-v1", model_dir = model,
#   scales = c(1L, 2L, 4L), minimum_free_gib = 100
# )


## ----supervised, eval=FALSE---------------------------------------------------
# head <- Bora.training(
#   images = image, annotations = "/data/roi-a-classes.tif",
#   feature_dir = prepared, output_dir = "/results/head-v1"
# )
# result <- Bora.prediction(
#   preparation = prepared, predictor = head,
#   output_dir = "/results/roi-a-prediction-v1",
#   source_mpp_xy = c(0.50, 0.50), # replace with measured X and Y values
#   output_mpp_xy = c(4, 4), minimum_area_um2 = 2000,
#   contour_sigma_um = 6, minimum_free_gib = 100
# )


## ----joint, eval=FALSE--------------------------------------------------------
# anonymous <- Bora.prediction(
#   preparation = prepared, predictor = NULL,
#   resolutions = c(0.5, 1, 2), output_dir = "/results/joint-v1",
#   source_mpp_xy = c(0.50, 0.50), # all inputs must have this measured spacing
#   minimum_free_gib = 100
# )


## ----spatial, eval=FALSE------------------------------------------------------
# spatial <- Bora.spatial(
#   image = image, object = clustered_object,
#   tissue_mask = tissue, output_dir = "/results/spatial-v1",
#   source_mpp_xy = c(0.50, 0.50), # measured H&E pixel spacing
#   coordinate_scale_xy = c(1, 1)
# )
# stopifnot(file.exists(spatial$mask), file.exists(spatial$geojson))

