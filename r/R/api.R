#' Prepare Virchow2 features for calibrated H&E images
#'
#' A supplied tissue mask limits encoder windows to tissue. Without a mask,
#' every image tile is encoded, including glass. Benchmark runs must provide
#' precomputed tissue masks. The result references on-disk full-channel arrays;
#' it does not copy whole-slide arrays into R.
#'
#' @param images Character vector of calibrated RGB H&E image paths.
#' @param tissue_masks Optional aligned raster or GeoJSON path per image.
#' @param feature_dir New directory for reusable feature caches.
#' @param model_dir Local licensed Virchow2 model directory.
#' @param scales Integer scale factors; defaults to 1, 2 and 4.
#' @return A manifest object with feature paths and provenance.
#' @export
Bora.preparation <- function(images, tissue_masks = NULL, feature_dir,
                             model_dir = Sys.getenv("BORA_VIRCHOW2_MODEL"),
                             scales = c(1L, 2L, 4L)) {
  images <- .bora_paths(images, "images")
  tissue_masks <- .bora_paired_paths(tissue_masks, images, "tissue_masks")
  feature_dir <- .bora_output_dir(feature_dir, "feature_dir")
  model_dir <- .bora_paths(model_dir, "model_dir")
  if (length(model_dir) != 1L || !dir.exists(model_dir)) {
    stop("model_dir must be one existing Virchow2 directory", call. = FALSE)
  }
  if (!is.numeric(scales) || !length(scales) || anyNA(scales) ||
      any(scales != as.integer(scales)) || any(scales < 1L) ||
      anyDuplicated(scales)) {
    stop("scales must be distinct positive integers", call. = FALSE)
  }
  .bora_backend()$prepare(images, tissue_masks, feature_dir, model_dir,
                          as.integer(scales))
  .bora_result(file.path(feature_dir, "PREPARATION_COMPLETE.json"),
               "bora-r-preparation-v1")
}

#' Train a supervised Bora class predictor
#'
#' Each annotation must encode class IDs, or class_labels must map its labels
#' to class IDs. A binary tissue-support mask alone cannot train a semantic
#' predictor. The returned model is stored on disk with a checksum receipt.
#'
#' @param images Calibrated H&E image paths.
#' @param annotations Aligned class-mask raster or GeoJSON path per image.
#' @param feature_dir An existing Bora.preparation manifest or cache directory.
#' @param class_labels Optional per-image label mappings.
#' @param output_dir New model output directory.
#' @return A predictor manifest object.
#' @export
Bora.training <- function(images, annotations, feature_dir,
                          class_labels = NULL, output_dir) {
  images <- .bora_paths(images, "images")
  annotations <- .bora_paired_paths(annotations, images, "annotations")
  if (is.null(annotations)) {
    stop("annotations are required for supervised training", call. = FALSE)
  }
  if (!is.null(class_labels) &&
      (!is.list(class_labels) || length(class_labels) != length(images))) {
    stop("class_labels must be a list with one mapping per image", call. = FALSE)
  }
  if (inherits(feature_dir, "bora-r-preparation-v1")) {
    feature_dir <- dirname(feature_dir$manifest)
  }
  feature_dir <- .bora_paths(feature_dir, "feature_dir")
  if (length(feature_dir) != 1L || !dir.exists(feature_dir)) {
    stop("feature_dir must be one existing preparation directory", call. = FALSE)
  }
  output_dir <- .bora_output_dir(output_dir, "output_dir")
  .bora_backend()$train(images, annotations, feature_dir,
                        class_labels, output_dir)
  .bora_result(file.path(output_dir, "TRAINING_COMPLETE.json"),
               "bora-r-training-v1")
}

#' Predict tissue masks from prepared Virchow2 features
#'
#' Without a predictor, tiles from all prepared images are jointly analyzed
#' by the frozen KODAMA route, then returned to their original image grids.
#' Multiple clustering resolutions produce separate output subdirectories.
#'
#' @param preparation Bora.preparation result or its cache directory.
#' @param predictor Optional Bora.training result or model directory.
#' @param output_dir New prediction directory.
#' @param resolutions Optional numeric clustering-resolution values.
#' @return A prediction manifest object.
#' @export
Bora.prediction <- function(preparation, predictor = NULL, output_dir,
                            resolutions = NULL) {
  if (inherits(preparation, "bora-r-preparation-v1")) {
    preparation <- dirname(preparation$manifest)
  }
  preparation <- .bora_paths(preparation, "preparation")
  if (length(preparation) != 1L || !dir.exists(preparation)) {
    stop("preparation must be one existing feature directory", call. = FALSE)
  }
  if (inherits(predictor, "bora-r-training-v1")) {
    predictor <- dirname(predictor$manifest)
  }
  if (!is.null(predictor)) {
    predictor <- .bora_paths(predictor, "predictor")
    if (length(predictor) != 1L || !dir.exists(predictor)) {
      stop("predictor must be one existing model directory", call. = FALSE)
    }
    if (!is.null(resolutions)) {
      stop("resolutions apply only to KODAMA predictions", call. = FALSE)
    }
  } else if (!is.null(resolutions) &&
             (!is.numeric(resolutions) || !length(resolutions) ||
              anyNA(resolutions) || any(!is.finite(resolutions)) ||
              any(resolutions <= 0) || anyDuplicated(resolutions))) {
    stop("resolutions must be distinct positive finite values", call. = FALSE)
  }
  output_dir <- .bora_output_dir(output_dir, "output_dir")
  .bora_backend()$predict(preparation, predictor, output_dir, resolutions)
  .bora_result(file.path(output_dir, "PREDICTION_COMPLETE.json"),
               "bora-r-prediction-v1")
}
