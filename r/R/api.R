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
#' @param max_seconds Finite maximum GPU seconds for each image-scale worker.
#' @param minimum_free_gib Minimum free-space reserve after projected features.
#' @return A manifest object with feature paths and provenance.
#' @export
Bora.preparation <- function(images, tissue_masks = NULL, feature_dir,
                             model_dir = Sys.getenv("BORA_VIRCHOW2_MODEL"),
                             scales = c(1L, 2L, 4L),
                             max_seconds = 3600L, minimum_free_gib = 1) {
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
  if (!is.numeric(max_seconds) || length(max_seconds) != 1L ||
      is.na(max_seconds) || max_seconds != as.integer(max_seconds) ||
      max_seconds < 60L || !is.numeric(minimum_free_gib) ||
      length(minimum_free_gib) != 1L || is.na(minimum_free_gib) ||
      !is.finite(minimum_free_gib) || minimum_free_gib < 0) {
    stop("Finite max_seconds and minimum_free_gib values are required", call. = FALSE)
  }
  .bora_backend()$prepare(as.list(images),
                          if (is.null(tissue_masks)) NULL else as.list(tissue_masks),
                          feature_dir, model_dir, as.list(as.integer(scales)),
                          max_seconds = as.integer(max_seconds),
                          minimum_free_gib = as.numeric(minimum_free_gib))
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
#' @param max_tokens_per_class Maximum sampled factor-1 token centres per class
#'   across the training images, allocated evenly by image.
#' @param channels_per_class Standardized contrast channels selected per class.
#' @param alpha Positive ridge penalty.
#' @return A predictor manifest object.
#' @export
Bora.training <- function(images, annotations, feature_dir,
                          class_labels = NULL, output_dir,
                          max_tokens_per_class = 8000L,
                          channels_per_class = 32L, alpha = 10) {
  images <- .bora_paths(images, "images")
  annotations <- .bora_paired_paths(annotations, images, "annotations")
  if (is.null(annotations)) {
    stop("annotations are required for supervised training", call. = FALSE)
  }
  if (!is.null(class_labels) &&
      (!is.list(class_labels) || length(class_labels) != length(images))) {
    stop("class_labels must be a list with one mapping per image", call. = FALSE)
  }
  if (!is.null(class_labels)) {
    class_labels <- lapply(class_labels, function(mapping) {
      if (!is.numeric(mapping) || !length(mapping) || anyNA(mapping) ||
          is.null(names(mapping)) || any(!nzchar(names(mapping))) ||
          anyDuplicated(names(mapping)) ||
          any(mapping != as.integer(mapping)) ||
          any(mapping < 1L | mapping > 254L)) {
        stop("Each class mapping must be a named integer vector with IDs 1..254",
             call. = FALSE)
      }
      # A named R list becomes a Python dict; a named atomic vector does not.
      value <- as.list(as.integer(mapping))
      names(value) <- names(mapping)
      value
    })
  }
  if (inherits(feature_dir, "bora-r-preparation-v1")) {
    feature_dir <- dirname(feature_dir$manifest)
  }
  feature_dir <- .bora_paths(feature_dir, "feature_dir")
  if (length(feature_dir) != 1L || !dir.exists(feature_dir)) {
    stop("feature_dir must be one existing preparation directory", call. = FALSE)
  }
  output_dir <- .bora_output_dir(output_dir, "output_dir")
  if (!is.numeric(max_tokens_per_class) || length(max_tokens_per_class) != 1L ||
      is.na(max_tokens_per_class) || max_tokens_per_class != as.integer(max_tokens_per_class) ||
      max_tokens_per_class < 1L || max_tokens_per_class > 100000L ||
      !is.numeric(channels_per_class) || length(channels_per_class) != 1L ||
      is.na(channels_per_class) || channels_per_class != as.integer(channels_per_class) ||
      channels_per_class < 1L || channels_per_class > 1280L ||
      !is.numeric(alpha) || length(alpha) != 1L || is.na(alpha) ||
      !is.finite(alpha) || alpha <= 0) {
    stop("Finite positive training limits and ridge penalty are required", call. = FALSE)
  }
  .bora_backend()$train(as.list(images), as.list(annotations), feature_dir,
                        class_labels, output_dir,
                        max_tokens_per_class = as.integer(max_tokens_per_class),
                        channels_per_class = as.integer(channels_per_class),
                        alpha = as.numeric(alpha))
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
