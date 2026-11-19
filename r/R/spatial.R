#' Refine registered expression groups on an H&E image
#'
#' This exploratory route treats each spot cluster as a sparse training label
#' at its registered image position. It fits a classwise Virchow2 ridge head
#' on those spot-centre tokens, projects the resulting response maps over the
#' H&E field, and applies Bora's usual contour, area and vector export. The
#' output group IDs are expression-derived; they are not expert tissue labels,
#' cell boundaries, or cell-specific expression estimates. Pixels away from
#' spots are image-guided extrapolations.
#'
#' Coordinates must be full-resolution pixel positions in the supplied image,
#' with the origin at its upper left. Use `coordinate_scale_xy` only when the
#' object stores coordinates at a known different image scale. A registered
#' tissue mask is recommended; without one, preparation also encodes empty
#' glass slide background outside the specimen. The function checks unique
#' spot IDs, finite coordinates, image bounds and conflicting group labels
#' within a Virchow2 token before fitting.
#'
#' @param image Path to one calibrated RGB H&E image.
#' @param object Seurat, Giotto, SpatialExperiment, or a data frame with spot
#'   IDs, coordinates and cluster labels.
#' @param output_dir A new output directory.
#' @param cluster_column Metadata column containing expression-group labels.
#'   For Seurat, `NULL` uses active identities. For other objects it selects
#'   `KODAMA_Louvain`, `seurat_clusters`, `leiden_clus`, or `cluster` in that order.
#' @param image_slot Seurat spatial image name; `NULL` uses the default image.
#' @param coordinate_columns Optional two column names in X,Y order.
#' @param coordinate_scale_xy Known multiplier from object coordinates to
#'   full-resolution H&E pixel coordinates, in X,Y order.
#' @param preparation Optional existing `Bora.preparation` result or directory
#'   for this exact image. If absent, full-channel Virchow2 features are made.
#' @param tissue_mask Optional aligned tissue raster or GeoJSON, used only when
#'   `preparation` is absent.
#' @param model_dir Local licensed Virchow2 model directory.
#' @param source_mpp_xy Required measured H&E X/Y pixel spacing in
#'   micrometres; never inferred from a nominal scan magnification.
#' @param output_mpp_xy Output X/Y pixel spacing in micrometres.
#' @param channels_per_group Maximum contrast channels selected per group.
#' @param alpha Positive ridge penalty.
#' @param minimum_area_um2 Minimum positive output component area.
#' @param contour_sigma_um Class-contour smoothing width.
#' @param minimum_free_gib Required free-space reserve.
#' @param max_new_gib Optional prediction peak storage cap.
#' @param preview_backend Preview implementation; see `Bora.prediction`.
#' @param backend `"cuda"` (default) or `"cpu"`. The same backend is used
#'   for new Virchow2 extraction, fitting and score projection.
#' @param n.cores Positive CPU thread limit for all three stages.
#' @param max_seconds Maximum runtime of each CUDA child or feature scale.
#' @return A list of paths to the final TIFF mask, source-pixel GeoJSON,
#'   preview, binary images, spot-to-group mapping and prediction manifest.
#' @export
Bora.spatial <- function(image, object, output_dir, cluster_column = NULL,
                         image_slot = NULL, coordinate_columns = NULL,
                         coordinate_scale_xy = c(1, 1), preparation = NULL,
                         tissue_mask = NULL,
                         model_dir = Sys.getenv("BORA_VIRCHOW2_MODEL"),
                         source_mpp_xy = NULL, output_mpp_xy = c(4, 4),
                         channels_per_group = 32L, alpha = 10,
                         minimum_area_um2 = 2000, contour_sigma_um = 6,
                         minimum_free_gib = 1, max_new_gib = NULL,
                         preview_backend = "vips-lanczos3",
                         backend = "cuda", n.cores = 2L,
                         max_seconds = 3600L) {
  execution <- .bora_execution(backend, n.cores)
  max_seconds <- .bora_max_seconds(max_seconds)
  if (execution$backend == "cuda") .bora_check_cuda()
  image <- .bora_paths(image, "image")
  if (length(image) != 1L) stop("image must be one H&E path", call. = FALSE)
  source_mpp_xy <- .bora_measured_spacing(source_mpp_xy)
  output_dir <- .bora_output_dir(output_dir, "output_dir")
  if (!is.numeric(coordinate_scale_xy) || length(coordinate_scale_xy) != 2L ||
      anyNA(coordinate_scale_xy) || any(!is.finite(coordinate_scale_xy)) ||
      any(coordinate_scale_xy <= 0)) {
    stop("coordinate_scale_xy must contain positive finite X/Y multipliers", call. = FALSE)
  }
  if (!is.null(preparation) && !is.null(tissue_mask)) {
    stop("tissue_mask belongs to feature preparation; omit it when reusing preparation",
         call. = FALSE)
  }
  spots <- .bora_spots(object, cluster_column, image_slot, coordinate_columns)
  spots$x <- spots$x * coordinate_scale_xy[1L]
  spots$y <- spots$y * coordinate_scale_xy[2L]
  rasterio <- reticulate::import("rasterio", convert = TRUE)
  dataset <- rasterio$open(image)
  on.exit(dataset$close(), add = TRUE)
  width <- as.integer(dataset$width)
  height <- as.integer(dataset$height)
  if (any(spots$x < 0 | spots$x >= width | spots$y < 0 | spots$y >= height)) {
    stop("Spot coordinates fall outside the supplied full-resolution H&E image; check registration and coordinate_scale_xy",
         call. = FALSE)
  }
  spots$token_x <- floor(spots$x / 14)
  spots$token_y <- floor(spots$y / 14)
  if (any(14 * spots$token_x + 7 >= width |
          14 * spots$token_y + 7 >= height)) {
    stop("A spot maps to an incomplete edge token without an image token centre",
         call. = FALSE)
  }
  classes <- sort(unique(spots$cluster))
  if (length(classes) < 2L || length(classes) > 32L) {
    stop("Require 2 to 32 observed expression groups", call. = FALSE)
  }
  spots$class_id <- match(spots$cluster, classes)
  key <- paste(spots$token_x, spots$token_y, sep = ":")
  if (any(vapply(split(spots$class_id, key), function(x) length(unique(x)) > 1L,
                 logical(1)))) {
    stop("Different expression groups occupy the same Virchow2 token; refine registration or spot selection",
         call. = FALSE)
  }
  # Multiple spots in one token with the same group provide one training token.
  spot_tokens <- spots[!duplicated(key), , drop = FALSE]
  dir.create(output_dir)
  tryCatch({
    spot_csv <- file.path(output_dir, "registered-spots.csv")
    utils::write.csv(spots[, c("spot_id", "x", "y", "cluster", "class_id",
                               "token_x", "token_y")], spot_csv, row.names = FALSE)
    annotation <- file.path(output_dir, "spot-centre-training.geojson")
    features <- lapply(seq_len(nrow(spot_tokens)), function(i) {
      row <- spot_tokens[i, ]
      x0 <- 14 * row$token_x
      y0 <- 14 * row$token_y
      list(type = "Feature",
           properties = list(class_id = as.integer(row$class_id),
                             spot_id = row$spot_id),
           geometry = list(type = "Polygon", coordinates = list(list(
             c(x0, y0), c(x0 + 14, y0), c(x0 + 14, y0 + 14),
             c(x0, y0 + 14), c(x0, y0)))))
    })
    jsonlite::write_json(list(type = "FeatureCollection", features = features),
                         annotation, auto_unbox = TRUE, digits = NA)
    if (is.null(preparation)) {
      preparation <- Bora.preparation(image, tissue_masks = tissue_mask,
                                      feature_dir = file.path(output_dir, "features"),
                                      model_dir = model_dir,
                                      scales = c(1L, 2L, 4L),
                                      minimum_free_gib = minimum_free_gib,
                                      backend = execution$backend,
                                      n.cores = execution$n.cores,
                                      max_seconds = max_seconds)
    } else {
      root <- if (inherits(preparation, "bora-r-preparation-v1"))
        dirname(preparation$manifest) else .bora_paths(preparation, "preparation")
      if (length(root) != 1L) stop("preparation must contain one image", call. = FALSE)
      record <- .bora_result(file.path(root, "PREPARATION_COMPLETE.json"),
                             "bora-r-preparation-v1")$data
      if (length(record$cases) != 1L ||
          normalizePath(record$cases[[1L]]$source$image) != image) {
        stop("preparation must be for the exact supplied H&E image", call. = FALSE)
      }
    }
    fitted <- Bora.training(image, annotation, preparation,
                            output_dir = file.path(output_dir, "spot-head"),
                            channels_per_class = channels_per_group, alpha = alpha,
                            backend = execution$backend,
                            n.cores = execution$n.cores,
                            max_seconds = max_seconds)
    predicted <- Bora.prediction(preparation, predictor = fitted,
                                 output_dir = file.path(output_dir, "prediction"),
                                 source_mpp_xy = source_mpp_xy,
                                 output_mpp_xy = output_mpp_xy,
                                 minimum_area_um2 = minimum_area_um2,
                                 contour_sigma_um = contour_sigma_um,
                                 minimum_free_gib = minimum_free_gib,
                                 max_new_gib = max_new_gib,
                                 preview_backend = preview_backend,
                                 backend = execution$backend,
                                 n.cores = execution$n.cores,
                                 max_seconds = max_seconds,
                                 class_names = stats::setNames(classes,
                                                               seq_along(classes)))
    case_path <- predicted$data$cases[[1L]]$receipt_path
    case <- jsonlite::fromJSON(case_path, simplifyVector = FALSE)
    result <- list(format = "bora-r-spatial-v1", status = "COMPLETE",
                   image = image, object_class = class(object)[1L],
                   coordinate_space = "source_image_full_resolution_pixels",
                   coordinate_scale_xy = as.list(as.numeric(coordinate_scale_xy)),
                   spot_count = nrow(spots), training_token_count = nrow(spot_tokens),
                   group_names = as.list(classes),
                   exploratory_same_section_fit = TRUE,
                   spot_table = spot_csv, annotation = annotation,
                   preparation_manifest = if (inherits(preparation, "bora_result"))
                     preparation$manifest else file.path(preparation, "PREPARATION_COMPLETE.json"),
                   predictor_manifest = fitted$manifest,
                   prediction_manifest = predicted$manifest,
                   mask = case$masks$final$path,
                   geojson = case$source_vector$source_geojson_path,
                   preview = case$preview$path,
                   bw_images = lapply(case$binary_class_images, function(x) x$path))
    receipt <- file.path(output_dir, "SPATIAL_COMPLETE.json")
    jsonlite::write_json(result, receipt, pretty = TRUE, auto_unbox = TRUE)
    result$manifest <- receipt
    result
  }, error = function(e) {
    jsonlite::write_json(list(format = "bora-r-spatial-failure-v1",
                              status = "FAILED_PRESERVE_PARTIAL",
                              error = conditionMessage(e)),
                         file.path(output_dir, "SPATIAL_FAILED.json"),
                         pretty = TRUE, auto_unbox = TRUE)
    stop(e)
  })
}

.bora_spots <- function(object, cluster_column, image_slot, coordinate_columns) {
  if (inherits(object, "Seurat")) {
    if (!requireNamespace("SeuratObject", quietly = TRUE))
      stop("SeuratObject is required for Seurat input", call. = FALSE)
    coords <- as.data.frame(SeuratObject::GetTissueCoordinates(
      object, image = image_slot, scale = NULL))
    meta <- as.data.frame(object[[]])
    if (is.null(cluster_column)) {
      active <- SeuratObject::Idents(object)
      meta$.bora_active_group <- as.character(active[rownames(meta)])
      cluster_column <- ".bora_active_group"
    }
    ids <- if ("cell" %in% names(coords)) as.character(coords$cell) else
      if ("barcode" %in% names(coords)) as.character(coords$barcode) else rownames(coords)
  } else if (inherits(object, c("giotto", "giottoObject"))) {
    if (!requireNamespace("Giotto", quietly = TRUE))
      stop("Giotto is required for Giotto input", call. = FALSE)
    coords <- as.data.frame(Giotto::getSpatialLocations(object, output = "data.table"))
    meta <- as.data.frame(Giotto::pDataDT(object))
    ids <- as.character(coords$cell_ID)
    rownames(meta) <- as.character(meta$cell_ID)
  } else if (inherits(object, "SpatialExperiment")) {
    if (!requireNamespace("SpatialExperiment", quietly = TRUE) ||
        !requireNamespace("SummarizedExperiment", quietly = TRUE))
      stop("SpatialExperiment and SummarizedExperiment are required", call. = FALSE)
    coords <- as.data.frame(SpatialExperiment::spatialCoords(object))
    meta <- as.data.frame(SummarizedExperiment::colData(object))
    ids <- colnames(object)
    rownames(meta) <- ids
  } else if (is.data.frame(object)) {
    coords <- meta <- object
    ids <- if ("spot_id" %in% names(object)) as.character(object$spot_id) else
      rownames(object)
    if (is.null(ids) || anyNA(ids) || any(!nzchar(ids)) || anyDuplicated(ids)) {
      stop("Spot IDs must be present and unique", call. = FALSE)
    }
    rownames(meta) <- ids
  } else {
    stop("object must be Seurat, Giotto, SpatialExperiment, or a spot data frame",
         call. = FALSE)
  }
  if (is.null(ids) || anyNA(ids) || any(!nzchar(ids)) || anyDuplicated(ids) ||
      !nrow(coords)) stop("Spot IDs must be present and unique", call. = FALSE)
  if (is.null(cluster_column)) {
    candidates <- c("KODAMA_Louvain", "seurat_clusters", "leiden_clus", "cluster")
    cluster_column <- candidates[candidates %in% names(meta)][1L]
  }
  if (length(cluster_column) != 1L || is.na(cluster_column) ||
      !(cluster_column %in% names(meta))) {
    stop("cluster_column is absent; select a metadata group column", call. = FALSE)
  }
  if (is.null(coordinate_columns)) {
    candidates <- list(c("pxl_col_in_fullres", "pxl_row_in_fullres"),
                       c("imagecol", "imagerow"), c("sdimx", "sdimy"),
                       c("x", "y"))
    available <- candidates[vapply(candidates,
      function(pair) all(pair %in% names(coords)), logical(1))]
    coordinate_columns <- if (length(available)) available[[1L]] else NULL
  }
  if (!is.character(coordinate_columns) || length(coordinate_columns) != 2L ||
      !all(coordinate_columns %in% names(coords))) {
    stop("Select two spatial coordinate columns in X,Y order", call. = FALSE)
  }
  if (anyDuplicated(rownames(meta)) || anyNA(match(ids, rownames(meta)))) {
    stop("Spot IDs do not match expression-group metadata", call. = FALSE)
  }
  groups <- as.character(meta[ids, cluster_column])
  xy <- coords[, coordinate_columns, drop = FALSE]
  if (!all(vapply(xy, is.numeric, logical(1))) ||
      anyNA(xy) || any(!is.finite(as.matrix(xy))) ||
      anyNA(groups) || any(!nzchar(groups))) {
    stop("Every registered spot needs finite X/Y coordinates and a group label",
         call. = FALSE)
  }
  keep <- rep(TRUE, length(ids))
  if ("in_tissue" %in% names(meta)) {
    keep <- as.character(meta[ids, "in_tissue"]) %in% c("1", "TRUE", "true")
  }
  if (!any(keep)) stop("No in-tissue spots remain", call. = FALSE)
  data.frame(spot_id = ids[keep], x = xy[[1L]][keep], y = xy[[2L]][keep],
             cluster = groups[keep], stringsAsFactors = FALSE)
}
