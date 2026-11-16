.bora_paths <- function(x, name, must_exist = TRUE) {
  if (!is.character(x) || !length(x) || anyNA(x) || any(!nzchar(x))) {
    stop(name, " must be a nonempty character vector of paths", call. = FALSE)
  }
  paths <- normalizePath(x, winslash = "/", mustWork = FALSE)
  if (must_exist && any(!file.exists(paths))) {
    stop(name, " contains a missing path: ", paths[which(!file.exists(paths))[1L]],
         call. = FALSE)
  }
  if (anyDuplicated(paths)) {
    stop(name, " contains a repeated path", call. = FALSE)
  }
  paths
}

.bora_paired_paths <- function(x, images, name) {
  if (is.null(x)) return(NULL)
  if (length(x) != length(images)) {
    stop(name, " must have one path per image", call. = FALSE)
  }
  .bora_paths(x, name)
}

.bora_output_dir <- function(x, name) {
  if (!is.character(x) || length(x) != 1L || is.na(x) || !nzchar(x)) {
    stop(name, " must be one directory path", call. = FALSE)
  }
  path <- normalizePath(x, winslash = "/", mustWork = FALSE)
  if (file.exists(path)) {
    stop(name, " already exists; use a new versioned output path", call. = FALSE)
  }
  parent <- dirname(path)
  if (!dir.exists(parent)) {
    stop(name, " parent directory does not exist: ", parent, call. = FALSE)
  }
  path
}

.bora_backend <- function() {
  if (!reticulate::py_module_available("bora.r_api")) {
    stop("The Bora Python backend is not installed in the active reticulate environment. ",
         "Install the matching Bora Python release first; see r/README.md.",
         call. = FALSE)
  }
  reticulate::import("bora.r_api", convert = TRUE)
}

.bora_result <- function(path, expected_format) {
  if (!file.exists(path)) {
    stop("The backend did not create its result manifest: ", path, call. = FALSE)
  }
  value <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  if (!identical(value$format, expected_format) ||
      !identical(value$status, "COMPLETE")) {
    stop("The result manifest is incomplete or has the wrong format", call. = FALSE)
  }
  structure(list(manifest = path, data = value), class = c(expected_format, "bora_result"))
}
.bora_measured_spacing <- function(value) {
  if (!is.numeric(value) || length(value) != 2L || anyNA(value) ||
      any(!is.finite(value)) || any(value <= 0)) {
    stop("Measured spacing source_mpp_xy must be supplied explicitly as two positive finite X/Y micrometre values", call. = FALSE)
  }
  as.numeric(value)
}
