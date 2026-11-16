#' Download the separately licensed Virchow2 model
#'
#' Downloads only `config.json` and `model.safetensors` from the official
#' gated `paige-ai/Virchow2` repository. You must first accept its terms with
#' your own institutional Hugging Face account and authenticate in the Python
#' environment used by `reticulate` (`HF_TOKEN` or `hf auth login`). Bora does
#' not include or redistribute these weights. The helper records the resolved
#' model commit and SHA256 of each file. An incomplete transfer stays outside
#' `model_dir` and cannot be used by `Bora.preparation()`.
#'
#' @param model_dir New destination directory for the two model files.
#' @param revision Hub branch, tag or commit; defaults to `"main"`. The exact
#'   resolved commit is recorded so the download can be reproduced.
#' @return A manifest object whose `model_dir` can be passed to
#'   `Bora.preparation()`.
#' @export
Bora.download_virchow2 <- function(model_dir, revision = "main") {
  model_dir <- .bora_output_dir(model_dir, "model_dir")
  if (!is.character(revision) || length(revision) != 1L ||
      is.na(revision) || !nzchar(revision)) {
    stop("revision must be one nonempty branch, tag or commit", call. = FALSE)
  }
  if (!reticulate::py_module_available("bora.r_model_download")) {
    stop("The matching Bora Python backend is not installed in the active reticulate environment.",
         call. = FALSE)
  }
  reticulate::import("bora.r_model_download", convert = TRUE)$download_virchow2(
    model_dir, revision)
  result <- .bora_result(file.path(model_dir, "DOWNLOAD_COMPLETE.json"),
                         "bora-r-virchow2-download-v1")
  result$model_dir <- model_dir
  result
}
