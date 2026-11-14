#' Cluster one joint-image PCA input with pinned KODAMA
#'
#' This internal routine runs once across all admitted tiles. It never uses
#' image coordinates or expert labels. Each requested graph resolution gets a
#' separate solution while sharing the same KODAMA embedding and neighbor graph.
#' @keywords internal
.bora_kodama_cluster <- function(stage_dir, cluster_dir, resolutions, seed = 1234L,
                                 neighbors = 20L, M = 100L) {
  if (!dir.exists(stage_dir) || file.exists(cluster_dir) ||
      !dir.exists(dirname(cluster_dir))) {
    stop("A completed stage and fresh cluster directory are required", call. = FALSE)
  }
  if (!is.numeric(resolutions) || !length(resolutions) || anyNA(resolutions) ||
      any(!is.finite(resolutions)) || any(resolutions <= 0) ||
      anyDuplicated(resolutions)) {
    stop("Distinct positive graph resolutions are required", call. = FALSE)
  }
  for (package in c("KODAMA", "RANN", "igraph")) {
    if (!requireNamespace(package, quietly = TRUE)) {
      stop("Missing KODAMA runtime package: ", package, call. = FALSE)
    }
  }
  if (as.character(utils::packageVersion("KODAMA")) != "0.99.7") {
    stop("Pinned KODAMA 0.99.7 is required", call. = FALSE)
  }
  stage <- jsonlite::read_json(file.path(stage_dir, "JOINT_STAGE_COMPLETE.json"),
                               simplifyVector = TRUE)
  if (!identical(stage$status, "COMPLETE") ||
      !identical(stage$format, "bora-r-joint-kodama-stage-v1")) {
    stop("A complete joint tile stage is required", call. = FALSE)
  }
  x <- as.matrix(utils::read.csv(file.path(stage_dir, "joint-pca30.csv"),
                                 check.names = FALSE))
  if (!is.numeric(x) || nrow(x) != stage$tile_count || nrow(x) < 40L ||
      ncol(x) != 30L || any(!is.finite(x))) {
    stop("Joint PCA30 rows differ from frozen tile identities", call. = FALSE)
  }
  if (nrow(x) <= neighbors || neighbors < 2L || neighbors > 100L ||
      M < 10L || M > 1000L) {
    stop("Invalid bounded KODAMA graph settings", call. = FALSE)
  }
  # The stage is capped before this call; its rows are never downsampled.
  set.seed(seed)
  fit <- KODAMA::KODAMA.matrix(x, spatial = NULL, M = M, n.cores = 1L,
                               seed = seed, return.graph = TRUE)
  set.seed(seed)
  embedding <- as.matrix(KODAMA::KODAMA.visualization(fit, method = "UMAP"))
  if (nrow(embedding) != nrow(x) || ncol(embedding) < 2L ||
      any(!is.finite(embedding))) {
    stop("KODAMA embedding differs from joint tile rows", call. = FALSE)
  }
  embedding <- embedding[, 1:2, drop = FALSE]
  nearest <- RANN::nn2(embedding, k = neighbors + 1L)$nn.idx[, -1L,
                                                              drop = FALSE]
  edges <- cbind(rep(seq_len(nrow(x)), each = ncol(nearest)),
                 as.vector(t(nearest)))
  graph <- igraph::simplify(igraph::graph_from_edgelist(edges, directed = FALSE),
                            remove.multiple = TRUE, remove.loops = TRUE)
  dir.create(cluster_dir)
  utils::write.csv(data.frame(row_index = seq_len(nrow(x)) - 1L,
                              kodama_1 = embedding[, 1],
                              kodama_2 = embedding[, 2]),
                   file.path(cluster_dir, "embedding.csv"), row.names = FALSE)
  solutions <- vector("list", length(resolutions))
  for (index in seq_along(resolutions)) {
    set.seed(seed)
    membership <- igraph::membership(igraph::cluster_louvain(
      graph, resolution = resolutions[[index]]))
    if (length(membership) != nrow(x)) {
      stop("Graph cluster rows differ from KODAMA input", call. = FALSE)
    }
    labels <- match(membership, unique(membership))
    folder <- file.path(cluster_dir, sprintf("solution-%03d", index))
    dir.create(folder)
    path <- file.path(folder, "groups.csv")
    utils::write.csv(data.frame(row_index = seq_len(nrow(x)) - 1L,
                                group = labels), path, row.names = FALSE)
    solutions[[index]] <- list(index = index, resolution = resolutions[[index]],
                               groups = length(unique(labels)),
                               group_counts = as.list(as.integer(tabulate(labels))),
                               group_csv = path)
  }
  receipt <- list(format = "bora-r-joint-kodama-clusters-v1", status = "COMPLETE",
                  stage_sha256 = reticulate::py_to_r(
                    reticulate::import("bora.r_feature_preflight")$sha256(
                      reticulate::import("pathlib")$Path(
                        file.path(stage_dir, "JOINT_STAGE_COMPLETE.json")))),
                  KODAMA_version = as.character(utils::packageVersion("KODAMA")),
                  igraph_version = as.character(utils::packageVersion("igraph")),
                  tile_count = nrow(x), image_count = stage$image_count,
                  spatial_coordinates_used = FALSE, expert_annotation_used = FALSE,
                  seed = seed, M = M, neighbors = neighbors,
                  embedding_csv = file.path(cluster_dir, "embedding.csv"),
                  solutions = solutions)
  jsonlite::write_json(receipt, file.path(cluster_dir, "CLUSTERS_COMPLETE.json"),
                       pretty = TRUE, auto_unbox = TRUE)
  receipt
}
