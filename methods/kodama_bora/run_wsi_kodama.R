#!/usr/bin/env Rscript
# Cluster a fixed sample of Virchow2 WSI tile features without slide coordinates.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Usage: Rscript run_wsi_kodama.R SAMPLE_PCA_CSV NEW_OUTPUT_DIR")
input <- normalizePath(args[[1]], mustWork = TRUE)
output <- normalizePath(args[[2]], mustWork = FALSE)
if (file.exists(output)) stop("Use a new output directory")
for (package in c("KODAMA", "RANN", "igraph", "jsonlite")) {
  if (!requireNamespace(package, quietly = TRUE)) stop("Missing R package: ", package)
}
x <- as.matrix(utils::read.csv(input, check.names = FALSE))
if (!is.numeric(x) || ncol(x) != 30L || nrow(x) < 100L || any(!is.finite(x)))
  stop("Expected finite sampled tile feature matrix with 30 principal components")
set.seed(1234)
fit <- KODAMA::KODAMA.matrix(x, spatial = NULL, M = 100, n.cores = 1, seed = 1234,
                             return.graph = TRUE)
set.seed(1234)
embedding <- as.matrix(KODAMA::KODAMA.visualization(fit, method = "UMAP"))
if (nrow(embedding) != nrow(x) || ncol(embedding) < 2L || any(!is.finite(embedding)))
  stop("KODAMA embedding differs from tile sample")
embedding <- embedding[, 1:2, drop = FALSE]
# Same 20-neighbor graph and Louvain resolution search as the Visium workflow.
neighbors <- RANN::nn2(embedding, k = 21L)$nn.idx[, -1, drop = FALSE]
edges <- cbind(rep(seq_len(nrow(embedding)), each = ncol(neighbors)),
               as.vector(t(neighbors)))
graph <- igraph::simplify(igraph::graph_from_edgelist(edges, directed = FALSE),
                          remove.multiple = TRUE, remove.loops = TRUE)
target_groups <- 6L  # fixed before inspecting expert labels; matches the older WSI case design
best <- NULL; best_gap <- Inf; best_resolution <- NA_real_
for (resolution in seq(0.10, 1.50, by = 0.05)) {
  set.seed(1234)
  labels <- igraph::membership(igraph::cluster_louvain(graph, resolution = resolution))
  gap <- abs(length(unique(labels)) - target_groups)
  if (gap < best_gap) {
    best <- labels; best_gap <- gap; best_resolution <- resolution
  }
  if (gap == 0L) break
}
if (is.null(best) || length(unique(best)) < 2L) stop("No usable unsupervised tile groups")
# Stable remapping by first occurrence; names are anonymous group IDs.
labels <- match(best, unique(best))
dir.create(output, recursive = TRUE, showWarnings = FALSE)
utils::write.csv(data.frame(kodama_1 = embedding[, 1], kodama_2 = embedding[, 2],
                            group = labels), file.path(output, "sample_clusters.csv"),
                 row.names = FALSE)
saveRDS(list(kodama = fit, embedding = embedding, labels = labels,
             louvain_resolution = best_resolution),
        file.path(output, "sample_clusters.rds"), compress = "gzip")
summary <- list(format = "bora-wsi-nonspatial-kodama-louvain-v1",
                n_tiles = nrow(x), pca_dimensions = ncol(x),
                spatial_coordinates_used = FALSE, expert_annotation_used = FALSE,
                kodama_M = 100L, kodama_seed = 1234L, louvain_neighbors = 20L,
                louvain_target_groups = target_groups,
                louvain_resolution = best_resolution,
                groups = length(unique(labels)),
                group_counts = as.list(as.integer(table(labels))),
                KODAMA_version = as.character(utils::packageVersion("KODAMA")),
                igraph_version = as.character(utils::packageVersion("igraph")))
grDevices::png(file.path(output, "kodama_groups.png"), width = 1200, height = 950, res = 150)
plot(embedding, col = grDevices::hcl.colors(length(unique(labels)), "Dark 3")[labels],
     pch = 16, cex = 0.48, xlab = "KODAMA 1", ylab = "KODAMA 2",
     main = "Virchow2 tile groups, no spatial input")
grDevices::dev.off()
writeLines(capture.output(sessionInfo()), file.path(output, "r_session.txt"))
jsonlite::write_json(summary, file.path(output, "CLUSTERS_COMPLETE.json"),
                     pretty = TRUE, auto_unbox = TRUE)
cat(jsonlite::toJSON(summary, auto_unbox = TRUE), "\n")
