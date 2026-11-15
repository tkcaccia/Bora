test_that("registered data-frame spots preserve identity and tissue filtering", {
  spots <- data.frame(spot_id = c("a", "b", "c"),
                      pxl_col_in_fullres = c(7, 21, 35),
                      pxl_row_in_fullres = c(7, 21, 35),
                      KODAMA_Louvain = c("one", "two", "three"),
                      in_tissue = c(1, 1, 0))
  result <- Bora:::.bora_spots(spots, NULL, NULL, NULL)
  expect_equal(result$spot_id, c("a", "b"))
  expect_equal(result$cluster, c("one", "two"))
  expect_equal(result$x, c(7, 21))
  expect_error(Bora:::.bora_spots(spots[, -2], NULL, NULL, NULL),
               "coordinate columns")
})

test_that("SpatialExperiment coordinates join cluster metadata by spot ID", {
  skip_if_not_installed("SpatialExperiment")
  spe <- SpatialExperiment::SpatialExperiment(
    assays = list(counts = matrix(1L, nrow = 2L, ncol = 3L,
                                  dimnames = list(c("gene1", "gene2"),
                                                  c("a", "b", "c")))),
    spatialCoords = matrix(c(7, 21, 35, 8, 22, 36), ncol = 2L,
                           dimnames = list(c("a", "b", "c"), c("x", "y"))))
  spe$KODAMA_Louvain <- c("A", "B", "A")
  result <- Bora:::.bora_spots(spe, NULL, NULL, NULL)
  expect_equal(result$spot_id, c("a", "b", "c"))
  expect_equal(result$x, c(7, 21, 35))
  expect_equal(result$cluster, c("A", "B", "A"))
})

test_that("Seurat image coordinates and cluster metadata are matched by cell", {
  skip_if_not_installed("SeuratObject")
  counts <- matrix(1L, nrow = 2L, ncol = 3L,
                   dimnames = list(c("gene1", "gene2"), c("a", "b", "c")))
  object <- suppressWarnings(SeuratObject::CreateSeuratObject(
    counts, assay = "Spatial"))
  fov <- SeuratObject::CreateFOV(data.frame(x = c(7, 21, 35),
                                            y = c(8, 22, 36),
                                            cell = c("a", "b", "c")),
                                  type = "centroids", assay = "Spatial")
  object[["slice"]] <- fov
  object$KODAMA_Louvain <- c("A", "B", "A")
  result <- Bora:::.bora_spots(object, "KODAMA_Louvain", "slice", NULL)
  expect_equal(result$spot_id, c("a", "b", "c"))
  expect_equal(result$cluster, c("A", "B", "A"))
  expect_equal(result$x, c(7, 21, 35))
})

test_that("Giotto spatial locations and metadata join by cell_ID", {
  skip_if_not_installed("Giotto")
  old <- options(giotto.check_core = FALSE, giotto.check_version = FALSE)
  on.exit(options(old))
  counts <- matrix(1L, nrow = 2L, ncol = 3L,
                   dimnames = list(c("gene1", "gene2"), c("a", "b", "c")))
  locations <- data.frame(sdimx = c(7, 21, 35), sdimy = c(8, 22, 36),
                          cell_ID = c("a", "b", "c"))
  object <- suppressWarnings(Giotto::createGiottoObject(
    expression = counts, spatial_locs = locations))
  object <- Giotto::addCellMetadata(
    object, new_metadata = data.frame(cell_ID = c("a", "b", "c"),
                                      KODAMA_Louvain = c("A", "B", "A")),
    by_column = TRUE, column_cell_ID = "cell_ID")
  result <- Bora:::.bora_spots(object, "KODAMA_Louvain", NULL, NULL)
  expect_equal(result$spot_id, c("a", "b", "c"))
  expect_equal(result$cluster, c("A", "B", "A"))
  expect_equal(result$x, c(7, 21, 35))
})
