test_that("preparation rejects ambiguous or unsafe extraction settings", {
  image <- tempfile(fileext = ".tif")
  writeBin(as.raw(1:8), image)
  on.exit(unlink(image))
  args <- list(images = image, feature_dir = tempfile(), model_dir = tempdir())
  expect_error(do.call(Bora.preparation, c(args, list(scales = c(1, 1)))), "distinct")
  expect_error(do.call(Bora.preparation, c(args, list(scales = c(1, 2.5)))), "integers")
  expect_error(do.call(Bora.preparation, c(args, list(max_seconds = 0))), "Finite")
  expect_error(do.call(Bora.preparation, c(args, list(minimum_free_gib = -1))), "Finite")
  expect_error(do.call(Bora.preparation, c(args, list(tissue_masks = tempfile()))),
               "missing path")
})

test_that("training rejects class mappings and limits before fitting", {
  image <- tempfile(fileext = ".tif")
  writeBin(as.raw(1:8), image)
  on.exit(unlink(image))
  args <- list(images = image, annotations = image, feature_dir = tempdir(),
               output_dir = tempfile())
  expect_error(do.call(Bora.training, c(args, list(class_labels = 1L))), "list")
  expect_error(do.call(Bora.training, c(args, list(class_labels = list(1L)))), "named")
  expect_error(do.call(Bora.training, c(args, list(class_labels = list(c(a = 0L))))),
               "IDs 1..254")
  expect_error(do.call(Bora.training, c(args, list(channels_per_class = 1281L))),
               "Finite positive")
  expect_error(do.call(Bora.training, c(args, list(alpha = 0))), "Finite positive")
})

test_that("prediction distinguishes supervised and joint KODAMA arguments", {
  args <- list(preparation = tempdir(), output_dir = tempfile(),
               source_mpp_xy = c(.5, .5))
  expect_error(Bora.prediction(tempdir(), output_dir = tempfile()),
               "Measured spacing")
  expect_error(do.call(Bora.prediction, c(args, list(resolutions = c(1, 1)))), "distinct")
  expect_error(do.call(Bora.prediction, c(args, list(max_joint_tiles = 39L))),
               "Measured spacing")
  expect_error(Bora.prediction(tempdir(), output_dir = tempfile(),
                               source_mpp_xy = c(.5, 0)),
               "Measured spacing")
  expect_error(do.call(Bora.prediction,
                       c(args, list(predictor = tempdir(), resolutions = 1))),
               "only to KODAMA")
  expect_error(do.call(Bora.prediction,
                       c(args, list(class_names = c(a = "tissue")))),
               "supervised prediction")
})

test_that("spatial route requires measured spacing before Python or output", {
  image <- tempfile(fileext = ".tif")
  writeBin(as.raw(1:8), image)
  on.exit(unlink(image))
  output <- tempfile()
  expect_error(Bora.spatial(image, data.frame(), output), "Measured spacing")
  expect_false(dir.exists(output))
  expect_error(Bora.spatial(image, data.frame(), output,
                            source_mpp_xy = c(0.5, NA_real_)), "Measured spacing")
})

test_that("spatial spot metadata catches missing and ambiguous inputs", {
  spots <- data.frame(spot_id = c("a", "a"), x = c(7, 21), y = c(7, 21),
                      cluster = c("A", "B"))
  expect_error(Bora:::.bora_spots(spots, "cluster", NULL, NULL), "unique")
  spots$spot_id <- c("a", "b")
  spots$x[[2L]] <- NA_real_
  expect_error(Bora:::.bora_spots(spots, "cluster", NULL, NULL), "finite")
  spots$x[[2L]] <- 21
  spots$cluster[[2L]] <- ""
  expect_error(Bora:::.bora_spots(spots, "cluster", NULL, NULL), "group label")
})
