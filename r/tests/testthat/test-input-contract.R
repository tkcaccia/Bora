test_that("preparation requires existing images and a new output", {
  image <- tempfile(fileext = ".tif")
  writeBin(as.raw(1:8), image)
  on.exit(unlink(image))
  expect_error(Bora.preparation(character(), feature_dir = tempfile()), "images")
  expect_error(Bora.preparation(image, tissue_masks = c(image, image),
                                feature_dir = tempfile()), "one path per image")
  expect_error(Bora.preparation(image, feature_dir = tempfile(),
                                model_dir = ""), "model_dir")
  expect_error(Bora.preparation(image, feature_dir = tempfile(),
                                model_dir = tempdir(),
                                precomputed_manifest = tempfile()),
               "missing path")
})

test_that("training rejects a tissue-only or unpaired reference", {
  image <- tempfile(fileext = ".tif")
  writeBin(as.raw(1:8), image)
  on.exit(unlink(image))
  expect_error(Bora.training(image, NULL, dirname(image),
                             output_dir = tempfile()), "annotations")
  expect_error(Bora.training(image, c(image, image), dirname(image),
                             output_dir = tempfile()), "one path per image")
})

test_that("prediction requires an existing preparation directory", {
  expect_error(Bora.prediction(tempfile(), output_dir = tempfile()), "missing path")
  expect_error(Bora.prediction(tempdir(), output_dir = tempfile(),
                               resolutions = c(1, 1)), "distinct")
  expect_error(Bora.prediction(tempdir(), output_dir = tempfile(),
                               preview_backend = "unknown"), "preview_backend")
})
