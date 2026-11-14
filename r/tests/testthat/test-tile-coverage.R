test_that("compiled tile coverage counts padding as background", {
  mask <- matrix(TRUE, 3, 3)
  observed <- Bora:::bora_tile_coverage_cpp(mask, 2L, 2L)
  expect_equal(observed, matrix(c(1, .5, .5, .25), 2, 2))
})

test_that("compiled tile coverage rejects missing pixels", {
  mask <- matrix(c(TRUE, NA), 1, 2)
  expect_error(Bora:::bora_tile_coverage_cpp(mask, 1L, 1L), "NA")
})
