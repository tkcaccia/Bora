#include <Rcpp.h>
#include <algorithm>
#include <cstdint>

// Exact tile tissue fraction. Pixels outside the image count as background.
// [[Rcpp::export]]
Rcpp::NumericMatrix bora_tile_coverage_cpp(const Rcpp::LogicalMatrix& tissue,
                                           const int tile_height,
                                           const int tile_width) {
  if (tile_height <= 0 || tile_width <= 0) {
    Rcpp::stop("Tile dimensions must be positive");
  }
  const int height = tissue.nrow();
  const int width = tissue.ncol();
  if (height <= 0 || width <= 0) {
    Rcpp::stop("The tissue mask must be nonempty");
  }
  const int tile_rows = (height + tile_height - 1) / tile_height;
  const int tile_cols = (width + tile_width - 1) / tile_width;
  const double denominator = static_cast<double>(tile_height) * tile_width;
  Rcpp::NumericMatrix result(tile_rows, tile_cols);
  for (int row = 0; row < tile_rows; ++row) {
    const int y0 = row * tile_height;
    const int y1 = std::min(height, y0 + tile_height);
    for (int col = 0; col < tile_cols; ++col) {
      const int x0 = col * tile_width;
      const int x1 = std::min(width, x0 + tile_width);
      std::int64_t count = 0;
      for (int x = x0; x < x1; ++x) {
        for (int y = y0; y < y1; ++y) {
          const int value = tissue(y, x);
          if (value == NA_LOGICAL) {
            Rcpp::stop("The tissue mask contains NA");
          }
          count += value == TRUE;
        }
      }
      result(row, col) = static_cast<double>(count) / denominator;
    }
  }
  return result;
}
