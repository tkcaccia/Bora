# Bora for R 0.1.6

- Reject duplicate spot IDs with an explicit registration error before R
  attempts to set row names.
- Expand function help and the reference manual with verified real extraction,
  equivalent raster/GeoJSON training, vector round-trip, coordinate scaling,
  joint-image regression, failure checks and current scientific limits.

# Bora for R 0.1.5

- `Bora.spatial()` reads registered expression-group positions from Seurat,
  Giotto, SpatialExperiment or a spot table, fits a sparse spot-guided Bora
  head on full-channel Virchow2 features, and writes a cleaned TIFF mask and
  source-pixel GeoJSON. This same-section exploratory fit is not an expert
  tissue classifier and does not reproduce the frozen historical Visium masks.

# Bora for R 0.1.4

- Align the hash-checked one-result guide and install command with the
  current R/Python release pair. The R 0.1.3 plus Python 0.3.4 installed-wheel
  replay matched the original training-region output hashes exactly; no
  prediction algorithm changed in 0.1.4.

# Bora for R 0.1.3

- Joint KODAMA predictions now record the correct SHA256-bound source preview
  cache name when the GDAL Lanczos backend is selected. This corrects a
  final-receipt failure after masks for all images and resolutions were saved.

# Bora for R 0.1.2

- `Bora.preparation(precomputed_manifest = ...)` adopts separately verified,
  source-bound full-1,280-channel Virchow2 arrays for multiple images without
  re-extraction or copying full arrays. Image, model, support, registration,
  feature and source-receipt identities are checked before output creation.
- A fabricated two-image three-scale regression passed the downstream training
  grid reader; tampered hashes, registration and source receipts fail closed.

# Bora for R 0.1.1

- `Bora.prediction(preview_backend = "gdal-lanczos")` uses compiled GDAL
  downsampling for large tiled H&E previews. The default
  `"vips-lanczos3"` retains byte-for-byte historical previews. On one
  source-only ovarian WSI, the explicit GDAL option reduced the thumbnail
  read from 120.82 to 1.37 seconds on warmed input. The full prediction
  replay took 29.96 seconds and kept 14 score, mask, binary TIFF, support
  and GeoJSON artifacts byte-identical to the frozen run. Only preview
  pixels changed slightly (mean absolute RGB difference 0.978/255).

# Bora for R 0.1.0

- First source release of `Bora.preparation`, `Bora.training` and
  `Bora.prediction` with full 1,280-channel Virchow2 caches, supervised
  classwise ridge masks and joint-image KODAMA group outputs.
- Source-bound raster/GeoJSON tissue support, native and clipped masks,
  cleaned class vectors, binary images, previews and checksum receipts.
- Verified on one real BCSS training ROI for preparation, fitting and
  prediction; one inspected BCSS ROI for geometry; one public ovarian WSI for
  supervised source-only projection; and a fabricated two-image,
  two-resolution KODAMA regression. These checks are not biological accuracy
  estimates.
- R source checks pass Linux, macOS and Windows CI. Python CPU tests pass
  Linux and macOS. GPU extraction is qualified on Chiamaka with the licensed
  checkpoint; other GPU runtimes and large joint KODAMA WSI fits have not
  been qualified.
- The MIT license covers Bora code. It does not license Virchow2 weights,
  H&E images, expert masks or other third-party data.
