# Whole-slide supervised Bora ridge substitution

These are copies of the executed Chiamaka CPU sources used to project the selected Benchmark 1 ridge head onto 13 public Benchmark 2 slides. They verify full 1,280-channel feature caches, use the 203-channel fitted head, choose slide-present classes with a training-only 2% winner-share floor, produce one full-resolution calibrated TIFF mask and native-coordinate GeoJSON per slide, then render the overview and nine matched zooms. They require frozen local images, features, reviewed Bora support geometry, fitted weights and receipts; none are bundled here.

The class-presence rule was chosen on 149 source-group-disjoint held-out training cases. The full-slide mask uses reviewed Bora tissue support, 6 µm contour smoothing and 2,000 µm² physical-area repair. The final native vector and TIFF were checked for exact pixel agreement, and all ten views come from the same final mask. The shared support is also used by the other displayed methods for background handling in the web panels.

The 13 outputs have no independent whole-slide expert masks. Visual alignment, mask hashes and GeoJSON raster agreement do not measure tissue-class accuracy. KODAMA–Bora and the four external methods on the pages remain their earlier predictions mapped to the broad ontology.
