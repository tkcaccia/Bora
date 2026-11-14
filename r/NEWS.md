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
