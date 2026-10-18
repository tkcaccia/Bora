# Release validation

The retained workflow is Virchow2 RGB evidence, Bora appearance fields, Neighbor
context tissue identification, and compact native-coordinate GeoJSON export.
The application exposes one classifier. Historical research and benchmark code
are preserved in https://github.com/tkcaccia/Bora-Analysis .

The native fastPLS C++ core and Python bindings are copied into this repository
under their original MIT license. `vendor/PROVENANCE.json` pins the upstream
revisions and source hashes. The package imports `bora._vendor.fastpls` and does
not depend on a separate fastPLS installation or a Git URL at runtime.

The CPU suite checks physical boundary weights, complete-group pooling,
calibration/eligibility rejection, the safe training-artifact round trip,
native raster/vector consistency, strict sub-20 area cleanup, diagonal fitting
and geometry safeguards. It passes on the local macOS build and the Linux
wheel built on Chiamaka.

The packaged Linux classifier was also fitted on the frozen 967 × 2560 float64
training matrix with 25 represented classes and 24 components. Predictions
matched the saved reference for all 116 evaluation images, including after
artifact save/load. Recomputed physical context descriptors were bitwise
identical for one field from each evaluation dataset. The report and audit
script are retained in Bora-Analysis. This verifies implementation continuity;
it does not constitute independent scientific validation. No GPU extraction or
new research experiment was launched for this audit.

Three deterministic CPU fixtures also compare the packaged RGB conversion with
the preserved reference implementation. Selected coordinates and resulting RGB
pixels are bitwise identical in all three fixtures. The published coverage
converter is unchanged (SHA256
`995f734250017c130a87ca79d81d54a5ed44fbf89808514314043f94626f7d46`).
