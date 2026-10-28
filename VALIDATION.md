# Release validation

## Frozen colorectal whole-slide release (0.2.2, 28 October 2026)

The selected route is one slide-wide RGB9+19 fit followed by direct
all-response H&E refinement, global 400 µm² topology cleanup and compact
diagonal contours. The accepted colorectal output is the reference;
the three-group PCA32 and field APIs are compatibility commands.
See [the frozen workflow](docs/FROZEN_COLORECTAL_WSI.md) and
[its source and output identities](src/bora/frozen_colorectal.json).

A fresh Linux native wheel was built and installed in an isolated CPU
environment. The complete suite passed: **114 passed, 1 skipped**. The skip
requires an optional private PM016 r4 fixture. New tests cover the complete
fit/project/refine/crop handoff, the strict physical area threshold, components
crossing diagnostic edges, nonzero native-coordinate origins, odd raster
sizes, streamed-versus-monolithic diffusion and smoothing, shared diagonal
fitting, annotation exclusion, receipt checks and changed-output rejection.

All extracted refinement function bodies were compared with the accepted
full-slide source and matched, apart from passing pixel spacing explicitly
instead of using a global variable. The converter/cleanup source files match
the recorded 0.2.1 baseline. A CPU replay using the frozen colorectal state
checked five whole-width token stripes (3,768,320 response values): all
float16 values were identical with the reference four-thread BLAS setting.
Five spatial refinement probes also had zero different pixels against the
saved global intermediate mask. Refitting on all 11,469 colorectal tiles
(183,504 spatial observations) reproduced all 17 saved model/scaler arrays
bit for bit. The packaged response functions enforce that
four-thread setting; arbitrary BLAS thread counts can change the last stored
float16 bit. Source and reference hashes accompany the release.

This is an implementation-continuity check, not a new full-slide rerun or an
independent accuracy experiment. No GPU inference or expert-label selection
was performed for the release audit. Dependency pins match the accepted
runtime, using the headless OpenCV distribution of the same version.

## Earlier release and classifier audits


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

The accepted-region identity handoff was validated on macOS with Python 3.12,
NumPy 2.2.6 and the built vendored fastPLS core: **103 tests passed, 2 skipped**.
The skips require optional pyvips and a local PM016 vector fixture. New tests
compare streamed descriptors against the original native-pixel overlap and
physical boundary context implementation across several stripe sizes. They
also check padded token coordinates, disconnected pieces, isolated groups,
allocation-driven stripe reduction, conflicting overlaps, changed input hashes,
frame/dimension/spacing rejection, the actual frozen native classifier, exact
geometry preservation, and exclusive new output paths. This synthetic suite
verifies implementation behavior; it does not establish tissue accuracy on a
new slide or independently prove caller-asserted image registration.
