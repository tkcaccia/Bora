# Synthetic RGB9+19 regression references

`rgb9plus19_legacy_2.npz` and `rgb9plus19_legacy_3.npz` contain the complete
uint8 RGB output and selected64 channel indices from the pre-extraction
`convert_rgb()` at Git commit `370ba9df782b7b1858d0222f13aeeedd2ca8524b`.
They were generated before its body was replaced by fit/replay delegation.

Inputs are produced by `tests/appearance_fixtures.py:spatial_case(2/3)`, seed
123. The producer environment used NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2,
OpenCV 5.0.0 and one BLAS thread. Selection indices remain an exact comparison.

Fitting the same formula on another CPU/BLAS platform can move a probability
slightly across an 8-bit half-integer rounding threshold. The initial Linux
Python 3.10/3.12 runs differed from these macOS fixtures in only one of 388,800
and four of 583,200 RGB channels, each by one intensity level. The frozen-raster
check therefore requires **both** maximum absolute channel difference <= 1 and
a mismatched-channel fraction <= 0.00001 (0.001%; at most three/five channels
for these fixtures). Tests explicitly reject a two-level error or too many
one-level errors. This is not a tolerance for a changed conversion formula.

An independent pre-extraction formula in `tests/appearance_fixtures.py` is also
evaluated on each test runtime. Full-resolution probabilities must agree at
absolute tolerance 2e-7, relative tolerance zero, and its RGB output may differ
by at most one intensity level. This checks formula, channel ordering and
padding/alignment without comparing separately fitted models across platforms.
Direct sklearn-versus-replay numeric checks cover scaler, Ridge and binary/
multiclass logistic predictions. Reloading the **same fitted JSON state** must
still reproduce every probability and RGB value bitwise exactly, with fitting
disabled; no cross-platform tolerance is applied to that replay assertion.

These are small synthetic software fixtures. They contain no patient images,
real encoder outputs, model weights or claimed segmentation ground truth.
