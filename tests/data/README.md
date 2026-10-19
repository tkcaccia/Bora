# Synthetic RGB9+19 regression references

`rgb9plus19_legacy_2.npz` and `rgb9plus19_legacy_3.npz` contain the complete
uint8 RGB output and selected64 channel indices from the pre-extraction
`convert_rgb()` at Git commit `370ba9df782b7b1858d0222f13aeeedd2ca8524b`.
They were generated before its body was replaced by fit/replay delegation.

Inputs are produced by `tests/appearance_fixtures.py:spatial_case(2/3)`, seed
123. The producer environment used NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2,
OpenCV 5.0.0 and one BLAS thread. The full-raster comparison guards the fixed
formula, channel ordering, padding/alignment and float32 RGB rounding.

These are small synthetic software fixtures. They contain no patient images,
real encoder outputs, model weights or claimed segmentation ground truth.
