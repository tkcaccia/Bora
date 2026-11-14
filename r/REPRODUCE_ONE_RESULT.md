# Reproduce one Bora R result

This is an end-to-end **technical reproduction on one BCSS training region**.
It is neither a held-out accuracy estimate nor a CAMELYON16 result. The source
image, expert annotation, derived tissue support and Virchow2 weights are not
redistributed by the MIT-licensed Bora code.

## Environment and exact inputs

Use Bora Python backend **0.3.1**, Bora R **0.1.0**, R **4.6.0**, and
Virchow2 revision `3158645804b69e3f3bc4439d4116edddf0840a72`. The
checkpoint SHA256 is
`8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b`;
the model configuration SHA256 is
`7db445b996bb165e88fe70e826c2ebb530539a2b1d136aa16eeb847df5f1e3db`.
The checkpoint must be obtained under its own terms.

The selected BCSS training region has this input contract:

| Input | Schema and SHA256 |
| --- | --- |
| Calibrated H&E | RGB TIFF, 1792 × 1792 pixels, 0.5 µm per pixel in both axes; `c5b03b006fc8f47e43856c43aaeb61734c364f4899760a5999fc5cec56d3e0b8` |
| Tissue support | aligned binary TIFF; `c3e99f453a75e89c4b70198b25f652ec5cfc031e1669acd8c9c326ef8b30f60a` |
| Training annotation | aligned uint8 class TIFF with IDs 1, 2, 3, 7 and zero for ignored/padding pixels; `fd0c5c4c34a75a2217406805c5b5dc7eac25dc3c5d8ac7ecd5c514d0d0ecea1b` |

The registered annotation is an eight-class mapping of the original BCSS
expert raster into the top-left of the padded calibrated image; class zero is
unscored. Exact H&E, support and annotation files are retained in the
checksum-bound Chiamaka migration receipts
`migration/bora-r-real-training-2026-11-14-v1/` and
`migration/bora-r-real-preparation-2026-11-14-v1/`. Outside Chiamaka, obtain
BCSS source data from its provider and reproduce the stated calibration,
support and ontology mapping before using this hash target. A file with a
different input hash is a different experiment.

Install the matching Python wheel into a Python environment with PyTorch,
`timm`, the licensed Virchow2 checkpoint and the package dependencies, then
install the R package from this same checkout. Set `RETICULATE_PYTHON` to
that environment's executable **before starting R**. KODAMA 0.99.7,
RANN and igraph are optional for this supervised example. Feature extraction
requires one supported NVIDIA GPU; fitting and projection are CPU tasks.
Provide at least **8 GiB host RAM** and **8 GiB GPU memory** for this ROI,
and preserve the Bora project's 100 GiB disk reserve on Chiamaka. A new
output directory is required.

```bash
python -m pip install '.[wsi]'
R CMD INSTALL r
export RETICULATE_PYTHON=/path/to/encoder-python
Rscript r/examples/reproduce_bcss_training_roi.R \
  /path/to/calibrated-he.tif \
  /path/to/tissue-support.tif \
  /path/to/registered-class-mask.tif \
  /path/to/Virchow2-model-directory \
  /path/to/new-output-directory
```

`Bora.preparation` writes factor-1/2/4 float16 NumPy arrays with all **1,280
channels**, token coordinates, input/model hashes and worker cleanup receipts.
The tissue support admits 48 source tiles and omits glass-only encoder tokens.
`Bora.training` samples factor-1 token centres, selects up to 16 classwise
contrast channels per class and fits a ridge head with `alpha = 10`.
`Bora.prediction` writes class binary TIFFs, native and cleaned multiclass
TIFFs, source-coordinate GeoJSON and a preview. The GeoJSON should rasterize
back to the final mask exactly.

## Expected hashes

The audited Chiamaka run produced these bytes with the frozen inputs and
runtime. Check SHA256 using `sha256sum` on Linux or a platform-equivalent
utility. Compressed TIFF/GeoJSON file hashes can depend on writer versions;
the decoded mask pixel hash is the more portable acceptance target.

| Output | Expected SHA256 |
| --- | --- |
| Factor-1 full-channel feature array | `e6087bfd74a78f4ffb0294a0c4d6f6ec3962ff6c49e177edb3783986efb1eb5c` |
| Factor-2 full-channel feature array | `3e2c48f35e8211aaf7a7e0f567afe5ea40d72fcb208c25d81fdd0096b7a98cb2` |
| Factor-4 full-channel feature array | `a77a532a2406c49ecc947a4880286fa38e8b1d700f0ac81a7e049737546c55f7` |
| Final 224 × 224 uint8 class-mask **decoded row-major pixels** | `9422d4637c79b8328c7a5c124dbfebb12be3de22b6f6c7cafa9282fdebb5e891` |
| Final `class-mask.tif` file in the audited environment | `15684950cfc00812e7bf99abbd95995f64b1654b727d313be661ef9846a1bc6f` |
| Preview PNG file in the audited environment | `e168e88f51a45e7811fca0bd6803cb18aa9b2b05dfa7d5f3ae521472d79faca5` |

For a portable decoded-pixel check, open
`prediction/image-000/class-mask.tif` with a TIFF reader, convert it to a
contiguous row-major uint8 array, then SHA256 its raw bytes. The source-coordinate
GeoJSON should contain 13 features and rasterize exactly to those same pixels.
Its raw file hash is path-bound through the model/provenance receipt and is not
a portable acceptance target. An isolated replay using the newly extracted
R features reproduced the final TIFF and preview byte-for-byte; its fitted
coefficients differed by at most `1.23e-6` from the prior cache-adoption run
under a different BLAS context. If a pixel hash differs,
preserve all receipts and compare model, source, support, annotation, spacing,
token registration and numerical runtime before interpreting the output.

The independently audited local receipts are
`migration/bora-r-real-preparation-2026-11-14-v1/REAL_PREPARATION_AUDIT_VERIFIED.json`
and
`migration/bora-r-real-training-2026-11-14-v1/R_TRAINING_AND_PREDICTION_AUDIT_VERIFIED.json`.
The separate [Python reproduction guide](../docs/reproduce-one-result.md)
covers the frozen eight-class development-case model.
