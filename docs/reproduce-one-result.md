# Reproduce one supervised Bora result

This recipe reproduces the **BCSS TCGA-A2-A04P ROI (case-001)** class mask
from the frozen Chiamaka-local input cache. It fits the accepted eight-class
training recipe, predicts without opening the case's expert mask, and checks
the pixel SHA256 of the result. It is a numerical reproduction of an
**inspected development image**, not a new external validation. The exact
source H&E, feature array and training sample are stored in the verified
`Bora-Data` folder; the repository does not redistribute those third-party
data or the Virchow2 checkpoint.

## Exact method and inputs

| Item | Frozen identity or schema |
| --- | --- |
| Package | `bora-segmentation==0.3.0`, `eight-class-ridge`; classwise top 32 standardized mean contrasts per class, union 203 channels, weighted `RidgeClassifier(alpha=10, solver=cholesky)`. |
| Encoder | Virchow2 revision `3158645804b69e3f3bc4439d4116edddf0840a72`; checkpoint SHA256 `8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b`. The encoder is upstream of this CPU recipe. |
| Training features | `training-features.f16.npy`, NumPy `(139736, 1280)` float16, SHA256 `67350fe358f8302ac745b4ab561ac6c09628725ee9b9715a564055107b3eb574`. The recipe maps its frozen index labels to classes 1–8 and ignores label 0. |
| Training index | `training-index.npz`, SHA256 `458996ed744a79e3360737da3767dd99ec277bed905770f79302e94304af4229`. The ontology is `benchmark-label-reduction-2026-11-11-v1/ONTOLOGY_FROZEN.json`. |
| ROI features | `virchow2-full1280-factor-1.npy`, NumPy `(128, 192, 1280)` float16, SHA256 `0f7ba45a1ea5714db19bdfd48bd8287371da1062331bcf50a03e630ec4d44e34`. Never use a preselected 203-channel cache as input. |
| Registration | Calibrated H&E `1897 × 1460` working pixels; response display `768 × 591` pixels; 14-pixel token stride. The frozen case receipt supplies these dimensions. |
| Predictions | Class IDs 1–8: tumor, stroma, necrosis, epithelium, blood, vessel, adipose, immune infiltrate. Zero is background/ignored. |

The fit manifest has schema `bora-supervised-fit-v1`: `split: train`,
`source_full_channels: 1280`, and `features`/`labels` objects containing
`path` and `sha256`. The prediction manifest has schema
`bora-supervised-predict-v1`: checksum-bound `model` and `features`, plus
`roi.working_size_wh`, `roi.display_size_wh`, `roi.allowed_classes`, and
`roi.polish: true`. The example script writes both manifests and passes them
to the stable `bora fit` and `bora predict` commands. See
[the complete command contract](supervised-commands.md) to supply your own
registered ROI.

## CPU/GPU requirements and command

Fit and prediction run locally on **CPU**. Use two BLAS threads and allow at
least **8 GiB RAM** for the 139,736-token fit. The case-level prediction needs
less memory. An NVIDIA GPU is required only if the complete Virchow2 cache
must be generated upstream; the command below does not start the encoder or
any GPU work. On Chiamaka, wait for any active Bora CPU/GPU bundle and its
terminal cleanup before running this recipe. Use a fresh output path and keep
the 100 GiB project free-space reserve.

```bash
cd /path/to/installed/Bora
python -m pip install -e '.[test,wsi]'
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python examples/reproduce_case001.py \
  --bora-root /mnt/sata_ssd/ChatGPT/Bora \
  --output /mnt/sata_ssd/ChatGPT/Bora-Data/reproduce-case001-2026-11-12-v1
```

The script refuses an existing output directory, verifies every input hash,
invokes `bora fit`, `bora verify`, `bora predict`, and `bora verify`,
then writes `REPRODUCE_RESULT.json`. You can inspect its generated
`fit.json` and `predict.json` and rerun the individual commands with
another fresh directory. Do not substitute an evaluation image for the
training arrays.

## Expected output and interpretation

| Output | Expected SHA256 |
| --- | --- |
| `prediction/class-mask.png` decoded 768 × 591 uint8 pixel bytes | `0249c9b1b37cfa1dfc0cedd941ec0875cae7c38ce0d9fcdf379c09cee7dedf16` |
| Accepted historical `case-001-class-mask.png` file bytes | `bc8e03ad2a13b197f34ac218349d3aa3973fcd7364fb735f99548d58d577fc8a` |
| Accepted historical scikit-learn `joblib` eight-class head file | `3027497160b131b4a8ca5c2fcd89ce5a3bad13d41abb667ce6c3b9a97ae26053` |

The portable package model is a **non-pickle NPZ**, so its file hash differs
from the historical `joblib` model. Its NPZ embeds the fit-manifest hash,
which includes input paths; relocation changes that file hash without changing
the fitted parameters. `REPRODUCE_RESULT.json` records the exact new model
and PNG file hashes for the chosen output path. The decoded mask pixel hash
is the portable acceptance target. A mismatched pixel hash is a failed
reproduction; preserve the outputs and inspect package/encoder/input versions.

The accepted historical mask and its metrics came from inspected development
data. The old training-selection folds had known cross-source TCGA patient-ID
overlap, so these hashes demonstrate reproducibility of that frozen method,
not patient-disjoint accuracy. No expert mask is read during prediction.
