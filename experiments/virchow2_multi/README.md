# Seven selected Virchow2/Bora methods across benchmark images

Runs algorithms **4, 5, 8, 9, 10, 15 and 19** on 49 new regions from TIGER
(16), BCSS (16), CAMELYON16 (7) and BEETLE (10). It saves every algorithm RGB,
before-Bora mask, final Bora mask and original/reference pair. The complete
gallery matrix contains all **343** input/final-mask combinations.

The presets retain the original hypotheses and parameters. Class handling is
generalized to the observed coarse labels, including binary cases, rather than
forcing five classes. The original example is excluded from this cohort and
used only for exact regression checks. It is not another trained RGB inverse.

## Reproduce on Chiamaka

Run from a checkout containing `experiments/` and `src/`, with the already
downloaded benchmark and approved, SHA-verified Virchow2 weights. No new
downloads or global environment changes are necessary.

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH=src
multi_python=/mnt/sata_ssd/Bora-benchmark/experiments/virchow2-inverse-2026-10-16/.venv/bin/python
multi_root=/mnt/sata_ssd/Bora-benchmark/results/virchow2-multicase-2026-10-16
multi_model=/mnt/sata_ssd/Bora-benchmark/models/Virchow2/3158645804b69e3f3bc4439d4116edddf0840a72
multi_previous=/mnt/sata_ssd/Bora-benchmark/results/virchow2-inverse-2026-10-16
multi_source=/mnt/sata_ssd/Bora-benchmark/results/gigatime-cnn-pilot-2026-10-16

"$multi_python" -m experiments.virchow2_multi.plan \
  --data-root /mnt/sata_ssd/Bora-benchmark --root "$multi_root"
"$multi_python" -m experiments.virchow2_multi.run \
  --root "$multi_root" --model-dir "$multi_model"
"$multi_python" -m experiments.virchow2_multi.validate \
  --root "$multi_root" --previous "$multi_previous" --source "$multi_source"
"$multi_python" -m experiments.virchow2_multi.score --root "$multi_root"
"$multi_python" -m experiments.virchow2_multi.publish --root "$multi_root"
"$multi_python" -m experiments.virchow2_multi.audit --root "$multi_root"
```

Use a **new** output directory for a fresh run; `plan` refuses to replace an
existing corpus. `run` resumes only when frozen-file hashes remain valid. GPU
allocation is capped at 40% and CPU thread pools at two. Feature caches stay on
Chiamaka; the Mac gallery copy omits `.npy` arrays but retains every image/mask.

The historical `gigatime` source-folder name is only used by `validate` to
locate the old image and coarse labels. No GigaTIME predictions are used.

## Scope and resolutions

All cases are **ROIs**, including native windows from CAMELYON/BEETLE WSIs.
The preparation stage selects boundary-containing fields without method-score
selection and generates reference-derived majority coarse labels. One
conservative source-case group per image is retained. TIGER is non-TCGA only,
avoiding its known reuse of TCGA BCSS cases. Original case 114 is excluded.

Native RGB/masks and coordinates are retained. Processing uses an approximately
0.5µm/pixel grid, 224px model windows (112µm), 14px tokens (7µm) and 90px coarse
tiles (45µm). Effective axis spacing after integer rounding is recorded. Native
scores compare nearest-neighbour projected predictions with the unresampled
cropped reference. This is not a claim of newly recovered native pixel detail.
Microscopy optical resolving power and objective magnification are unknown.

BCSS uses its reported 0.25µm/px mirror sampling, TIGER approximately 0.5µm/px,
CAMELYON its embedded per-axis WSI spacing and BEETLE the original TIFF rational
X/YResolution with physical units (approximately 0.505µm/px). No magnification is
converted into an assumed pixel spacing. CAMELYON's tumor masks are
expert-derived; its negative tissue label is automatically detected rather than
expert stroma. Unreconciled masks or uncalibrated datasets are documented as
excluded in PLAN.json. One CAMELYON field lacked two usable coarse classes,
leaving 49 eligible regions from the requested 50.

## Reading outputs

- `gallery.html`: dataset-balanced ranking and interactive native mask overlays.
- `all_outputs.html`: every source image, algorithm input and final Bora overlay.
- `all_inputs.jpg`, `all_overlays.jpg`: complete contact sheets, thumbnails only.
- `cases/<case>/contact.jpg`: all seven inputs/results for one region.
- `cases/<case>/sample.json`: source, ROI/WSI origin, crop, class legend, sampling.
- `original.native.png` / `reference.native.tif`: native cropped reference pair.
- `original.png` / `coarse.tif`: common-grid image and 90px majority-tile input.
- `<method>/representation.png`: actual common-grid RGB sent to Bora.
- `<method>/before.native.tif`, `<method>/bora.native.tif`: native-coordinate
  compact-label masks; `sample.json` maps their labels to original tissue codes.
- `features/base.npy`: all 1280 spatial channels at the working sampling; the four variant
  caches retain the64 selected coordinates. This directory also stores the plain
  feature-RGB/Bora control, making it both a cache and a control-output folder.
- `scores.csv`, `ranking.csv`, `results.json`: every score and before/after delta.
- `ALL_PREDICTIONS_FROZEN.json`, per-case `FROZEN.json`, `AUDIT.json`: checksums
  and execution audit; numerical features and model weights are not in Git.

The before mask for method5 is a separate projection-classifier control. Other
methods already produce class evidence; their argmax is retained before Bora.
The primary rank uses equal-weight dataset means of per-image native macro IoU;
per-dataset ranks and an alternative excluding CAMELYON are also provided.
Missing coarse classes are retained in scoring, not dropped. Per-case best
methods are retrospective descriptions, not an automatic deployment selector.

An initial rounding discrepancy in the context head was discovered through
original-example regression. Only 10/15 and the plain-feature control needed
regeneration. The old revision, outputs and scores are archived in diagnostics;
no new-case metric was used to select a crop or hyperparameter. Fresh runs use
the corrected head directly; `--refresh-context-head` exists solely to reproduce
that documented correction on a complete earlier cache.

## Validation

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src \
  python -m pytest tests/test_virchow2_multi.py -q
```

Tests cover binary RGB geometry, nonconsecutive class/probability alignment,
case grouping, coarse-only support, ignored labels, missing true classes,
physical boundary tolerances and equal dataset weighting. The original
five-class image and its cached embeddings are used for exact algorithm
RGB/Bora-mask regression. The new cohort uses the same pinned encoder, with a
forward-inference batch size of four. The final
audit checks every frozen input/mask, source-group uniqueness, physical frame
mapping, feature-cache hashes and all 343 gallery outputs.
