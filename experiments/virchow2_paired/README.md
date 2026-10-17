# Paired Virchow2 RGB reconstruction

This separate proof of concept accompanies the twenty experiments in
`experiments/virchow2_inverse/`. It tests a learned RGB decoder of **all final
spatial-token channels**, followed by bounded edits of coarse-class contrasts.
It does not use the original RGB pixels as reconstruction inputs or residuals.

The attached arXiv:2609.03947 paper motivates a trainable spatial-token decoder,
but its output is segmentation, not regenerated H&E. Our RGB decoding and
intervention are exploratory adaptations.

## Reproduce on Chiamaka

Use the dedicated environment already configured for the twenty-idea study.
Run from a code checkout containing `experiments/` and `src/`. The following
variables refer only to this task's data and environment:

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH=src
paired_root=/mnt/sata_ssd/Bora-benchmark/experiments/virchow2-paired-reconstruction-2026-10-16
paired_results=/mnt/sata_ssd/Bora-benchmark/results/virchow2-paired-reconstruction-2026-10-16
paired_python=/mnt/sata_ssd/Bora-benchmark/experiments/virchow2-inverse-2026-10-16/.venv/bin/python
paired_source=/mnt/sata_ssd/Bora-benchmark/results/gigatime-cnn-pilot-2026-10-16
paired_previous=/mnt/sata_ssd/Bora-benchmark/results/virchow2-inverse-2026-10-16
paired_model=/mnt/sata_ssd/Bora-benchmark/models/Virchow2/3158645804b69e3f3bc4439d4116edddf0840a72

"$paired_python" -m experiments.virchow2_paired.prepare --root "$paired_root" \
  --data /mnt/sata_ssd/Bora-benchmark/raw/tiger/extracted/roi-level-annotations \
  --model-dir "$paired_model"
"$paired_python" -m experiments.virchow2_paired.train --root "$paired_root"
"$paired_python" -m experiments.virchow2_paired.reconstruct --training "$paired_root" \
  --source "$paired_source" --previous "$paired_previous" --root "$paired_results"
# Optional single-tile diagnostic, after the primary RGB/Bora outputs are frozen:
"$paired_python" -m experiments.virchow2_paired.consistency --root "$paired_results" \
  --previous "$paired_previous" --model-dir "$paired_model"
"$paired_python" -m experiments.virchow2_paired.evaluate --training "$paired_root" \
  --source "$paired_source" --previous "$paired_previous" --root "$paired_results"
"$paired_python" -m experiments.virchow2_paired.publish --training "$paired_root" \
  --previous "$paired_previous" --root "$paired_results"
```

The historical source folder's name includes `gigatime`; only the original RGB,
coarse labels and final evaluation reference are used. No GigaTIME predictions
are inputs. No additional model or dataset downloads are required.

`prepare` writes the full crop manifest before extraction, excludes all source
case 114 images and separates other case proxies across train/validation/test.
TIGER's ROI release reports 0.5 µm/pixel for both its tissue-cells and tissue-bcss
subsets. The decoder takes native 224 px crops; no resampling is performed.
Tokens are cached in FP16, RGB in uint8. Roughly 4 GB of disk space is needed.
GPU allocation is capped at 40% of device memory and thread pools at two.

`train` uses training-only normalization and validation checkpoint selection.
The test set is evaluated after training. `reconstruct` never reads the fine
mask; it saves a frozen SHA256 manifest for all four RGB outputs and their Bora
masks. `evaluate` verifies those hashes before opening fine tissue annotations.
`publish` creates a local gallery and a manuscript-oriented report. The native
direct tile and overlap-assembled full ROI are both retained.

Read the generated report before interpreting outputs. A visually plausible
image or increased class contrast does not establish anatomical preservation.
Negative reconstruction results limit this decoder and budget, not the
information content of all possible Virchow2 inverses.

The consistency diagnostic starts from each decoder image and optimizes pixels
against **all** full or edited spatial-token targets, with no original-pixel
target. It uses FP32 input gradients because unscaled FP16 pilot gradients were
unstable. An actual-model finite-difference audit is saved in
`GRADIENT_AUDIT.json`. This diagnostic covers only the selected native tile;
it does not replace the primary full-ROI or Bora assessment.

Reproduce the gradient audit with:

```bash
"$paired_python" -m experiments.virchow2_paired.gradient_audit \
  --root "$paired_results" --previous "$paired_previous" --model-dir "$paired_model"
"$paired_python" -m experiments.virchow2_paired.audit \
  --root "$paired_results" --training "$paired_root"
```

The selective edit leaves all
1,216 unselected coordinates exactly unchanged. Before capping, it lies within
the learned class-contrast subspace; elementwise capping can add an orthogonal
component within the 64 selected coordinates. No channel is removed.

## Checks

```bash
PYTHONPATH=src python -m pytest tests/test_virchow2_paired.py -q
```

Tests cover source-case grouping, unchanged unselected components (and
orthogonal components when the component cap is inactive),
bounded latent edits, dependence on every spatial channel, output geometry,
identity reconstruction loss and one-to-one object matching. They do not replace
the held-out image reconstruction and expert tissue-mask evaluation.
