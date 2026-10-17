# Full tissue, independent quality support, all algorithms side by side

Rerun the same 49 ROIs with retained methods **5, 8, 9, 10, 15 and 19**, the
equal-probability **9 + 19** combination, and matched H&E/plain-feature controls.
Algorithm 4 remains rejected. The new run preserves previous experiments.

The predictor reuses SHA-verified, full-field Virchow2 caches and verifies every
original-image token was encoded. Coarse-only feature/class fitting is retained.
Zero coarse tiles are completed from the same plain-feature class evidence for
all algorithms, preserving original labelled tiles. Method 15 computes its
object/context candidates on full RGB tissue. Bora runs on the completed input.
Its final native mask is clipped to a separate RGB-derived tissue support:
positive on every detected tissue pixel, zero on background. No fine reference
is read until all predictions have been frozen and hashed.

Quality is measured on the **original native reference > 0**, without intersecting
that region with estimated tissue. Unannotated tissue is predicted and displayed
but unscored. Annotated tissue erroneously removed remains penalized. All original
classes are scored, including those missing from the coarse labels. Physical
pixel spacing is preserved. RGB support is an algorithmic estimate, with fixed
physical morphology and explicit lumen/adipose handling; it is not expert truth.

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src
python -m experiments.virchow2_tissue.run \
  --source /mnt/sata_ssd/Bora-benchmark/results/virchow2-multicase-2026-10-16 \
  --root /mnt/sata_ssd/Bora-benchmark/results/virchow2-full-tissue-2026-10-16
python -m experiments.virchow2_tissue.score \
  --source /mnt/sata_ssd/Bora-benchmark/results/virchow2-multicase-2026-10-16 \
  --root /mnt/sata_ssd/Bora-benchmark/results/virchow2-full-tissue-2026-10-16
python -m experiments.virchow2_tissue.publish \
  --source /mnt/sata_ssd/Bora-benchmark/results/virchow2-multicase-2026-10-16 \
  --root /mnt/sata_ssd/Bora-benchmark/results/virchow2-full-tissue-2026-10-16
python -m experiments.virchow2_tissue.audit \
  --source /mnt/sata_ssd/Bora-benchmark/results/virchow2-multicase-2026-10-16 \
  --root /mnt/sata_ssd/Bora-benchmark/results/virchow2-full-tissue-2026-10-16
```

Use the existing isolated experiment environment on Chiamaka. No new model
download, GPU inference, global environment change or desktop restart is needed.

Serve the parent results directory with the original multi-image folder beside
the new one. **gallery.html** compares all seven selected conversions/masks side
by side, with shared image, zoom, centre and opacity. **all_outputs.html** shows
the entire 49-image matrix, including every actual RGB conversion above its
final mask. **detail.html** provides a larger five-panel view for any method.
All controls, native class TIFFs, coverage masks and per-class scores are linked.

RGB-only support review exposed grey scanner glass missed by the initial dark
neutral threshold. The threshold was changed from OD .12 to .30 before new
reference scoring, after inspecting original RGB/support pairs for all 49 cases.
Preliminary partial predictions were archived separately. Final predictions use
the fixed support.py policy; no metric-driven threshold optimization is used.

These are exploratory reference-derived coarse-label experiments on previously
inspected cases. Unannotated predictions are not validated by their existence.
CAMELYON's automatically generated non-tumor reference may disagree with RGB
background detection; references are retained unchanged. The report and coverage
counts expose those disagreements. Models, images, caches and results stay out
of Git; code and the manuscript-oriented report are committed.
