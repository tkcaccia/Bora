# Paired Bora label smoothing ablation

Use the frozen full-tissue Virchow2 experiment as input. Compare the existing
median9 setting, no label postfilter, median3, categorical majority3, and an
image-edge-weighted Potts regularizer. All nine representations/controls use
the same completed tile mask, background support and native reference geometry.
Algorithm 4 remains excluded. Existing defaults and outputs remain unchanged.

`none` removes categorical median filtering only: LAB Gaussian preblur, local
appearance-model averaging and detached-component cleanup remain fixed. Edge
regularization uses the original finite wand costs and four-neighbor contrast
weights inside the existing band. Majority and edge modes are class-identity
operations; unlike numeric median filtering, they do not impose a numerical
ordering on the class IDs. Majority ties retain the current class. Edge updates
require a strict energy reduction.

Run in the existing isolated Chiamaka experiment environment, with CPU thread
limits 2 and low scheduling priority. No Virchow2 inference is needed.

```bash
export PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -m experiments.virchow2_smoothing.run --source "$source" --root "$root"
python -m experiments.virchow2_smoothing.score --source "$source" --root "$root"
python -m experiments.virchow2_smoothing.publish --source "$source" --root "$root"
python -m experiments.virchow2_smoothing.audit --source "$source" --root "$root"
```

Prediction verifies previous input hashes and reproduces all 441 baseline masks
pixel for pixel. All 2,205 predictions are frozen and verified before this
ablation reads fine references. Scoring retains the original annotated support,
including penalties for RGB tissue detector false negatives. Ignore interfaces
are excluded from physical boundary metrics. All tissue remains predicted.

The primary summary gives equal weight to the seven retained conversions in
each case, then averages within dataset and equally over four datasets. A paired
case bootstrap retains correlated conversions together. Per-representation and
H&E/plain-feature control summaries are also saved. These are exploratory,
previously inspected cases with reference-derived coarse supervision. CAMELYON
non-tumor reference is algorithmic tissue support, not expert stroma.

The gallery includes original RGB, expert annotation, coarse tile input,
converted RGB, every final mask, exact difference maps and the editable band.
Every case exposes native pixel spacing and the current kernel footprint in
micrometres. The unchanged band and spatial prior are explicit remaining
constraints, so similar outputs should not automatically be attributed to
smoothing.
