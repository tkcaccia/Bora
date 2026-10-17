# Twenty Bora strategies on identical converted inputs

Compare 20 inference mechanisms (B01–B20), including current Bora, on the same
49 ROIs and seven retained RGB conversions. These produce 6,860 native masks.
Conversion IDs and strategy IDs are separate. The registry describes each
hypothesis and freezes numerical parameters before scoring. These are distinct
experimental implementations, not twenty claims of methodological novelty.

Most new strategies allow whole-field relabelling from coarse-only appearance
supervision and sparse tile-center anchors. They do not inherit the current
wand's fixed boundary band. Exceptions and constraints are explicit in the
registry and report. Completed coarse labels initialize geometry or a soft area
prior in specified strategies; supervised targets remain original coarse labels.
Fine-reference pixels are never read by the predictor.

Use Chiamaka's existing isolated experiment environment. No Virchow2 encoding,
model download, global installation or desktop modification is required.

```bash
export PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -m experiments.bora_strategies.run --workers 3 \
  --source "$source" --root "$root"
python -m experiments.bora_strategies.score --workers 3 \
  --source "$source" --root "$root"
python -m experiments.bora_strategies.publish --source "$source" --root "$root"
python -m experiments.bora_strategies.audit --source "$source" --root "$root"
```

Production scope is fixed at 49 cases × seven conversions × 20 strategies.
`--pilot --cases tiger_10 --conversions 09 --workers 1` enables an explicitly
separate smoke/timing run. Use a different output directory. A preliminary
sequential timing run was archived unscored before the bounded parallel run.

Original native geometry and per-axis physical sampling are preserved. New
solvers use approximately 1 µm/px; B01 retains its exact previous native output.
All detected tissue is predicted, final background is removed, and scoring
retains the original annotated support. Unannotated tissue is displayed but
unscored. False-negative tissue removal remains penalized. The report retains
CAMELYON's algorithmic non-tumor-reference caveat and the previously inspected,
reference-derived coarse-supervision limitation.

The primary aggregate weights conversions equally within each case and datasets
equally. Paired bootstrap resampling keeps all conversions of a case together.
All weak outputs, collapsed classes and per-class scores remain available.
The viewer supplies all 20 native masks at synchronized zoom and opacity,
original/reference/tile/converted inputs, the current editable band and exact
magenta differences. The all-image page links every comparison sheet.

Default production Bora behavior remains unchanged; the new mechanisms live
in this experiment package until they have been assessed and validated.
