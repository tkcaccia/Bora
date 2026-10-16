# Fixed combination and four-panel mask comparison

Combine the class evidence of algorithms 9 and 19 with equal weights, before
palette conversion or 8-bit quantization, then run the unchanged Bora preset. Algorithm 4 was rejected by the user and
is excluded from the active combination. The earlier three-way trial is archived;
reproduce it explicitly with `--components 9 19 4` and a separate output root.
This reuses the exact 49-region cohort and cached Virchow2 features from
`experiments/virchow2_multi`; it does not run the encoder or alter prior outputs.

Run from the repository root with the existing experiment environment:

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src
python -m experiments.virchow2_ensemble.run \
  --source /mnt/sata_ssd/Bora-benchmark/results/virchow2-multicase-2026-10-16 \
  --root /mnt/sata_ssd/Bora-benchmark/results/virchow2-ensemble-9-19-2026-10-16
python -m experiments.virchow2_ensemble.score \
  --source /mnt/sata_ssd/Bora-benchmark/results/virchow2-multicase-2026-10-16 \
  --root /mnt/sata_ssd/Bora-benchmark/results/virchow2-ensemble-9-19-2026-10-16
python -m experiments.virchow2_ensemble.publish \
  --source /mnt/sata_ssd/Bora-benchmark/results/virchow2-multicase-2026-10-16 \
  --root /mnt/sata_ssd/Bora-benchmark/results/virchow2-ensemble-9-19-2026-10-16
```

Prediction never reads fine reference pixels or existing scores. All 98
component RGB images and pre-Bora masks must reproduce exactly before the
combined evidence is accepted. All 49 predictions are frozen and hashed before
scoring. The scorer also reproduces every old algorithm 9 score exactly.
Actual class IDs, including nonconsecutive IDs, are retained. There is no weight
search on the references. The previous methods module has one optional return
flag to expose its existing probabilities without changing its default output.

Serve the parent `results` directory so the unchanged prior folder is available
at the gallery's relative links. The viewer shows four matched anatomical
panels: original H&E, original reference annotation, input tiled labels, and
Bora-refined labels. It supports the six retained preceding algorithms, the combination
and the H&E control, with shared opacity, zoom, centre and physical scale bars.
The evidence argmax is explicitly distinguished from the input tiled mask.
`all_outputs.html` retains every paired final mask and score; `scores.csv` and
`results.json` retain numerical and per-class metrics. Each case also has a
four-panel export and a five-panel #9/combination comparison.

The primary mean gives four datasets equal weight, with 3,000 paired case
bootstrap resamples within dataset. Patient mappings are incompletely verified.
The selected methods and the combination are exploratory reuse of these cases.
Reference-derived coarse labels, partial annotation, missing coarse classes and
CAMELYON's automatically detected non-tumor reference retain the preceding
study's limitations. No clinical or held-out validation claim is made.
