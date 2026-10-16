# Virchow2 experiments for Bora

## Equal combination of algorithms 9, 19 and 4

The fixed equal-probability combination followed by Bora scores **0.7625**,
compared with **0.7611** for algorithm 9 on the same 49 ROIs, using equal dataset
weight. It improves 34 images, worsens 12 and ties 3. This is a small exploratory
gain on reused cases. The new viewer shows the original H&E, original reference
mask, actual input tiled mask and selected Bora refinement in four synchronized
panels. All earlier seven methods can also be selected.

- [Combination report](ENSEMBLE_REPORT.md)
- [Code, protocol and reproduction commands](../../experiments/virchow2_ensemble/)

Outputs on Chiamaka:
`/mnt/sata_ssd/Bora-benchmark/results/virchow2-ensemble-2026-10-16/`

Local Mac copy:
`/Users/stefano/Documents/ChatGPT/Bora/benchmark/results/virchow2-ensemble-2026-10-16/`

Keep the preceding multi-image result folder alongside this one. Open
`gallery.html` for the four-panel viewer and `all_outputs.html` for all 49
original/reference/tiled/algorithm-9/combined comparisons.

## Seven selected methods across 49 new regions

Algorithms 4, 5, 8, 9, 10, 15 and 19 were executed on 49 new regions from TIGER (16),
BCSS (16), CAMELYON16 (7) and BEETLE (10), producing 343 algorithm images and
final Bora masks. Algorithm 9 has the highest equal-dataset mean macro IoU:
0.7611, versus 0.7295 for matched H&E/Bora and 0.7548 for plain Virchow2 feature
RGB/Bora. It improves 41 of 49 regions over the H&E control. Every result,
including weaker cases and before/after masks, is retained.

- [Multi-image report and rankings](MULTICASE_REPORT.md)
- [Code and reproduction commands](../../experiments/virchow2_multi/)

All images are ROIs. Processing uses a calibrated approximately 0.5µm/px grid;
scores compare projected native-coordinate predictions with native reference
masks. Coarse labels are reference-derived. CAMELYON's negative tissue label is
automatically detected, and its cohort is also ranked separately. These are
controlled refinement experiments, not independently predicted coarse labels.

Remote output:
`/mnt/sata_ssd/Bora-benchmark/results/virchow2-multicase-2026-10-16/`

Local Mac output:
`/Users/stefano/Documents/ChatGPT/Bora/benchmark/results/virchow2-multicase-2026-10-16/`

Open `gallery.html` for ranking and adjustable native-mask overlays, or
`all_outputs.html` to inspect every algorithm input and final mask. Models,
images and numerical feature caches remain outside Git.

## Twenty original hypotheses on one region

This benchmark tests twenty distinct interventions on one expert-annotated
TIGER breast tissue ROI. The image is 1224 × 1166 pixels at provider-reported
approximately 0.5 µm/px. It is **an ROI, not a full WSI**; objective magnification
and optical resolving power are unknown.

- [Experimental protocol and reproduction commands](../../experiments/virchow2_inverse/README.md)
- [Experiment source code](../../experiments/virchow2_inverse/)
- Results report: `REPORT.md` in this directory after execution.

The source image and coarse mask come from the downloaded benchmark. No
GigaTIME prediction is used. Fine reference masks are loaded for scoring only
after all twenty outputs are frozen. The coarse labels themselves were derived
from expert annotations, so the task is controlled boundary reconstruction.

On Chiamaka, complete numerical outputs are saved under:

`/mnt/sata_ssd/Bora-benchmark/results/virchow2-inverse-2026-10-16/`

The Mac gallery copy is saved under:

`/Users/stefano/Documents/ChatGPT/Bora/benchmark/results/virchow2-inverse-2026-10-16/`

Open `gallery.html` to compare the original image, expert reference, coarse
tiling mask, each experimental image and each Bora result. Overlays have
adjustable transparency. TIFF masks, per-class scores, model/input hashes,
physical sampling, candidate object maps and donor provenance remain with the
full run. Models, benchmark images and generated outputs are excluded from Git.

Dataset: [TIGER](https://tiger.grand-challenge.org/Data/).
Model: [Virchow2](https://huggingface.co/paige-ai/Virchow2).

## Paired full/selective reconstruction proof of concept

The additional experiment trains a frozen-Virchow2 spatial-token RGB decoder on
3,920 native TIGER tiles from other source cases, with separate validation and
test cases. It compares the unchanged full token representation with bounded
coarse-class feature edits using the same decoder. No original-pixel residual
supplies image detail. A separate FP32 token-consistency diagnostic checks the
same focal tile.

- [Paired experiment report and results](PAIRED_RECONSTRUCTION.md)
- [Paired source code and reproduction commands](../../experiments/virchow2_paired/)
- [Exact dataset scope](paired_dataset_scope.json)

The decoder regenerates recognizable tissue but did not pass the prespecified
fidelity screen. The primary edited image did not improve Bora over original
H&E (macro IoU 0.7446 versus 0.7476 on this one ROI). This does not establish a
new preprocessing method. The earlier twenty outputs remain unchanged.

The paired output directory on Chiamaka is
`/mnt/sata_ssd/Bora-benchmark/results/virchow2-paired-reconstruction-2026-10-16/`.
The local Mac copy is
`/Users/stefano/Documents/ChatGPT/Bora/benchmark/results/virchow2-paired-reconstruction-2026-10-16/`.
Both contain `gallery.html`, paired images, masks, scores and provenance.
