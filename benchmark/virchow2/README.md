# Virchow2 experiments for Bora

## Expanded benchmark: eight retained strategies on 149 ROIs

The active benchmark retains **B02, B03, B05, B08, B09, B10, B12 and B19**,
with RGB conversions **#9, #10, #15, #19 and equal 9+19**. It adds **100 new
ROIs** (24 TIGER and 76 BCSS groups) to the original 49, giving **5,960 final
native masks**. New groups are disjoint from the original plan. New-only and
full-corpus rankings are reported separately against the common input tile mask.

Across the five retained RGB inputs, **B09 geodesic competition** has the highest
observed mean native macro IoU on the **100 new images: 0.694649**, versus
**0.628784** for their completed input tiles (difference **+0.065865**, descriptive
paired 95% interval **[+0.055613, +0.076594]**). B05 watershed is second at
**0.681854**. B09 also ranks first on the full 149-image, four-dataset aggregate
at **0.770765**, versus input tiles **0.717299**. New-only results balance TIGER
and BCSS; full results equally weight those plus CAMELYON16 and BEETLE, so the
two aggregates describe different dataset mixtures. These remain controlled
refinement experiments using reference-derived per-image coarse supervision.


Rejected B04/B20/B16/B17/B14/B13/B01/B06/B07/B11/B15/B18 and conversions
#4/#5/#8 are absent from the active benchmark. Their implementations were removed.
**B19 revision 2** now combines only B02/B05/B09/B10, with appearance resolving
ties. The production Bora package remains intact. Earlier sections below describe
historical studies; their code is retained in recorded Git revisions and output
runtime snapshots, while the current code implements the retained scope.

- [Expanded report](BORA_RETAINED_EXPANDED_REPORT.md)
- [Locked input plan](BORA_RETAINED_EXPANDED_PLAN.json)
- [Exact execution protocol](BORA_RETAINED_EXPANDED_PROTOCOL.json)
- [Complete output audit](BORA_RETAINED_EXPANDED_AUDIT.json)
- [Active source and commands](../../experiments/bora_strategies/)

Chiamaka results:
`/mnt/sata_ssd/Bora-benchmark/results/bora-retained-expanded-2026-10-17/`

Mac copy:
`/Users/stefano/Documents/ChatGPT/Bora/benchmark/results/bora-retained-expanded-2026-10-17/`

Open `gallery.html` for the new-image filter, all converted RGBs, original H&E,
original annotation, input tiles and all eight final masks with synchronized
semi-transparent overlays. `all_outputs.html` includes every image/conversion.
Sampling, class legends, quality support and per-class scores are preserved.

## Archived twenty-strategy study on 49 ROIs

Twenty distinct strategies, **B01–B20**, now run across all seven retained RGB
conversions and all 49 benchmark ROIs: **6,860 final native masks**. B01 is the
exact current balanced-wand baseline. Alternatives include whole-field appearance
classification, graph cuts, harmonic propagation, geodesic competition, region
merging, topology constraints, elastic deformation and a coarse-only small CNN.
RGB conversion IDs and Bora strategy IDs are separate; conversion 4 stays rejected.

The highest observed input/strategy pairing is **9 + 19 RGB with B09 geodesic
competition**: native macro IoU **0.7598**, versus **0.7586** for current Bora
on the same input. Boundary F1 at 8 µm is **0.5095 versus 0.4955**. The paired
IoU difference interval includes zero, so this is a candidate for inspection,
not an established general improvement. It improves 27 cases, worsens 21 and
ties one. This pairing was selected after comparing the full result matrix.

Current Bora has the highest measured aggregate native macro IoU, **0.7556**.
Geodesic competition B09 is close at **0.7553**, with a paired descriptive interval
for its difference spanning zero. Its mean boundary F1 at 8 µm is higher:
**0.5051 versus 0.4922**. Thus the study does not establish a general overlap
improvement; the complete galleries preserve the boundary/area tradeoffs and all
weak results for visual assessment. Means weight conversions equally within each
case and the four datasets equally. Coarse labels are reference-derived and the
cases have been previously inspected.

On the expert-tissue subset excluding CAMELYON's automatically generated
non-tumor reference, B09 has slightly higher mean IoU: **0.7261 versus 0.7247**.
This planned secondary comparison remains exploratory.

- [Twenty-strategy report](BORA_STRATEGIES_REPORT.md)
- [Exact final protocol](BORA_STRATEGIES_PROTOCOL.json)
- [Geometry correction and unchanged-output lineage](BORA_STRATEGIES_CORRECTION.json)
- [Code and reproduction commands](../../experiments/bora_strategies/)
- [Separate five-setting smoothing ablation](SMOOTHING_REPORT.md)

Outputs on Chiamaka:
`/mnt/sata_ssd/Bora-benchmark/results/bora-strategies-2026-10-17/`

Mac copy:
`/Users/stefano/Documents/ChatGPT/Bora/benchmark/results/bora-strategies-2026-10-17/`

Open **gallery.html** for original/reference/tile/converted inputs and all 20
native-mask comparisons at synchronized zoom, opacity and crop. Native TIFFs and
PNGs, exact difference maps, physical sampling, per-class scores and the original
annotation-only quality support are retained. Display masks are nearest-neighbor
previews capped at 1,536 pixels; full native files remain linked. **all_outputs.html**
contains all 49 cases and every retained conversion. B17's sampled-coordinate
interpolation was corrected using a synthetic identity test; the initial run is
archived, the other 19 outputs remain byte-identical, and unchanged scores are
reused only after matching native hashes, reference geometry and evaluator code.

## Full tissue, background removal and all algorithms side by side

The latest rerun separates **RGB-detected tissue** from **annotated scoring
support**. All detected tissue receives labels, including 21,935,352 native
pixels without annotations; those pixels are displayed but unscored. Background
is removed after Bora. The original annotation-valid quality region is retained,
so annotated tissue removed by the RGB detector is still penalized.

The same 49 ROIs run methods **5, 8, 9, 10, 15 and 19**, **9 + 19**, and matched
H&E/plain-feature controls. Algorithm 4 remains rejected. The main viewer shows
every retained algorithm side by side, with its actual RGB conversion above its
final Bora overlay. Shared zoom, crop, opacity, native scales and scores make
the columns directly comparable. A separate matrix includes all 49 fields.

The new equal-dataset native macro IoU ranking starts with **9 + 19: 0.7586**,
**9: 0.7582**, and **19: 0.7564**. The combination's interval versus 9 includes
zero. Completion and background removal change the pipeline; this ranking should
not be interpreted as the earlier support-limited comparison. All reference
classes and original annotation pixels remain in scoring.

- [Full-tissue report and exact protocol](FULL_TISSUE_REPORT.md)
- [Code and reproduction commands](../../experiments/virchow2_tissue/)

Chiamaka output:
`/mnt/sata_ssd/Bora-benchmark/results/virchow2-full-tissue-2026-10-16/`

Mac copy:
`/Users/stefano/Documents/ChatGPT/Bora/benchmark/results/virchow2-full-tissue-2026-10-16/`

Open **gallery.html** for all algorithms side by side on a selected image, or
**all_outputs.html** for every image, conversion and final mask. Keep the original
multi-image results folder alongside it. Earlier limited-support experiments
below are preserved as historical results.

## Active selection: combine algorithms 9 and 19

Algorithm **4 was rejected by the user after reviewing the complete gallery**.
The active combination averages only methods 9 and 19, with equal probability
weights, then runs the unchanged Bora preset on all the same 49 ROIs.

Dataset-balanced native macro IoU is **0.7614**, versus **0.7611** for algorithm 9:
27 improved images, 19 worse, 3 tied. The mean difference is only +0.00029,
with paired descriptive bootstrap 95% interval [-0.00089, +0.00149]. This does
not show a clear advantage over algorithm 9. BEETLE is slightly worse; all
per-case and per-class outcomes are retained. These are exploratory reused cases.

- [Active two-method report](ENSEMBLE_9_19_REPORT.md)
- [Current human method selection](METHOD_SELECTION.json)
- [Code and reproduction commands](../../experiments/virchow2_ensemble/)
- [Archived three-way trial report; algorithm 4 subsequently rejected](ENSEMBLE_REPORT.md)

Active outputs on Chiamaka:
`/mnt/sata_ssd/Bora-benchmark/results/virchow2-ensemble-9-19-2026-10-16/`

Local Mac copy:
`/Users/stefano/Documents/ChatGPT/Bora/benchmark/results/virchow2-ensemble-9-19-2026-10-16/`

Keep the preceding multi-image result folder alongside this one. Open
`gallery.html` for the four-panel original/reference/tiled/refined viewer, or
`all_outputs.html` for all 49 original/reference/tiled/algorithm-9/combined
comparisons. Algorithm 4 is excluded from the active selector. The original
seven-method study and completed three-way trial remain archived.

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
