# Equal evidence combination: algorithms 9 + 19

Same 49 tissue ROIs as the preceding study: 16 TIGER, 16 BCSS, 7 CAMELYON16 and 10 BEETLE. No case, crop or coarse label was changed. This is ROI processing, not full-WSI inference.

| Input / method | Dataset-balanced native macro IoU |
|---|---:|
| Input tiled mask | 0.709799 |
| Original H&E + Bora | 0.729457 |
| Algorithm 9 + Bora | 0.761104 |
| Equal combination 9 + 19, then Bora | 0.761393 |

Ensemble minus algorithm 9: **+0.000289** macro IoU; paired case bootstrap 95% interval [-0.000893, +0.001492]. Wins / losses / ties: **27 / 19 / 3**. These are descriptive intervals for reused cases, not independent confirmation or clinical efficacy.

| Dataset | ROIs | Algorithm 9 | Combined | Difference | Wins / losses / ties |
|---|---:|---:|---:|---:|---|
| bcss | 16 | 0.545875 | 0.546369 | +0.000494 | 9 / 7 / 0 |
| beetle | 10 | 0.822047 | 0.820417 | -0.001630 | 2 / 5 / 3 |
| camelyon16-subset | 7 | 0.896456 | 0.897688 | +0.001232 | 5 / 2 / 0 |
| tiger | 16 | 0.780039 | 0.781100 | +0.001061 | 11 / 5 / 0 |

## Fixed method and controls

Class probabilities from algorithms 9 + 19 are aligned by actual class ID and averaged with fixed weights 1/2 each. The mean is converted to the same class-palette RGB, then processed with the unchanged Bora internal-wand preset (downsample 2). It is a fusion of class evidence, not an H&E photographic reconstruction. Probabilities are averaged before RGB quantization; no mask voting or reference-driven weight search is used.

All selected component RGB images and pre-Bora masks were reconstructed from the cached features and verified pixel-identical to the preceding experiment for every case. No encoder inference was repeated. All ensemble predictions were frozen and hashed before the scorer opened fine references. Existing algorithm 9 masks were preserved and every baseline score was reproduced exactly.

The primary score averages native-reference macro IoU within each dataset and gives each of four datasets equal weight. The interval uses 3,000 paired resamples within dataset. Code 0 is ignored according to the fine reference; classes missing from coarse inputs remain in the score. Boundary F1 uses native physical coordinates. Complete per-class IoU/Dice and boundary metrics are in results.json and per-case scores.json.

Working sampling is approximately 0.5 µm/px with recorded axis-specific values; native sampling and physical field size are inherited unchanged in each sample record. Coarse tiles are 90 working pixels (about 45 µm), Virchow2 windows 224 pixels (about 112 µm), spatial tokens 14 pixels (about 7 µm). Native masks are nearest-neighbour projections of working predictions. Optical resolving power and objective magnification are unknown, not inferred from pixel spacing.

Coarse labels are majority labels derived from the reference for a controlled boundary-refinement test. The methods were selected on these same cases, so this is exploratory reuse, not a held-out comparison. TIGER/BCSS/BEETLE provide their released tissue annotations; CAMELYON tumor is expert-derived, whereas its non-tumor tissue reference is automatically detected and must not be called expert stroma. Some fields have only partial annotations. Dataset-specific results are retained.

## Reading the images

Algorithm 4 was rejected by the user after visual inspection and is excluded from this combination and its algorithm selector. Its earlier outputs and the completed three-way trial remain archived.

gallery.html presents exactly four matched panels: original H&E; original annotated reference over H&E; actual input tiled mask over H&E; Bora refinement over H&E. Switch the algorithm to compare any of the retained preceding methods or the new combination. Mask opacity, zoom and crop centre are shared. The input tiled mask is the coarse supervision supplied to Bora; it is not the algorithm probability argmax. The reference is used for scoring and display, not prediction. Transparent areas mean ignore/unlabelled, not a tissue class.

all_outputs.html shows every ROI with original H&E, reference, tiled input, algorithm 9 refinement and combined refinement. All native numeric TIFF masks, full-resolution coloured masks, actual RGB supplied to Bora and scores are linked. No best-case filtering is applied. Side-by-side figures use a common 35% overlay opacity.

The prior result folder must remain alongside this one: gallery links use ../virchow2-multicase-2026-10-16/ for the unchanged originals, reference, tiled masks and algorithm outputs. New predictions are in cases/<case>/; PROTOCOL.json and FROZEN manifests record scope, weights, provenance and checksums. Source images and generated results are excluded from Git.
