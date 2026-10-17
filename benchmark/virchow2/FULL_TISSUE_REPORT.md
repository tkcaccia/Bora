# Full-tissue Virchow2 / Bora rerun

The same 49 native ROIs are rerun with separate analysis, background and quality supports. Retained methods are 5, 8, 9, 10, 15 and 19, plus the equal-evidence 9 + 19 combination. Algorithm 4 remains rejected. H&E and plain-feature controls use the same completed input and background removal. All 441 final masks and actual RGB inputs are retained.

## Coverage and quality

All RGB-detected tissue pixels receive a positive class label in every final mask; every pixel outside that support is zero. The masks include **21,935,352 unannotated tissue pixels** across the 49 fields. These pixels are displayed but excluded from quality scores. Evaluation uses the original **166,566,267 annotation-valid pixels**, with no intersection with predicted tissue. **3,028,863 annotation-valid pixels fall outside the RGB tissue estimate**; their removal remains penalized. These counts include all datasets, with the CAMELYON caveat below.

The primary metric is per-image native-reference macro IoU, averaged within each dataset and then equally over four datasets. Every original annotation class remains in scoring, including classes absent from the coarse inputs. Label 0 is ignored solely according to the original reference. Boundary F1 uses native physical spacing and excludes annotation-ignore interfaces.

| Rank | Representation + Bora | Dataset-balanced native macro IoU |
|---|---|---:|
| 1 | ensemble Equal evidence 9 + 19 | 0.758606 |
| 2 | 09 Scale-persistent evidence | 0.758203 |
| 3 | 19 Predictable latent structure | 0.756417 |
| 4 | 10 Context chooses, detail places | 0.755702 |
| 5 | 08 Stain-perturbation invariance | 0.753534 |
| 6 | 15 Object relationships | 0.753404 |
| 7 | 05 Worst-class colour geometry | 0.753246 |

Completed input tiles: 0.725648; H&E + Bora: 0.725238. Current #9: 0.758203; current 9 + 19: 0.758606. The paired descriptive 95% bootstrap interval for combination minus #9 is [-0.0006634430131655848, 0.0015108188339942753]. All scores and changes from the prior limited-support protocol are in scores.csv. Both coarse completion and background removal change the pipeline; differences from old outputs cannot be attributed only to an RGB conversion.

## Prediction protocol

Existing SHA-verified Virchow2 caches encode the entire original RGB field, not just annotated tiles. Every token covering the original image is checked to have at least one encoder window. The full-image caches are reused; no redundant encoder forward pass is needed. Feature selection and class heads are fit only to the original nonzero coarse labels. Unknown 90-pixel tiles are filled from mean plain-feature class evidence over detected tissue, or all pixels if the tile contains no detected tissue. Known coarse labels are preserved. All representations and controls receive exactly the same completed tiled input; pseudo labels never retrain their classifiers.

All selected transformations operate over the whole field. Method 15 now computes its candidate nuclei/lumen/context maps on full RGB-detected tissue, instead of restricting those maps to annotated coarse support. Its class head is still trained only from original labelled tile summaries. Predictions can use only the tissue classes represented in their original coarse training labels; newly labelled unannotated areas are extrapolations without reference validation.

Bora uses the unchanged internal-wand preset with downsample 2. It runs on the completed full-field labels, including temporary labels over background. Those labels are removed only afterward with the independent native RGB tissue mask. The working output is separately clipped to its working support. The final native mask is projected from the unmasked Bora prediction before native clipping, so fine support pixels are never lost because a resampled working support was zero.

## RGB tissue/background estimate

The fixed detector uses mean optical density > 0.025 with RGB saturation > 0.04, or mean optical density > 0.30 for dark neutral pixels. It applies elliptical closing of radius 2 µm, removes connected components smaller than 4 µm² and fills enclosed holes. Enclosed gland lumina and adipose spaces are retained as part of tissue architecture; exterior glass remains excluded. The rule is an algorithmic tissue estimate, not an expert annotation. Each support mask is exposed for visual inspection.

An initial OD 0.12 neutral threshold misclassified grey scanner glass in two CAMELYON RGB fields. Before any new fine-reference scoring, RGB-only review of all 49 images led to OD 0.30. Preliminary partial predictions are archived under virchow2-full-tissue-support-pilot-2026-10-16. The selected support rule was fixed before the final predictions and their scores. No score-driven support threshold search is performed.

## Scale, annotation and interpretation

All inputs are ROIs, not whole-WSI inference. Native RGB, crop coordinates, reference geometry and per-axis pixel spacing are inherited unchanged. Processing uses approximately 0.5 µm/px, 224-pixel encoder windows (about 112 µm), 14-pixel tokens (about 7 µm) and 90-pixel coarse tiles (about 45 µm). Native predictions are nearest-neighbour projections followed by native support clipping. Optical resolving power and objective magnification remain unknown.

Coarse supervision is reference-derived, and the methods and these cases were previously inspected. This remains exploratory controlled boundary refinement, not independent held-out classification or clinical validation. TIGER and BCSS avoid known TCGA reuse as documented in the original scope. CAMELYON tumor is expert-derived, but its non-tumor tissue reference is automatically detected, not expert stroma. Such automatic non-tumor labels may disagree with the new RGB background estimate; the reference is retained unchanged and those disagreements are visible in coverage counts.

## Files and views

- all_outputs.html: all 49 fields; original/reference/original tiled labels/completed tiled input/coverage; every retained converted RGB above its final Bora overlay and native score. H&E and plain-feature controls are also linked.
- gallery.html: synchronized original, reference, completed tiled input, converted RGB and final Bora panels, shared zoom and opacity, native masks, tissue/quality supports and per-class scores.
- all_inputs.jpg / all_masks.jpg: complete thumbnail sheets; cases/<case>/contact.jpg contains every converted input and final mask for that field.
- tissue.native.tif / tissue.working.tif: RGB-derived analysis support; evaluation.native.tif is the original annotation-valid support, created only after freezing predictions.
- coarse.completed.tif: exact working input to Bora, with original labels preserved and unknown tiles filled; coarse.full.native.tif is its native projection; coarse.native.tif is its tissue-clipped comparison.
- <method>/representation.png: actual working-grid conversion supplied to Bora; bora.pre_background.working.tif: output before the final background-removal operation; bora.native.tif: delivered native final class mask.
- results.json / scores.csv: every native score and coverage count; ALL_PREDICTIONS_FROZEN.json and FROZEN.json: immutable prediction hashes. Original reference/images remain in the preceding multi-image folder; keep it alongside this result folder.

Generated images, data, caches and model weights stay outside Git. Code, protocol and this manuscript-oriented report are saved in the repository.
