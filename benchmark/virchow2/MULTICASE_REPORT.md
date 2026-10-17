# Selected Virchow2 methods across benchmark tissue regions

Executed all seven selected algorithms on **49 new source-case/slide groups**, yielding **343 algorithm RGB images and Bora masks**. Every output is available in gallery.html, all_outputs.html, per-case contact.jpg and the two full contact sheets. No original case 114 image is included.

## Ranking

The primary score is native-reference per-image macro IoU, averaged within each dataset and then equally across datasets. This prevents the larger cohorts from dominating. Before-Bora scores and all per-case/per-class values are retained; no best-case-only selection. Bootstrap intervals describe paired case resampling within dataset, not clinical efficacy.

| Rank | Method | Dataset-balanced IoU | Δ H&E Bora | Paired bootstrap 95% Δ | Wins / losses / ties vs H&E |
|---|---|---:|---:|---|---|
| 1 | #09 Scale-persistent evidence | 0.7611 | +0.0316 | [+0.0229, +0.0410] | 41 / 6 / 2 |
| 2 | #19 Predictable latent structure | 0.7593 | +0.0298 | [+0.0214, +0.0392] | 42 / 5 / 2 |
| 3 | #04 Class-specific concepts | 0.7587 | +0.0292 | [+0.0216, +0.0374] | 42 / 5 / 2 |
| 4 | #10 Context chooses, detail places | 0.7583 | +0.0289 | [+0.0206, +0.0379] | 41 / 6 / 2 |
| 5 | #08 Stain-perturbation invariance | 0.7565 | +0.0270 | [+0.0188, +0.0362] | 40 / 7 / 2 |
| 6 | #15 Object relationships | 0.7561 | +0.0267 | [+0.0182, +0.0357] | 37 / 10 / 2 |
| 7 | #05 Worst-class colour geometry | 0.7560 | +0.0265 | [+0.0176, +0.0362] | 40 / 7 / 2 |

Matched controls, with the same dataset weighting: coarse **0.7098**, original H&E + Bora **0.7295**, plain selected-feature RGB + Bora **0.7548**.

## Dataset-specific rankings

| Dataset | Regions/groups | Ordered methods (native macro IoU) |
|---|---:|---|
| tiger | 16 | #04 0.7805; #09 0.7800; #05 0.7774; #19 0.7765; #10 0.7762; #08 0.7743; #15 0.7716 |
| bcss | 16 | #09 0.5459; #19 0.5444; #10 0.5430; #15 0.5420; #04 0.5416; #08 0.5400; #05 0.5368 |
| camelyon16-subset | 7 | #09 0.8965; #19 0.8954; #10 0.8941; #04 0.8928; #05 0.8911; #08 0.8909; #15 0.8909 |
| beetle | 10 | #09 0.8220; #08 0.8209; #19 0.8208; #15 0.8201; #10 0.8200; #04 0.8199; #05 0.8186 |

Excluding CAMELYON's automatically detected non-tumor reference, the equal-weight TIGER/BCSS/BEETLE order is: #09 0.7160; #04 0.7140; #19 0.7139; #10 0.7131; #08 0.7117; #15 0.7112; #05 0.7109.

## Exact scope and physical scale

These are **ROIs**, not full-WSI inference. TIGER and BCSS use released ROI images; CAMELYON16 and BEETLE use native crops from downloaded WSI windows. One conservative case/source-slide group per image is retained. TIGER uses non-TCGA cases, BCSS uses TCGA cases, avoiding their known cross-dataset reuse. Patient identity remains unverified for some source identifiers, especially CAMELYON slides.

Native RGB and reference masks are preserved in each case directory. Larger fields are cropped to at most about 640×640µm; minimum working side 896 px. Window selection uses annotated-class entropy and coarse-label eligibility, never algorithm scores. This enriches the cohort for boundaries and is not an estimate of clinical tissue prevalence. The input PLAN.json lists every source ID, group, native coordinate, mask hash, exclusion and class mapping.

Virchow2 inference uses a common approximately 0.5µm/px working grid, 224 px model windows (about 112 µm), 14 px spatial tokens (about 7 µm) and 112 px window stride. Integer output sizes cause small recorded axis-specific deviations from 0.5. Coarse majority tiles are 90 working px (about 45 µm). Native spacing is 0.5 for TIGER, 0.25 for the colour-normalized BCSS mirror, embedded per-axis WSI spacing for CAMELYON, and embedded TIFF X/YResolution with physical units for BEETLE (about 0.505µm/px). Do not infer optical resolving power or objective magnification from these values; both are unknown here.

Bora runs on each algorithm's common-grid RGB output, with the same internal-wand preset and downsample 2. Native evaluation masks are obtained by nearest-neighbour projection back to the original cropped reference grid. They preserve native coordinates but do not recover detail lost in resampling. The H&E control undergoes the same calibrated working-grid processing and Bora settings. No reconstructed H&E decoder or GigaTIME markers are used.

## Methods and before/after interpretation

Presets are inherited from the original five-class experiment: 64 balanced-MRMR coordinates, class-balanced logistic C=0.3, 12 NMF components / keep 8, max-min RGB geometry with 350 steps, two fixed OD stain transforms, base, 2× and 4× context at the working sampling, original object scales and four-quadrant ridge completion with an exclusion buffer of four tokens. Class handling is generalized to the actual observed coarse classes, retaining numeric IDs and a stable dataset palette. For binary cases, the three-axis geometry initializes an extra null-space axis; the context top-two gate includes both classes. These changes permit the same hypotheses to run across different annotation vocabularies without forcing a five-class task.

For methods 4, 8, 9, 10, 15 and 19, the before-Bora mask is the argmax of that method's class evidence. For method 5, its RGB projection has no intrinsic class mask, so a coarse-trained logistic head on the three projection coordinates supplies the explicitly labelled before-Bora control; it never affects the actual RGB/Bora result. Before masks share the same coarse-zero support restriction as Bora. A direct evidence classifier and a local boundary-refinement wand have different constraints, so a stronger before score does not imply the RGB is a photographic reconstruction.

A regression check against the original five-class example detected a small rounding difference in the initial class head. It was corrected by normalizing the original float64 tile means, matching the earlier implementation. Methods 10 and 15 and the plain-feature control were regenerated from cached features; the earlier revision, outputs and scores are archived under diagnostics/. Crop selection and method hyperparameters were not changed in response to new-case scores. FIVE_CLASS_REGRESSION.json records the final comparison with all seven original outputs.

## Annotation and scoring limits

TIGER supplies manual tissue compartments; BCSS supplies structured crowdsourced tissue labels; BEETLE supplies released expert tissue annotations. CAMELYON tumor is expert-derived, but code 1 is automatically detected non-tumor tissue, not expert stroma. Its results are shown separately and an alternative ranking excludes that cohort. Unreconciled SICAP JPEG masks, CoCaHis foreground semantics and uncalibrated RINGS/WSSS sampling are excluded rather than assigned guessed labels or micrometre scales.

The preparation stage uses fine annotations to select annotated fields and generate majority coarse labels. The prediction module reads only the original RGB, coarse labels and class metadata. All native and working predictions are frozen and hashed for every case before the scorer opens any fine reference. No parameters are selected from the resulting scores. This is controlled boundary reconstruction with reference-derived coarse labels, not independently predicted tissue classification or a prospective patient benchmark. It is a larger follow-up to methods selected on the earlier single ROI; further untouched cases would be needed to confirm a chosen winner.

**14 regions contain reference classes absent from their coarse labels.** Those classes remain in native macro IoU with zero IoU when unpredicted; they are not silently dropped or reassigned. Per-case class coverage is retained in scores.json. Label 0 is ignored only where the native reference is 0. Boundary F1 is computed per class in native physical coordinates, excluding annotation-ignore interfaces. Missing predicted contours score 0 in boundary F1; the supplemental HD95 omits unmatched contours and must not be interpreted alone.

## Files

- gallery.html: rankings, every-case contact overview, selectable native expert/coarse/before/after mask overlays with opacity control.
- all_outputs.html: every algorithm image and every Bora overlay, grouped by source region; no hidden failures or only-best subset.
- all_inputs.jpg / all_overlays.jpg: complete 49-region sheets (thumbnail overviews).
- scores.csv / ranking.csv / results.json: all native scores, before/after comparisons and rankings.
- cases/<case>/sample.json: source image, class legend, hashes, sampling, ROI/WSI origin and exact crop.
- cases/<case>/original.native.png and reference.native.tif: native reference pair.
- cases/<case>/<method>/representation.png: actual RGB supplied to Bora (working grid).
- cases/<case>/<method>/before.native.tif and bora.native.tif: numeric compact class masks in native coordinates.
- PLAN.json, ALL_PREDICTIONS_FROZEN.json and per-case FROZEN.json: scope and immutable prediction manifests.

Model: paige-ai/Virchow2, revision 3158645804b69e3f3bc4439d4116edddf0840a72, SHA256 8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b. Models, source images, features and generated visual outputs are kept outside Git; code and this report are saved in the repository.
