# Expanded Bora benchmark: 149 annotated ROIs

100 newly added ROIs: {'tiger': 24, 'bcss': 76}. Full corpus: {'tiger': 40, 'bcss': 92, 'camelyon16-subset': 7, 'beetle': 10}. Five retained RGB conversions (#9, #10, #15, #19, equal 9+19); eight retained Bora strategies; 5,960 native final masks.

Rejected strategies B04, B20, B16, B17, B14, B13, B01, B06, B07, B11, B15 and B18, and RGB conversions #4, #5 (Worst-class geometry), #8 (Stain perturbation), are excluded. Their benchmark implementations have been removed from the active source. Previous results remain an explicitly historical archive.

B19 is revision 2: equal votes from retained B02/B05/B09/B10, with appearance resolving ties. It contains no rejected solver. Its revised definition is distinct from historical B19. Other retained solvers and RGB conversions retain their prior parameters.

## Cohorts and selection

The original 49 cases are retained as previously inspected. Additional TIGER and BCSS groups are absent from that 49-case plan. TIGER uses non-TCGA cases and excludes group 114; BCSS uses TCGA. Each dataset contributes at most one ROI per recorded group. New images were selected deterministically with seed 8811, choosing annotated boundary-containing fields by class entropy, never by strategy scores. All previously available BEETLE cases and nearly all eligible CAMELYON groups were already used; there are no new cases from those datasets in this extension. Group IDs follow provider metadata; complete patient identity verification is unavailable.

All inputs are ROIs, including crops originating from full WSIs. Working encoder sampling targets 0.5 µm/px; exact actual per-axis sampling, native dimensions, source crop and field of view are recorded in PLAN.json and each SAMPLE.json. Virchow2 uses overlapping 224-pixel windows, 112-pixel stride and 14-pixel spatial token spacing. Nominal scale variants target 1 and 2 µm/px; actual sampling after integer resizing is saved. Optical resolving power and objective magnification are unknown. No resampling creates extra optical detail.

## Supervision and tissue policy

Feature selection remains per-image, supervised by the original reference-derived coarse tiles: pair-balanced Fisher separation with a correlation penalty selects 64 of 1280 spatial feature dimensions; #9 further retains 32 with cross-scale persistence. Coarse tiles are 90 working pixels (nominally 45 µm). Fine boundaries are excluded from prediction code. Entire original RGB fields are encoded, including unannotated tissue. Plain selected-feature evidence completes unlabelled coarse tiles identically for all solvers; completed pseudo labels never train the feature selector or class heads.

The original RGB alone determines tissue/background support. Every detected tissue pixel receives a class. Background is removed after segmentation. The scoring domain is the original reference >0 and is never intersected with detected tissue: unannotated tissue is unscored, while annotated tissue mistakenly removed by the detector remains penalized. All expert reference classes contribute to macro IoU, including classes absent from coarse supervision. CAMELYON non-tumor reference is automatic tissue support, not expert stroma.

## Ranking

Native macro IoU is averaged across five conversions within each case, then cases within each dataset, then equally across the datasets represented in that cohort. New-only ranking therefore balances two datasets; full ranking balances four. Baseline is the completed input tile mask with the same final background clipping. Differences and 3000-resample paired within-dataset case bootstrap intervals are relative to that input, with conversions kept together. Boundary F1 at 8 µm is secondary. Intervals are descriptive because coarse labels derive from the reference and method choices followed prior visual inspection.

## 100 new images

| Rank | Strategy | Macro IoU | Delta vs tiles | 95% paired interval | Boundary F1 8 µm |
|---|---|---:|---:|---|---:|
| 1 | B09 Geodesic competition | 0.694649 | +0.065865 | [+0.055613, +0.076594] | 0.532669 |
| 2 | B05 Seeded watershed | 0.681854 | +0.053070 | [+0.041326, +0.065282] | 0.522983 |
| 3 | B19 Retained-solver consensus | 0.667233 | +0.038449 | [+0.021153, +0.054982] | 0.533926 |
| 4 | B12 Hierarchical region merging | 0.666939 | +0.038154 | [+0.025066, +0.050902] | 0.537903 |
| 5 | B03 Multiscale posterior agreement | 0.666636 | +0.037852 | [+0.021602, +0.053817] | 0.512389 |
| 6 | B10 Convex total-variation partition | 0.666041 | +0.037256 | [+0.019785, +0.054001] | 0.529636 |
| 7 | B08 Finite-range CRF | 0.662472 | +0.033688 | [+0.015888, +0.050529] | 0.531285 |
| 8 | B02 Whole-field appearance classification | 0.660586 | +0.031802 | [+0.013740, +0.048915] | 0.530594 |
## All images

| Rank | Strategy | Macro IoU | Delta vs tiles | 95% paired interval | Boundary F1 8 µm |
|---|---|---:|---:|---|---:|
| 1 | B09 Geodesic competition | 0.770765 | +0.053466 | [+0.045115, +0.062614] | 0.524193 |
| 2 | B05 Seeded watershed | 0.762706 | +0.045407 | [+0.036670, +0.054414] | 0.515713 |
| 3 | B03 Multiscale posterior agreement | 0.754771 | +0.037473 | [+0.026016, +0.050345] | 0.512654 |
| 4 | B19 Retained-solver consensus | 0.753735 | +0.036436 | [+0.024316, +0.049707] | 0.525257 |
| 5 | B10 Convex total-variation partition | 0.753149 | +0.035850 | [+0.023626, +0.049239] | 0.522492 |
| 6 | B12 Hierarchical region merging | 0.751553 | +0.034255 | [+0.023637, +0.045493] | 0.519073 |
| 7 | B08 Finite-range CRF | 0.751378 | +0.034079 | [+0.021929, +0.047258] | 0.523519 |
| 8 | B02 Whole-field appearance classification | 0.750169 | +0.032870 | [+0.020473, +0.046455] | 0.522347 |

## Interpretation and files

This is a controlled refinement experiment using reference-derived coarse supervision, not independent whole-slide classification or a test using realistic classifier errors. The new groups extend image diversity but do not remove per-image supervision. Aggregate ranks can conceal class loss or poor boundaries; every retained output and per-class score is visible. Native outputs preserve source geometry; browser mask previews use nearest-neighbor resizing capped at 1536 pixels and disclose display sampling. No diffusion model or photorealistic inverse reconstruction is used.

gallery.html supplies original H&E, original annotation, completed input tiles, every retained converted RGB and all eight synchronized semi-transparent final masks. all_outputs.html contains every case/conversion sheet. scores.csv and results.json retain all scores, per-class metrics and cohort-specific rankings. PROTOCOL.json, input/prediction freezes, runtime_sources, source hashes and AUDIT.json record reproducibility and coverage. CPU inference and scoring use three low-priority workers with two threads each; Virchow2 uses the cached checkpoint and a bounded GPU batch of four. No desktop settings or unrelated analyses are changed.
