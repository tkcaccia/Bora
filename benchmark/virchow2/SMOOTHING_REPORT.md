# Bora categorical smoothing comparison

The same 49 ROIs, seven retained Virchow2 representations (5, 8, 9, 10, 15, 19, and 9 + 19), plus H&E and plain-feature controls are compared under five postfilter settings. Algorithm 4 remains rejected. All 2,205 final native masks are retained.

## Controlled change

Only the label regularizer changes. Each prediction receives the exact SHA-verified previous converted RGB and completed tile mask. Working downsample 2, LAB preblur sigma 1.4, local color-model averaging, distance penalty, editable band, seed-connectivity cleanup and final native RGB background clipping remain fixed. Thus “no label smoothing” does not remove Gaussian averaging used to estimate the appearance model. No model encoding or feature selection is repeated.

- median9: original numeric median, window 9 × 9 at the Bora working scale; every one of 441 baselines is reproduced pixel for pixel.
- none: skip the categorical postfilter; keep seed-connected component cleanup.
- median3: reduce the existing numeric median window to 3 × 3.
- majority3: class-identity vote in a 3 × 3 window, unique winners only; ties retain the current class. This avoids imposing numeric order on tissue IDs.
- edge: no median; minimize the original wand class cost plus a contrast-weighted four-neighbor Potts penalty. Fixed weight 0.35, edge sigma 12 in weighted LAB, six deterministic red/black coordinate sweeps, strictly decreasing local energy updates. Changes remain in the same editable band and cannot claim classes with infinite unary cost.

Parameters were declared before scoring this ablation and were not tuned to these references. Native per-axis sampling is preserved in each case. A kernel footprint in micrometres is reported in the viewer. All inputs remain ROIs rather than full WSI inference.

## Aggregate results

Primary summary: average native macro IoU across the seven representations within each case, average cases within each dataset, then weight the four datasets equally. Controls are separate. This is a paired ablation; the seven conversions are not treated as independent specimens. Confidence intervals use 3,000 paired case resamples within dataset, retaining conversions together. Boundary F1 at 8 µm is secondary; 4 and 16 µm are also saved.

| Setting | Native macro IoU | Change from current | Paired descriptive 95% interval | Boundary F1 at 8 µm |
|---|---:|---:|---|---:|
| Current median 9 × 9 | 0.755587 | +0.000000 | [+0.000000, +0.000000] | 0.492152 |
| Small median 3 × 3 | 0.755108 | -0.000479 | [-0.000774, -0.000168] | 0.496701 |
| Categorical vote 3 × 3 | 0.755105 | -0.000483 | [-0.000779, -0.000171] | 0.496658 |
| Edge-aware competition | 0.755068 | -0.000519 | [-0.000845, -0.000184] | 0.497283 |
| No label smoothing | 0.755049 | -0.000538 | [-0.000852, -0.000206] | 0.497364 |

## Per representation

| Input | Best measured setting | IoU | Change from current |
|---|---|---:|---:|
| 05 | Current median 9 × 9 | 0.753246 | +0.000000 |
| 08 | Current median 9 × 9 | 0.753534 | +0.000000 |
| 09 | Current median 9 × 9 | 0.758203 | +0.000000 |
| 10 | Current median 9 × 9 | 0.755702 | +0.000000 |
| 15 | Current median 9 × 9 | 0.753404 | +0.000000 |
| 19 | Current median 9 × 9 | 0.756417 | +0.000000 |
| ensemble | Edge-aware competition | 0.758713 | +0.000107 |
| he | Current median 9 × 9 | 0.725238 | +0.000000 |
| features | Current median 9 × 9 | 0.752031 | +0.000000 |

## Why different RGB inputs can produce similar masks

For tiger_10, 79.21% of detected tissue is outside the editable band. Classifier-derived RGB colors can differ strongly while their class boundaries agree. The viewer reports the fraction changed from the completed tiles and from the current median9 result, and exposes magenta pixel-difference maps. This ablation isolates smoothing; it does not relax the band or spatial prior.

## Quality and limitations

All RGB-detected tissue is assigned; final background is zero. Scoring uses the original annotation-valid pixels only, unchanged from the prior full-tissue experiment. Unannotated tissue is displayed but excluded. Annotated pixels removed by the RGB background detector remain penalized. All original reference classes are scored even when absent from coarse supervision. CAMELYON non-tumor reference is automatically detected tissue, not expert stroma.

These images and conversions have already been inspected and coarse labels are reference-derived. Results are exploratory boundary-refinement comparisons, not independent held-out validation. A best measured setting is descriptive, not a claim that the setting generalizes. Existing Bora defaults and previous results are preserved.

## Deliverables

- gallery.html: original/reference/tile/conversion and all five final masks at synchronized zoom, with pixel-difference toggle and editable-band toggle. Select any of 49 cases and any retained representation/control.
- all_outputs.html: all 49 cases × seven primary representations, all five masks, with linked native TIFFs and per-case scores.
- scores.csv / results.json: all 2,205 scores, boundary metrics and per-class details.
- cases/<case>/<method>/<variant>/: native categorical mask, raw working mask, native RGBA visualization, difference map, overlay thumbnail. contact.jpg compares all settings.
- PROTOCOL.json / FROZEN.json / ALL_PREDICTIONS_FROZEN.json / AUDIT.json: exact parameters, input/output hashes and integrity checks.
- Keep the previous full-tissue and multi-image result folders alongside this folder for original inputs and expert masks.
