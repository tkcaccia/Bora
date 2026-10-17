# Cycle four: complete review and selection

Four of the twenty requested cycles are fully executed, reviewed and selected.
All 149 frozen ROIs produced all 3278 native masks from twenty candidates and
two controls. Every all-method contact sheet was actually viewed, with concrete
observations tied to its SHA256. Four full-run native crops were also viewed;
three earlier pilot native views are separate evidence, linked to identical
full-run pixels by the 176-mask parity check. Exporting a crop was not counted
as inspecting it: the four full-run records cover `labels.png` only.

Chiamaka and the local mirror passed the integrity audit: 6854 frozen files,
3278 native masks with exact display-raster agreement, and 745 exact comparisons
of unchanged-parent masks. Selection modified no prediction, score or reviewed
contact sheet. `CYCLE_FOUR_SELECTION.json` records a decision for all twenty
candidates; `CYCLE_FOUR_EXECUTION.json` contains the complete numerical evidence.

## Five survivors

| ID | Retained role | Mean image macro IoU | Mean seconds/image | Mean small-region recall |
| --- | --- | ---: | ---: | ---: |
| F01 | Accuracy leader; uncertain local prototypes | 0.679529 | 1.9140 | 0.189664 |
| C17 | Fast, converged palette inverse | 0.671604 | 0.1812 | 0.183608 |
| C01 | Tile-plurality weak supervision | 0.663706 | 0.0874 | 0.203302 |
| D08 | Fastest candidate; analytic within-tile nuisance | 0.656050 | 0.0505 | 0.194626 |
| C15 | Sensitivity research branch; false-positive risk | 0.644943 | 0.3219 | 0.225562 |

All five are unchanged from cycle three. No new child improves F01's primary
accuracy. Retaining existing methods is an experimental outcome, not an
improvement claim. C15 remains a sensitivity branch, not the default method.

G01 is the strongest new candidate: IoU 0.674400, 2.1064 seconds/image, small
recall 0.201831. Against F01 it wins 47 images, loses 101 and ties one. Its
accuracy and speed are worse, while its small recall is better. G01 is not
Pareto-dominated across these three metrics. The five-slot portfolio instead
keeps F01 for accuracy, C17 for much faster high overlap, C01 for greater small
recall at much lower cost, D08 for speed and C15 for the highest small recall.

The exact pair-count child G08 gives only 0.000631 greater mean small recall
than C01, with lower IoU, longer time, ten more completely missed components
and fifteen unsuccessful optimizer flags. It does not replace C01. G13–G15
improve C15's IoU but lose the sensitivity for which it is retained. Their
aggregate accuracy remains below other retained branches. Every other rejection
and its numerical/visual tradeoff is recorded in the signed selection.

## What actual inspection changed

- **G01's gains are partial and its losses include tissue identity.** Native
  BCSS74 at (0,492,256,256), the largest case loss against F01, shows a small
  reference-purple core replaced by false grey. Native BCSS87 at
  (828,268,256,256), selected from its strongest case gain, shows G01 reducing
  a displaced gold region but still missing the reference central gold circle
  and green ring. Both crops were selected post hoc using scored disagreement;
  they are explanations of observed extremes, not independent validation.
- **A bright void is not a universal background label.** Native BCSS18 at
  (637,947,256,256) shows G03 opening a real cleft that reference leaves zero.
  Native BCSS07 at (616,29,256,256) shows the rule still filling a broad pale
  cleft. TIGER11/25/28 overviews show deletion of annotated lumina/adipose.
  G03 leaves 5726334 additional reference-positive pixels unassigned, equivalent
  to 762023.13 um² after respecting each image's pixel spacing. These remain
  scored errors; reference-zero clefts remain qualitative evidence only.
- **Several new models change whole interiors incorrectly.** G06 introduces
  false gold halos in TIGER32/35 and class replacement in TIGER01/23. G11 makes
  false objects, merged islands and grey/purple mottling. G05's exact coarse-area
  constraint and G14's colour modes can suppress legitimate rare compartments
  or change whole tissue interiors. A mathematical constraint or successful
  optimization does not establish a valid biological assumption.
- **Small regions remain a central failure.** The atlas contains 5578 reference
  components <=400 um²; 377 have a class absent from the original coarse mask.
  No candidate reaches 50% correct coverage for 4053 components, and all 22
  completely miss 3471. These are overlapping descriptive categories. Comparing
  the all-method hit count to cycle three is not proof the retained parents
  worsened: that cycle also included rejected, highly fragmented candidates.
- **Sparse reference coverage hides extrapolation differences.** BEETLE04/10
  have nearly perfect scores despite substantial changes outside annotation.
  Reference-zero tissue is still analysed, but cannot validate its predicted
  boundary. All visible tissue is not necessarily annotated tissue.

## Numerical and CPU evidence

Unchanged C01 retains 67 unsuccessful optimizer flags; its original termination
messages are unavailable. G06 reaches its finite budget on 43 cases, with
maximum projected-gradient residual 0.000366965. G08 has fifteen unsuccessful
flags. G02's 1515 LP fits and G15's 505 class fits report success. G05 satisfies
its mass constraint on all 149 images. These flags and residuals remain explicit.

The new C++ pair-count kernel matches independent numerical and gradient
checks. On Chiamaka's synthetic 2048-bag workload, one/two-thread medians are
0.01856/0.01073 seconds (1.730x), with exact one/two-thread outputs. At 64 bags
the ratio is 1.011x, no useful threading gain. These include wrapper allocation
and are kernel measurements; no controlled end-to-end multicore speedup is
claimed. Full-cycle elapsed times use three low-priority image processes with
two numerical/OpenMP threads each and one measurement per method/image.

## Delivery and next cycle

The signed `selected.html` viewer displays only these five methods across all 149
images, alongside original H&E, reference, coarse input, unchanged #9+#19 RGB,
class/resolution metadata and adjustable semitransparent overlays. All twenty
candidate outputs remain archived for reproducibility and internal review.

The next cycle must implement fifteen new, distinct mathematical interventions,
three per survivor, alongside the five unchanged parents. It has not yet run.
The full objective remains twenty completed cycles followed by the unchanged
five-conversion comparison. All149 cases are repeatedly reused development
data, not independent validation.
