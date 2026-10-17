# Cycle three: complete review and selection

All 149 frozen ROIs and twenty candidates plus two controls produced 3278
successful native masks. Every all-method contact sheet was actually viewed
and recorded with its SHA256, followed by six full-run native-coordinate
crops. Four earlier native pilot crops can be linked to identical full-run
pixels by the saved 88-mask parity audit; these are not new visual inspections.
Both the local and Chiamaka integrity audits passed before review. No
prediction or score changed during inspection or selection.

The signed result is `CYCLE_THREE_SELECTION.json`. Actual overview and native
observations are in the runtime result folder and separate durable journals.
The numerical execution snapshot `CYCLE_THREE_EXECUTION.json` predates this
completed review and intentionally retains its historical pending status.

## Five survivors

| ID | Role | Mean image macro IoU | Mean seconds/image | Mean small-region recall |
| --- | --- | ---: | ---: | ---: |
| F01 | Integrated local-prototype uncertainty; accuracy leader | 0.679529 | 1.7402 | 0.189664 |
| C17 | Checked palette inverse; restrained compact masks | 0.671604 | 0.1715 | 0.183608 |
| C01 | Tile-winner weak supervision | 0.663706 | 0.0868 | 0.203302 |
| D08 | Analytic within-tile covariance; fastest candidate | 0.656050 | 0.0498 | 0.194626 |
| C15 | Copula density; sensitivity research branch | 0.644943 | 0.3073 | 0.225562 |

F01 improves mean IoU over D10 by 0.007132, winning 96 images and losing 52
with one tie. It is about twice as slow and has lower small-region recall.
The native BCSS44 crop confirms a whole tumor-compartment recovery, while
BCSS02 confirms a real loss of tumor margin. Retention is not a claim of
uniform improvement. C17 gives almost D10 accuracy at about one fifth its
time; retaining it and F01 leaves a more distinct portfolio than retaining
both local affine parent and child. D10 is not Pareto-dominated.

The accuracy/time/small-recall frontier also includes F05, F09, F12, F14 and
F15. They are not selected automatically. F14/F15's extra component hits
come with false internal regions and major accuracy losses. C15 is retained
only as the more restrained sensitivity branch, with its false-positive
risks made explicit. C01 and D08 share a linear output family, but estimate
different observation models; their overlap is acknowledged.

## Native evidence and next failure hypotheses

- **Broad gaps are still filled.** In BCSS54 at native crop
  (500,560,256,256), all eight viewed candidates, including F03, paint a
  broad bright cleft as red/green tissue. The reference leaves it zero, so
  this is a qualitative background failure, not an accuracy target. An
  earlier manually aimed BCSS54 crop missed the gap and is recorded as a
  tissue-interior view instead of being misrepresented as gap evidence.
- **Brightness and elongation are insufficient.** TIGER36 at
  (5,838,256,256) shows F03 removing branching bright spaces between cells
  inside invasive-tumor reference labels. Those positive reference pixels
  remain scored errors. A new glass hypothesis must distinguish genuine
  broad voids from bright anatomy without relying on this same inertia rule.
- **Convex support can invert whole interiors.** In native BCSS39 at
  (1,300,256,256), reference, coarse and fixed RGB are red. Seven candidates
  remain red, but F14 changes all 65536 pixels to lymphocytic purple.
  Overview/native inspection also shows false rings and grid speckle.
- **Typicality is not specificity.** F15 sometimes gives clean binary masks,
  but creates false gold/blue rims and whole-compartment changes elsewhere.
  Its recall advantage does not establish valid recovered tissue structure.
- **Many tiny regions lack usable coarse or RGB evidence.** The coordinate
  atlas finds 5578 components <=400 um², including 377 whose class is absent
  from the original coarse labels. No candidate reaches 50% correct coverage
  for 3643 components, and all 22 completely miss 2676. These overlapping
  failure groups are descriptive, not a complete causal decomposition.
  Unknown-colour proposals keep unknown semantic identity.
- **Partial annotation can conceal large extrapolation differences.**
  BEETLE04/10 have very high scores despite visibly different predictions
  outside sparse annotation. All detected tissue is analysed; only original
  reference-positive pixels contribute to accuracy.

## Numerical and runtime interpretation

C01 has 67 unsuccessful optimizer flags; its original termination messages
are unavailable. F12 has 56 unsuccessful flags at its 150-iteration budget.
F07 converges in all 149 cases, yet has lower accuracy than D08; convergence
is not a surrogate for mask quality. F04 records a projected-gradient
residual after at most 160 iterations, not a proven exact solution.

Times include candidate preparation, prediction, background policy and native
projection, excluding shared loading, scoring, compression and rendering.
The full cycle used three low-priority image processes with two numerical or
OpenMP threads each, and one timing per candidate/case. These elapsed times
do not establish controlled end-to-end CPU scaling. A separate synthetic
C++ hull-kernel profile measured 1.932x scaling from one to two threads with
exact output parity; its scope is kernel-only, including wrapper allocation.

The primary metric is the unweighted mean of native per-image macro IoU over
all 149 cases. Small-region recall is averaged over the 130 eligible cases
at >=50% correct component coverage. All reference-positive pixels remain
scored, including reference classes absent from coarse and pixels removed by
support or a candidate. Annotation-zero pixels remain unscored.

Three of twenty required cycles are now fully reviewed and selected. All 149
cases are repeatedly used for exploratory development selection; these are
not independent validation results. The next cycle must implement three
distinct mathematical children per selected parent, keeping these five and
the RGB conversion unchanged.
