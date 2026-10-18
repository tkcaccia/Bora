# Cycle nine: fifteen prospective interventions

Cycle eight is signed after349 actually inspected all-method overviews and14
native crops. Its selected parents are J01, L02, K05, L03 and C17. The next
twenty candidates comprise these unchanged five and three new models per
parent. B21/B22 remain controls. No new model is an established improvement.

| Parent | Children | Changed assumption |
| --- | --- | --- |
| J01 | M01 latent true-class fields; M02 conserved observed class appearance; M03 series-resistance H&E transport | Coarse classes may be wrong; field transport should preserve the observed first moment; subgrid stain barriers need not disappear on averaging |
| L02 | M04 positive sampling uncertainty; M05 normalized Student prediction; M06 expected appearance distance | Observation noise differs from a Bayesian inverse diagonal; finite appearance observations imply uncertain variance; expected classification loss differs from predictive density |
| K05 | M07 free-boundary thin plate; M08 Hermite value/gradient observations; M09 repulsive class prototypes | Affine fields should have zero curvature energy; observations may contain useful derivatives; competing class means should not collapse locally |
| L03 | M10 accumulated-stain exterior access; M11 eight-ray exterior visibility; M12 binary H&E glass mincut | Background may be defined by integrated stain cost, straight access, or one global unary/perimeter energy rather than route width |
| C17 | M13 quantized observation intervals; M14 counterfactual class necessity; M15 adversarial joint palette error | RGB bytes have finite precision; class necessity differs from mixture size; palette calibration can be uncertain |

`nine_manifest.py` freezes all equations, numerical budgets and failure
hypotheses before the engineering pilot. `cycle_nine.py` has no reference-mask
I/O. These are separate models, with one stated intervention each, not voting
combinations or parameter sweeps. M01 infers semantic observation labels,
whereas H01 retained semantic labels and fitted within-class appearance modes.
M02 conserves colour moments, unlike the earlier predicted-area constraint.
M07 changes free-boundary curvature, not a Laplacian coefficient. M13 models
byte precision, unlike C14's broad-cell feasible probability intervals.

The349 frozen ROIs and09+19 converted RGB are unchanged. The parents must match
all349 cycle-eight native masks exactly. Unknown tissue is analyzed but fine
reference0 is unscored; all positive reference regions, including absent coarse
classes and deleted tumor, count. RINGS outside-tumor0 is a valid negative and
is not an expert stroma label. Thirty constant-RGB images use the same frozen
coarse completion for every candidate; explicitly H&E-based glass models can
still delete white pixels. No tissue-class refinement is claimed there.

All scale values use calibrated micrometres where verified, otherwise explicit
pixels on the80 WSSS4LUAD/RINGS inputs. M10 and M12 use a four-unit H&E grid for
bounded graph memory; original-resolution white pixels gate every removal.
M12 can remove internal lumina and therefore has no exterior-access guarantee.
M14's palette hull solves use a1e-7 numerical tie ridge; counterfactual necessity
is not a tissue posterior. Palette children all use occupied24-bin colour nodes
and an averaged coarse prior, whereas the affine C17 branch uses a direct pixel
inverse. They are whole algorithm comparisons, not isolated loss ablations.

Primary utility remains mean_i[macro_IoU_i-0.05 max(seconds_i-1,0)]. Candidate
mask projection and glass deletion are timed; shared encoder preparation,
storage, rendering and scoring are separate. Max three nice12 image workers,
two numerical/OpenMP threads each. Mathematical tests precede a separate13-case
engineering pilot. All predictions freeze before reference scoring. Full-cycle
completion still requires actual inspection of every all-method overview,
useful native crops, explicit20-method tradeoffs, a signed five and its viewer.
All349 are repeatedly used development images, not independent validation.

Engineering corrections before visual pilot inspection: the first pilot froze
all286 masks but stopped at a scoring guard that still recognized only L03's
glass deletions. The second snapshot registers the three new glass mechanisms
explicitly, including constant-input fallback metadata. That failed pilot is
preserved. A mathematical simplification then removed M06's proposed minimum
Cantelli-bound calculation: the class with smallest expected distance has all
positive rival margins, whereas every other class has a nonpositive margin
against it. Thus that ranking collapses to minimum expected distance. The final
M06 implements this exact, simpler Gaussian quadratic-risk rule directly and
checks it against independent six-point Gaussian quadratic integration. These
are engineering snapshots of one cycle, not additional completed cycles.

Real-input numerical checks of the third pilot, before actual visual review,
found M09 gradient residual0.263 and M15 projected residual0.0156. The finite
solvers were inadequate. M09 now uses sparse quadratic majorization with a
proven global upper curvature bound for its repulsion, reusing factorizations.
M15 now solves its unsmoothed convex minimax energy using exact simplex
quadratics and an exact two-norm proximal map, with explicit ADMM primal,
dual and projected Lagrangian residuals. Independent solvers verify both small
optima; no numerical correction is counted as an additional model or cycle.
