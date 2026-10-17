# Cycle four: fifteen review-derived hypotheses

Five unchanged parents are F01, C17, C01, D08 and C15, as signed after all 149
cycle-three visual inspections. No RGB representation, encoder, fine reference,
or semantic training label is changed. Every candidate still analyses all
detected tissue. Three children per parent give twenty candidates plus two
controls. This document describes hypotheses, not established improvements.

| ID | Parent | Mathematical intervention | Main risk |
| --- | --- | --- | --- |
| G01 | F01 | Bounded-channel Beta predictive densities using local mean/variance | Endpoint artefacts and lost cross-channel dependence |
| G02 | F01 | Global spatial-affine Laplace likelihood, fitted by exact LAD linear programs | Too little local flexibility; mixed coarse medians |
| G03 | F01 | Physical disk clearance and white-component reconstruction | Large true lumina/fat can also be removed |
| G04 | C17 | Poisson/I-divergence palette observation on the simplex | Synthetic RGB is not photon data |
| G05 | C17 | Minimum-KL posterior change with prescribed global class totals | Coarse winner proportions are not true areas |
| G06 | C17 | Logistic observations of within-pixel channel rank order | Only six strict orders; weak class identifiability |
| G07 | C01 | Intrabag design-effect correction to count variance | Heterogeneity can masquerade as dependence |
| G08 | C01 | Exact signed categorical count comparisons by C++ dynamic programming | Independence remains false; pairwise composite, not joint event |
| G09 | C01 | Analytic one-versus-rest between-tile ranking risk | Ranking scores are not calibrated probabilities |
| G10 | D08 | Regularized Tyler angular scatter of within-tile residuals | Relevant radial colour variation is downweighted |
| G11 | D08 | One pooled rank-normal RGB geometry before bag discrimination | Biased pooled ranks distort class contrasts |
| G12 | D08 | Matched neighbouring-tile contrast normal equations | Boundary appearance may differ from interiors |
| G13 | C15 | Seven-projection patch-to-class quantile transport | Finite projections and mixed class distributions |
| G14 | C15 | Full-tissue colour-density mode basins, coarse-only class counts | Shared modes merge classes; unsupported modes have unknown identity |
| G15 | C15 | Joint normalized maximum-entropy RGB density from Legendre moments | Rare modes lost; cubature and fitting are approximate |

## Why these follow the actual review

G01 and G02 question F01's Gaussian appearance assumption rather than adding
another postprocessing vote. G02 changes both the error model and the unit of
spatial fit to one robust surface per class. It is a single Laplace regression
model, not a claimed exact local Gaussian integral. G03 tests the opposite
failure pattern from the rejected inertia-based F03: a broad cavity can pass
the clearance rule while a narrow branching intercellular space cannot. Its
explicit large-lumen/adipose risk must be judged against positive-reference
errors and actual native views. The original H&E white threshold stays fixed.

G04 and G06 change distinct aspects of the palette observation: amplitude
likelihood versus ordinal information. G05 instead tests a single global
class-mass-constrained relative-entropy objective, with an intentionally
falsifiable assumption about extrapolating known coarse areas. These three
share a palette ancestry but are not solver ports or coefficient sweeps.

G07 questions effective sample size, G08 tests the normal tail approximation,
and G09 changes the observation to between-tile ranking. The G08 kernel uses
exact heterogeneous categorical factors and deterministic count ties. Its
gradient is derived from forward/backward convolutions. Exact arithmetic for
the stated pair events does not validate independent tissue pixels or turn
their product into a joint multiclass likelihood. The C++ port is machinery
within this one count-model intervention, not an additional candidate.

G10 changes nuisance scatter, G11 changes the shared observation geometry,
and G12 changes the matched observational design for class contrasts. G12's
small graph indexes classes, not image regions; it does not diffuse a mask.
G13 compares local distributions by transport, G14 uses density mode basins
learned on all tissue, and G15 fits a single joint maximum-entropy density.
The G14 density may observe unannotated tissue, but only original coarse
labels train its semantic counts. Unsupported modes and connected regions
are reported with unknown identity. Their hard-mask tie is documented, not
presented as discovery of a named tissue class.

## Evaluation and execution

Use all 149 frozen cases and the same 9+19 conversion. First run the focused
numerical/semantic tests and a four-dataset engineering pilot with exact
unchanged-survivor native-mask parity. A pilot does not complete a cycle.
Freeze executed source and compiled-library hashes before any fine-reference
scoring. Predictions must all finish before reference-mask scoring starts.

Primary selection remains the mean native per-image macro IoU over all 149.
Runtime includes each candidate's own preparation and native projection.
Report physical small-region recall/misses, all positive-reference deletions,
background evidence and outlier proposals separately. The five-survivor
decision requires actual inspection of every all-method contact sheet and
targeted native crops. Numerical residuals and optimizer flags are retained;
finite budgets are not a claim of exact convergence.

Chiamaka normally uses three low-priority image processes with two numerical
threads each. Local engineering fallback uses one image process with two
threads on the 8-GB Mac, preserving the desktop's memory. Local elapsed times
must not be compared as speedups over the earlier Chiamaka cycle. The new C++
count kernel is checked against exhaustive categorical outcomes, analytic
gradient tests and exact one-versus-two-thread parity. Controlled kernel
profiling is separate from end-to-end elapsed-time measurements.

Only three of the required twenty cycles are currently reviewed and selected.
The fourth cycle is an implementation until full execution and actual review
are complete; no improvement is asserted by this proposal document.

## Engineering pilot and numerical repair

The first local pilot (source f748a64) completed all 88 method/image outputs,
with all twenty unchanged-survivor native masks exactly matching cycle three.
All four overview sheets were actually inspected and journalled. All twenty
hypotheses remain in the full-cycle roster; no candidate is selected from
these four scores. G03 visibly removes large internal adipose/lumen-like
cavities in CAM01 that the reference includes in green. This is a scored
deletion risk, not proof of expert glass segmentation.

G06's initial 160-step projected solver retained stationarity residuals up
to 0.007355. Its objective and observation model remain unchanged, but its
numerical implementation now uses feasible damped Newton iterations with
checked C++ quadratic simplex subproblems. An independent constrained
optimization test was added. The first pilot stays preserved; a fresh source
snapshot and pilot must verify the replacement and exact non-G06 parity.
The revised focused suite has 122 passing tests.

A separate synthetic count-kernel profile includes wrapper allocation and
seven repetitions. At 64 bags, two C++ threads are slightly slower than one
(median 0.000567 versus 0.000493 s): threading has overhead. At 2048 bags,
medians are 0.01842 s for one thread and 0.01015 s for two, a 1.814x kernel
ratio. NumPy's independent convolution/gradient implementation takes 1.0273 s
for the larger workload. Numerical errors are below 5e-15 for that workload,
and the two C++ thread outputs are exactly equal. These synthetic ratios are
not end-to-end Bora speedups; CPU affinity and machine exclusivity were not
controlled. The script and both complete timing records are retained.

## Failed local full attempt and return to Chiamaka

The local full attempt stopped before the all-prediction freeze: 21 completed
case records, seven exact-parent parity failures, and a worker SIGSEGV in the
system Accelerate BLAS `DOUBLE_matmul` path. No full-cycle scores or reviews
were produced. Its folder, log, crash report and terminal-failure manifest
remain preserved. The four-case pilot did not expose either failure, which
shows why its scope must not be overstated.

An isolated OpenBLAS build avoids the observed BCSS06 matrix crash in targeted
checks. Matching the saved Chiamaka NumPy2.2.6/SciPy1.15.3 versions still leaves
small native-mask differences on several TIGER parents; disabling OpenCV
optimizations also leaves those differences. Their precise cross-platform
cause is not established. Exact mask checks were not relaxed or bypassed.

Chiamaka is reachable again. Subsequent full execution will use its existing
runtime and the same unchanged-parent checks, after the new code passes its
tests and engineering pilot there. Protocols now record package versions,
NumPy BLAS build, loaded numerical-library hashes and OpenCV build identity,
so identical version labels cannot conceal different numerical runtimes.
This is engineering provenance, not another algorithmic intervention.

The user now requests a viewer at the end of every completed cycle. Selection
updates only the gallery's report status and writes VIEWER_READY.json; masks,
scores and actually reviewed contact sheets stay unchanged. Open each viewer
for inspection while continuing the full twenty-cycle objective.

The user's later display preference narrows the delivered viewer to the five
selected algorithms. `selected.html` shows those five only, with all 149 cases,
original H&E, reference annotation, input tile mask, fixed converted RGB,
class/resolution metadata and adjustable semitransparent overlays. It is a
report built from the signed selection; all-candidate evidence remains for
the required internal review. Every later cycle's selection creates this view.

## Chiamaka validation and full execution

The immutable 03c77ad snapshot passed 132 focused tests on Chiamaka. Its expanded
eight-case pilot includes TIGER07/09/14 and BCSS06 from the local failures,
plus the original four-dataset cases. All 176 predictions and display rasters
passed integrity checks; all forty unchanged-parent native masks match cycle
three exactly. All eight all-method sheets were actually inspected and saved
with concrete observations. G06 reached its forty-iteration budget on TIGER07
with projected-gradient residual 2.0494e-6; seven other pilot cases met its
stopping tolerance. This status remains visible rather than implying exact
optimization. No candidates were removed using pilot scores.

The full 149-case run has started on Chiamaka in a fresh `cycle-04` folder.
Its local mirror is named `cycle-04-chiamaka` to preserve the failed Mac full
attempt. The executed model snapshot remains unchanged. Posthoc auditing has
a separate source snapshot and explicitly selects the host's profile files.

Chiamaka's synthetic count-kernel medians at 2048 bags are 0.01856 s with one
C++ thread and 0.01073 s with two (1.730x); the two thread outputs are exact.
At 64 bags the ratio is 1.011x, effectively no useful threading gain. These
kernel measurements include wrapper allocation, not whole-mask execution;
CPU affinity and machine exclusivity were not controlled. The historical Mac
profiles remain separately labelled.

## Historical execution checkpoint before review completion

The full Chiamaka run is terminal with all 3278 predictions, scores and all
149 all-method comparison sheets. Both Chiamaka and the local mirror passed
the integrity audit: 6854 frozen files, 3278 native masks and exact display
rasters, and 745 unchanged-parent comparisons. All 176 pilot/full native
comparisons are exact. The small-region atlas and cached colour-proposal
checks are complete. These checks do not replace actual visual review.

No new child exceeds F01's mean image macro IoU of 0.679529. The closest new
method, G01, reaches 0.674400 and wins 47 images, loses 101 and ties one against
F01. It increases mean small-region recall from 0.189664 to 0.201831 but takes
2.106 rather than 1.914 seconds/image in this run. These single-run elapsed
times include preparation and native projection; they are not controlled CPU
scaling measurements. The full numerical ranking, parent comparisons and
execution hashes are in CYCLE_FOUR_EXECUTION.json. No survivors have yet been
chosen for this cycle.

The review checkpoint has 24 actual all-method overviews and two new native
crops. BCSS18 shows G03 opening a real bright cleft left zero by the reference;
BCSS07 shows it leaving a broad pale cleft filled. Earlier pilot crops, linked
to identical full-run masks, also show deletion of positively annotated fat.
G03 adds 5726334 unassigned positive-reference pixels across mixed native
spacings; the report also sums physical area using each case's spacing. Such
deletions are scored, while reference-zero clefts remain qualitative evidence.

Numerical budgets remain explicit: G06 has 43 unconverged cases (largest
projected-gradient residual 0.000366965), G08 has fifteen unsuccessful optimizer
flags, and unchanged C01 has 67. G02's 1515 linear-program fits and G15's 505
class fits report success, but this does not imply better tissue masks or
arbitrarily small gradients. G05 meets its mass tolerance on all149 cases yet
has lower accuracy; satisfying a mathematical constraint does not validate
its assumed biological class totals. All149 cases remain development data.

## Completed review and selection

All149 all-method contact sheets have now actually been viewed and journalled,
plus four full-run native crops. The signed selection retains F01, C17, C01,
D08 and C15 unchanged: none of the fifteen children improves the primary
accuracy leader. [CYCLE_FOUR_REVIEW.md](CYCLE_FOUR_REVIEW.md) records the
accuracy/time/small-region tradeoffs, native evidence and remaining failures.
`CYCLE_FOUR_SELECTION.json` gives all twenty decisions. Four of twenty cycles
are complete; cycle five and the final five-conversion evaluation remain.
