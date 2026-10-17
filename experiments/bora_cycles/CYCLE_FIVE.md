# Cycle five: fifteen distinct hypotheses under a one-second runtime penalty

The signed cycle-four survivors are F01, C17, C01, D08 and C15. They remain
unchanged in this roster, with three new children each. All149 frozen ROIs,
the original coarse inputs and the equal #9+#19 conversion remain unchanged.
No Virchow2 inference or RGB conversion is repeated.

The new primary selection utility is the mean over all149 images of
`macro_IoU - 0.05 * max(candidate_seconds - 1, 0)`. Raw IoU is still reported.
The penalty is per image before averaging; time includes each candidate's
preparation, inference, glass policy and native projection. A slow raw-IoU
leader is no longer guaranteed a survivor slot. This policy is frozen in
protocol.json before any cycle-five predictions or reference scoring.

## Implementations and falsifiable assumptions

| Child | Parent | New mathematical observation/model | Principal failure risk |
| --- | --- | --- | --- |
| H01 | F01 | Two latent Gaussian appearances per class, with spatial mixture weights | Weak labels create false appearance states |
| H02 | F01 | One global screened-Laplace class-appearance field | Smooth fields erase abrupt local drift |
| H03 | F01 | White component state conditioned on two-class evidence in its external ring | Real interface lumina are deleted |
| H04 | C17 | Joint bounded palette and latent mixture fitting on known coarse tissue | Nonconvex calibration follows coarse contamination |
| H05 | C17 | Class-specific nonnegative brightness/neutral-offset nuisance planes | Different class planes overlap |
| H06 | C17 | Smooth nonlinear cardinal coordinates anchored at palette vertices | Nonlinear interpolation misdecodes true colour mixtures |
| H07 | C01 | A latent set of at least ceil(25/K) supporting pixels per tile | Necessary plurality support is not the actual winner event |
| H08 | C01 | Explicit forward channel from latent tissue to corrupted coarse label | Palette fractions estimate the wrong noise mechanism |
| H09 | C01 | Multinomial information-volume bias correction | Finite-sample correction distorts weak labels |
| H10 | D08 | Analytically Gaussian-noise-marginalized exponential discrimination | Correlated tissue violates the mean-error model |
| H11 | D08 | Full-tissue likelihood estimation of class prevalence, with fixed semantic densities | Unlabelled appearance shift collapses rare classes |
| H12 | D08 | Whitened geometric medians of class tile centres | Genuine minority appearances are downweighted |
| H13 | C15 | Class-specific Chow-Liu dependence trees among RGB variables | Two edges miss genuine three-way dependence |
| H14 | C15 | Adaptive nearest-neighbour colour-density ratios | Duplicates and finite sampling concentrate false evidence |
| H15 | C15 | Positive, normalized continuous tensor Bernstein/Beta density mixtures | Low polynomial order erases rare modes |

The exact rules, constants, parent differences and risks are in
CHILDREN_FIVE in registry.py. These are separate models, not a voting ensemble.
No candidate is a port or a parameter sweep counted as an innovation.

H01 fits at most4096 original known pixels per class by twenty finite EM
steps, using the parent's shared noise covariance. It then estimates local
mixture proportions in a48um window, avoiding a matrix inverse at every
image pixel. The new C++ OpenMP kernel evaluates the spatially varying
mixture by streaming log-sum-exp. Its threading is engineering inside H01,
not an extra candidate. Latent states do not establish new biological classes.

H02 instead minimizes one screened quadratic energy on an8um appearance grid,
with16um gradient regularity and global-mean anchoring. Sparse equations have
recorded residuals. Only an appearance surface is interpolated: no label mask
or class posterior is diffused. H03 tests a semantic cavity hypothesis rather
than the failed inertia or clearance geometry. A bright component must have
at least32um² and a2um outer ring supporting two classes by at least0.25 mean
posterior each. Lumina at real class interfaces remain a stated deletion risk.

H04's simplex and palette updates minimize one anchored objective. Palette
fitting uses only original known coarse tissue, never semantic pseudo-labels
from unknown tissue. Fifteen alternating steps are a finite nonconvex fit,
not a global-optimality claim. H05 profiles gain/offset analytically; H06 uses
regularized constant-reproducing radial interpolation, splitting duplicate
palette vertices equally. Neither changes the input conversion image.

H07 changes the tile observation to a combinatorial supporting set; H08
instead models class-dependent corruption of the observed coarse label.
H09 has a fixed reference-class gauge and analytic information-determinant
gradients. Its small information ridge means it is Firth-type correction,
not an exact invariant Jeffreys prior. All three fit at most60 iterations and
report termination and gradients. They do not claim an exact count likelihood.

H10's Gaussian expectation gives a single convex exponential-risk objective.
H11 uses all detected tissue to estimate mixture proportions, while retaining
coarse-trained class appearance parameters unchanged. This is a label-shift
hypothesis, not prescribed coarse-area conservation or semantic self-training.
H12 keeps D08's nuisance covariance and changes only class location to a
smoothed multivariate geometric median. Missing original bag classes trigger
an explicit recorded linear fallback, as in the parent.

H13's tree is over RGB variables, not spatial regions or colour-space bins.
Its discrete density product normalizes exactly. H14 uses eight neighbours,
at most2048 weighted-quantile colour samples per class, and a one-level RGB
radius floor. All-class radius outliers are reported with unknown identity;
the primary mask still uses only supported original class IDs. H15 uses64
normalized cubic tensor Beta components, positive EM weights, a weak symmetric
Dirichlet term and a forty-step budget with likelihood history. Colour-node
representatives approximate training values in H04–H06/H11/H14/H15.

## Execution and verification

Run independent analytic-gradient, convex-objective, normalization, physical
policy and C++ parity checks first. The eight-case engineering pilot covers
all four dataset groups and previous numerical edge cases: TIGER32/07/09/14,
BCSS17/06, BEETLE01 and CAMELYON01. All forty unchanged-parent native masks
must match cycle four exactly. Pilots cannot complete a cycle or select five.

Chiamaka runs three low-priority image processes with two numerical/OpenMP
threads each. Profile the new mixture kernel separately at one and two
threads; do not turn a kernel-only ratio into an end-to-end speedup claim.
After pilot checks, freeze executed source and compiled-library hashes and
run all149 cases before opening any fine reference for scoring. All positive
reference pixels remain scored, including deleted white anatomy and classes
absent from coarse. Annotation-zero tissue is still predicted and unscored.

After execution, audit native masks and unchanged parents, measure missed
small regions and inspect every all-method overview plus targeted native
crops. Only then select five and open the selected-five viewer. Four of twenty
cycles are complete at this proposal stage; no cycle-five gain is asserted.

The frozen ef9222b implementation passed172 local and172 Chiamaka checks.
All176 pilot predictions succeeded. The first read-only audit identified an
auditor compatibility error: it recomputed raw-IoU aggregation while the
pilot correctly saved the frozen runtime-adjusted aggregation. The auditor
now recomputes using the policy inside each result's protocol, with a new
tamper-rejection regression test. No model, prediction or score is changed.
The corrected auditor runs from a separate posthoc source snapshot; the
executed model snapshot remains immutable for the full-run parity check.

## Full execution and selection

All 149 ROIs have 3,278 successful native predictions, zero failed methods and
149 contact sheets. Chiamaka and local integrity audits pass, including 745
exact unchanged-parent comparisons. All 176 pilot/full native masks match.
The separate posthoc job passed nine checks and completed the small-region
atlas. All five saved artifact hashes match locally.

H02 leads the new utility: IoU 0.671889 at 0.180560 seconds, with no image above
one second. H10 improves D08 by 0.011375 mean IoU at 0.059427 seconds versus
0.049341 seconds: 98 wins, 45 losses and six ties. F01 still leads raw IoU at
0.679529, but 121 over-limit images lower its utility to 0.645593 at a mean
1.654383 seconds. These are repeatedly reused development images.

All 149 full-run all-method contact sheets were actually inspected and recorded
against their frozen hashes, plus four native label sheets. The signed five
are H02, C17, H10, H04 and C01. H02 is the runtime-adjusted leader; the slow
raw leader F01 and its H03 child are rejected. H04 ranks fourth but is slower
and less accurate than C17; its retention is not a parent-improvement claim.

BCSS07's cleft remains filled; H03 opens BCSS18's cleft with noisy edges but
deletes positive fat/lumina elsewhere. H02 mistakes blood vessel for fat in
BCSS45, whereas H10 corrects that error against D08 in BCSS26. Fat and vessel
are different mask IDs with similar gold display colours. Many TIGER gland
islands remain merged or connected, and thin tissue fingers remain lost.
Five of twenty cycles are now complete. The next fifteen models have not
been implemented by this selection. CYCLE_FIVE_REVIEW.md and the signed
CYCLE_FIVE_SELECTION.json record the decision for every candidate.

H02's largest linear residual is 2.143e-14. H10 has eight unsuccessful optimizer
terminations, H07 has six, and unchanged C01 has 67. All 505 H15 class fits hit
the finite EM budget without convergence. Scores retain the fits as executed;
finite masks do not establish exact optimization or biological correctness.
CYCLE_FIVE_EXECUTION.json records all ranks, cases, hashes and status.
