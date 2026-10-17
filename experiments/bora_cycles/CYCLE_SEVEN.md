# Cycle seven: prospective interventions

Six of twenty cycles are complete. This cycle retains the signed J01, H02,
C17, J07 and H10 implementations unchanged, and adds fifteen distinct models.
All 149 frozen development ROIs, original coarse labels, class IDs, physical
spacing and equal #9+#19 RGB conversion remain fixed. No encoder rerun, fine
reference supervision, voting between methods or post-hoc class invention is
permitted. All unannotated tissue is analyzed; reference zero remains unscored.
Deleted positive reference pixels and missed small components remain errors.

Selection uses mean per-image [IoU -0.05 max(seconds -1,0)]. Raw IoU, time,
small-region recall and the penalty remain separate. One second refers to the
frozen ROI workload, not full-WSI processing. All cases are reused development
data. No numerical test or pilot alone completes the cycle.

## Fifteen independent hypotheses

| Child | Parent | Change | Main risk |
| --- | --- | --- | --- |
| K01 | J01 | Orientation-guided appearance | Cell orientations can be irrelevant to tissue boundaries and can propagate the wrong colour along a bundle. |
| K02 | J01 | Covariance-coupled vector field | A pooled noise covariance may over-smooth a rare discriminating colour direction. |
| K03 | J01 | Log-normal appearance field | Near-zero synthetic channels become disproportionately influential; converted colours are not physical stain intensities. |
| K04 | H02 | Fused appearance field | Piecewise-constant appearance can introduce false plateaux or suppress gradual real changes. |
| K05 | H02 | Bending appearance field | Curvature regularity can overshoot across genuine abrupt appearance changes and increase sparse-solve cost. |
| K06 | H02 | Positive quasi-Poisson field | Synthetic RGB channels are not counts; the mean-variance relation can be wrong and Newton solves may exceed the time budget. |
| K07 | C17 | Worst-channel palette inverse | Protecting the worst channel can let small errors in several reliable channels accumulate; finite gaps may remain. |
| K08 | C17 | Heteroscedastic palette inverse | Real tissue mixtures can be mistaken for noise and tile-specific metrics can produce grid artifacts. |
| K09 | C17 | Colour-cloud harmonic coordinates | Colour-cloud bottlenecks can connect unrelated tissue classes; coarse-prior screening can dominate far from palette vertices. |
| K10 | J07 | Worst-case colour margin | Protecting against adversarial colours can underfit small but genuine class differences. |
| K11 | J07 | Quadratic Gaussian Brier risk | Quadratic scores can extrapolate implausibly and Brier regression need not match a tile plurality event. |
| K12 | J07 | Joint Gaussian logistic risk | Finite cubature can miss tail behaviour and probabilistic tile-mean labels can still hide rare pixel classes. |
| K13 | H10 | Empirical tile-noise risk | The sampled residual law may not represent correlated tile-mean noise and outliers can dominate exponential risk. |
| K14 | H10 | Worst-class exponential risk | A severely contaminated or scarcely observed class can dictate every decision and hurt the average. |
| K15 | H10 | Shared low-rank contrasts | A rare class may require the suppressed third contrast direction; finite proximal iterations can remain unconverged. |

## Children of J01

### K01: Orientation-guided appearance

Use a symmetric eight-neighbour physical graph weighted by the H&E structure tensor: borrow along tissue orientation more freely than across it; fit the same screened appearance fields.

Replace scalar edge contrast by directional transport geometry, not a different scalar bandwidth.

Cell orientations can be irrelevant to tissue boundaries and can propagate the wrong colour along a bundle.

### K02: Covariance-coupled vector field

Fit vector appearance with Mahalanobis data/prior error but isotropic RGB graph energy; diagonalize the covariance and solve one sparse normal system per colour eigenmode.

The parent solves three identically regularized scalar fields; this model couples their regularity through measured covariance.

A pooled noise covariance may over-smooth a rare discriminating colour direction.

### K03: Log-normal appearance field

Fit H&E-conducted appearance fields to log((255 RGB+1)/256), estimate the pooled log-colour covariance, and classify with conditional log-normal likelihood.

Change the observation geometry and likelihood from additive colour error to multiplicative log-colour error; fixed image bytes remain unchanged.

Near-zero synthetic channels become disproportionately influential; converted colours are not physical stain intensities.

## Children of H02

### K04: Fused appearance field

Minimize weighted squared data error plus a smooth vector total-variation graph penalty and the global-mean anchor; use finite majorizing quadratic solves and report the original objective/residual.

Change the field prior from squared differences to bounded-influence vector variation, without smoothing output labels.

Piecewise-constant appearance can introduce false plateaux or suppress gradual real changes.

### K05: Bending appearance field

Replace the Laplace Dirichlet energy by squared physical-Laplacian energy, solving L-transpose L plus observed mass and the same global anchor.

Penalize bending rather than first differences; the model supports gentle ramps instead of forcing locally flat prototypes.

Curvature regularity can overshoot across genuine abrupt appearance changes and increase sparse-solve cost.

### K06: Positive quasi-Poisson field

Fit positive RGB means by Poisson quasi-deviance with spatial quadratic regularity and a positive global-mean anchor; use positivity-preserving damped Newton and classify by the same deviance.

Change the observation model to a convex positive-intensity field, not a Gaussian prototype or another label propagation stage.

Synthetic RGB channels are not counts; the mean-variance relation can be wrong and Newton solves may exceed the time budget.

## Children of C17

### K07: Worst-channel palette inverse

Minimize 0.025 times maximum absolute channel reconstruction error plus 0.006 squared distance to the original colour prior on the simplex; report a primal-dual gap from a C++ independent-row solver.

Replace summed squared channel error by a minimax observation model with an explicit dual certificate, not a changed iteration count.

Protecting the worst channel can let small errors in several reliable channels accumulate; finite gaps may remain.

### K08: Heteroscedastic palette inverse

Estimate each input tile covariance from all its converted colours, shrink and normalize its inverse to trace three, then solve the palette simplex quadratic in that local Mahalanobis geometry.

Use spatially varying multichannel observation uncertainty in the forward inverse; no class mean, palette vertex or RGB image is recalibrated.

Real tissue mixtures can be mistaken for noise and tile-specific metrics can produce grid artifacts.

### K09: Colour-cloud harmonic coordinates

Join occupied colour cells and canonical palette vertices in a symmetric six-neighbour colour graph; pin vertices to their semantic unit vectors and solve screened harmonic coordinates with the original colour prior.

Replace per-colour mixture reconstruction by one graph boundary-value interpolation on the observed distribution, not fraction smoothing under the old quadratic.

Colour-cloud bottlenecks can connect unrelated tissue classes; coarse-prior screening can dominate far from palette vertices.

## Children of J07

### K10: Worst-case colour margin

Fit squared hinge risk for the worst mean-colour perturbation in a whitened radius-0.2 ball: margin is 1+delta dot x+0.2 norm(delta_colour), plus the original ridge.

Replace average Gaussian uncertainty by a support-set robust optimization with a closed-form worst perturbation.

Protecting against adversarial colours can underfit small but genuine class differences.

### K11: Quadratic Gaussian Brier risk

Predict one-hot class scores with all degree-at-most-two whitened colour monomials; analytically integrate their Gaussian Brier risk using moments through degree four, then solve the class-balanced ridge normal equations.

Replace margin fitting by a closed-form Gaussian polynomial risk; cross-channel interactions and uncertainty enter one analytic normal system.

Quadratic scores can extrapolate implausibly and Brier regression need not match a tile plurality event.

### K12: Joint Gaussian logistic risk

Integrate multiclass log-softmax cross entropy over shared three-dimensional Gaussian tile-mean errors using 27 fixed Gauss-Hermite points, with the original ridge and finite optimizer budget.

Replace the sum of independent rival hinges by one normalized joint multiclass observation likelihood; all rivals share each sampled colour perturbation.

Finite cubature can miss tail behaviour and probabilistic tile-mean labels can still hide rare pixel classes.

## Children of H10

### K13: Empirical tile-noise risk

Replace the Gaussian moment-generating function by the empirical mean of exponentials over the 25 within-tile centred residuals divided by five; retain class-balanced all-rival risk and ridge.

Use each tile empirical noise distribution, including non-Gaussian shape and local covariance, instead of assuming one pooled normal law.

The sampled residual law may not represent correlated tile-mean noise and outliers can dominate exponential risk.

### K14: Worst-class exponential risk

Minimize the maximum class-conditional log Gaussian exponential risk plus coefficient ridge, using a single convex epigraph problem with analytic constraint Jacobians.

Change the training population objective from average class risk to worst-class protection; no reference-derived class scores enter fitting.

A severely contaminated or scarcely observed class can dictate every decision and hurt the average.

### K15: Shared low-rank contrasts

Add 0.02 times the nuclear norm of the class-by-three colour coefficient matrix to the original Gaussian exponential risk; use singular-value proximal steps with monotone backtracking and report the mapping residual.

Learn a shared low-dimensional contrast geometry through a convex matrix prior rather than separately shrinking coefficients or selecting RGB channels.

A rare class may require the suppressed third contrast direction; finite proximal iterations can remain unconverged.

## Fixed numerical scope

K01 uses a positive symmetric eight-neighbour orientation graph. Its directional
weights are not claimed to be an exact finite-element tensor-PDE discretization.
K02 solves the full vector normal equations by covariance eigendecomposition;
an independent dense Kronecker solve checks its equivalence. K03 uses log colour
with an offset of one digital count; its class-independent Jacobian cancels in
the class decision. No RGB source file is modified.

K04 uses a vector variation epsilon0.01, weight0.02 and at most12 majorization
solves. Its original objective and stationarity residual are reported. K05
uses the squared 16-micrometre physical Laplacian. K06 uses at most15 damped
Newton steps, an anchor0.015 and strictly positive means. Its Poisson form is
a quasi-likelihood for synthetic RGB, not a claim of actual photon counts.

K07 uses beta0.025 and ridge0.012, at most600 primal-dual iterations and gap
tolerance1e-7. A C++ OpenMP kernel handles independent colour nodes. Gaps and
unconverged nodes remain explicit. K08 uses each tile's full converted-colour
covariance, 0.8 full plus0.2 diagonal shrinkage and a digital-count variance
floor; inverse precision is normalized to trace3. K09 uses the existing24-bin
occupied colour nodes, symmetric six-neighbour affinities and screening0.02.
Palette vertices have fixed semantic unit-vector Dirichlet data. Unknown
colour novelty is separately reported with unknown biological identity.

K10 uses a whitened uncertainty radius0.2, consistent with tile-mean variance
1/25. Its norm is smoothed by1e-12 solely at zero. K11 integrates Gaussian
moments through degree4 for ten quadratic features and solves a10-by-10
ridge normal system. K12 uses27 shared three-dimensional Gauss-Hermite nodes.
K10/K12 have60 L-BFGS iterations and ridge0.0025; unsuccessful terminations
are retained as finite-budget fits, not asserted optima.

K13 uses all25 sampled residuals divided by5. Optimization uses the logarithm
of risk plus ridge, preserving its minimizer but not claiming the transformed
objective convex. K14 minimizes a single epigraph of class log risks with at
most80 SLSQP iterations; constraint violation is explicit, and an objective
gradient alone is not a KKT certificate. K15 uses nuclear penalty0.02, at most
100 monotone proximal steps, singular-value shrinkage and a mapping residual.
Ports and CPU profiles are engineering checks, not additional innovations.

## Execution and review gates

Run independent analytic-gradient, moment/cubature, constrained-optimization,
field-objective and one/two-thread checks. Freeze the passing source. Run an
eight-case engineering pilot covering four dataset groups and known failure
examples. Audit exact unchanged-parent masks and actually inspect every pilot
overview plus targeted native crops. Then run all149 with three low-priority
image processes and two numerical/OpenMP threads each. Freeze every prediction
before fine-reference scoring. Perform native audits and missed-region analysis,
actually inspect all149 all-method contact sheets, then sign five survivors
with benefit/rejection reasons for every candidate. Deliver only selected.html.
Do not design cycle-eight children before that signed selection.

## Engineering pilot and full launch

All 241 focused checks pass locally and on Chiamaka. The frozen launcher ran
205 checks and omitted the 36 J-series checks; those 36 passed separately on
the exact snapshot before the full run. The repository wrapper is corrected.
No frozen model source or prediction was changed. The executing model commit
is 1ad219254115963a2b9efd2396d942732fd89bfb.

All 176 pilot masks succeed. Both audits pass 176 exact display rasters and
40 unchanged-parent comparisons. All eight all-method overview sheets and
three native label sheets were actually inspected, with hash-pinned notes.
Native BCSS45 shows K01 reintroducing H02's false fat stripe in a vessel.
K02 largely preserves J01's correction; C17/K07/K08/K09 preserve the vessel
class more consistently. Native BCSS26 exposes fat and border-class errors.
Every displayed method still truncates the native TIGER23 stromal wedge.
Shared white-support errors remain; this cycle has no new glass-exclusion rule.

K02 has a small pilot gain, not an established full-benchmark improvement.
K04 exceeds one second on three pilot images and receives the frozen penalty.
All K04 field fits are finite but unconverged at the specified budget. K07
flags unfinished gaps on seven cases, K10 unsuccessful termination on six,
K13 on three, and K15 its proximal residual on all eight. These are evaluated
finite-budget masks, not asserted fitted optima.

Serial large Chiamaka Chebyshev-kernel medians are 1.58275 seconds on one thread
and 0.799729 on two, ratio 1.979, with exact outputs. 29,564/32,768 synthetic
nodes miss the 1e-7 gap tolerance at 600 iterations; maximum gap is 1.393e-5.
These measurements are kernel-only, not whole-image speed claims.

The same frozen model completed all149 ROIs with zero failed methods. Both
main PID1090822 and posthoc PID1093871 are terminal/missing. Both native audits
confirm3278 exact displayed masks and745 unchanged-parent comparisons. All176
pilot/full mask arrays also agree exactly, and596 reused diagnostic files have
verified hashes. No restart is needed.

All149 full-run all-method overviews and eight native label sheets have now
been actually inspected with hash-pinned observations. The signed five are
J01,K05,C17,J07,K13. No child exceeds J01's overall accuracy. K05 improves H02
by0.10329 IoU percentage points with94 wins/53 losses/2 ties, while K13 improves
H10 by0.09265 points with73 wins/64 losses/12 ties. Both stay below one second
on every image, but neither is a uniform anatomical winner.

Native BCSS45/40 distinguish blood-vessel from similar-coloured fat errors.
Native BCSS53 shows J01 deleting a tumour band retained by K05. Every model
truncates the TIGER23 stromal wedge and colours the bright BCSS07 cleft.
The input audit finds56 cases with a fine class absent from coarse and4,079,674
positive reference pixels outside support. Primary scores include positive
reference pixels and are not normalized to an optimistic cap.

Seven of twenty cycles are complete. See CYCLE_SEVEN_REVIEW.md and the signed
CYCLE_SEVEN_SELECTION.json for every benefit, limitation and rejection reason.
All200 requested/additional input sets are prepared and independently audited,
for349 future ROIs. Expanded feature inference is now complete and its600 arrays
pass the independent audit. Unchanged conversions, task/scaling integration and
review gates remain to be implemented before an expanded cycle.
Historical149-image cycle-seven predictions and scores remain unchanged.
