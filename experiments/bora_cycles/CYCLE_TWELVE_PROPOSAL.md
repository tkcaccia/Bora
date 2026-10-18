# Cycle12: prospective mechanisms and engineering checks

Eleven of20 cycles are complete. This proposal follows the signed cycle11
selection P11,N08,M04,J01,C17. No cycle12 result or selection is claimed by
this document. These are experimental alternatives, not claims of literature
novelty, calibrated probabilities, or improved biological segmentation.

The469 calibrated development ROIs, their coarse masks, five RGB conversions
and equal09+19 primary conversion remain unchanged. At least two observed
classes and the existing real-tile/physical-field limits remain required.
All native reference-positive pixels count, including absent-coarse classes
and tissue removed by a support hypothesis. Reference0 remains unscored.
P11,N08,M04,J01,C17 must reproduce their cycle11 native predictions exactly.
B21/B22/B23 remain diagnostic controls and cannot survive selection.

Candidate timing includes Bora preparation and native projection but excludes
cached Virchow2 encoding, RGB conversion, loading and rendering. The primary
utility remains mean_i[IoU_i-0.05*max(seconds_i-1,0)]. Use three nice12 workers,
two numeric/OpenMP threads each. New code and result rasters go on Chiamaka's
NVMe because the SATA SSD has less than20 GiB free; no datasets are deleted.

The three native support models keep the J01 class evidence used by P11.
Only Q01 can remove original nonwhite pixels, through its explicitly
registered affine blank model. Gray glass is not silently ignored in scoring:
removed annotated nonwhite pixels are errors. Q02/Q03 preserve all original
nonwhite pixels. Other twelve children use common native support.

## Fifteen independently labelled interventions

### Q01 — Illumination-calibrated glass (parent P11)

Estimate an affine RGB blank plane from bright neutral border samples by Huber regression. Replace the fixed white predicate by a three-channel standardized blank residual (squared distance<=16, channel range<=24); retain exactly the parent Beta-binomial occupancy decision on this stain definition.

Difference: Change only the photometric observation of blank glass, allowing neutral gray glass to be removed; do not add a class-dependent background model or alter class scores.

Failure to inspect: The border may contain tissue rather than glass, and a blank illumination plane can misclassify pale tissue. An insufficient border sample triggers the explicitly reported original photometric predicate.

### Q02 — Local stain enclosure depth (parent P11)

On a4 micrometre grid compute the stained fraction in each of eight oriented half-disks of radius64 micrometres. Keep every nonwhite pixel and white pixels whose minimum oriented half-disk stained fraction exceeds0.025; lift the scalar depth bilinearly to native pixels.

Difference: Replace isotropic occupancy by a local enclosure statistic that requires stain on every side, unlike ray access, a global hull or an alpha triangle radius.

Failure to inspect: Open adipose edges lack surrounding stain and may be cut; scattered debris can falsely enclose glass. Eight orientations only approximate continuous halfspace depth.

### Q03 — Texture-versus-blank evidence (parent P11)

Estimate blank noise variance from adjacent neutral border RGB differences on a4 micrometre grid. In each64 micrometre square compare an isotropic blank covariance against a rank-one texture covariance (strength plus two directions: three additional parameters, with fixed blank noise). Keep white where n/2*(r-1-log(r)) >1.5log(n), r=max(1,lambda_max/noise), and retain all nonwhite.

Difference: Decide tissue extent by covariance model evidence rather than stained pixel density or geometric enclosure; this does not claim an expert-calibrated tissue posterior.

Failure to inspect: Spatial correlation makes the nominal patch sample size optimistic. Scanner texture and compression can look like tissue; unstained fat may have insufficient texture.

### Q04 — Wasserstein scale transport (parent N08)

Keep the parent bending means and residual variance targets. Transport their positive standard deviations, using a screened first-order M-matrix and a unit prior, then square the lifted field for normalized diagonal Gaussian likelihood.

Difference: Replace log-variance geometry by one-dimensional Gaussian Wasserstein coordinates, not a variance strength or probe-count change.

Failure to inspect: A different variance geometry can overbroaden classes. The variance transport uses a positive first-order operator because the bending inverse does not obey a maximum principle.

### Q05 — Asymmetric residual appearance (parent N08)

Keep parent bending means. Fit separate positive/negative residual second-moment targets for each whitened channel, transport their logarithms with the same bending solve, and use the normalized two-piece normal score -sum(log(s_plus+s_minus)+delta²/(2*s_sign²)).

Difference: Learn asymmetric class likelihoods around the same appearance mean, rather than transporting a symmetric covariance or changing a tail-loss constant.

Failure to inspect: Small coarse classes may poorly estimate one residual side; a two-piece normal is a model of synthetic RGB rather than a biological measurement.

### Q06 — Conditional residual innovations (parent N08)

Keep parent bending mean and log-variance fields. Standardize residuals, estimate each class/channel four-neighbour autoregression rho in[-0.8,0.8] by original coarse weighted least squares with0.015 ridge, and score normalized Gaussian innovations z-rho*neighbour_mean(z), including fitted innovation variance and the original variance determinant.

Difference: Use spatial dependence of fitted class residuals, rather than appending raw local covariance features to a global discriminant or smoothing final labels.

Failure to inspect: Conditional scores are a pseudolikelihood, not a normalized joint image distribution. Coarse contamination and wrong neighbours can reward an incorrect class.

### Q07 — Credible-ball appearance risk (parent M04)

Keep exactly the parent whitened means and positive sampling variance. Choose the class minimizing (norm(x-mu)+sqrt(7.814727903*v))², the worst distance over its95 percent three-dimensional Gaussian prototype credible ball.

Difference: Replace predictive density by a worst-credible-set decision, not expected squared risk, Gaussian hinge training or joint winner sampling.

Failure to inspect: A conservative uncertainty penalty can remove a correctly rare class. The sampling variance is not expert calibrated and the nominal95 percent level is only a model statement.

### Q08 — Tolerance-ball prototype probability (parent M04)

Keep parent means and sampling variance. Score the analytic probability that a Gaussian prototype lies within a unit-radius ball of the query in whitened RGB; use parent predictive density only to break finite-precision ties.

Difference: Integrate a finite acceptance region around the query rather than evaluate a point density, compare distribution affinity, or estimate which prototype wins jointly.

Failure to inspect: A unit whitened tolerance is an engineering decision. Tiny prototype variance can saturate probabilities and make tie decisions important.

### Q09 — Residual sandwich uncertainty (parent M04)

Keep the parent H&E-conducted means. Propagate16 fixed Rademacher probes multiplied by per-node, per-channel fitted residuals through each screened field system. Their positive outer-product average, scaled0.015, is a full3x3 sampling covariance; score N(mu,I+V).

Difference: Replace homoscedastic scalar observation noise by empirical residual influence covariance, distinct from transporting the class appearance covariance itself.

Failure to inspect: Residuals aggregated at field nodes underestimate within-node variation. The16-probe sandwich is an approximation and may be costly for many classes.

### Q10 — Stain-contour advection (parent J01)

Replace the symmetric parent transport operator by its sum with an upwind directed graph generated by the rotated spatial gradient of H&E optical density. Solve the resulting screened M-matrix for each class; report equation residuals and constant reproduction.

Difference: Introduce nonsymmetric contour-directed transport, rather than an orientation tensor in a symmetric diffusion energy or a new regularization magnitude.

Failure to inspect: Contour circulation can transport appearance along an irrelevant microscopic stain contour. Directed transport is not an anatomical flow measurement.

### Q11 — Shared skew colour field (parent J01)

Fit one shared three-component field omega(s) under the observation x=mu_g+omega(s) cross (mu_g-barycentre), with the parent H&E Laplacian and0.015 zero anchor. Solve one joint sparse quadratic normal system; compare queries against these locally coupled prototypes.

Difference: Tie local class changes through one skew-symmetric sensor law, rather than independent fields, shared additive drift, scalar gain, or a global Procrustes calibration.

Failure to inspect: A skew linear sensor field cannot express all local changes and large rotations are only a first-order approximation. Degenerate class colours leave some directions prior-determined.

### Q12 — Fuzzy tissue connectivity (parent J01)

On a4 micrometre grid take parent class probabilities as gray ceilings. Seed each class at original coarse interior observations whose continuous weight exceeds0.8. Compute the exact maximum over paths of the minimum ceiling/seed value, separately per class, using C++ priority queues; normalize and lift these connectivity evidences.

Difference: Replace pointwise density decisions by class-specific widest-path connectivity through probability ceilings, unlike a glass-seeded forest or tree Potts energy.

Failure to inspect: A small true island without a coarse seed can disappear; a weak bridge can still connect a false region. This is a seeded connectivity model, not discovery of an absent class.

### Q13 — Tile-integral plurality projection (parent C17)

Keep parent palette probabilities. On a4 micrometre quadrature grid minimize mean KL(q||p)+sum_tiles sum_rivals max(mean(q_rival-q_observed),0)². Solve the nonnegative smooth dual by L-BFGS-B, at most100 iterations; lift each tile logit offset without freezing any pixel and report projected dual residual.

Difference: Keep the palette fixed and impose soft tile-integral plurality evidence on the fractions, rather than learn endmember colours, impose hard pixel margins or match global class area.

Failure to inspect: An incorrect coarse winner can be reinforced and quadrature only approximates full-pixel tile integrals. The bounded optimizer may fail stationarity or exceed one second.

### Q14 — Censored channel inverse (parent C17)

At occupied colour nodes minimize the fixed palette squared error for interior channels and sigma² times negative Gaussian log-tail probability for RGB channels within one byte of0/255, with sigma0.05 and the parent0.006 prior ridge. Use projected gradient on the simplex, at most160 steps, and report projected stationarity.

Difference: Model clipping as a censored observation with one-sided probability, distinct from quantization intervals, heavy-tailed residuals or compositional channel ratios.

Failure to inspect: The converted RGB may not arise from a censored Gaussian sensor. A finite optimizer can miss tolerance, and changing the observation model can spoil a useful exact linear mixture.

### Q15 — Bayesian contiguous regions (parent C17)

On a4 micrometre grid sum log parent palette probabilities per cell. A region has log evidence logsumexp(class log scores)-log(K). Repeatedly merge adjacent regions with the largest positive log Bayes factor for a shared class, update neighbours, and label by its pooled posterior. C++ implements the deterministic greedy graph operation.

Difference: Make contiguous decisions by integrated class evidence, unlike stain-only region partitioning, tree Potts penalties, a fixed quadtree or spatial fraction TV.

Failure to inspect: Correlated probability cells overstate evidence and a greedy merge is not a globally optimal partition. Narrow channels can merge away and weak evidence can promote excessive pooling.

## Validation and review gates

The C++ widest-path implementation is compared with all-path max-min algebra;
the region queue with exhaustive greedy graph search on random small grids.
Probability calculations are compared with independent distribution functions.
Sparse transport and skew systems are checked against independent dense
normal equations. Both convex objectives have finite-difference gradient
checks, with an original-primal/dual gap for tile plurality. New support
contracts require prediction-stage tests with deliberately unreadable fine
reference files. All methods require deterministic finite categorical outputs,
unchanged inputs, physical geometry and explicit solver residuals.

After these checks, run a score-blindly specified engineering pilot covering
all ten datasets, the nine requested focus cases, and the documented glass /
architecture failures. Review every pilot overview and native crops before
launching the full469-case run. A pilot cannot complete a cycle or select five.
Predictions and supports must freeze before any fine-reference scoring starts.

Small-reference-region scores and rare-RGB region hypotheses remain separate.
These children do not assign an expert biological identity to an appearance
outlier, and they cannot recover a semantic class absent from the coarse IDs.
Actual native crops must distinguish exterior glass, adipose and internal
lumens rather than treat every white pixel as glass. Report every candidate's
accuracy, time, small-region misses, residuals, benefits and failures. Select
five only after all469 case overviews have actually been inspected; publish
only their selected viewer for the user. Cycles13–20 and the final all-five-RGB
comparison remain outstanding after this proposal.

## Engineering revision after the first pilot

The immutable be54524 pilot finished all24 cases, passed552 native-mask checks,
120 parent-native comparisons and120 parent-score comparisons. Every overview
and eight targeted native label crops were actually inspected. No five were
selected and no full run was released from that pilot.

Native PUMA11 exposed Q12's zero-ceiling fallback: a4µm support cell can miss a
thin real tissue fragment. Normalizing its zero class vector gave a uniform
tie and an artificial lowest-ID class fringe. A failing regression test
reproduced this independently. The corrected implementation restores the
parent's nonzero class evidence in unseeded cells before fine-grid lifting;
zero graph ceilings still block propagation through excluded cells. This is
an implementation repair of the declared parent fallback, not a new hypothesis.

Q15's C++ neighbour sets now use hash tables. The Bayes-factor objective,
priority ordering and merge law are unchanged; a port is not an extra model.
Q13/Q15 now retain parent palette-fit metadata under an explicit field rather
than overwrite it with the lineage parent ID. A fresh pilot must check output
parity for all22 unchanged methods, all native supports, Q12's repaired crop,
and the measured speed of Q15 before the469-case run can start. The first
pilot remains preserved as engineering evidence, including its errors.
