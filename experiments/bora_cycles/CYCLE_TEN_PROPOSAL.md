# Cycle 10 prospective implementation plan

Nine of twenty cycles are signed. The actual cycle-nine selection (SHA256
7a75cc4a19016b718c067eaf22cdd4c99e69fd374afd5bf258e9d049f29f1704)
retains J01,M04,K05,M11,C17. Only after that selection were N01–N15 implemented.
Each parent remains unchanged; every child has one stated model intervention.

The frozen349-image scope, all input bytes and equal09+19 RGB remain unchanged.
The original149 and additional200 are separate reported strata. All tissue,
including unannotated tissue, is analyzed. Fine references enter only after the
complete prediction freeze. Missing positive classes and deleted labelled
pixels remain errors. RINGS raw0 is valid outside-tumour, not expert stroma;
its established scoring adapter is unchanged.80 uncalibrated WSSS/RINGS cases
remain pixel-only; numerical coordinate units are not claimed micrometres.
30 constant-RGB inputs retain the shared no-refinement coarse-completion rule.

Primary utility remains mean_i[macroIoU_i-0.05 max(seconds_i-1,0)]. Raw accuracy,
mean seconds, excess seconds and slow-image counts remain separate. There is
no subsecond bonus and no controlled cross-cycle speedup claim. Maximum three
nice12 image workers and two numerical threads each. Encoder cost is separate.

N03 uses a four-coordinate-unit categorical tree grid, so its exact optimum is
for that tree/grid, not the full pixel image. N07 runs at most60 ADMM updates;
finite residuals and cap flags remain visible. N14 runs at most180 projected
updates. N08 positive log-variance exponentiation clips only numerical log
values outside[-30,30] and records every clipping. These numerical changes
are not proof of improved anatomical accuracy.

All glass children keep J01 class decisions, changing only additional white
exclusion. The common frozen working/native tissue support still clips every
method, so none can restore tissue already excluded by that input support.
Their removals on valid positive reference pixels still count as errors.
N13's categorical codewords, N14's conditional channel proportions and N15's
geometric mixture are competing hypotheses about artificial converted RGB,
not established physical sensor laws.

The pilot must pass independent mathematical/graph/lineage checks, prediction
and parent parity audits, and actual visual review. Full computation alone
cannot finish cycle10: all349 all-method overviews require explicit actual
inspection, followed by native crops and twenty written selection decisions.
Only a signed selection permits cycle11. At completion show only selected.html
with five survivors and all cases.

## Fifteen interventions

### N01 — Integral tile observations (parent J01)

Replace pointwise class-node observations by one weighted RGB average per original coarse tile and class. B averages field nodes with original pixel weights; solve (L+B.T W B+0.015I)u=B.T W tileRGB+0.015 globalMean. W preserves total parent observation mass.

Difference: Treat an original tile as one integral observation rather than expand its class into independent node observations or impose a tile-winner likelihood.

Failure hypothesis: Tile averages discard within-tile architectural detail and the nonlocal normal equations can extrapolate appearance incorrectly.

### N02 — Shared multiplicative sensor (parent J01)

Fit one spatial scalar gain a from RGB=a(s) globalClassMean using the original coarse observations, H&E conductance and a unit-gain quadratic anchor. Solve one positive screened system; score all classes using a(s) times their global mean and the original shared covariance.

Difference: Share a multiplicative appearance field across classes, unlike J02 shared additive drift or H05 independently profiled per-pixel gain and offset.

Failure hypothesis: A common scalar sensor cannot describe class-specific or chromatic drift and may underfit real tissue appearance.

### N03 — Exact anatomical tree competition (parent J01)

On a four-coordinate-unit spatial grid aggregate parent negative-log scores as area-weighted unaries. Build an H&E optical-density minimum spanning tree and minimize unary plus six times edge length times exp(-ODcontrastSquared/0.02) Potts cost by exact tree dynamic programming. Project categorical cells by nearest neighbour.

Difference: Optimize a global discrete tissue partition on an anatomical tree, not a prototype prior, quadtree partition or directional sequence.

Failure hypothesis: A spanning tree drops image cycles and the coarse grid can produce block edges or join unrelated tissue through one retained edge.

### N04 — Coherent tile noise (parent M04)

Keep the mean fit; assign a shared Rademacher perturbation to all field nodes in the same original tile. Propagate sixteen block-noise probes through each class system and use their mean squared solutions as positive predictive variance.

Difference: Change the observation covariance to original-tile blocks rather than independently perturb field nodes or clip an inverse diagonal.

Failure hypothesis: A tile-wide systematic error model may overstate correlation and broaden weakly supported classes.

### N05 — Common-noise class contrasts (parent M04)

Use common field-node noise probes across class fits, estimate positive marginal variances and pairwise covariances, and compare class squared-distance scores with a first-order Gaussian contrast law plus query noise. Sum log pairwise win probabilities and normalize as composite scores.

Difference: Cancel shared appearance noise in class contrasts rather than classify by independent marginal predictive densities or expected squared distances.

Failure hypothesis: Linearization neglects quadratic prototype noise and composite pairwise scores need not be calibrated multiclass probabilities.

### N06 — Residual degrees-of-freedom noise (parent M04)

Keep the class mean system, estimate residual degrees of freedom n-2 tr(S)+tr(S squared), S=sqrt(D) A inverse sqrt(D), with sixteen probes. Estimate class noise from whitened node-mean residuals with six pseudo-degrees at parent variance0.015, then propagate that variance through the mean estimator.

Difference: Estimate observation noise using the smoother residual law instead of fixing its scale, changing the predictive distribution, or merely increasing probe count.

Failure hypothesis: Weak class means contain mixture bias and finite trace estimates can distort degrees of freedom; this is not expert-label calibration.

### N07 — Sparse curvature fields (parent K05)

Replace squared curvature by sum of Euclidean norms of L u in whitened RGB. Minimize weighted squared observations plus0.015 global anchoring and this convex sparse-curvature penalty using at most sixty ADMM steps with rho1; report primal,dual and stationarity residuals.

Difference: Sparsify second-order curvature, unlike K04 first-gradient total variation or L05 robust observation residuals.

Failure hypothesis: Finite ADMM may be slow or nonstationary, while sparse curvature can create artificial piecewise-affine appearance transitions.

### N08 — Positive log-variance transport (parent K05)

Retain the bending mean; compute nonnegative coarse-weighted residual second moments in whitened RGB and shrink them0.8 toward data plus0.2 unit variance. Fit three log-variance bending fields and exponentiate their interpolants; use the resulting normalized diagonal Gaussian density.

Difference: Transport log variances with guaranteed positivity, rather than interpolate raw second moments and clip negative covariance eigenvalues as L06 did.

Failure hypothesis: Coarse mixtures inflate local variance and diagonal whitened covariance can miss rotating class-specific scatter.

### N09 — Morphology-guided prolongation (parent K05)

Keep the parent bending solve but replace bilinear mean-field lifting by normalized four-corner bilinear weights multiplied by0.02+exp(-H&E optical-density difference squared/0.02). Score the unchanged RGB against those lifted means.

Difference: Change the coarse-to-fine appearance interpolation operator, leaving both converted RGB and field-fitting mathematics unchanged.

Failure hypothesis: Nuclear edges may select an inappropriate neighbouring prototype and preserve noise rather than tissue boundaries.

### N10 — Convex tissue envelope (parent M11)

Keep J01 tissue labels. Construct the convex hull of all non-white H&E pixels and remove only white pixels outside it; enclosed white areas are preserved. Use row extrema to build the same hull efficiently.

Difference: Infer a global geometric tissue envelope instead of tracing eight directional rays, connected white paths or local physical openings.

Failure hypothesis: Deep true exterior clefts inside the convex hull remain filled and isolated stained specks can enlarge the envelope.

### N11 — Seeded barrier watershed (parent M11)

Keep J01 tissue labels. On a four-coordinate-unit H&E grid seed white border pixels as glass and strongly stained pixels as tissue. Sort four-neighbour OD-contrast edges and construct the exact marker-constrained Kruskal minimum spanning forest; remove only original white pixels in glass trees.

Difference: Use marker-constrained single-link barrier competition rather than additive route cost, harmonic access or unary-plus-perimeter mincut.

Failure hypothesis: Minimax contrast barriers and deterministic ties may assign a bright internal lumen to exterior glass; coarse sampling remains.

### N12 — Two independent escape routes (parent M11)

Keep J01 tissue labels. On a four-coordinate-unit white grid join every border-white node to an exterior root. Tarjan biconnected decomposition identifies white nodes sharing a nontrivial block with that root, equivalent to two internally vertex-disjoint escape routes. Also remove white border nodes; gate the lifted result by original white pixels.

Difference: Use a topological redundancy condition on exterior access, not one straight ray, a minimum route width or a shortest path threshold.

Failure hypothesis: A genuine exterior cul-de-sac can be preserved while a broadly open fat cavity still has two escape routes.

### N13 — Categorical palette codebook (parent C17)

Assume each pixel emits one supplied palette codeword and choose the exact least squared Euclidean RGB residual among eligible class codewords. Return its one-hot class score without mixture fitting or uncertain coarse correction.

Difference: Decode one categorical codeword rather than recover fractions on a simplex or use learned appearance prototypes.

Failure hypothesis: Boundary RGB can be a genuine mixture and nearest codeword can systematically favor a visually intermediate class.

### N14 — Conditional channel proportions (parent C17)

Normalize positive RGB channel sums and palette rows. Fit brightness-weighted mixture r by convex channel cross-entropy plus0.006 squared distance to the brightness-weighted coarse prior; recover tissue fractions q proportional to r divided by palette row brightness. Use at most180 projected-gradient steps and report residuals.

Difference: Use a conditional multinomial observation law and explicitly undo brightness-weighted mixing, unlike Poisson intensities or L2 directional color geometry.

Failure hypothesis: Constructed RGB bytes are not measured photon counts and brightness conditioning can remove real class information.

### N15 — Multiplicative palette mixture (parent C17)

Transform occupied-node RGB averages and eligible palette rows by minus log((value+1/255)/(1+1/255)); solve the original quadratic simplex inverse in this optical-density geometry with the same coarse prior and confidence correction.

Difference: Change the palette forward observation from arithmetic RGB mixing to a geometric-channel mixture, not a log-normal class-appearance distribution.

Failure hypothesis: Converted RGB is artificial; multiplicative color mixing need not describe its forward construction, and node averaging remains approximate.
