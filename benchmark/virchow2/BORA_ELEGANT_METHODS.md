# Whole-field Bora with unchanged evidence RGB

Development date: 2026-10-17. Develop on tiger_32, tiger_33, tiger_36,
tiger_37, tiger_38, bcss_17, bcss_18, bcss_19 and bcss_24. The primary conversion
is fixed equal 9+19, selected before developing the new solvers. Rank by the
arithmetic mean of nine native per-image macro IoUs, not a dataset-balanced
mean, pooled pixel accuracy, best conversion per case, or an average over RGB
conversions. Other conversions remain byte-identical and are evaluated only
after method selection.

## Main change

The original benchmark architecture transforms class evidence RGB into LAB,
learns class appearance in coarse interiors, and fixes sparse tile-center labels.
The new methods interpret the evidence directly. All tissue may change; no
fixed boundary margin or hard seed points are used. Coarse labels remain soft
supervision, not ground truth for each pixel.

B05, B09, B10 and B12 are removed from active inference implementations, in
addition to all earlier rejected solvers. B19 revision 3 contains only retained
B02/B03/B08. These three comparison solvers keep their previous numerical
parameters. Production boundary-refinement backends remain separate from the
new explicit `bora evidence` command; benchmark strategy IDs do not silently
replace existing command defaults.

## B21: shared-covariance evidence (development E09)

Converted RGB is scaled to [0,1]. Each known coarse class supplies a weighted
mean. The covariance is shared across classes:

    Sigma = 0.8 Sigma_empirical + 0.2 diag(Sigma_empirical) + 0.0025 I
    score_c(x) = x^T Sigma^-1 mu_c - 0.5 mu_c^T Sigma^-1 mu_c

Equal class priors are used. Every pixel receives the largest score. Sharing
covariance gives a linear decision rule and avoids per-pixel iterative
optimization. The covariance estimate pools the weighted training pixels; it
does not impose equal class sample counts.

Training uses original known coarse tissue only. Reliability is continuous:

    w(x,y) = 0.15 + 0.85 sin^2(pi phase_x) sin^2(pi phase_y)

where phases use the original 90-working-pixel tile coordinates. It is zero
outside original known tissue and positive everywhere inside it. There is no
excluded training band, frozen interior, or requirement that a tile center
retain its original class. Unknown completed labels never train this model.

## B22: amortized simplex evidence (development E11)

A converted pixel is a probability-weighted combination of known class colours:

    x ~= p C,  p_c >= 0,  sum_c p_c = 1

C contains the exact conversion palette for observed coarse classes. When the
augmented palette has full column rank, use its affine pseudoinverse and project
onto the probability simplex. Rounded RGB introduces small quantization error.

With more affine-dependent classes than RGB can distinguish, inversion is not
unique. Use a coarse-trained colour likelihood prior pi(x) and solve:

    minimize_p  ||p C - x||^2 + 0.012 ||p - pi(x)||^2
    subject to p on the probability simplex

The prior is a class-normalized 20-bin-per-axis colour histogram, with Gaussian
smoothing sigma 0.85 bins, a small pseudocount and equal class priors. Original
known coarse labels and the same continuous reliability weights train it.

For speed, average x and pi within occupied nodes of a 24-bin-per-axis RGB
lattice and solve the convex problem there, rather than at every pixel. Use
18 projected-gradient iterations with step 0.9/(||C||_2^2 + 0.012), then assign
the corresponding solution to each pixel. The finite colour quantization is an
approximation; development scores and masks quantify its practical effect.

Finally, weakly support the original coarse class only where posterior
separation is small:

    gap = largest_probability - second_largest_probability
    q_c proportional to p_c + 0.35 (1-gap)^2 w 1[coarse_class=c]

Normalize and take the largest class probability. This is a bounded soft
correction, not an immovable label or boundary restraint. Under-determined
inversion remains ambiguous despite regularization; reconstructed probabilities
must not be described as uniquely recovered ground-truth tissue fractions.

## Development and stopping

Sixteen candidate mechanisms were evaluated in five documented rounds. They
include robust full-covariance evidence, multimodal colour likelihoods, guided
regression, local/nonlocal harmonic graphs, tile-rank constraints, colour-simplex
inversion, confidence correction, shared covariance, Gaussian mixtures,
amortized inversion, global majority calibration, quadratic regression,
uncertainty-guided processing, noise deconvolution and information-geometric
diffusion. They are not claimed to be individually novel in the literature.

All candidate predictions in each round were frozen before fine-reference
scoring. Choices between rounds use development scores. Stop after two
consecutive rounds without a qualifying new accuracy/speed Pareto improvement
(0.001 mean-IoU accuracy tolerance; 5% speed tolerance). Rounds 4 and 5 increased
cost without meeting that accuracy threshold. This is a practical stopping
criterion, not a claim that mathematical innovation is exhausted.

Choose the fastest method within 0.002 mean IoU of the highest development
score: B22. Keep B21 as the faster alternative. There is no per-image or
per-conversion winner selection. Development scores and timings for every
candidate are in elegant-development/round1.json through round5.json; the
protocol and selection are in experiments/bora_elegant/PLAN.json and
SELECTION.json. Archived source hashes reproduce every round.

## Full evaluation

Freeze B21/B22 and the remaining four comparison solvers before predicting and
scoring all 149 ROIs with five unchanged RGB conversions: 4,470 native masks.
Report the nine development images separately from the other 140. Those 140
are additional evaluation cases, not an untouched external test: they belong
to a previously inspected corpus. The original coarse labels derive from the
same reference masks. This is controlled refinement, not independent tissue
classifier generalization.

Virchow2 already covers the entire RGB field, including unannotated tissue.
Every detected tissue pixel receives a label; the independent RGB-derived
background support is removed after segmentation. Quality is measured on the
unchanged original reference >0, never intersected with tissue detection.
Unknown tissue is predicted but unscored; falsely removed annotated tissue is
penalized. All reference classes contribute to macro IoU, including classes
absent from coarse supervision. CAMELYON non-tumor references are automatic
tissue support, not expert stroma annotations.

Native, encoder, analysis and browser sampling are recorded per axis. The new
methods analyze approximately 1 um/px and project labels back to native
coordinates with nearest-neighbor interpolation. All fields are ROIs. Pixel
spacing is not optical resolving power, and projection does not create new
microscopic detail.

CPU timing includes each method's preprocessing, inference and native label
projection. It excludes common image loading, compression, scoring and the
fixed upstream RGB/Virchow2 computation. B19 includes its component solvers.
Development timings repeat two or three times; full-corpus timings are one pass
with three low-priority processes and two numerical threads each. Speed claims
apply to the Bora stage, not the complete foundation-model pipeline.
