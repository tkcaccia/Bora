"""Cycle-one hypotheses; implemented status is explicit, never inferred."""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Candidate:
    title: str
    mechanism: str
    hypothesis: str
    risk: str
    implemented: bool = False
    parent: str | None = None


CYCLE_ONE = {
    "C01": Candidate("Tile plurality likelihood", "count-observation-likelihood",
        "Fit class logits from a normal approximation to the probability of the observed tile winner.",
        "Spatially correlated pixels violate independent count approximations.", True),
    "C02": Candidate("Cross-fitted supervision trust", "held-out-observation-reweighting",
        "Refit appearance using original coarse pixels weighted by spatially held-out predictive support.",
        "Rare correct appearances may be downweighted as if erroneous.", True),
    "C03": Candidate("Locally shrinking class appearance", "nonstationary-empirical-bayes",
        "Let local class means vary while shrinking sparse observations to global means.",
        "Local coarse errors may be reinforced.", True),
    "C04": Candidate("Pairwise class decisions", "pairwise-logistic-coupling",
        "Learn each class contrast separately and reconcile log pairwise win probabilities.",
        "Inconsistent pairwise decisions can hurt multiclass calibration.", True),
    "C05": Candidate("Colour co-occurrence", "second-order-local-appearance",
        "Classify RGB together with local covariance and lag-correlation features.",
        "Texture scales may obscure small components.", True),
    "C06": Candidate("Sparse patch reconstruction", "class-dictionary-reconstruction-residual",
        "Classify by the L1-penalized reconstruction energy in a class-specific orthogonal patch dictionary.",
        "Weakly labelled mixed patches may produce poor dictionaries.", True),
    "C07": Candidate("Low-rank structure and sparse residual", "patch-low-rank-sparse-decomposition",
        "Project probability patches onto a low-rank basis and retain large residual exceptions.",
        "Small true components may be mistaken for noise; this is a one-step approximation, not converged robust PCA.", True),
    "C08": Candidate("Wavelet probability field", "haar-domain-sparse-regularization",
        "Shrink small Haar detail coefficients instead of spatially blurring the mask.",
        "Grid-phase sensitivity and removal of real fine structure.", True),
    "C09": Candidate("Minimum-description spatial partition", "global-quadtree-dynamic-programming",
        "Choose homogeneous spatial cells by global fit-versus-complexity dynamic programming.",
        "Axis-aligned cells can introduce geometric artefacts.", True),
    "C10": Candidate("Semi-Markov tissue runs", "duration-aware-sequence-inference",
        "Use dynamic programming over labelled runs along image axes.",
        "Directional bias or loss of thin components.", True),
    "C11": Candidate("Continuous spline probability field", "adaptive-continuous-spatial-basis",
        "Fit class evidence using a compact cubic spline field with coarse-only supervision.",
        "Insufficient basis detail or excessive spatial bias.", True),
    "C12": Candidate("Higher-order class motifs", "three-class-context-likelihood",
        "Learn soft triad likelihoods from known coarse class configurations.",
        "Coarse tile geometry may contaminate context statistics.", True),
    "C13": Candidate("Expected macro-IoU decision", "posterior-expected-set-utility",
        "Choose a coupled hard assignment using posterior expected intersection/union.",
        "Miscalibrated probabilities can overexpand rare classes.", True),
    "C14": Candidate("Interval palette evidence", "feasible-polytope-decision-bounds",
        "Use feasible class-probability bounds to expose ambiguous RGB interpretations.",
        "Bounds may be too loose to change decisions, while adding cost.", True),
    "C15": Candidate("Colour copula likelihood", "rank-marginal-dependence-density",
        "Estimate non-Gaussian class marginals separately from their rank dependence.",
        "Sparse class observations can destabilize tail estimates.", True),
    "C16": Candidate("Training-tile influence", "delete-group-decision-uncertainty",
        "Penalize predictions whose class score is sensitive to deleting individual training tiles.",
        "Rare but valid regions may depend on a few tiles.", True),
    "C17": Candidate("Converged palette active set", "simplex-quadratic-active-set",
        "Solve the palette quadratic to checked KKT tolerance instead of a fixed iteration count.",
        "More accurate optimization need not improve biological segmentation.", True),
    "C18": Candidate("Adaptive colour precision", "certified-colour-partition-refinement",
        "Refine colour cells only where a decision-error bound cannot certify their winner.",
        "Conservative bounds can erase the speed advantage.", True),
    "C19": Candidate("Adaptive spatial resolution", "certified-spatial-subdivision",
        "Use fine sampling wherever within-cell RGB variation can change a class decision.",
        "Subdivision overhead can exceed dense linear classification.", True),
    "C20": Candidate("Exterior white-space geometry", "border-connected-glass-exclusion",
        "Remove border-connected bright low-saturation white space while preserving enclosed lumina.",
        "Border-connected adipose or lumen can be removed incorrectly.", True)
}


CYCLE_TWO_SURVIVORS = ("C18", "C17", "C01", "C03", "C15")

# These are model interventions, not a parameter schedule. The manifest is
# frozen with executed source before any cycle-two reference mask is opened.
CHILDREN_TWO = [
    dict(id="D01", parent="C18", title="Two-component palette explanation",
         intervention="cardinality-constrained-colour-inverse",
         equation_or_rule="Minimize palette reconstruction error plus the original quadratic prior on every simplex edge; choose the globally best pair, so at most two class fractions are nonzero.",
         failure_hypothesis="A third genuine tissue at a junction may be suppressed together with false third-class colour bands.",
         difference_from_parent="Replace the full simplex by its union of edges; enumerate exact one-dimensional optima instead of changing a regularization coefficient."),
    dict(id="D02", parent="C18", title="H&E-conditioned palette inference",
         intervention="joint-original-he-conditional-prior",
         equation_or_rule="Use a coarse-trained shared-covariance model of fixed RGB and aligned original H&E optical densities as the prior in the palette quadratic; solve each pixel without RGB-only quantization.",
         failure_hypothesis="Original stain variation or mixed coarse supervision may introduce false microscopic structures.",
         difference_from_parent="Condition the inverse on independently available original morphology, which the fixed RGB may have erased; no representation file is altered."),
    dict(id="D03", parent="C18", title="Relative-entropy palette inverse",
         intervention="relative-entropy-simplex-geometry",
         equation_or_rule="Minimize half squared palette residual plus 0.012 KL(q || coarse-trained colour prior), subject to the probability simplex, with feasible damped Newton iterations.",
         failure_hypothesis="An overconfident coarse colour prior can resist a valid tissue change even with accurate numerical optimization.",
         difference_from_parent="Change the probabilistic prior from Euclidean distance to relative entropy; the objective is a different convex model, not an iteration-count sweep."),
    dict(id="D04", parent="C17", title="Robust palette residual",
         intervention="huber-observation-error-model",
         equation_or_rule="Minimize the sum of channelwise Huber residuals with transition 0.04 plus the original quadratic prior; solve the iteratively reweighted simplex quadratics.",
         failure_hypothesis="A real class-specific colour coordinate may be mistaken for an aberrant reconstruction channel.",
         difference_from_parent="Replace Gaussian RGB observation error with bounded-influence error; retain an explicit single convex objective."),
    dict(id="D05", parent="C17", title="Regularize only invisible mixtures",
         intervention="palette-nullspace-only-prior",
         equation_or_rule="Minimize half squared palette residual plus 0.006 times squared projection of q-prior onto the nullspace of the augmented palette; apply a numerical 1e-9 ridge only for stability.",
         failure_hypothesis="RGB noise can make identifiable mixture fractions unstable when the palette is nearly dependent.",
         difference_from_parent="Apply supervision only in colour-unidentifiable directions; observable class contrasts no longer shrink toward the coarse prior."),
    dict(id="D06", parent="C17", title="Directional palette geometry",
         intervention="brightness-nuisance-directional-inverse",
         equation_or_rule="Normalize observed RGB and every palette vector to unit Euclidean length, then solve the original simplex quadratic in directional colour geometry.",
         failure_hypothesis="Classes separated mainly by brightness, especially grey debris versus other tissue, may become indistinguishable.",
         difference_from_parent="Treat brightness as a nuisance variable instead of a tissue contrast; this changes the observation geometry without editing the fixed RGB image."),
    dict(id="D07", parent="C01", title="Poisson tile-winner probability",
         intervention="poisson-order-statistic-supervision",
         equation_or_rule="Approximate each tile's class counts by independent Poisson variables with means equal to summed pixel probabilities; maximize the exact count-order event probability, with deterministic class-index tie breaking.",
         failure_hypothesis="Poisson independence and weak tile labels may remain inaccurate; summation is more expensive than the parent's normal approximation.",
         difference_from_parent="Replace pairwise Gaussian count comparisons with a joint discrete order-statistic event; evaluate its analytic gradient."),
    dict(id="D08", parent="C01", title="Within-tile nuisance covariance",
         intervention="bag-centred-discriminant-contrast",
         equation_or_rule="Estimate class centroids from equally weighted known tile means and nuisance covariance from deviations around each tile mean; use the resulting shared-covariance pixel discriminant.",
         failure_hypothesis="True tissue mixtures inside a coarse tile may be downweighted as nuisance variation.",
         difference_from_parent="Change the unit of statistical supervision to tile contrasts and separate within-tile variation from between-class evidence; no pixel pseudo-label EM is used."),
    dict(id="D09", parent="C01", title="Information-volume tile design",
         intervention="determinantal-training-bag-selection",
         equation_or_rule="Keep the original normal tile-winner likelihood, but greedily select class-covered tile observations by log-determinant gain of standardized mean-colour design vectors, up to 12 bags per known class.",
         failure_hypothesis="Geometrically extreme tiles may overrepresent noise or omit common mixed appearances.",
         difference_from_parent="Use an explicit experimental-design optimization to reduce redundant training bags; this is a selection algorithm, not a smaller arbitrary sample count."),
    dict(id="D10", parent="C03", title="Local affine class appearance",
         intervention="local-polynomial-appearance-regression",
         equation_or_rule="For each class and pixel, fit a spatially affine RGB prototype by compact-window weighted normal equations and global-prototype ridge; classify by shared-covariance residual.",
         failure_hypothesis="Sparse local class observations can make an affine extrapolation less reliable than a constant local mean.",
         difference_from_parent="Estimate local appearance slopes as well as intercepts; box sufficient statistics replace Gaussian constant-mean smoothing."),
    dict(id="D11", parent="C03", title="Shared spatial colour nuisance",
         intervention="class-shared-polynomial-drift-model",
         equation_or_rule="Fit one quadratic spatial RGB drift to residuals from known class means, subtract this shared field, then fit and evaluate a single class discriminant.",
         failure_hypothesis="A real large-scale tissue transition may be absorbed into the nuisance field.",
         difference_from_parent="Share one low-dimensional drift across all classes instead of estimating an independent local mean field for every class."),
    dict(id="D12", parent="C03", title="Competing glass observation model",
         intervention="class-conditional-white-glass-competition",
         equation_or_rule="Learn each known tissue class's white-pixel rate from original H&E and coarse observations. Combine that Bernoulli likelihood with local class evidence and an explicit glass state of prior 0.08; assign zero only when glass wins.",
         failure_hypothesis="Coarse mixtures can underestimate a genuine lumen's white rate; glass and adipocyte interiors remain locally photometrically ambiguous.",
         difference_from_parent="Add a separately identified glass state and learned tissue-specific white likelihood, rather than removing all bright or border-connected pixels."),
    dict(id="D13", parent="C15", title="Colour dependency tree",
         intervention="chow-liu-class-density-factorization",
         equation_or_rule="Estimate smoothed one- and two-channel histograms per class, choose the maximum-mutual-information spanning tree of RGB channels, and evaluate its normalized joint density.",
         failure_hypothesis="A tree cannot represent a genuine three-way colour interaction; sparse pair histograms may create artificial ridges.",
         difference_from_parent="Replace Gaussian rank dependence with a nonparametric, explicitly normalized dependency-tree distribution."),
    dict(id="D14", parent="C15", title="Hierarchical colour probability tree",
         intervention="polya-tree-predictive-density",
         equation_or_rule="Partition all tissue RGB recursively by median splits of the widest colour coordinate; multiply class-conditional Beta(1,1) predictive branch probabilities to obtain leaf likelihoods.",
         failure_hypothesis="Axis-aligned leaf boundaries can become false class interfaces and depth limits may erase rare colours.",
         difference_from_parent="Replace separately fitted marginal/copula densities with one hierarchical joint predictive distribution learned on a shared unsupervised colour partition."),
    dict(id="D15", parent="C15", title="Density in discriminating colour plane",
         intervention="supervised-information-plane-density",
         equation_or_rule="Solve the between-class versus within-class generalized eigenproblem, project RGB onto its top two contrast directions, and classify using joint two-dimensional empirical class densities.",
         failure_hypothesis="A rare class requiring the discarded third direction may be merged with another tissue.",
         difference_from_parent="Select a low-dimensional class-discriminating sufficient contrast before joint density estimation; no independent channel-copula approximation remains."),
]

CYCLE_TWO = {key: CYCLE_ONE[key] for key in CYCLE_TWO_SURVIVORS}
CYCLE_TWO.update({item["id"]: Candidate(item["title"], item["intervention"],
    item["equation_or_rule"], item["failure_hypothesis"], True, item["parent"])
    for item in CHILDREN_TWO})


def candidates(cycle=1):
    if cycle not in (1, 2):
        raise ValueError("This cycle has no implemented, reviewed lineage")
    return CYCLE_ONE if cycle == 1 else CYCLE_TWO


def roster(cycle=1):
    return {key: asdict(value) for key, value in candidates(cycle).items()}
