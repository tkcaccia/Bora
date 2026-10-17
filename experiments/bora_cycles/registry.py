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


def roster():
    return {key: asdict(value) for key, value in CYCLE_ONE.items()}
