# Cycle eight: expanded development benchmark

Seven cycles are signed. Cycle eight keeps J01, K05, C17, J07 and K13 exactly
and adds fifteen models in `eight_manifest.py` and `cycle_eight.py`. These are
implemented hypotheses, not established improvements. Every model sees the
same frozen09+19 RGB and original coarse labels. Aligned H&E is used only by
the explicitly morphology-dependent models. There are no fixed label anchors
or forbidden boundary margins.

| Parent | New models | Principal limitation |
| --- | --- | --- |
| J01 | L01 nonlocal H&E appearance graph; L02 predictive variance from the screened system; L03 wide exterior-white routes | Similar stain can join unrelated tissue; uncertainty is approximate; border fat can resemble glass |
| K05 | L04 box-constrained bending; L05 radial Huber observations; L06 spatial first/second moments | Iterative solves can be slow; minority appearances may be downweighted; covariance can be weakly identified |
| C17 | L07 convex Hellinger observation; L08 radial Student-type errors; L09 minimum-entropy mixture energy | Dark channels can dominate; heavy-tailed and entropy objectives have only local nonconvex solutions |
| J07 | L10 joint maximum-rival hinge; L11 per-tile Gaussian uncertainty; L12 noise-integrated kernel mean distance | Weak winner labels and correlated samples still limit identifiability |
| K13 | L13 upper-quarter tile CVaR; L14 row/column cluster noise; L15 average per-tile log risk | Incorrect coarse tiles can dominate a protected tail; limited blocks do not identify the full correlation law |

The manifest records each equation, numerical budget, parent difference and
failure hypothesis. L03 only removes labels from J01 and never swaps its
nonzero tissue identities. Its white core must have clearance of3 coordinate
units along an entire route to the exterior; a3-unit rim is restored only
inside bright H&E. On calibrated images these are micrometres; on80 unknown-
resolution additions they are explicitly pixels. The latter is exploratory
pixel-scale processing, not an assumed microscopy calibration.

The349 inputs are132 BCSS,80 TIGER,40 WSSS4LUAD,40 RINGS tumor,40 OCELOT,
10 BEETLE and7 CAMELYON16 windows. All are ROIs. Reference0 stays unscored,
and all positive references count, including deleted tumor and coarse-absent
classes. For RINGS only, removed background0 is interpreted as its valid
outside-tumor class during scoring; this class is not expert stroma. The saved
mask remains unchanged. Single-eligible-class RGBs cannot supply absent class
contrasts and are reported separately.

The primary utility remains mean_i[macro_IoU_i-0.05*max(seconds_i-1,0)].
Native mask projection and candidate background processing are timed. Shared
Virchow inference, I/O, scoring and rendering are separate. Three nice12 image
workers each use at most two numerical/OpenMP threads; desktop and other jobs
are left alone. Reused parents must match all149 historical native masks
exactly; the200 new cases have no previous predictions to compare.

Engineering pilot: eight conversion-pilot additions plus WSSS10, RINGS40 and
six historical stress cases (TIGER32/23, BCSS45/53/18/87). Pilot and full outputs
use separate folders. No full cycle is complete until all349 all-method
contact sheets and useful native crops are actually inspected, all20 methods
have explicit tradeoffs, five are signed, and their selected-only viewer is
opened. Repeated selection makes this development, not independent validation.

Preflight amendment, before any pilot mask or fine-reference result inspection:
the original L09 penalty -0.004 sum(q²) was dominated by the0.012 quadratic
Hessian floor and was therefore still convex. Its nonconvex description was
incorrect. L09 now uses concave smoothed Shannon entropy, with an explicit
tangent-majorization inequality test. The first engineering pilot remains
archived; its implementation is not a separate candidate or completed cycle.

The second pilot passed its352-mask integrity audit and all16 overviews were
actually inspected. WSSS04/10 exposed a distinct input-identifiability issue:
their constant RGB produced contradictory uniform classes and tiny islands
from numerical differences in class appearance fits. No such gain is accepted.
`expanded_prediction.py` now applies the same frozen completed-coarse fallback
to every candidate/control for the30 explicitly degenerate inputs. An explicit
H&E glass rule may still remove white pixels. This is an input contract, not
another model or a change to the RGB conversion. All319 nonconstant cases and
every historical parent parity requirement retain their original algorithms.
The second pilot is archived and superseded before full launch; revised pilot
verification is still required. Single-class IoU1 on ignored background or
RINGS valid negatives is not a claim of fine-boundary recovery.
