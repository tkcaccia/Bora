# Cycle 2: changes derived from all 149 reviewed ROIs

Cycle 1 is fully reviewed and selected. Its five survivors are **C18, C17,
C01, C03 and C15**. Every one lies on the measured accuracy/time/small-region
recall Pareto frontier. C18 is the accuracy leader. C17 is a faster, closely
related palette inverse; these two are explicitly not independent biological
models. C01 supplies the fastest weak-label model, C03 local appearance, and
C15 higher small-component recall at the cost of false regions and lower IoU.
Fifteen rejected candidates and their reasons remain in the signed selection.

All 149 images remain development data. No new result in this repeatedly
selected study constitutes independent validation. The five RGB conversions
are unchanged; primary selection still uses the equal 9+19 conversion.

## Evidence motivating this cycle

Native review shows that exact palette inversion can remove a false third
class between two real classes, but cannot recover nuclei or glands erased
from the converted RGB. C11 sometimes improves broad regions, yet often
rounds or merges genuine small structures. C15 recovers more small reference
components but creates false chromatic rims. White spaces are ambiguous:
C20 misses enclosed tissue clefts while deleting annotated adipose in
TIGER28. Sparse BEETLE references can score radically different full-field
masks almost identically. These limitations must remain visible in selection.

## Fifteen children, three per survivor

| ID | Parent | Single intervention | Main risk |
|---|---|---|---|
| D01 | C18 | Explain each colour with at most two palette components; enumerate exact edge optima. | Remove genuine three-class junctions. |
| D02 | C18 | Condition the palette inverse on original H&E optical densities as well as fixed RGB. | Introduce stain-sensitive microscopic artefacts. |
| D03 | C18 | Replace the quadratic prior with relative entropy, using a convex simplex model. | Reinforce an overconfident coarse colour prior. |
| D04 | C17 | Use Huber RGB residuals to bound the influence of aberrant conversion channels. | Discard a meaningful channel contrast. |
| D05 | C17 | Regularize only class mixtures invisible to the augmented RGB palette. | Amplify noise in nearly identifiable contrasts. |
| D06 | C17 | Remove brightness as a nuisance and invert directional RGB geometry. | Merge classes distinguished by intensity. |
| D07 | C01 | Fit a joint Poisson count-order probability for the observed tile winner. | Count independence may be wrong; summation adds cost. |
| D08 | C01 | Estimate nuisance covariance within bags and class contrast between bag means. | Treat true intratile tissue variation as nuisance. |
| D09 | C01 | Select class-covered training bags by greedy information-volume gain. | Overrepresent unusual extreme tiles. |
| D10 | C03 | Fit local affine spatial appearance rather than local constant means. | Extrapolate poorly where local observations are sparse. |
| D11 | C03 | Estimate one shared quadratic spatial colour drift across classes. | Absorb a real tissue transition into the drift. |
| D12 | C03 | Compete tissue-specific white-pixel likelihoods with an explicit glass state. | Confuse genuine lumina and glass under mixed coarse labels. |
| D13 | C15 | Model channel dependencies with a nonparametric Chow–Liu density tree. | Miss three-way interactions or create histogram ridges. |
| D14 | C15 | Estimate joint class density with hierarchical Beta predictive colour partitions. | Introduce axis-aligned colour artefacts. |
| D15 | C15 | Estimate joint density in two statistically discriminating colour directions. | Lose a rare class separated by the omitted direction. |

Equations, individual failure hypotheses and differences from each parent
are machine-readable in `registry.py:CHILDREN_TWO`, and copied into the frozen
run protocol. These are new objectives, statistical assumptions or a training
design algorithm. C++ exports and numerical ports are implementation details,
not additional candidates. None restores a user-rejected B strategy.

D01 deliberately omits the additive coarse correction after its constrained
inverse: that correction could reintroduce a third active fraction. D02 solves
per pixel because an RGB-only cache would merge distinct original H&E
observations. D12 learns white fractions only from original H&E under known
coarse tissue; annotation-0 is not treated as glass. It may assign background
zero, and any annotated tissue it removes remains an error in native scoring.

## Execution and checks

The five survivors execute their original prediction implementations. Every
survivor's native output is compared with the previous frozen prediction;
a mismatch fails the method. New C++ simplex and relative-entropy kernels are
checked against independent SciPy constrained optimizers. Poisson winner
probabilities must partition count events, and their derivatives are checked
by finite differences. Glass tests distinguish white-rich tissue from a
white cleft without using reference masks.

Cycle 2 cannot start without all 149 actual cycle-1 visual reviews and the
signed selection. The runner freezes all predictions before scoring. The
compiled scorer is used after full cycle-1 Python/C++ parity on all 3278
masks. Candidate times include their own preparation and native projection;
parity verification, scoring, rendering and disk compression are excluded.

Run three low-priority processes, two numerical/OpenMP threads each. An
engineering pilot checks one ROI from each of the four dataset groups,
then the same frozen implementations run on all 149 ROIs. Every case needs
actual visual review before the next five survivors are selected.

This is cycle 2 of the required 20. A pilot, a successful inference job or a
numerical test suite is not completion of the full requested study.
