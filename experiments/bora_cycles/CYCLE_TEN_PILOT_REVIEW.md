# Cycle 10 engineering pilot: observed behavior

Thirteen fixed cases and 22 methods were predicted, frozen, scored and rendered.
All 13 all-method overview sheets and 3 native label crops were actually viewed.
The 286 native/display rasters, 65 unchanged-parent masks,65 unchanged-parent
accuracy records and 44 constant-input contracts pass their checks.
The source is 2f25e21; no implementation was changed after this pilot.
This small engineering sample does not select survivors or finish cycle 10.
The unchanged implementation is running all 349 frozen development ROIs.

| Child | Parent | Mean IoU | IoU change from parent | Mean seconds | Images >1s | Main observed limitation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| N01 | J01 | 0.776222 | -0.001938 | 0.371 | 1 | Integral tile averages; loses within-tile detail and broadens some strips. |
| N02 | J01 | 0.766639 | -0.011522 | 0.248 | 0 | One shared multiplicative field; visibly underfits several class interfaces. |
| N03 | J01 | 0.776303 | -0.001858 | 0.269 | 1 | Exact tree optimum on coarse grid; visible block steps, isolated slivers. |
| N04 | M04 | 0.778468 | +0.000192 | 0.267 | 0 | Tile-correlated positive variance; small changes, no claim of calibrated uncertainty. |
| N05 | M04 | 0.778756 | +0.000481 | 0.452 | 1 | Shared-noise class contrast; preserves TIGER divider but has composite-score assumptions. |
| N06 | M04 | 0.778174 | -0.000101 | 0.264 | 0 | Residual degrees-of-freedom variance; slight changes with a finite trace estimate. |
| N07 | K05 | 0.773777 | -0.003978 | 0.937 | 5 | Sparse curvature; all 11 informative pilot cases fail the stationarity criterion at 60 steps. |
| N08 | K05 | 0.777474 | -0.000281 | 0.532 | 1 | Positive log-variance; expands some green pockets, with extra runtime. |
| N09 | K05 | 0.777751 | -0.000004 | 0.875 | 3 | H&E-guided interpolation; still merges thin strands and exceeds one second on 3 cases. |
| N10 | M11 | 0.778160 | +0.001053 | 0.241 | 0 | Convex-envelope white rule; little visible change and cannot restore shared lost support. |
| N11 | M11 | 0.776263 | -0.000844 | 0.250 | 0 | Rooted barrier forest; removes real fat interiors as well as some real glass. |
| N12 | M11 | 0.776716 | -0.000391 | 0.242 | 0 | Two-route exterior connectivity; less aggressive than N11, still deletes fat spaces. |
| N13 | C17 | 0.768550 | -0.012398 | 0.098 | 0 | Nearest codeword; very fast, false grey TIGER fields and false red RINGS islands. |
| N14 | C17 | 0.774817 | -0.006131 | 0.264 | 0 | Conditional channel model; some cleaner BCSS strips, false grey fringes; 2 stationarity flags. |
| N15 | C17 | 0.774211 | -0.006738 | 0.244 | 0 | Geometric palette model; fast but false grey fringes and binary false red islands. |

The proposed mathematics and fixed numerical limits are in
[CYCLE_TEN_PROPOSAL.md](CYCLE_TEN_PROPOSAL.md). Actual notes and numerical
evidence are in [CYCLE_TEN_PILOT_REVIEW.json](CYCLE_TEN_PILOT_REVIEW.json).

N07 is a finite-budget approximation, not a converged optimum. Its independent
small-problem mathematical checks pass, which does not establish full-image
convergence. N14 also remains a finite solve; residuals are retained rather
than replaced by a success claim.

Native TIGER23 shows that nearest-colour decoding creates grey areas where
green/violet tissue is annotated. Native RINGS01 shows false red islands in
a valid-negative window, suppressed by the screened-field methods. Native
BCSS45 contains unannotated fat: H&E exposes the difference between cell walls
and the gold/white lace output, but absent fine reference cannot establish
an expert-correct segmentation there.

Constant WSSS04 and RINGS03 provide no spatial converted-RGB information.
Their common coarse fallback and high scores do not demonstrate architectural
recovery. RINGS negative denotes outside tumour, not expert stroma.

The primary utility remains mean_i[IoU_i-0.05 max(seconds_i-1,0)],
with no speed bonus below 1 second. Full 349 results, actual visual inspection
and 20 written tradeoffs are required before five survivors are signed.
Encoder preparation is separate from Bora runtime; these are development data.

Each pilot method has one timed invocation per image. This is operational
runtime evidence, not a controlled repeated hardware-speed experiment or a
statistical claim of improvement. The full run uses the same timing protocol.
