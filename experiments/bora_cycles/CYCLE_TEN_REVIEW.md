# Cycle 10: signed review of 349 images

Ten of twenty cycles are complete. All349 all-method overview sheets and15 planned native label sheets were actually viewed. Exported native overlays are not counted as viewed. Selected five: J01,M04,N08,N10,C17. J01 is the mandatory runtime-adjusted leader. N10 replaces M11; N08 replaces K05 as a declared accuracy/appearance-variance tradeoff. These are not the five highest utility ranks.

## Scope and integrity

The frozen349 ROIs retain all149 original inputs and add200:40 each BCSS,TIGER,WSSS4LUAD,RINGS tumour and OCELOT. Totals are132 BCSS,80 TIGER,40 WSSS,40 RINGS,40 OCELOT,10 BEETLE and7 CAMELYON16 windows.269 have verified pixel spacing;80 WSSS/RINGS remain pixel-only.30 constant converted-RGB inputs use common coarse completion, with no class-refinement gain claim. RINGS outside-tumour includes glass/lumina and is not expert stroma; deleted positive tumour still counts as error. All tissue is analysed; unannotated tissue is unscored. Sparse BEETLE annotations cannot validate complete reconstructed anatomy.

Source2f25e218a173265c15923e16eab2571bcf8fa2db completed prediction before fine-reference scoring. All7678 native masks/display rasters,1745 historical-parent native comparisons,1745 historical-parent eight-field accuracy comparisons and660 constant-contract checks pass. The13 overlapping pilot/full cases match in286 native mask and286 accuracy comparisons, with zero differences. This is scoped repeatability, not biological validation, controlled timing, or full349 rerun reproducibility. Six posthoc report digests were verified unchanged. Native labels and provenance digests are verified for all15 viewed crops.

## Accuracy and time

Utility is mean_i[macro_IoU_i-0.05*max(seconds_i-1,0)], calculated per image with no subsecond bonus. Elapsed time includes method preparation,inference,glass policy and native projection; shared encoder/loading,scoring,compression and rendering are separate. Three nice12 workers use two numerical/OpenMP threads each. One timed invocation per method/image on a shared machine is not controlled cross-cycle speedup evidence. Mean time below one second does not imply all images meet the budget.

| Method | Raw mean IoU | Mean seconds | Images >1s | Utility |
| --- | ---: | ---: | ---: | ---: |
| J01 | 0.720017 | 0.2148 | 13/349 | 0.719594 |
| M04 | 0.720016 | 0.2349 | 17/349 | 0.719417 |
| N10 | 0.720007 | 0.2531 | 19/349 | 0.719260 |
| C17 | 0.717562 | 0.1691 | 6/349 | 0.717474 |
| N08 | 0.721170 | 0.4860 | 30/349 | 0.715089 |


N10 gains0.08772 IoU percentage points over M11, with258 wins,19 losses and72 ties. This small paired development benefit primarily preserves tissue; it is not a claim of significant independent generalization. It still cannot restore tissue removed by common support, and it leaves deep exterior clefts inside a convex envelope filled. Its mean time is0.2531s versus M11's0.2730s in this run; the comparison is not a controlled speedup experiment.

N08 gains only0.05646 IoU percentage points over K05 and loses on more cases than it wins. Its38 ambiguous-palette cases improve0.83108 points on average, while WSSS overall loses0.63754 points and WSSS25 alone loses7.24736 points. Native BCSS01 shows fewer false gold bands; BCSS44/132 show improved mesh detail. Both calibrated mean small-region recall(0.21144 versus K05's0.21613) and pixel-only recall(0.57711 versus0.60347) decline, despite fewer completely missed calibrated components(6578 versus6594). This survivor is an acknowledged accuracy/heteroscedastic-model tradeoff with lower utility, not dominance over K05 or J01.

## What the actual images show

False gold bands at red/green interfaces persist under J01/M04 and several field children in native BCSS01. C17 remains cleaner there; N08 reduces but does not eliminate the artifact. N13 creates false grey fringes in many BCSS/TIGER views. N03 has conspicuous sampled-grid steps despite an exact discrete tree solver. WSSS and OCELOT gland images repeatedly lose narrow separators and fill internal lumina. Broad overlap can be high while fine architecture is wrong.

Common support deletes reference-positive pale fat/lumina in BCSS66/76/82/88/107/112,TIGER28/47/60/79/80,RINGS05/25 and many OCELOT cases. N11/N12 can open real glass clefts, but additional valid tissue deletion makes them unsuitable successors. Native TIGER28 and CAMELYON01 show the problem clearly. N10 avoids the directional-ray cuts of M11 but cannot repair shared support. This is a priority for a prospective next cycle, not a posthoc modification of this cycle's predictions.

All10185 small components remain scored, including643 in classes missing from coarse labels.15 cannot reach50% under common support;7513 have no roster method reaching50%;6637 are entirely missed by all;319 are at least50% recovered by coarse labels but below50% for every refinement. These are roster-union diagnostics mixing distinct micrometre and pixel criteria, not a per-method recall or independent-validation claim. Coordinate-specific results are preserved in SMALL_REGION_ATLAS.json. All classes missing from coarse input still count in pixel accuracy and missed-region scoring.

## Numerical and evaluation limitations

N07 reaches its60-step budget without stationarity on every one of319 informative cases(max gradient3.3491462,primal0.0656584,dual3.3491462); its frozen finite outputs are scored honestly, without calling them optima. N14 has25 stationarity flags(max projected gradient5.77427e-5). Absence of flags for other methods is not proof of calibration or anatomical validity. Exact C++ tree/forest/Tarjan kernel tests certify their small discrete tasks, not biological performance or end-to-end acceleration.

These349 images are repeatedly reused development data. TIGER source overlap and unverified RINGS patient identity limit independence. BCSS/TIGER palettes are dataset-specific; no broad gold/grey tissue interpretation is inferred from display colour. The five fixed RGB conversions remain unchanged and the primary cycle uses equal09+19. Final selected methods across all fixed conversions remain a required later analysis. Ten further cycles remain.

## All twenty decisions

### J01

SELECT: mandatory primary utility leader,0.719594170. Its0.2148s mean still includes13 over-budget images. Native BCSS01 shows false gold interfaces; dense OCELOT/WSSS glands remain merged, and common support deletes labelled fat. Retain the strongest speed-adjusted reference, without claiming new accuracy or a universal one-second bound.

Raw IoU 0.720016592; mean 0.2148s; 13/349 over1s; utility 0.719594170.

### M04

SELECT as the positive sampling-variance branch, not an accuracy advance or numerical top-five requirement. IoU differs from J01 by only-0.000001003 and actual masks are visually near-identical. N04/N05/N06 offer no convincing quality/utility improvement over it. Variance is positive by construction but remains uncalibrated;17 images exceed one second.

Raw IoU 0.720015589; mean 0.2349s; 17/349 over1s; utility 0.719417227.

### K05

REJECT this cycle in favor of N08 as an explicit raw-accuracy/appearance-variance experiment. K05 is faster and has higher utility and better calibrated and pixel-only mean small-region recall than N08; this tradeoff is acknowledged. Native BCSS01 has large false gold spill, whereas N08 reduces that failure and improves ambiguous-palette mean IoU. The replacement is not dominance.

Raw IoU 0.720605555; mean 0.3419s; 30/349 over1s; utility 0.718161432.

### M11

REJECT in favor of N10. Eight-direction glass rays produce visible diagonal cuts in native BCSS07/TIGER28 and delete positive tissue. N10 gains0.000877194 mean IoU with258 wins,19 losses,72 ties and0.001085085 higher measured utility. Both remain limited by shared support and can fill true exterior clefts.

Raw IoU 0.719130217; mean 0.2730s; 20/349 over1s; utility 0.718175406.

### C17

SELECT as the fast checked palette-mixture branch. Mean0.1691s and6 images over one second, with lower overall IoU than J01. Native BCSS01 has the cleanest red/green interface among retained alternatives and avoids many field-induced false gold bands. Elsewhere palette ambiguity and merged glands persist; do not call this anatomical reconstruction.

Raw IoU 0.717561833; mean 0.1691s; 6/349 over1s; utility 0.717474413.

### N01

REJECT despite0.001029218 mean raw-IoU gain over J01 and192 wins/125 losses/32 ties. Integral tile observations lower utility by0.002614240, cost30 over-budget images, and lose0.013052176 mean IoU on38 ambiguous-palette cases. Native BCSS01 shows major false gold spill; original149 mean IoU also falls0.002176056. Gains on additional200 do not erase these regressions.

Raw IoU 0.721045811; mean 0.3591s; 30/349 over1s; utility 0.716979930.

### N02

REJECT: one common multiplicative colour-gain field lowers J01 raw IoU by0.013423574,81 wins/236 losses/32 ties. Visual examples repeatedly show class overexpansion and lost branches, including OCELOT17/25 and CAMELYON05. Its11 over-budget images and similar mean time do not compensate for systematic class distortion.

Raw IoU 0.706593019; mean 0.2062s; 11/349 over1s; utility 0.706215946.

### N03

REJECT: exact tree-Potts optimization on the sampled graph does not justify the visibly stepped nearest-projected masks seen throughout the349 overviews and native crops. IoU falls0.000510429 versus J01,134 wins/184 losses/31 ties; small-region recall falls in both coordinate groups.18 images exceed one second. An exact discrete optimum is not anatomical correctness.

Raw IoU 0.719506164; mean 0.2509s; 18/349 over1s; utility 0.718523160.

### N04

REJECT: tile-correlated noise probes gain only0.000004383 raw IoU over M04,141 wins/171 losses/37 ties, while utility falls0.000142405. Native BCSS01 retains the same false gold fringe and most overviews are nearly indistinguishable.17 slow cases and uncalibrated variance do not establish a useful replacement.

Raw IoU 0.720019972; mean 0.2319s; 17/349 over1s; utility 0.719274822.

### N05

REJECT: cross-class covariance/log-probit contrast lowers M04 raw IoU0.000212769 and utility0.002046302,125 wins/173 losses/51 ties. Most actual masks retain the same merged glands and false interfaces while mean time rises to0.4068s with26 slow cases. Covariance modelling did not produce a convincing visual benefit.

Raw IoU 0.719802820; mean 0.4068s; 26/349 over1s; utility 0.717370925.

### N06

REJECT: residual-degree-of-freedom noise estimation lowers M04 raw IoU0.000073211 and utility0.000292583,129 wins/177 losses/43 ties. Small changes in near-parent contours do not resolve native false gold bands or missing gland channels.17 cases exceed one second; positive noise estimates are not calibrated uncertainty.

Raw IoU 0.719942378; mean 0.2419s; 17/349 over1s; utility 0.719124644.

### N07

REJECT this60-step group-L1-curvature implementation: all319 informative cases miss the declared stationarity tolerance. Its finite masks lower K05 raw IoU0.001920942 and utility0.014373866,106 wins/211 losses/32 ties;84 images exceed one second. The rejection concerns the measured budgeted solver, not proof that the mathematical L1 objective cannot work.

Raw IoU 0.718684613; mean 0.9234s; 84/349 over1s; utility 0.703787565.

### N08

SELECT as an explicit raw-accuracy tradeoff replacing K05, not a top-five-utility method. Mean IoU0.721170133 is the raw leader, up0.000564578 over K05, but144 wins versus174 losses/31 ties, utility down0.003072207 and30 slow cases. Ambiguous-palette mean IoU rises0.008310781 and native BCSS01 reduces false gold bands; BCSS44/132 improve visibly. WSSS25 loses0.072473596, pixel-only small recall drops and calibrated mean small recall also drops. Retain this distinct heteroscedastic likelihood to test its failures, without claiming general superiority.

Raw IoU 0.721170133; mean 0.4860s; 30/349 over1s; utility 0.715089225.

### N09

REJECT: H&E-guided prolongation lowers K05 raw IoU0.000077253,92 wins/223 losses/34 ties, and utility0.013758325. Mean0.8970s still includes59 images over one second. Native BCSS20 adds false red flecks and fine glands remain merged. Slight small-region recall changes do not justify its runtime and artifacts.

Raw IoU 0.720528302; mean 0.8970s; 59/349 over1s; utility 0.704403107.

### N10

SELECT as the tissue-envelope branch replacing M11. Convex stained-tissue support removes the parent straight-ray artifacts and improves raw IoU0.000877194 and utility0.001085085,258 wins/19 losses/72 ties. Positive-fat preservation improves BCSS66/88/112. It is almost J01 in dense fields,19 cases exceed one second, and true exterior clefts inside the hull stay filled. Native WSSS/OCELOT lumina and shared positive support deletions remain unresolved.

Raw IoU 0.720007412; mean 0.2531s; 19/349 over1s; utility 0.719260491.

### N11

REJECT: seeded minimax-barrier watershed opens useful white clefts in some overviews but also removes positive fat, gland lumina and pale tissue. Native TIGER28/CAMELYON01 and RINGS22 plus OCELOT04/06/36 show this failure. Mean IoU falls0.002331163 versus M11 despite192 wins/86 losses/71 ties;18 images exceed one second. More glass opening is not reliable tissue preservation.

Raw IoU 0.716799055; mean 0.2602s; 18/349 over1s; utility 0.715990885.

### N12

REJECT: two independent exterior escape routes are less aggressive than N11 in many views but still delete labelled fat/lumina. Mean IoU falls0.000885631 versus M11 despite220 wins/57 losses/72 ties; TIGER28 loses0.200137856 and TIGER80 loses0.063751941.18 images exceed one second. Topological redundancy alone does not distinguish exterior glass from valid pale tissue.

Raw IoU 0.718244586; mean 0.2488s; 18/349 over1s; utility 0.717556977.

### N13

REJECT despite mean0.0889s and no over-budget case. Nearest categorical codewords lower C17 raw IoU0.019476820,68 wins/249 losses/32 ties. Actual BCSS/TIGER overviews and native TIGER23 show false grey fringes. Mixed artificial RGB cannot reliably be decoded by a nearest single palette colour.

Raw IoU 0.698085013; mean 0.0889s; 0/349 over1s; utility 0.698085013.

### N14

REJECT: conditional channel-proportion likelihood lowers C17 raw IoU0.003695886 and utility0.004380100,96 wins/221 losses/32 ties.25 informative finite optimizations miss stationarity; artificial RGB channels are not validated multinomial measurements. Some native fringes shrink but others remain and fine glands are not recovered;17 images exceed one second.

Raw IoU 0.713865947; mean 0.2596s; 17/349 over1s; utility 0.713094313.

### N15

REJECT: multiplicative optical-density palette mixing lowers C17 raw IoU0.005692637 and utility0.006141338,106 wins/212 losses/31 ties. Native BCSS01 has a clean interface, but BEETLE04/09/10 and other overviews show distorted red/gold structures. Improved pixel-only small recall accompanies overall losses;14 cases exceed one second. Artificial converted RGB does not justify the changed forward mixing law.

Raw IoU 0.711869196; mean 0.2475s; 14/349 over1s; utility 0.711333075.
