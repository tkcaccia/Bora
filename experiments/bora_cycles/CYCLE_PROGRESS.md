# Accuracy, speed and the new one-second penalty

Seven of twenty cycles are complete on all 149 repeatedly reused development
ROIs. The user now requests penalizing algorithms that take longer than one
second per image. The explicit working policy is:

    adjusted utility = mean_i[macro_IoU_i - 0.05 * max(seconds_i - 1, 0)]

Cycle8 expands prospectively to349 ROIs, retaining the original149 exactly and
adding40 each from BCSS, TIGER, WSSS4LUAD, RINGS tumor and OCELOT. All inputs and
1745 fixed RGB conversions are frozen and audited. The full349x22 run is active
after16 actual pilot overview reviews and four native-label crops; it is not
yet a completed cycle. See `CYCLE_EIGHT_EXECUTION.json`. Historical149 and new200
scores are reported separately. Unknown-resolution pixel-area summaries are
separate from physical-area summaries. A common completed-coarse fallback
prevents numerical class artifacts on30 constant-RGB inputs; no refinement
claim is made for those cases.

This subtracts five IoU percentage points per excess second, with no bonus
below one second. Calculate excess on each image before averaging: a method
whose mean time is below one second can still incur a penalty on slow cases.
The adjusted utility is not IoU. Raw IoU, measured time, over-budget image
count and the penalty remain separately visible.

The coefficient is an explicit engineering tradeoff chosen in response to the
user's request, not an estimated biological parameter. The policy is frozen
prospectively from cycle 5. Cycles 1–4 receive a clearly labelled posthoc report;
their signed selections, protocols, predictions and raw scores are unchanged.
The slow raw-accuracy leader no longer has a mandatory survivor slot from
cycle 5; the runtime-adjusted candidate leader must survive instead.

## What improved across completed cycles

| Cycle | Main mathematical development | Best raw mean IoU | Leader's mean seconds in that cycle |
| --- | --- | ---: | ---: |
| 1 | C18 adapts colour precision; C17 solves palette mixtures to checked KKT tolerance | C18: 0.671789 | 0.1959 |
| 2 | D10 uses local spatial-affine class appearance rather than a stationary prototype | D10: 0.672397 | 0.8287 |
| 3 | F01 integrates uncertainty in local prototype estimates, avoiding certainty from sparse local support | F01: 0.679529 | 1.7402 |
| 4 | Fifteen new hypotheses tested; none exceeds F01 | F01: 0.679529 | 1.9140 |
| 5 | H02 solves a screened spatial appearance field; H10 integrates Gaussian colour noise in a discriminant objective | F01: 0.679529 (rejected for runtime) | 1.6544 |
| 6 | J01 uses original H&E conductance; J07 integrates squared-hinge Gaussian risk | J01: 0.674658 | 0.1932 |

The cycle 2 raw gain is 0.000608, or 0.061 IoU percentage points. Cycle 3 adds
0.007132, or 0.713 points. The total raw gain from cycle 1's leader is 0.774
points, with a significant computational cost. Native review also identifies
regressions, so this is not uniform improvement on every image.

Cycle 2 introduced the D08 analytic within-tile covariance model: estimate
nuisance from variation within coarse tiles, then solve a small discriminant
system. In cycle 4 it takes 0.05045 seconds/image with IoU 0.656050. This is a
useful speed branch with lower accuracy, not a claim that it achieves F01
quality at that time. C17 was the strongest subsecond accuracy branch among
the cycle-four survivors; its accuracy was already present in cycle 1, so the
fast accuracy frontier did not advance through cycle 4.

Cycle 5 changes the surviving frontier. H02 estimates slowly varying class
appearance by solving a sparse screened-Laplace system, rather than repeatedly
inverting local uncertainty models at every pixel. It obtains IoU 0.671889 at
0.180560 seconds/image, with no image above one second. Its advantage over C17
is only 0.0285 IoU percentage points, and only 0.0100 points over cycle 1's C18;
this is a small fast-frontier gain. Native BCSS45 exposes a vessel-to-fat
regression. Smoothing acts on estimated appearance, not labels or posteriors,
and the converted RGB is unchanged.

The clearer paired improvement is H10. It integrates Gaussian uncertainty in
tile mean colours directly into an exponential discrimination objective. Mean
IoU rises from its parent D08's 0.656050 to 0.667425: **1.1375 percentage points**,
98 case wins, 45 losses and six ties. Its measured time is 0.059427 versus
0.049341 seconds/image. Native BCSS26 confirms corrected vessel recognition;
BCSS83/87 and TIGER27 still regress. Eight finite optimizer runs report
unsuccessful termination, so the result does not establish exact optimization.

H04's joint palette fitting survives as the fourth runtime-adjusted candidate,
but it is less accurate and slower than C17. It is a distinct retained model,
not evidence of an improvement over its parent. H03's white-cavity rule is
rejected: opening BCSS18's cleft does not justify deleting positively labelled
fat and lumina elsewhere. H14's nested contours and false islands are another
clear rejection despite apparently higher small-region recall.

Background and small-region experiments also exposed limitations. G03 opens
the reference-zero BCSS18 cleft, but deletes positively annotated adipose and
lumina elsewhere and lowers overall IoU. G01 improves some cases but produces
false grey tissue in native BCSS74 and misses the true central structure in
BCSS87. Those outcomes were rejected or retained as failure evidence, not
reported as successful general improvements. Small-region scoring, an atlas
of missed components, annotation-independent colour novelty proposals and
exact native-mask audits make these failures measurable.

## Cycle 5: selected five under the new penalty

| Method | Raw mean IoU | Mean seconds | Images >1s | Mean penalty | Adjusted utility |
| --- | ---: | ---: | ---: | ---: | ---: |
| H02 | 0.671889 | 0.1806 | 0/149 | 0 | 0.671889 |
| C17 | 0.671604 | 0.1580 | 0/149 | 0 | 0.671604 |
| H10 | 0.667425 | 0.0594 | 0/149 | 0 | 0.667425 |
| H04 | 0.665594 | 0.2563 | 0/149 | 0 | 0.665594 |
| C01 | 0.663706 | 0.0867 | 0/149 | 0 | 0.663706 |

F01 remains the raw leader at IoU 0.679529. It takes 1.654383 seconds/image in
cycle 5, exceeds one second on 121/149 images, and loses 0.033936 mean utility:
its adjusted score is 0.645593. It is rejected, along with its slow H03 child.
The penalty is based on each image, not on the 1.654383 mean. All five selected
methods have zero measured over-budget images. These are ROI timings at the
frozen analysis settings, not claims of full-WSI processing in one second.

Selection follows all 149 actually inspected all-method contact sheets and
four native crops. Reasons for all twenty candidates are recorded in
`CYCLE_FIVE_SELECTION.json` and `CYCLE_FIVE_REVIEW.md`. Earlier signed selections
are unchanged; their runtime ranking is explicitly posthoc.

Elapsed time includes candidate preparation, inference, background policy and
native projection, excluding shared loading, scoring, compression and rendering.
Each completed cycle used Chiamaka with three low-priority image processes
and two numerical/OpenMP threads each, one timing per method/image. The
unchanged F01 masks take different times across cycles 3/4/5, illustrating timing
variability; those differences are not controlled speedup measurements.

The C++ pair-count kernel separately shows 1.730x one-to-two-thread scaling on
the larger synthetic workload, with exact output parity, and no useful gain
on the small workload. This is kernel evidence, not whole-algorithm acceleration.

`runtime_report.py` derives `RUNTIME_PENALTY_HISTORY.json` and `runtime_history.html`
from signed completed-cycle evidence. The selected-five viewer shows the
penalty and orders its five methods by adjusted utility. Seven of twenty cycles
are complete. All later selections must respect the one-second preference.

## Cycle 6: measurable gains below one second

J01 changes appearance-field conductance using original H&E optical-density
edges, limiting inappropriate class-colour borrowing. It improves H02 from
0.671889 to 0.674658 mean IoU (**0.27685 percentage points**), with 90 wins,
57 losses and two ties. Mean time is 0.1932 versus 0.1923 seconds in the same run;
every image is below one second. This is a 0.2869-point gain over cycle 1's C18
leader. Native BCSS45 improves vessel recognition, but native BCSS53 loses
part of a tumour band and small-region recall falls. H02 therefore survives too.

J07 replaces H10's expected exponential loss with an analytically integrated
squared hinge, improving mean IoU from 0.667425 to 0.668524 (**0.10996 points**)
at 0.0753 versus 0.0613 seconds. It corrects a false fat island in native BCSS87,
but loses more stroma in native TIGER23 and has 25 unsuccessful finite optimizer
terminations. H10 remains the fastest survivor. The RGB conversion is unchanged.

The signed five are **J01, H02, C17, J07 and H10**. All149 overviews and six
native crops were actually inspected. J03 is rejected for extra annotated-fat
deletion without general cleft recovery; J15 for false rings despite higher
small-region recall. All3,278 masks and745 unchanged-parent comparisons pass
the native audits. See CYCLE_SIX_REVIEW.md for every decision. These remain
development results, with 14 cycles and the final full-conversion comparison
still outstanding.

## Cycle 7: complementary subsecond improvements

All149 all-method overviews and eight native label crops are actually reviewed.
The signed survivors are **J01,K05,C17,J07,K13**. Both native audits pass3278
masks and745 exact parent comparisons. All241 focused checks pass on both hosts.

J01 remains the leader at IoU0.674658 and0.2008 seconds; no new child beats it.
K05 replaces H02: a biharmonic curvature penalty on class appearance raises
IoU0.671889 to0.672922 (**0.10329 points**,94 wins/53 losses/2 ties).
Mean time rises0.1946 to0.3080 seconds, still zero images over one second.
Small-region recall improves0.192574 to0.202911. It retains BCSS53's tumour
band lost by J01, but misclassifies vessel interiors in BCSS26/45.

K13 replaces H10: empirical within-tile residual integration raises IoU0.667425
to0.668351 (**0.09265 points**,73 wins/64 losses/12 ties), with mean time
0.0569 to0.0714 seconds. Native BCSS87 removes the false-fat interior.
Optimizer unsuccessful flags rise8 to18; maximum gradient norm0.004826.
J07 is slightly more accurate at0.668524/0.0735 seconds; the two-millisecond
difference is not a controlled speedup. Retaining both compares distinct
Gaussian and empirical noise assumptions. C17 retains the direct palette model.

K04 receives a penalty on21 images despite mean time below one second.
K06 exceeds one second on10. K07's C++ kernel scales1.979x from one to two
threads on the large synthetic case, but accuracy falls and126/149 image
fits contain unfinished gaps. Engineering gains do not override anatomical
regressions. See CYCLE_SEVEN_REVIEW.md for all20 decisions.

All200 additional native/coarse/reference input sets are prepared and audited:
40 each BCSS,TIGER,WSSS4LUAD,RINGS,OCELOT. The planned corpus is349 ROIs,
retaining all149 historical cases. Future cycles must add unchanged conversions,
single-class/task/calibration adapters and a manifest-based review gate.
WSSS/RINGS remain uncalibrated pixel-only data. Historical subset scores stay
separately comparable; all reused images are development data.
Thirteen cycles and the final all-five-conversion evaluation remain.

Expanded-input update: all200 additions now have600 pinned Virchow2 arrays,
with full-field coverage, source/cache integrity, finite values and per-scale
calibration verified by a model-free audit.120 are calibrated and80 pixel-only.
Thirty expose a single eligible training class (11 WSSS,19 RINGS), including
three WSSS cases with other coarse labels outside eligible central summaries.
No additional class is inferred from reference metadata. Shared encoder time
is recorded separately from Bora timing. RGB conversions and expanded cycle8
are not yet run; see ../bora_extension/FEATURE_EXECUTION.json.
