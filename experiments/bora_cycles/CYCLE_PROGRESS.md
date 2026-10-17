# Accuracy, speed and the new one-second penalty

Five of twenty cycles are complete on all 149 repeatedly reused development
ROIs. The user now requests penalizing algorithms that take longer than one
second per image. The explicit working policy is:

    adjusted utility = mean_i[macro_IoU_i - 0.05 * max(seconds_i - 1, 0)]

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
penalty and orders its five methods by adjusted utility. Five of twenty cycles
are complete. The next fifteen innovations and all later selections must
respect the one-second preference; they have not been implemented or evaluated
by the cycle-five selection.
