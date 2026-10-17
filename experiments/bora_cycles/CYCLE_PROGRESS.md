# Accuracy, speed and the new one-second penalty

Four of twenty cycles are complete on all 149 repeatedly reused development
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

The cycle 2 raw gain is 0.000608, or 0.061 IoU percentage points. Cycle 3 adds
0.007132, or 0.713 points. The total raw gain from cycle 1's leader is 0.774
points, with a significant computational cost. Native review also identifies
regressions, so this is not uniform improvement on every image.

Cycle 2 introduced the D08 analytic within-tile covariance model: estimate
nuisance from variation within coarse tiles, then solve a small discriminant
system. In cycle 4 it takes 0.05045 seconds/image with IoU 0.656050. This is a
useful speed branch with lower accuracy, not a claim that it achieves F01
quality at that time. C17 remains the strongest subsecond accuracy branch among the current survivors;
its accuracy was already present in cycle 1, so the fast accuracy frontier has
not yet advanced through cycle 4.

Background and small-region experiments also exposed limitations. G03 opens
the reference-zero BCSS18 cleft, but deletes positively annotated adipose and
lumina elsewhere and lowers overall IoU. G01 improves some cases but produces
false grey tissue in native BCSS74 and misses the true central structure in
BCSS87. Those outcomes were rejected or retained as failure evidence, not
reported as successful general improvements. Small-region scoring, an atlas
of missed components, annotation-independent colour novelty proposals and
exact native-mask audits make these failures measurable.

## Current five under the new penalty

| Method | Raw mean IoU | Mean seconds | Images >1s | Mean penalty | Adjusted utility |
| --- | ---: | ---: | ---: | ---: | ---: |
| C17 | 0.671604 | 0.1812 | 0/149 | 0 | 0.671604 |
| C01 | 0.663706 | 0.0874 | 0/149 | 0 | 0.663706 |
| D08 | 0.656050 | 0.0505 | 0/149 | 0 | 0.656050 |
| C15 | 0.644943 | 0.3219 | 0/149 | 0 | 0.644943 |
| F01 | 0.679529 | 1.9140 | 138/149 | 0.046304 | 0.633225 |

C17 now ranks first among the historical five. F01 gives 0.007925 higher raw
IoU than C17, but its runtime cost outweighs that gain under the requested
penalty. G04 and G08 also rank above F01 in the full cycle 4 posthoc numeric
report; the displayed historical five remain the signed selection rather
than a retroactively substituted roster. Cycle 5 will select using the new rule.

Elapsed time includes candidate preparation, inference, background policy and
native projection, excluding shared loading, scoring, compression and rendering.
Each completed cycle used Chiamaka with three low-priority image processes
and two numerical/OpenMP threads each, one timing per method/image. The
unchanged F01 masks take different times across cycles 3/4, illustrating timing
variability; those differences are not controlled speedup measurements.

The C++ pair-count kernel separately shows 1.730x one-to-two-thread scaling on
the larger synthetic workload, with exact output parity, and no useful gain
on the small workload. This is kernel evidence, not whole-algorithm acceleration.

`runtime_report.py` derives `RUNTIME_PENALTY_HISTORY.json` and `runtime_history.html`
from signed completed-cycle evidence. The selected-five viewer shows the
penalty and orders its five methods by adjusted utility. The next fifteen
innovations and all later selections must respect the one-second preference.
