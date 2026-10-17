# Cycle two: complete review and selection

All 149 frozen ROIs were executed with the same 9+19 ensemble RGB conversion.
Twenty candidates plus two controls produced 3278 native masks, with zero
missing method outputs. All 149 contact sheets were actually inspected across
all 22 panels, followed by six native-coordinate crops comparing seven
candidates. Every mask hash was rechecked and all 745 comparisons of the five
unchanged survivors with cycle one were identical. No segmentation was rerun
or changed during review.

The signed result is `CYCLE_TWO_SELECTION.json`. Runtime result folders also
contain `SELECTION_INPUT_AUDIT.json`, `OPTIMIZER_AUDIT.json`, the full per-case
scores, actual hashed review journals, and provenance for native crops.

## Five survivors

| ID | Role | Mean image macro IoU | Mean seconds/image | Mean small-region recall |
| --- | --- | ---: | ---: | ---: |
| D10 | Local affine class appearance; accuracy leader | 0.672397 | 0.829 | 0.198091 |
| C17 | Converged palette inverse; compact clean regions | 0.671604 | 0.160 | 0.183608 |
| C01 | Tile-winner likelihood; weak-supervision branch | 0.663706 | 0.082 | 0.203302 |
| D08 | Within-tile covariance; fastest analytic branch | 0.656050 | 0.048 | 0.194626 |
| C15 | Copula density; highest small-region sensitivity | 0.644943 | 0.306 | 0.225562 |

Selection first retains the primary accuracy leader, then uses the
accuracy/time/small-region Pareto frontier and explicit visual tradeoffs.
C01 and D08 share a linear discriminant output family but estimate it through
different observation models. Their retention is an explicit speed/accuracy
tradeoff. C15 remains solely for its sensitivity advantage, with substantial
false-positive risks; it is not the preferred general-purpose mask.

D10 improves 112 images and worsens 35 compared with C03, with two ties.
Against C18, it improves 95 and worsens 52, with two ties. Its mean IoU gain
over C18 is only 0.000608. This is a small development gain, not evidence of a
large or independently validated biological improvement. C18 is rejected in
favour of the faster, almost equally accurate C17 palette branch. The signed
selection records the benefits and rejection reasons of all fifteen others.

## Findings that must drive the next innovations

- **True tears are still filled.** Native BCSS37 and BCSS50 show broad pale
  empty cavities painted necrosis-grey by every inspected candidate. D12
  clears only a horizontal band or tiny patch. Its pixelwise white likelihood
  is sensitive to coarse tile context instead of following the whole cavity.
- **D12 also has real successes.** In native BCSS64 it clears a long white
  tear between stained tissue banks that the other masks fill. However, it
  frequently perforates stained stroma, glands and lymphocytic tissue. Its
  aggregate performance and runtime do not justify survival. A future glass
  model must differ mathematically from this rejected Bernoulli competition.
- **Bright anatomy is not automatically glass.** Many TIGER lumina and fat
  cavities are included in expert tissue-region labels. The common input
  support already deletes peripheral adipose in TIGER28. Native BCSS40
  shows enclosed adipocytes, false C15 purple/green islands, and small D12
  white deletions. Most of that crop is unannotated, so its qualitative
  concern must not be presented as a measured reference error.
- **Thin known structures are lost internally.** Native BCSS38 contains a
  long gold-labelled vessel, but all candidates recover only the lower part.
  This is a failure of internal tissue structure, not merely an outer margin.
- **A recovered region can also be nonspecific.** D09 recovers the duct-labelled
  focus in native BCSS90 but also colours an adjacent vessel as duct. The
  recovered reference component is 411.56 um², just above the <=400 um²
  small-region bin; it is not proof of improvement in that bin.
- **Absent coarse classes remain absent.** Tiny red foci in TIGER16 and grey
  necrosis foci in BEETLE06 have no coarse class support and are missed by all
  candidates. Their fixed RGB often lacks a distinct interior signal. Colour
  rarity proposals retain unknown semantic identity; they cannot establish a
  new named tissue type. The cached annotation-free proposal maps are verified
  identical across cycles and remain available for all 149 ROIs.
- **Some nonlinear densities make false rims or merge objects.** C15 and
  several children enlarge glands or add third-class contours, including
  false tissue islands inside fat. C17's palette inverse often removes those
  rims, while still losing true fine structures.
- **Sparse annotations limit visual score interpretation.** BEETLE10 has
  nearly perfect scores despite large differences in unannotated tissue.
  All tissue is analysed, but only original reference values >0 are scored.

## Numerical and timing limitations

All stored outputs are finite and were scored as executed. An output existing
does not prove its statistical fit converged. The optimizer audit finds:

| Candidate | Optimizer reports success=false | At the 35-iteration budget | At zero iterations |
| --- | ---: | ---: | ---: |
| C01 | 67 | 66 | 1 (BCSS50) |
| D07 | 37 | 37 | 0 |
| D09 | 91 | 90 | 1 (BCSS50) |

The zero-iteration statuses are recorded without inferring an unavailable
optimizer termination message. These fits are budget-limited or unsuccessful
optimization results, not established optima. In contrast, D03's numerical
stationarity checks and D04's final probability-change checks pass on all cases;
that did not make their masks more accurate than C17.

Times include each candidate's preparation, inference, background policy and
native projection. Shared loading, scoring, compression and rendering are
excluded. Execution uses three low-priority CPU processes with two numerical
or OpenMP threads each. Each full-cycle candidate was timed once; these are
observed elapsed times, not a controlled CPU scaling or speedup study. C++
ports, compiler changes and report-export optimizations are engineering work,
not separate algorithmic innovations.

All 149 cases are repeatedly used for development selection. Results are not
an independent validation set. Small-region recall averages case-wise recall
at >=50% correctly labelled component coverage over 130 cases with eligible
components. Annotation-zero pixels are never used as accuracy targets, and
annotated pixels deleted by a candidate remain errors.

Two of twenty required cycles are fully reviewed and selected. The next gate
is three distinct mathematical innovations per survivor, fifteen in total,
while keeping the five survivors and RGB conversion unchanged. No cycle-three
implementation or result is claimed by this report.
