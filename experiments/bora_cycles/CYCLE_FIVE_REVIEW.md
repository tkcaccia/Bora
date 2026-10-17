# Cycle five: runtime-penalized review and selection

Five of twenty cycles are complete. All 149 frozen development ROIs produced
3,278 native masks from twenty candidates and two controls. All 149 contact
sheets were actually inspected, covering every method, with four additional
native label sheets. Local and Chiamaka audits passed: 6,854 frozen files,
3,278 exact native/display rasters, 745 exact unchanged-parent comparisons,
and 176 exact pilot/full native masks. Selection changes no prediction,
raw accuracy score or reviewed sheet.

The prospective selection utility is the average over images of
`macro_IoU - 0.05 * max(candidate_seconds - 1, 0)`. The rate is five IoU
percentage points per excess second. There is no bonus below one second.
The user requested the runtime preference; the coefficient is an explicit
engineering tradeoff. The five selected candidates are the five highest
eligible adjusted scores. B21/B22 are reference controls, not candidates.

| ID | Model | Raw mean IoU | Mean seconds | Images >1 s | Adjusted utility |
| --- | --- | ---: | ---: | ---: | ---: |
| H02 | Screened spatial appearance field | 0.671889 | 0.180560 | 0/149 | 0.671889 |
| C17 | Converged simplex palette model | 0.671604 | 0.158000 | 0/149 | 0.671604 |
| H10 | Noise-marginalized exponential discriminant | 0.667425 | 0.059427 | 0/149 | 0.667425 |
| H04 | Coarse-supervised joint palette fitting | 0.665594 | 0.256255 | 0/149 | 0.665594 |
| C01 | Tile-plurality likelihood | 0.663706 | 0.086688 | 0/149 | 0.663706 |

F01 has the highest raw IoU, 0.679529, but its mean 1.654383 seconds and
121 over-budget images lower its utility to 0.645593. It is rejected.
H03 is also rejected at utility 0.641733, with 125 over-budget images.
Earlier signed selections remain historical; only their display is rescored
post hoc under this policy.

## What improved, and what did not

H10 improves its parent D08 by 0.011375 mean IoU (1.1375 percentage points),
with 98 wins, 45 losses and six ties. It takes 0.059427 rather than 0.049341
seconds/image. It changes the statistical risk by integrating a Gaussian
tile-colour error model; it does not modify the RGB conversion or run
Virchow2 again. Native BCSS26 shows vessel recovery where D08 predicts fat.

H02 leads adjusted accuracy, but exceeds C17 by only 0.000285 IoU
(0.0285 points). Relative to the fast cycle-one C18, the gain is only
0.000100 IoU (0.0100 points). Its sparse appearance-field formulation is
inexpensive, but BCSS45 shows a large false fat stripe in true vessel.
H04 loses 0.006010 IoU against C17 and is slower; it survives as the fourth
eligible candidate, not because it improves its parent.

Native BCSS07's broad reference-zero cleft remains filled. H03 opens the
BCSS18 cleft with irregular edge flecks but deletes positively annotated
fat/lumina in other cases. TIGER38/40 show why bright spaces cannot simply
be removed. Positive reference pixels remain scored even when a method
deletes them; reference-zero clefts provide qualitative evidence only.
The fixed source conversion and mask IDs are unchanged. Fat and blood
vessel have distinct IDs but close gold display colours.

The small-region atlas contains 5,578 reference components <=400 um²,
including 377 with a class absent from the original coarse mask. No method
recovers 50% correct coverage for 4,017, all 22 completely miss 3,421, and
143 recovered by the coarse input fall below 50% in every refinement.
These categories overlap. All-method counts cannot establish improvement
against cycle four because the candidate roster changes.

## Numerical and runtime limits

H02's maximum linear-system residual is 2.143e-14. H04's alternating objective
decreases, with maximum latent/prediction KKT residuals 9.921e-9/6.661e-16;
finite nonconvex fitting is not a global optimum. C01 retains 67 unsuccessful
optimizer flags, H07 six and H10 eight. C01's unavailable gradient norm is
null, not zero. All 505 H15 class fits hit the finite EM budget without
convergence. These are the executed finite fits, retained in the scores.

Elapsed time includes candidate preparation, prediction, background policy
and native projection; shared loading, scoring, compression and rendering are
excluded. These are ROI timings at frozen analysis settings. Three low-priority
image processes each use two numerical/OpenMP threads. Cross-cycle times
are not controlled speedup measurements: unchanged F01 masks have different
measured times in cycles three, four and five.

The separate H01 C++ mixture kernel gives 0.04949/0.03329 seconds on the
million-pixel one/two-thread workload (1.486x), with exact thread-output parity.
This is kernel evidence and does not turn the rejected H01 into a successful
segmentation method.

## Delivery and continuation

`CYCLE_FIVE_SELECTION.json` records concrete benefits, limitations and rejection
reasons for all twenty candidates. `selected.html` shows only the five signed
survivors, with all 149 images selectable. Original H&E, reference, tile input,
fixed converted RGB, adjustable semitransparent masks, class legend, pixel
spacing and scores are available. Full candidate results remain archived.

The goal remains active: fifteen further cycles and the final unchanged
five-conversion comparison remain. The next fifteen innovations must be
three distinct new interventions per selected parent, not parameter sweeps.
All 149 repeatedly reused ROIs are development data, not independent validation.
