# Cycle six: reviewed development results

Six of twenty cycles are complete. All 149 frozen development ROIs produced all
22 masks (3,278 total), with zero failed predictions. Both native audits pass,
including 745 unchanged-parent comparisons. All 176 pilot masks are array-identical
to their full-run counterparts. The RGB conversion remains unchanged.

All 149 all-method overview sheets were actually viewed, with hash-pinned notes.
Six targeted native label sheets were also viewed: BCSS06/45/53/74/87 and TIGER23.
This is not exhaustive native-pixel verification. No seventh-cycle model existed
before selection.

## Selected five

Utility is mean per-image [IoU - 0.05 max(seconds - 1, 0)]. Every cycle-six
candidate is below one second on every measured ROI, so its penalty is zero.
J01 is the utility leader. H02 preserves complementary small regions, C17 is a
direct palette model, and J07/H10 retain different fast statistical losses.
These are not simply the five highest raw-IoU rows.

| Method | Mean IoU | Mean seconds | Images >1s | Small-region recall |
| --- | ---: | ---: | ---: | ---: |
| J01 | 0.674658 | 0.1932 | 0/149 | 0.184554 |
| H02 | 0.671889 | 0.1923 | 0/149 | 0.192574 |
| C17 | 0.671604 | 0.1628 | 0/149 | 0.183608 |
| J07 | 0.668524 | 0.0753 | 0/149 | 0.197088 |
| H10 | 0.667425 | 0.0613 | 0/149 | 0.199198 |

Small-region recall is averaged over the 130 images containing small reference
components; mean IoU uses all 149. Small means at most 400 square micrometres.

J01 improves H02 by **0.27685 IoU percentage points**, with 90 wins, 57 losses
and two ties, at essentially the same measured runtime. Original H&E conductance
changes the appearance field, not the fixed RGB conversion or label smoothing.
Native BCSS45 largely corrects a vessel-to-fat stripe, but native BCSS53 loses
a tumour band; TIGER33 loses 1.94 points. Small-region recall falls from
0.19257 to 0.18455.

J07 integrates squared-hinge risk under Gaussian tile-mean uncertainty. It improves
H10 by **0.10996 points**, with 77 wins, 67 losses and five ties, at 0.0753 versus
0.0613 seconds. Native BCSS87 removes a false fat island, while native TIGER23
loses more of the stromal wedge. H10 remains the fastest survivor with slightly
better small-region recall.

J03 adds **43,377 positively annotated native pixels** to the unassigned count
relative to H02. Native BCSS06 shows extra fat deletion; broad white clefts remain
filled in many overview cases. Deleted positive labels remain errors. CAMELYON
and TIGER references sometimes include white cavities inside positive regions,
so cleaner glass removal and better reference agreement are distinct measurements.

J02 loses 0.4172 points versus H02, including 8.18 in BCSS26 and 4.07 in TIGER23.
J07 is faster and more accurate overall. J15 has the highest small-region recall,
but creates false rings and nested contours; it is rejected on morphology,
accuracy and numerical grounds.

## Numerical evidence and limitations

J01's largest linear-system residual is 1.36e-14; J02's is 9.10e-12.
J07 has unsuccessful optimizer termination on 25/149 cases (maximum gradient
infinity norm 0.003052), and H10 on 8/149. These finite masks are evaluated as
executed, not asserted converged optima. J09 is unsuccessful on 67, J14 on 12
and J15 on 118. Missing C01 gradient diagnostics are unavailable, not zero.

All 209 focused tests passed locally and on Chiamaka; nine posthoc checks passed.
The new C++ Gaussian and dominance kernels match independent numerical references
and one/two-thread outputs exactly. Serial large synthetic Chiamaka profiles give
1.983x and 1.908x one-to-two-thread ratios. These are kernel measurements, not
whole-algorithm speedups; the models using those kernels are both rejected.

All 149 images are repeatedly reused development ROIs, not an independent test
set. Timings include candidate preparation, inference, background policy and
native projection; shared loading, scoring, compression and rendering are excluded.
One timing per method/image was measured under three low-priority processes
with two numerical threads each. Cross-cycle timing variation of unchanged
masks is not a controlled speedup.

## Decision for every candidate

### H02

SELECT: unchanged fast appearance-field comparator, IoU 0.671889 at 0.192s, small-region recall 0.192574. It preserves regions that J01 loses, including the native BCSS53 tumour band and TIGER33 geometry. Its slight time advantage over J01 is not claimed as a stable speed improvement. Broad white clefts and false grey in BCSS60 remain; it is retained for complementary accuracy and an unchanged reference branch.

### C17

SELECT: unchanged exact palette-mixture inverse, IoU 0.671604 at 0.163s. The native BCSS45 and BCSS87 vessels avoid the fat islands seen in Gaussian discriminants, and BCSS60 avoids the false grey stripe. It gives a structurally different direct model with no learned colour classifier. It still fragments lymphocytic tissue in native BCSS74 and misses unrepresented coarse classes; small-region recall 0.183608.

### H10

SELECT: unchanged expected exponential Gaussian discriminant, IoU 0.667425 at 0.0613s, fastest retained method. It has higher small-region recall 0.199198 and fewer unsuccessful optimizers 8/149 than J07, despite a slightly lower aggregate IoU. Vessel-to-fat islands in native BCSS45/87 and excess red in TIGER37 remain. Keep as a transparent speed/accuracy comparator; no artificial bonus is awarded below one second.

### H04

REJECT: unchanged calibrated palette model gives IoU 0.665594 at 0.2576s, below C17 while slower. Native BCSS74 inserts grey necrosis/debris absent from the reference. Its child J11 slightly improves it, but neither reaches the retained palette or Gaussian methods; iterative calibration can distort a useful fixed conversion.

### C01

REJECT: unchanged tile-count likelihood gives IoU 0.663706 at 0.0851s. H10 is faster and more accurate on average; slightly better small recall 0.203302 does not offset its false-region contours and lower mean segmentation accuracy. It preserves more of the native TIGER23 wedge than J07, but is not selected on that single crop. Missing gradient diagnostics are unavailable, not assumed zero.

### J01

SELECT: primary runtime-adjusted leader, mean IoU 0.674658 at 0.193s. Original-H&E edge conductance adds 0.27685 IoU percentage points over H02, with 90 wins/57 losses/2 ties and linear residual at most 1.36e-14. Native BCSS45 corrects much of the vessel-to-fat error, but native BCSS53 erases a tumour band. Mean small-region recall drops from 0.19257 to 0.18455; retain H02 alongside it. White clefts and missing coarse classes remain unresolved.

### J02

REJECT: joint convex shared-nuisance field is fast 0.1088s and small recall 0.20368, but IoU 0.667717 is 0.4172 percentage points below H02. J07 has higher mean IoU and lower runtime. Native TIGER23 shows excessive in-situ expansion deleting the stromal wedge; BCSS26 loses 8.18 points relative to H02. The restricted shared drift model has too large a class-specific regression despite a sound linear solve.

### J03

REJECT: harmonic exterior-glass probability is slower than H02 (0.2282 versus 0.1923s) and slightly less accurate 0.671798 versus 0.671889. It leaves 100/149 case scores unchanged, loses 44 and gains only 5 by tiny amounts. Native BCSS06 deletes additional annotated fat, while overview clefts remain filled across many BCSS cases. Its extra support deletion is fully penalized; it is not a demonstrated general white-cleft solution.

### J04

REJECT: posterior dominant-class integration yields IoU 0.662780 at 0.2494s, below C17 in both accuracy and speed. Finite Sobol importance quadrature remains approximate. Native/overview evidence gives no consistent gain in gland separation or small-region recovery; BCSS74 has false grey and BCSS60 remains sensitive to palette confusion.

### J05

REJECT: colour-graph fraction coupling yields IoU 0.665982 at 0.2587s, lower accuracy and slower than C17. Small recall 0.183127 is also lower. Overview false-grey tissue on BCSS74/86 and retained broad clefts show that smoothness across neighbouring colours does not reliably establish anatomical similarity. Finite projected-gradient residuals remain reported.

### J06

REJECT: coarse-label log-fraction decoder gives IoU 0.665271 at 0.2094s, below C17. Small recall 0.204917 increases, but overview BCSS25/26/41/63 and TIGER02 show extra speckles and false contour classes; an additional tiny region is not necessarily recovered anatomy. Its tile-label calibration distorts an already useful palette inverse.

### J07

SELECT: expected Gaussian squared-hinge risk, IoU 0.668524 at 0.0753s,0.10996 percentage points above H10 with 77 wins/67 losses/5 ties. Native BCSS87 removes H10's false fat island, offering a fast statistical branch. It loses more of the stromal wedge in native TIGER23, and its mean small recall 0.197088 is below H10. 25/149 optimizers return unsuccessful termination, maximum gradient infinity norm 0.003051; these are finite-budget masks, not asserted converged optima.

### J08

REJECT: noise-subtracted quadratic class densities yield IoU 0.656234 at 0.0620s,1.1190 points below H10 with 104 losses/41 wins. Native BCSS74 invents a broad grey necrosis band; native BCSS45 inserts fat inside a vessel. Native/pilot BCSS26 likewise confuses vessel and fat. New C++ kernel and exact thread parity are useful engineering evidence, not justification for a less accurate model.

### J09

REJECT: additive nonlinear expected colour risk gives IoU 0.668078 at 0.2559s. J07 is more accurate, faster and has higher small-region recall.67/149 optimizations return unsuccessful flags. Overview TIGER38 still misses purple structure and BEETLE10 overmerges unannotated red regions; the extra basis complexity does not establish a useful advantage.

### J10

REJECT: proper rigid palette calibration gives IoU 0.665690 at 0.2595s. It preserves inter-endmember distances but barely improves H04 and remains below C17 with greater cost. Native BCSS45 retains vessel class, yet overview BCSS74/86 false grey and lower accuracy persist. Rigid calibration cannot explain the relevant conversion errors.

### J11

REJECT: dominance-constrained latent fractions improve H04 by 0.11957 points toIoU 0.666790, at 0.2533s. Native TIGER23 preserves more stroma than J02/J07, but overall accuracy and speed remain below J07 and C17. Native BCSS74 still inserts grey, while finite dominance-QP and outer-alternation residuals prevent calling the joint fit a certified global optimum.

### J12

REJECT: propagated endmember covariance gives IoU 0.665654 at 0.2713s, almost unchanged from H04 and slower. The interior-Gaussian uncertainty ignores active endmember bounds. Native BCSS45 vessel classification is reasonable, but the false-grey compartment in BCSS74 remains and small recall 0.182992 does not improve H04.

### J13

REJECT: shared-own-count Gaussian integration yields IoU 0.661554 at 0.1605s, below C01 while slower and with similar small recall. Independent Gaussian counts still omit multinomial anticorrelation. Overview BCSS60 false grey persists; analytical conditioning alone does not repair the coarse winner-likelihood mismatch.

### J14

REJECT: latent uniform contamination gives IoU 0.663931 at 0.1678s, only0.02242 points above C01 and approximately twice its runtime. Small recall falls to 0.197246. BCSS74 loses6.17 points relative to C01; gains in TIGER26/27 do not justify the inconsistent fit.12/149 optimizers are unsuccessful.

### J15

REJECT: eight-unit nonlinear colour-vote field gives IoU 0.662102 at 0.3077s despite higher small recall 0.219928.118/149 optimizers are unsuccessful. Overview BCSS04/25/63/78/80, CAM04 and TIGER03/38 show nested contours, false rings and fragmentation; these explain why raw recovery counts alone would reward artifacts. The nonconvex extra capacity does not preserve anatomy.

## Frozen evidence

Executed source: de01909b1c0c451d55684139e32821af065c2761.
Snapshot archive SHA256: 9a8ef921c2d13715d25b55e5606fdfa280b6e4ad4b3c58accbf8836d749a7789.
Selection SHA256: bd7696427f6368f979c499de0d632e34728e4e8c6e0231d49284b87aa9b91e76.
Local audit SHA256: b2474de99abcbd345ae6b4c4fe92892f8a8b83242cf209ef980d99ef5bd5ee13.

The exact signed selection is CYCLE_SIX_SELECTION.json. Complete local records
include REVIEW_EVIDENCE.json, VISUAL_REVIEW.json, NATIVE_VISUAL_REVIEW.json and
PILOT_FULL_NATIVE_PARITY.json under cycle-06-chiamaka.

Only selected.html is delivered, with all 149 cases selectable. The archived
all-method gallery/contact sheets remain available. The next 15 innovations
may be designed after this signed selection. The 20-cycle goal remains active.
