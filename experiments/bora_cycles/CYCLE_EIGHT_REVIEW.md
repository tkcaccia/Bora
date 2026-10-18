# Cycle8: signed review of349 images

Eight of twenty cycles are complete. All349 all-method contact sheets and14 native label crops from10 unique cases were actually inspected. Exported overlays were not counted as reviewed native crops. The selected five are J01,L02,K05,L03,C17: the five highest candidate adjusted utilities.

## Scope

Original149 inputs remain byte-exact;200 additions include40 each BCSS,TIGER,WSSS4LUAD,RINGS tumor,OCELOT. Total132 BCSS,80 TIGER,40 WSSS4LUAD,40 RINGS,40 OCELOT,10 BEETLE,7 CAMELYON16 ROIs.269 have verified pixel spacing;80 WSSS/RINGS remain pixel-only. RINGS outside-tumour is not expert stroma.30 constant RGB cases use a common coarse-completion contract, without a spatial-refinement claim. All are reused development data.

J01 mean IoU0.720017 on349 versus0.674658 on149 is a corpus-composition change. All745 historical-parent comparisons and all eight original score fields are exact.7678 native masks and display rasters,660 constant-input checks pass. Integrity is not anatomical correctness or optimizer convergence.

## Selected five

Utility is mean_i[IoU_i-0.05*max(seconds_i-1,0)], calculated per image with no subsecond bonus. Timing includes candidate preparation,background and native projection, excluding shared loading/encoder/scoring/rendering. Single low-priority elapsed measurements are not controlled speedup evidence.

| Method | Raw IoU | Mean seconds | Images >1s | Utility |
| --- | ---: | ---: | ---: | ---: |
| J01 | 0.720017 | 0.1997 | 6/349 | 0.719806 |
| L02 | 0.719998 | 0.2647 | 19/349 | 0.719096 |
| K05 | 0.720606 | 0.3211 | 29/349 | 0.718810 |
| L03 | 0.718520 | 0.2780 | 19/349 | 0.717638 |
| C17 | 0.717562 | 0.1588 | 5/349 | 0.717482 |

## Actual visual findings

Thin strands and white lumina remain unresolved. Native TIGER23 verifies false grey fringes in L07/L08/L09/L12; annotated native BCSS25 verifies false gold rims in J01/K05/L01/L04/L05/L06. C17 suppresses those rims but misses other geometry. BCSS60 favours palette fringe suppression, whereas BCSS74/86 show palette-child grey fringes. C17 uses a direct pixel inverse on281 affine cases; the children quantize occupied colours and add a prior, so these are complete-implementation comparisons, not isolated loss ablations.

L03 opens the BCSS54 cleft and native BCSS07 glass, but misses BCSS84 fissures and many gland lumina. Positive-reference deletions remain errors. Full tissue outside reference annotations is displayed and unscored. BEETLE10 demonstrates that perfect sparse-reference scores can coexist with filled unannotated lumina.

Coarse-absent classes are unrecovered in BCSS50/64/98 and TIGER16/48/50; their positive pixels stay scored. The small-region atlas has10185 components,643 with coarse-absent class,7513 with no method reaching50% coverage and351 where coarse reaches50% but all refinements fall below50%. Unknown spacing is never assigned a physical area.

## All20 decisions

### J01

SELECT: primary adjusted-utility leader. Original-H&E-conducted appearance is the best aggregate compromise. Native BCSS25 confirms false gold rims and TIGER23 still loses thin stroma; shared lumen filling remains. Retain as leader, not a uniformly correct anatomical result.

IoU 0.720017; mean 0.1997s; 6/349 over1s; utility 0.719806.

### L02

SELECT: second adjusted utility. Adds predictive appearance variance with almost unchanged aggregate IoU and some case-specific border improvements, but no gain over J01 (150 wins,155 losses,44 ties). All319 informative cases clip negative stochastic diagonal estimates; uncertainty is approximate and uncalibrated. Retain a prospectively distinct hypothesis for replacement by better numerical models.

IoU 0.719998; mean 0.2647s; 19/349 over1s; utility 0.719096.

### K05

SELECT: third adjusted utility and strongest raw accuracy among the selected five. Bending regularity preserves some narrower tissue separations that J01 merges. It is slower,29 images exceed one second, and false gold rims and vessel/fat confusion persist. No claim of dominance over J01.

IoU 0.720606; mean 0.3211s; 29/349 over1s; utility 0.718810.

### L03

SELECT: fourth adjusted utility; retains exactly J01 tissue decisions except additional H&E glass deletion. Actual BCSS54 and native BCSS07 show real cleft opening, but many internal lumina stay filled and positive-reference deletions lower IoU (12 wins,132 losses,205 ties). Keep this explicit background mechanism for improvement, not as an accuracy success.

IoU 0.718520; mean 0.2780s; 19/349 over1s; utility 0.717638.

### C17

SELECT: fifth adjusted utility and distinct direct palette inverse. Removes false gold rims in annotated native BCSS25 and grey fringes in BCSS60, but can erase small structures and still fills lumina. Its affine direct pixel inverse differs from the quantized/prior-regularized palette children; complete implementations, not loss-only ablations, are compared.

IoU 0.717562; mean 0.1588s; 5/349 over1s; utility 0.717482.

### J07

REJECT from next roster: fast Gaussian hinge model, but below the five retained utility values and shared thin-structure losses persist.35 informative-case optimizers terminate unsuccessfully. False gold rims in BCSS129 and false satellites remain despite useful individual cases.

IoU 0.716486; mean 0.0696s; 0/349 over1s; utility 0.716486.

### K13

REJECT from next roster: fast empirical-noise exponential risk lies below the selected five in accuracy/utility.22 informative-case optimizer failures and false islands/fringes remain; retained only in immutable history as a comparator.

IoU 0.716100; mean 0.0718s; 0/349 over1s; utility 0.716100.

### L01

REJECT: nonlocal H&E graph lowers parent IoU,65 wins versus252 losses and32 ties, while increasing runtime. Contact sheets show rougher and fragmented edges without recovering fine networks; stain similarity can transport appearance between unrelated regions.

IoU 0.716546; mean 0.6090s; 31/349 over1s; utility 0.704439.

### L04

REJECT: all349 native masks are byte-identical to K05; all947 constrained fits terminate at zero iterations because bounds are inactive. Extra runtime has no output benefit. A distinct mathematical hypothesis did not yield a distinct solution on this corpus.

IoU 0.720606; mean 0.4765s; 30/349 over1s; utility 0.716822.

### L05

REJECT: radial-Huber bending lowers K05 mean IoU slightly and substantially increases runtime,169 images over one second.61 cases contain a nonconverged nested fit. Fine lumen and septal errors persist; tiny local changes do not justify cost.

IoU 0.720507; mean 1.6158s; 169/349 over1s; utility 0.679668.

### L06

REJECT despite highest raw mean IoU: improvement over J01 is only0.11966 IoU percentage points at1.9414 seconds mean and258 images over one second. Covariance eigenvalue clipping occurs on all319 informative cases. Some branches improve, but many lumina and fine structures stay wrong and utility is far below survivors.

IoU 0.721213; mean 1.9414s; 258/349 over1s; utility 0.667355.

### L07

REJECT: Hellinger palette implementation lowers C17 IoU,102 wins versus215 losses, and adds grey fringes in native TIGER23 and BCSS74/86. Five cases have unconverged fits. Affine quantization and prior also differ from C17, so do not attribute failure to divergence alone.

IoU 0.714277; mean 0.2407s; 12/349 over1s; utility 0.713863.

### L08

REJECT: heavy-tailed palette fit lowers C17 IoU,97 wins versus219 losses;20 cases are not converged. Grey fringes and erased thin tissue persist. Nonconvex downweighting of large colour discrepancies has no general benefit in the complete implementation.

IoU 0.714438; mean 0.2309s; 10/349 over1s; utility 0.714205.

### L09

REJECT: minimum-entropy mixture lowers C17 IoU,100 wins versus217 losses;158 cases fail the outer convergence criterion. Purifying fractions does not recover anatomy and grey fringes persist. No proof that finite majorization reaches the optimum.

IoU 0.713728; mean 0.2332s; 10/349 over1s; utility 0.713508.

### L10

REJECT: joint-rival cubature hinge is slightly worse than J07 and below retained utility.52 optimizers terminate unsuccessfully,maximum gradient0.1103. Fine strands remain lost and false satellites/fringes persist.

IoU 0.716389; mean 0.0751s; 0/349 over1s; utility 0.716389.

### L11

REJECT: tile-specific uncertainty gains just0.01231 IoU percentage points over J07,107 wins versus183 losses.48 optimizer failures exceed parent35. It remains below all five selected utility values and does not resolve anatomical errors.

IoU 0.716609; mean 0.0711s; 0/349 over1s; utility 0.716609.

### L12

REJECT: kernel class-distribution distance lowers J07 mean IoU by0.84717 points,85 wins versus231 losses. Grey/gold fringes are visible in many BCSS/TIGER interfaces and thin structures are not recovered. Nonlinearity alone is not a useful improvement.

IoU 0.708015; mean 0.2120s; 8/349 over1s; utility 0.707905.

### L13

REJECT: upper-quarter CVaR lowers K13 mean IoU,98 wins versus215 losses.106 optimizers terminate unsuccessfully,maximum gradient0.5529. Some fringe suppression is offset by rougher islands and sensitivity to difficult coarse labels.

IoU 0.712484; mean 0.0916s; 0/349 over1s; utility 0.712484.

### L14

REJECT: row/column correlated-noise law lowers K13 IoU,109 wins versus172 losses,with24 optimizer failures. Fast output is not sufficient: false fringes and lost branches persist, and lower utility gives no survivor slot.

IoU 0.715603; mean 0.0658s; 0/349 over1s; utility 0.715603.

### L15

REJECT: equal-tile log risk has zero unsuccessful optimizer flags and is fast, but lowers K13 IoU,100 wins versus209 losses. Improved numerical stopping does not overcome lower accuracy and unresolved false islands and thin-structure losses.

IoU 0.714330; mean 0.0715s; 0/349 over1s; utility 0.714330.

## Evidence

Selection binds protocol,results and349 actual overview records. CYCLE_EIGHT_NATIVE_REVIEW.json records native crop observations/provenance; paired and numerical reports retain per-cohort differences and finite-solver flags. The frozen audit is unchanged; its review count reflects its earlier execution, before subsequent manual review.
