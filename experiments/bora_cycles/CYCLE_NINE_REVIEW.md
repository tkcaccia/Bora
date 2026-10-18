# Cycle 9: signed review of 349 images

Nine of twenty cycles are complete. All 349 all-method contact sheets and nine native label crops were actually inspected. Exported overlays are not counted as reviewed native crops. Selected: J01, M04, K05, M11, C17. J01 is the mandatory utility leader. The other four preserve distinct uncertainty, bending, background and palette hypotheses; they are **not the five highest numerical utility ranks**. Every tradeoff is explicit below.

## Scope and execution

The frozen 349 ROIs comprise 132 BCSS,80 TIGER,40 WSSS4LUAD,40 RINGS tumour,40 OCELOT,10 BEETLE and7 CAMELYON16 subset images. All original149 are retained exactly.269 have verified spacing;80 WSSS/RINGS are pixel-only.30 constant RGB cases use shared coarse completion without a refinement claim. RINGS outside-tumour is not expert stroma. All images are reused development data, with documented group-overlap limitations; this is not independent validation.

Source ea8ce72608f172d53350e3ee1aee354406c25a12 was frozen before scoring. All7678 native masks and exact display rasters,1745 historical-parent native checks,1745 historical-parent accuracy checks and660 constant-contract checks passed, with zero failed predictions. Full posthoc evidence is preserved unchanged. Integrity checks do not establish biological accuracy.

## Selected five and speed policy

Utility is mean_i[macro_IoU_i -0.05*max(seconds_i-1,0)], applying the excess per image before averaging, with no subsecond bonus. Candidate preparation,background and native projection are timed; shared encoder/loading/scoring/rendering are separate. One elapsed measurement per case in a shared low-priority3-worker run is not controlled cross-cycle speedup evidence.

| Method | Raw IoU | Mean seconds | Images >1s | Utility |
| --- | ---: | ---: | ---: | ---: |
| J01 | 0.720017 | 0.2041 | 2/349 | 0.719998 |
| M04 | 0.720016 | 0.2175 | 3/349 | 0.719991 |
| K05 | 0.720606 | 0.3277 | 28/349 | 0.719837 |
| M11 | 0.719130 | 0.2536 | 2/349 | 0.719011 |
| C17 | 0.717562 | 0.1594 | 0/349 | 0.717562 |

M04's variance is positive by construction but uncalibrated. Its near-equality with J01 is not a meaningful accuracy advance. M11 improves L03 mean IoU by0.06105 percentage points despite58 wins versus219 losses and72 ties; the mean benefit is concentrated in a few tissue-preservation cases. It still loses to J01, and straight-ray artifacts remain. M09 has the highest raw IoU but its90 slow cases outweigh the small gain under the fixed utility.

## Actual visual findings

Native BCSS01 exposes false gold rims in the field methods, reduced by C17 but accompanied by other palette losses. Native BCSS20 and the BCSS58 overview show M14 deleting genuine grey tissue. Native TIGER23 shows M13/M15 grey halos and continuing thin-tissue loss. Native WSSS13 shows a local red-rim improvement in K05/M09, whereas WSSS22 loses green arms under M09. OCELOT21/31/37 demonstrate merged glands despite high pixel overlap; OCELOT12/32 show M03's false hollow tumour rings.

Native TIGER28 shows M11 preserving fat interiors deleted by L03/M10; native CAMELYON01 shows M12 lattice-like fat deletion. RINGS22 removes labelled tumour lumina under L03/M12; RINGS25 shows a major shared bright-lumen false-negative problem. These positive-reference deletions remain scored errors. Constant negative RINGS cases are controls, not successful architecture reconstruction. BEETLE06 loses a coarse-absent grey class; BEETLE10's almost-perfect sparse annotation score does not validate unannotated tissue. All tissue is analyzed, while absent annotations remain unscored.

The common small-region atlas contains10185 components,643 with coarse-absent classes,7490 for which no method reaches50% coverage,6643 completely missed by all, and289 with coarse coverage at least50% but every refinement below50%. These are diagnostic comparisons within this method set, not a normalized accuracy claim or independent validation. The optimistic input-support IoU bound is not generally attainable and is not used to rescale scores.

## Numerical limitations

M09 has three explicit nonstationary finite solves (max gradient4.28336e-5). M15 has100 finite-cap cases (max primal5.46167e-5,dual7.39931e-5,Lagrangian projected residual5.08314e-5). L02/M05/M06 inherit clipped stochastic inverse-diagonal estimates; their uncertainty is not calibrated. M04 avoids negative estimates with a different positive sampling-variance model, which is also assumed and approximate. Palette children use occupied-colour aggregation/prior terms unlike C17's affine direct-pixel branch; comparisons are complete implementations rather than isolated loss ablations.

## All twenty decisions

### J01

SELECT: mandatory primary runtime-adjusted leader, IoU 0.720016592 and utility 0.719998232. Original-H&E-conducted appearance remains the strongest aggregate compromise. Native BCSS01 still shows false gold fringes, TIGER23 loses thin tissue, and OCELOT21/31 merge gland profiles. Retain the leader without claiming anatomical correctness on every image.

IoU 0.720017; mean 0.2041s; 2/349 over1s; utility 0.719998.

### L02

REJECT from next roster: M04 has slightly higher raw IoU and adjusted utility while using a positive sampling-variance law instead of the clipped approximate inverse-system diagonal. M05 also improves L02 raw IoU but costs more. Keep the signed historical implementation for reproducibility, not as another near-duplicate survivor.

IoU 0.719998; mean 0.2631s; 6/349 over1s; utility 0.719859.

### K05

SELECT: strongest raw IoU among these five and a distinct bending-field hypothesis; small-region recall 0.2161 versus J01 0.1985 on calibrated cases. Native BCSS01 also exposes a false gold rim and TIGER23 remains wrong. Its 28 images over one second are penalized; it does not beat J01 utility. M07/M08/M09 do not justify their additional runtime.

IoU 0.720606; mean 0.3277s; 28/349 over1s; utility 0.719837.

### L03

REJECT from next roster: M11 preserves valid fat in native TIGER28 and avoids the large labelled-tumour lumen deletion in RINGS22. L03 still opens the real BCSS07/54 clefts, but its broader glass removal deletes annotated tissue elsewhere. M11 has higher aggregate IoU/utility, although its ray artifacts and many case regressions remain documented.

IoU 0.718520; mean 0.2791s; 6/349 over1s; utility 0.718251.

### C17

SELECT as the fastest distinct palette inverse: 0.15936 seconds mean, zero images over one second, with checked native inverse residuals. Native BCSS01 and BCSS60 favor its suppression of false fringes. It still erases or merges some small regions and loses fine gland topology. All three palette children have lower raw IoU and utility; direct affine pixel inversion differs from their occupied-node/prior approximations.

IoU 0.717562; mean 0.1594s; 0/349 over1s; utility 0.717562.

### M01

REJECT: latent true-class EM reduces J01 IoU by 0.002569992 with 132 wins,185 losses,32 ties and 43 images over one second. It produces fragmented or false islands, including WSSS22, without restoring absent tissue classes. Monotone aggregated-observation EM energy does not ensure anatomical accuracy or convergence after eight updates.

IoU 0.717447; mean 0.6105s; 43/349 over1s; utility 0.712013.

### M02

REJECT despite high adjusted rank: exact first-moment conservation changes little, loses mean IoU versus J01 with154 wins,163 losses,32 ties, and retains the same fringe and topology failures. Its measured mean time is slightly lower but not controlled speedup evidence. M04 retains a more useful uncertainty intervention without taking multiple near-duplicate field slots.

IoU 0.719989; mean 0.1959s; 2/349 over1s; utility 0.719943.

### M03

REJECT: series-resistance stain transport lowers J01 IoU by 0.002806643 with117 wins versus200 losses. Actual OCELOT12/32 and CAMELYON01 show false hollow tumour rings; BCSS/TIGER show repeated rough outlines. Subgrid nuclear barriers create artificial holes rather than recover the fine true gland/stromal network.

IoU 0.717210; mean 0.2195s; 2/349 over1s; utility 0.717111.

### M04

SELECT: second adjusted utility, replacing L02 with positive observation-noise sampling variance. It removes the need to clip negative inverse-diagonal estimates in this new variance law; finite probes and assumed noise remain uncalibrated. IoU is effectively J01-level, not a meaningful accuracy gain. Retain this numerically coherent uncertainty branch rather than several near-identical field candidates.

IoU 0.720016; mean 0.2175s; 3/349 over1s; utility 0.719991.

### M05

REJECT narrowly: Student predictive density improves L02 raw IoU by only0.000227619 (0.02276 percentage points),169 wins versus146 losses, with9 images over one second. Native WSSS22 improves some green arms but BCSS60 regresses and fine glands remain lost. M04 has higher adjusted utility and avoids inherited inverse-diagonal clipping; no broad gain over J01 is established.

IoU 0.720225; mean 0.2810s; 9/349 over1s; utility 0.719951.

### M06

REJECT: exact expected squared appearance risk is slightly worse than L02 in raw IoU and materially lower in adjusted utility. It retains the inherited approximate uncertainty estimator, with7 images over one second. Native TIGER23 and the complete overviews show no recovery of the thin tissue partitions that would justify another similar field branch.

IoU 0.719992; mean 0.2683s; 7/349 over1s; utility 0.719696.

### M07

REJECT: free-boundary Hessian regularity gains just0.000074210 raw IoU over K05 (0.00742 points),169 wins versus145 losses, but lowers utility by0.000719290. It slightly raises small-region recall without resolving false gold rims or merged glands.29 images over one second and near-parent masks do not justify replacing K05.

IoU 0.720680; mean 0.3542s; 29/349 over1s; utility 0.719118.

### M08

REJECT: within-tile Hermite observations lower K05 raw IoU,116 wins versus200 losses, and cost more.29 images exceed one second. Native TIGER23 and OCELOT gland examples remain structurally wrong; reusing observed gradients does not restore the lost fine networks.

IoU 0.720580; mean 0.3539s; 29/349 over1s; utility 0.719162.

### M09

REJECT despite highest raw IoU0.722214629: cross-class repulsion gains0.001609074 over K05 but90 images exceed one second and adjusted utility drops0.009188170. Native WSSS13 improves one red rim while WSSS22 regresses; BCSS01 false gold and gland merges persist. Three finite solves miss stationarity, maximum gradient4.28336e-5. Accuracy gains do not justify cost under the frozen penalty.

IoU 0.722215; mean 0.7881s; 90/349 over1s; utility 0.710649.

### M10

REJECT: integrated-stain exterior paths slightly lower L03 raw IoU (77 wins,90 losses,182 ties). It deletes valid fat in native TIGER28 and narrows tissue despite some useful cleft access. Lower mean runtime than L03 is not sufficient; M11 preserves more labelled tissue and has higher utility.

IoU 0.718458; mean 0.2428s; 2/349 over1s; utility 0.718377.

### M11

SELECT as the explicit background branch, replacing L03: mean IoU rises 0.000610519 and utility 0.000760207 versus L03. Native TIGER28 shows preservation of valid fat interiors that L03 deletes, and RINGS22 avoids a large luminal deletion. However it loses on 219 cases versus 58 wins and creates straight-ray artifacts; this is a deliberate tissue-preservation tradeoff, not broad dominance or a top-five-utility claim.

IoU 0.719130; mean 0.2536s; 2/349 over1s; utility 0.719011.

### M12

REJECT: global glass mincut lowers L03 IoU by0.001540780 and has the most additional annotated unassignment among the glass candidates. Native CAMELYON01 shows lattice-like fat holes; RINGS22 deletes central tumour lumen and upper gland lumina. A cut certificate proves optimization of its discrete energy, not correctness of the tissue/glass interpretation.

IoU 0.716979; mean 0.2529s; 5/349 over1s; utility 0.716856.

### M13

REJECT: byte-interval deadzone inverse lowers C17 IoU by0.003084567 with105 wins versus212 losses and adds grey halos in native TIGER23 and BCSS examples. Half-byte modeling does not fix larger conversion/identifiability errors. Occupied-node quantization and prior also differ from C17, so this is a complete-implementation comparison.

IoU 0.714477; mean 0.2591s; 3/349 over1s; utility 0.714344.

### M14

REJECT: counterfactual class necessity lowers C17 IoU by0.008055151,88 wins versus229 losses. It deletes true grey tissue, dramatically in BCSS58 and native BCSS20, despite suppressing some false fringes. Palette vertices inside rival hulls can be unnecessary even when biologically present. Higher small-region recall does not compensate for systematic class loss.

IoU 0.709507; mean 0.2211s; 2/349 over1s; utility 0.709463.

### M15

REJECT: robust calibration inverse lowers C17 IoU by0.002318870,115 wins versus202 losses, and adds native TIGER23 grey halos.100 cases reach the finite ADMM cap without all residual criteria; max primal5.46167e-5,dual7.39931e-5,Lagrangian projected residual5.08314e-5. Robustness and checked inner QP residuals do not establish the outer solution or improved anatomy.

IoU 0.715243; mean 0.3384s; 8/349 over1s; utility 0.715178.

## Evidence

SELECTION.json binds the unchanged protocol,results and349 actual overview records. Native provenance and explicit observations are in CYCLE_NINE_NATIVE_REVIEW.json. CYCLE_NINE_PAIRED_ANALYSIS.json and CYCLE_NINE_NUMERICAL_STATUS.json retain the paired score and solver evidence. Frozen audit review counts refer to their earlier execution and are not retrospectively rewritten. Cycle10 is not implemented by this selection.
