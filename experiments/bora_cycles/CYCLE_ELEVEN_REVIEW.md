# Cycle 11: reviewed selection on 469 calibrated tissue ROIs

Eleven of twenty cycles are complete. All469 all23-method contact-sheet overviews,25 preplanned native label sheets and2 posthoc native label sheets were actually inspected. Exported native overlays are not counted as viewed. Selected five: **P11,N08,M04,J01,C17**. P11 replaces N10. These are an explicit accuracy/runtime/visual/model-diversity selection, not the five highest numerical ranks. P10/P12 score highly but introduce severe foreground geometry artifacts; C17 preserves the fastest distinct palette branch. M04 and J01 remain visually almost identical and are retained as different mathematical branches, not a demonstrated uncertainty gain.

## Scope, calibration and integrity

The fixed469 consist of132 BCSS,80 TIGER,40 OCELOT,10 BEETLE,7 CAMELYON16 windows, and40 each IGNITE,PESO,GCSS,PUMA,GleasonXAI. All have documented per-axis physical sampling and meet the prospective multi-group/size/real-tile rules. The five RGB conversion formulas are unchanged; this cycle uses equal09+19. WSSS/RINGS and Histo-Seg images with unresolved microscopy spacing remain outside inference. Magnification, optical resolving power and pixel spacing are different quantities. Resampling does not restore missing optical detail.

Executed source5a255e0 froze all10,787 predictions before fine-reference scoring. Full native/display audit passes;1,345 unchanged-parent native comparisons and1,145 unchanged-ontology accuracy comparisons pass. Another200 OCELOT comparisons are prospectively rescored under the declared broad-negative adapter, not historical score parity. The17 overlapping pilot/full cases match391 native masks/accuracy cores,68 supports and34 rarity arrays. This is scoped repeatability, not a full independent rerun or controlled timing experiment. All frozen scores and masks are unchanged by review/publication.

Inference covers visible tissue in reference-zero regions; those pixels are unscored. All positive reference classes count, including classes absent from coarse input. PESO/OCELOT negatives are broad non-target regions, not expert stroma. PUMA annotations cover a central window. GleasonXAI uses broad sparse region polygons, which do not provide precise foreground truth everywhere.

## Scores and time

Utility is the image-average of IoU minus0.05 times seconds above1. No subsecond bonus applies. Time measures Bora from cached converted RGB, including its preparation and native projection, but excludes encoder/conversion/loading/scoring/rendering. Three nice12 inference workers use two numerical threads each. A single shared-machine invocation per image is not a controlled cross-cycle speedup measurement. The mean below one second is not an all-image guarantee.

| Method | Mean IoU | Mean seconds | Images >1s | Utility | Small recall | Fully missed small |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B23 control | 0.723733 | 0.393 | 11/469 | 0.723632 | 0.191759 | 9129 |
| P11 | 0.722393 | 0.424 | 19/469 | 0.722216 | 0.192429 | 9121 |
| P10 | 0.722349 | 0.447 | 17/469 | 0.722133 | 0.193947 | 9072 |
| P12 | 0.721836 | 0.429 | 32/469 | 0.721260 | 0.194135 | 9098 |
| N08 | 0.714123 | 0.385 | 23/469 | 0.713619 | 0.206970 | 9051 |
| M04 | 0.713193 | 0.202 | 0/469 | 0.713193 | 0.193908 | 9120 |
| J01 | 0.713193 | 0.183 | 0/469 | 0.713193 | 0.193801 | 9116 |
| C17 | 0.710460 | 0.142 | 0/469 | 0.710460 | 0.190774 | 9301 |

B23 is a diagnostic control outside the selectable candidate roster. Its mean IoU and utility exceed every candidate; no improvement over the best control is established. P11 gains0.009199681 mean IoU versus the paired current-cohort J01 outputs (218 wins,56 losses,195 ties), but this includes foreground-support/annotation effects. It has lower mean small-region recall. GleasonXAI mean IoU is0.847518 for P11 versus0.802365 for J01, while PESO falls to0.815349 from0.820370. These are reused development data, not evidence of statistically significant independent generalization.

## What the actual images reveal

Native support restores some reference-positive pale adipocyte spaces in TIGER47/53/57/60/79/80. It also fills actual exterior glass, often dramatically in CAMELYON06/07,PESO12/26/38,PUMA11 and GleasonXAI05/08/20/26/40. P10 creates angular spikes; P12 creates rounded rectangular blankets and can remove valid pale borders in BCSS71/76/80. P11 generally avoids those geometric extremes but still leaks. It is a retained unresolved research branch, not validated background removal.

Background colour alone is insufficient. Three posthoc visible-background rectangles document the photometric problem, not independent foreground ground truth. A reproducible raw-source check confirms TIGER71 has a selected all-white rectangle annotated100%stroma; frozen scores reward filling it. Native TIGER11 review instead shows an actual empty ductal lumen included in the in-situ tumour semantic label. Exterior glass, adipose spaces and semantic lumens cannot be treated as one white-pixel category.

GleasonXAI11/12/29/35/40 show why polygon overlap and anatomical foreground can disagree: annotation polygons span real tears, lumens and clear exterior regions. Native filling raises overlap while destroying visible gaps. The original labels remain unchanged. Reference-zero visible tissue is still inferred and must not be mistaken for glass simply because it is unscored.

Fine architecture remains weak. PUMA27 merges red nests across thin green septa, PUMA19 breaks a continuous green channel, OCELOT21/31 merge gland walls, and IGNITE papillae remain fused. PUMA40 preserves blue when it exists in coarse labels; absent-coarse blue vessels in many other PUMA fields are missed by every closed-class method. GCSS18 similarly misses absent grey islands. BCSS75 and IGNITE33 expose interior semantic competition that foreground support cannot fix. GCSS32 and TIGER52 show transformed-RGB mottling turning broad regions into small islands.

P03 repeatedly destroys broad compartments into dominant-class nuclear-scale speckle. P15 sometimes extends true fine linings in PESO18/38 or red strips in PUMA23, but frequently invents rare-colour halos and overexpands classes in GCSS24/30/37 and PUMA09/12/39/40. RGB rarity maps are appearance diagnostics, not discoveries of biological tissue types.

All11,768 small components remain assessed:948 have a class absent from coarse labels,118 cannot reach50% under common support,7,195 are below50% for every method,6,171 are wholly missed by all, and243 recovered by coarse labels fall below50% in every refinement. These are roster-union diagnostics, not per-method recalls; the common-support bound does not bind native-support variants.

## Numerical limits and next cycle

P01 misses its declared convergence criterion in469/469 finite runs (maximum gradient0.0296121); P13 misses in469/469 (maximum projected residual0.0243154), costs14.556s on average and is penalized on every image. P14 misses stationarity in136/469 (maximum0.00837574). Finite masks are scored honestly without calling them optima. C++ correctness checks concern their discrete kernels, not biological validity or end-to-end speedup.

Nine cycles remain. Next design must produce15 distinct mathematical children, three per signed survivor, rather than parameter sweeps or ports. Priorities include separating photometric glass from semantic lumens, recovering fine regional connectivity and handling interior colour ambiguity without further complexity. Every next cycle retains all469 calibrated fields and unchanged RGB formulas, then requires full scoring, runtime penalties and actual visual review. The final all-five-fixed-conversion analysis remains outstanding.

## All twenty decisions

### J01: H&E edge-conducted appearance

SELECT as the fastest high-accuracy appearance-field reference: mean IoU 0.713193, mean 0.183s, no case over one second in this run. M04 is visually almost identical. It follows exterior glass more faithfully than native-envelope variants in GleasonXAI, PESO and PUMA, but deletes some valid pale tissue and merges thin gland walls; false palette fringes persist. Retain its simple H&E-conducted field and speed, not an accuracy advance or universal one-second guarantee.

Raw mean IoU 0.713192969; mean 0.1827s; 0/469 over1s; utility 0.713192969.

### M04: Positive sampling uncertainty

SELECT as the positive sampling-uncertainty branch: mean IoU 0.713193, mean 0.202s, no case over one second. Its IoU differs from J01 by only 0.000000206 and the actual masks are nearly identical, so this is mathematical branch retention, not demonstrated benefit from uncertainty. Fine septa, absent-coarse classes and common-support pale-tissue deletion remain unresolved. P04/P05/P06 do not give a convincing quality/runtime improvement.

Raw mean IoU 0.713193175; mean 0.2021s; 0/469 over1s; utility 0.713193175.

### N08: Positive log-variance transport

SELECT as the best common-support candidate under the frozen utility: mean IoU 0.714123, adjusted 0.713619, mean 0.385s, 23 over-budget images. Some native interfaces and mixed-colour fields improve, but many overviews are almost unchanged; GleasonXAI28 remains mottled and PUMA thin channels still break. Small-region recall 0.206970 remains poor. Positive log-variance transport is retained as a distinct likelihood with a measured speed/accuracy tradeoff, not dominance on every image.

Raw mean IoU 0.714123146; mean 0.3850s; 23/469 over1s; utility 0.713618792.

### N10: Convex tissue envelope

REJECT this common-support convex-envelope branch. Mean IoU 0.713186 and adjusted 0.713182 are slightly below J01/M04, with mean 0.212s and one over-budget image. Most full-tissue masks are nearly identical to those parents, and it cannot repair tissue removed before its envelope acts. P11 tests native support directly and replaces this branch, with its foreground leakage explicitly retained as a limitation.

Raw mean IoU 0.713186233; mean 0.2121s; 1/469 over1s; utility 0.713182482.

### C17: Converged palette active set

SELECT the fast palette active-set branch for its distinct forward model and practical behaviour on mixed RGB: mean IoU 0.710460, mean 0.142s, no over-budget image. This sacrifices aggregate utility relative to P10/P12/P09/P02. Actual BCSS interfaces and IGNITE33/GCSS32 show cases where a simpler palette decision avoids field-induced mottling or false colour bands. It still merges glands and misses absent-coarse classes; cleaner selected examples do not establish overall accuracy superiority.

Raw mean IoU 0.710460258; mean 0.1423s; 0/469 over1s; utility 0.710460258.

### P01: Ordered appearance fields

REJECT the budgeted ordered-field ADMM implementation. Mean IoU 0.714061 brings little visual change over J01, while 142 images exceed one second and adjusted utility falls to 0.689956. Every one of 469 finite runs misses the declared convergence criterion; the maximum gradient residual is 0.0296121. Scores refer to those finite outputs, not optimized solutions. Added constraints and runtime do not resolve thin-channel and small-class failures.

Raw mean IoU 0.714060905; mean 1.0854s; 142/469 over1s; utility 0.689955532.

### P02: Fractional spatial transport

REJECT despite a small aggregate utility advantage over C17: mean IoU 0.710574, mean 0.184s and no over-budget image. Fractional transport usually yields masks close to C17/J01 while retaining their merged glands, misplaced small islands and common-support deletion. Retain C17's faster and simpler palette branch for the fifth diversity slot. This is an explicit speed/model-choice tradeoff, not proof that P02 is numerically worse than C17.

Raw mean IoU 0.710573984; mean 0.1835s; 0/469 over1s; utility 0.710573984.

### P03: Adaptive stain components

REJECT: stain-component regions frequently collapse broad tissue compartments to the dominant class plus nuclear-scale speckle. PUMA16/24/34 and GleasonXAI05/11/26 make this failure unmistakable. Mean IoU is 0.356613, adjusted 0.356447; its better small-component recall does not compensate for destroyed broad architecture. Exact C++ component operations certify the discrete operation, not biological segmentation or useful full-pipeline acceleration.

Raw mean IoU 0.356613440; mean 0.3754s; 16/469 over1s; utility 0.356447181.

### P04: Uncertain nearest-appearance winner

REJECT: the 32-probe uncertain-winner decision produces almost the same mask as M04/J01 on most cases. Mean IoU 0.713200 is only a tiny change, but mean 2.330s and 377 over-budget images reduce utility to 0.644515. Independent local Gaussian sampling does not restore PUMA blue classes or narrow gland separations. Numerical perturbation expense has not produced a useful visual or accuracy improvement.

Raw mean IoU 0.713199544; mean 2.3302s; 377/469 over1s; utility 0.644515314.

### P05: Expected absolute colour error

REJECT: expected absolute colour error has mean IoU 0.712621 and utility 0.711424, below M04/J01, with 30 over-budget cases. Many overviews are almost identical; others broaden an incorrect class patch, as at PUMA17 upper-left. The changed loss does not solve shared small-region omissions, fragmented septa or background support. Its different formula is insufficient without measured practical benefit.

Raw mean IoU 0.712621074; mean 0.3983s; 30/469 over1s; utility 0.711424095.

### P06: Gaussian affinity decision

REJECT: Gaussian affinity yields mean IoU 0.713191, nearly indistinguishable from M04/J01, but mean 0.259s and five over-budget cases lower utility to 0.713150. Actual masks retain the same false fringes and merged structures across PUMA, PESO and GCSS. It adds density comparison without a demonstrated quality improvement over the simpler retained appearance branches.

Raw mean IoU 0.713191121; mean 0.2594s; 5/469 over1s; utility 0.713150306.

### P07: Full matrix covariance transport

REJECT despite mean IoU 0.716366, higher than N08. Full SPD covariance transport occasionally improves boundaries, including GleasonXAI30, but mean 1.942s and 393 over-budget cases reduce utility to 0.667278. Most actual masks remain close to faster methods and fine channels are still lost. This expressly honours the one-second penalty rather than choosing raw overlap alone.

Raw mean IoU 0.716366097; mean 1.9423s; 393/469 over1s; utility 0.667278334.

### P08: Derivative-based density decision

REJECT despite local improvements in GleasonXAI28/30. Mean IoU 0.712728 and utility 0.711642 are below N08; 28 cases exceed one second. Derivative-density decisions change island widths but often retain the same fragmented compartments and can overexpand red, for example in PUMA30. It is a real alternative objective with insufficient cohort benefit at its observed runtime.

Raw mean IoU 0.712728110; mean 0.4297s; 28/469 over1s; utility 0.711641912.

### P09: Shared latent noise scale

REJECT: mean IoU 0.713745 is slightly above J01/M04, but utility 0.712583 is lower after 28 over-budget cases and mean 0.456s. Shared latent noise scale changes small islands and can improve part of a contour, yet misses the same fine PUMA vessels/channels and weakens broad compartments in other fields. N08 has better aggregate IoU and utility among these likelihood alternatives.

Raw mean IoU 0.713744767; mean 0.4561s; 28/469 over1s; utility 0.712582632.

### P10: Alpha-complex tissue envelope

REJECT despite second-best selectable utility 0.722133 and mean IoU 0.722349. Alpha-complex support creates conspicuous angular glass spikes and polygon fills in BEETLE03/08, GleasonXAI13/28/39/40 and other cases. Its high scores partly reward coverage under broad semantic polygons rather than reliable foreground anatomy. P11 retains the native-support branch with fewer angular artifacts, though it still leaks. Mean 0.447s includes 17 over-budget cases.

Raw mean IoU 0.722349124; mean 0.4473s; 17/469 over1s; utility 0.722133292.

### P11: Probabilistic stain occupancy

SELECT the mandatory selectable utility leader: mean IoU 0.722393, adjusted 0.722216, mean 0.424s, 19 over-budget images. Native probabilistic stain occupancy restores some labelled pale adipose tissue and generally avoids P10 triangular spikes. It also fills substantial real glass in CAMELYON/PESO/PUMA/GleasonXAI, including PUMA11 and GleasonXAI40; small-region recall is only 0.192429. Broad polygons sometimes reward that filling. This is an unresolved research branch, not validated background removal, and the B23 diagnostic control still scores higher.

Raw mean IoU 0.722392650; mean 0.4243s; 19/469 over1s; utility 0.722216046.

### P12: Diffusive glass residence

REJECT despite third-best selectable utility 0.721260 and mean IoU 0.721836. Diffusive glass residence often creates broad rounded rectangular foreground blankets over clear glass, especially GleasonXAI05/08/20/26/40, while also cutting some valid pale borders in BCSS71/76/80. Score gains do not resolve this conflicting foreground behaviour. Mean 0.429s includes 32 over-budget images. P11 is the retained native-support experiment, with its own limitations disclosed.

Raw mean IoU 0.721835952; mean 0.4285s; 32/469 over1s; utility 0.721259511.

### P13: Spatially coupled mixture field

REJECT this finite 120-step spatial-mixture TV solver: mean IoU 0.707518 and mean 14.556s give utility 0.029699; every image exceeds one second. All 469 runs miss the stated convergence criterion (maximum projected residual 0.0243154). Smooth-looking masks generally retain the same missing structures as fast C17 and do not justify the cost. This rejects the measured implementation, not the possibility of a better TV optimizer.

Raw mean IoU 0.707517912; mean 14.5564s; 469/469 over1s; utility 0.029698778.

### P14: Log-ratio palette observation

REJECT: log-ratio palette inversion has mean IoU 0.706798 and adjusted 0.706736, below C17, with five over-budget cases. Some interfaces smooth, but fragmented broad regions and gland mergers remain. 136 of 469 runs miss projected stationarity tolerance (maximum 0.00837574). Artificial converted RGB channel proportions are not validated physical compositional measurements, and this tested change offers no convincing aggregate advantage.

Raw mean IoU 0.706797782; mean 0.2567s; 5/469 over1s; utility 0.706736146.

### P15: Negative-class tail competition

REJECT despite occasional recovery of fine epithelial linings in PESO18/38 and some PUMA23 strips. Negative-tail competition often overexpands rare colours into false halos or islands, notably PUMA09/12/39/40 and GCSS24/30/37. Mean IoU 0.689847 and adjusted 0.689735 are poor; six cases exceed one second. Local tail response is not reliable new-class discovery, and all closed-class methods still miss classes absent from coarse labels.

Raw mean IoU 0.689846744; mean 0.2992s; 6/469 over1s; utility 0.689734541.

