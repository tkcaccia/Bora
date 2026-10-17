# Cycle 7: signed review and five survivors

Seven of twenty cycles are complete. All149 all-method overviews and eight native label crops were actually inspected. Three pilot crops are additional engineering evidence, not substituted for full-run review. All3278 native masks and745 unchanged-parent comparisons pass both audits. No cycle-eight candidate was implemented before this selection.

The selected five are **J01, K05, C17, J07 and K13**. No new child exceeds J01's overall accuracy. K05 replaces H02 with a small paired gain and complementary tumour-band preservation. K13 replaces H10 with a small paired gain and corrected vessel/fat confusion in BCSS87, at increased optimizer failures and cost. These repeatedly inspected images remain development data; paired differences are descriptive and do not establish independent validation or statistical significance.

| Method | Mean IoU | Mean seconds | Images >1s | Adjusted utility | Small-region recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| J01 | 0.674658 | 0.2008 | 0/149 | 0.674658 | 0.184554 |
| K05 | 0.672922 | 0.3080 | 0/149 | 0.672922 | 0.202911 |
| C17 | 0.671604 | 0.1681 | 0/149 | 0.671604 | 0.183608 |
| J07 | 0.668524 | 0.0735 | 0/149 | 0.668524 | 0.197088 |
| K13 | 0.668351 | 0.0714 | 0/149 | 0.668351 | 0.198487 |

The frozen utility subtracts .05 per second above one second on each image before averaging. It gives no bonus below one second. Timings include candidate preparation and native projection, exclude shared loading/scoring/rendering, and use one measurement per method/image with three low-priority image workers and two numerical threads each. Millisecond differences are not controlled speed claims.

## What the native review changes

Gold colours encode different classes: BCSS fat and blood-vessel colours are close. Native BCSS45 and40 reveal errors almost invisible in the overview. J01 preserves vessels better in BCSS45 yet loses a tumour band in BCSS53; K05 retains that band but misclassifies vessel interior. J07/K13 avoid H10's false-fat interior in BCSS87. Every model truncates the TIGER23 stromal wedge, and every displayed model colours the large BCSS07 bright cleft. Reference-zero portions of that cleft remain unscored. Background, missing coarse classes and fine branches are unresolved.

The input-limit audit finds56/149 cases with a reference class absent from coarse and4,079,674 positive reference pixels outside frozen support. Positive reference pixels remain scored; no optimistic score-cap normalization is used. Full native notes and provenance are in CYCLE_SEVEN_NATIVE_REVIEW.json; post-scoring contrast-selected crops are labelled explicitly. Exported overlays were not included in the eight reviewed label-sheet count.

## Every candidate decision

### J01

SELECT: primary adjusted-utility leader, mean IoU 0.674658 at 0.201 seconds; every image is below one second. Native BCSS45 largely avoids the vessel-to-fat error and TIGER23 preserves more stroma. Native BCSS53 loses a tumour band and BCSS21 fragments true fat; small-region recall is only 0.184554. Keep as the overall leader, not a uniform anatomical winner.

### H02

REJECT: unchanged screened-Laplace parent gives IoU 0.671889 at 0.195 seconds, with small recall 0.192574. K05 improves both aggregate IoU and small-region recall, retains the native BCSS53 tumour band, and stays subsecond on every image, although it costs more time. H02 remains archived as a comparator; replacing it does not mean K05 wins every case.

### C17

SELECT: direct palette-mixture inverse remains IoU 0.671604 at 0.168 seconds with no slow cases. Native BCSS45/87 retain vessel identity without the prominent false-fat interiors of several statistical models. It avoids some broad false-grey expansions in overviews. Native BCSS21 shows fat/vessel confusion at a thin edge and it still merges fine glands. Retain a structurally distinct direct inverse without adding a fitted colour classifier.

### J07

SELECT: Gaussian-integrated squared-hinge model gives IoU 0.668524 at 0.0735 seconds and small recall 0.197088. Native BCSS87 avoids H10's false fat island and BCSS21 retains the fat region that J01 fragments. Native TIGER23 still loses part of the stromal wedge. 25/149 optimizers terminate unsuccessfully, maximum gradient infinity norm 0.003051; retain finite-budget predictions without claiming convergence.

### H10

REJECT: unchanged Gaussian exponential-risk parent is fastest at 0.0569 seconds and small recall 0.199198, but IoU 0.667425 is below K13. Native BCSS87 has a false-fat island corrected by K13/J07. Its eight unsuccessful optimizer flags are fewer than K13's 18; that stability and its small-recall advantage remain explicit limitations of the replacement, not hidden.

### K01

REJECT: orientation-guided appearance field gives IoU 0.672422 at 0.225 seconds, 0.22357 percentage points below J01 with 58 wins, 89 losses and two ties. It retains the native BCSS53 tumour band but reintroduces a large false-fat interior in the BCSS45 vessel. K05 has slightly higher aggregate accuracy and small recall; no additional slot is justified by orientation alone.

### K02

REJECT: covariance-coupled vector field nearly matches J01, IoU 0.674566 versus 0.674658, but costs 0.278 versus 0.201 seconds and loses 82 cases versus 65 wins. Small recall gains only 0.001813. Native BCSS53 partially improves the tumour band but leaves a false stromal hole; BCSS21 still fragments true fat. A near-duplicate frontier point is not preferred over distinct retained mechanisms.

### K03

REJECT: lognormal appearance likelihood gives IoU 0.669758 at 0.214 seconds, 0.48994 percentage points below J01, with 42 wins and 105 losses. Some TIGER outlines improve, but BCSS52 loses 7.03 points and BCSS26 loses 3.81. Native/overview evidence does not show a general structural benefit from treating synthetic RGB logarithmically.

### K04

REJECT: vector-TV appearance yields a small raw gain over H02, IoU 0.672194, but takes 0.677 seconds and exceeds one second on 21 images; adjusted utility falls to 0.671074. 148/149 cases contain an unconverged fit, maximum gradient infinity norm 0.013681. K05 is more accurate and cheaper without slow cases. Fine branches and white clefts remain unresolved.

### K05

SELECT: biharmonic appearance field improves H02 by 0.10329 IoU percentage points, with 94 wins, 53 losses and two ties. Mean IoU 0.672922 at 0.308 seconds, zero images above one second; small-region recall rises from H02's 0.192574 to 0.202911. Native BCSS53 retains the tumour band lost by J01, but BCSS26/45 misclassify vessel interior as fat. This is a complementary appearance model, not an improvement over J01.

### K06

REJECT: positive Poisson quasi-deviance gives IoU 0.667472 at 0.571 seconds, with ten images over one second. It loses 0.44168 points versus H02 and visibly introduces false-grey expansion/lower stripes in several BCSS/TIGER cases. Fits meet their reported gradient condition, but converted RGB is not Poisson count data and numerical convergence does not validate the biological model.

### K07

REJECT: maximum-channel-error palette inverse gives IoU 0.665668 at 0.248 seconds, 0.59365 points below C17, with 44 wins and 103 losses. Overviews show false-grey expansion in BCSS74/78 and loss of the TIGER23 wedge. 126/149 cases contain nodes failing the frozen gap tolerance; maximum primal-dual gap is 1.068e-5. Two-thread kernel scaling is engineering evidence, not segmentation improvement.

### K08

REJECT: tile-specific covariance metric gives IoU 0.662708 at 0.344 seconds, below C17 in both accuracy and speed, with 36 wins and 112 losses. BEETLE08 improves on sparse annotation, but BCSS74/78 produce false-grey regions and TIGER32 loses red strands. Local covariance can encode tissue mixture rather than observation noise.

### K09

REJECT: harmonic occupied-colour coordinates give IoU 0.659283 at 0.243 seconds, 1.23206 points below C17 with 46 wins and 102 losses; small recall falls to 0.168395. TIGER07/23/37 and BCSS74/78 show broad false compartments or erased wedges. A well-solved colour graph does not establish anatomical connectivity.

### K10

REJECT: ellipsoidal worst-case hinge gives IoU 0.667130 at 0.0689 seconds, below J07 by 0.13941 points with 48 wins, 87 losses and 14 ties. Native BCSS40 labels vessel pixels as fat even though overview gold hues look similar. 37/149 optimizers are unsuccessful, maximum gradient infinity norm 0.072578. The uncertainty ball does not provide a robust improvement.

### K11

REJECT: analytic Gaussian quadratic Brier regression gives IoU 0.661844 at 0.223 seconds, below J07 by 0.66803 points with 43 wins and 105 losses. Small recall rises to 0.206435 but overviews add false islands and deform TIGER17's isolated gland. Exact moment/normal-equation calculations do not make the tile-mean regression a valid pixel plurality model.

### K12

REJECT: joint Gaussian logistic cubature gives IoU 0.664449 at 0.0860 seconds, below J07 while slower, with 32 wins and 111 losses. All optimizer flags are successful, but overviews/native-vessel comparisons still show false-fat structures and BCSS87 loses 5.90 points versus J07. The finite quadrature is not a certified exact expectation.

### K13

SELECT: empirical within-tile residual exponential risk improves H10 by 0.09265 IoU percentage points, 73 wins, 64 losses and 12 ties. IoU 0.668351 at 0.0714 seconds, no slow cases, small recall 0.198487. Native BCSS87 removes H10's false-fat interior, but BCSS45 keeps one and TIGER23 still loses stroma. 18/149 optimizer flags are unsuccessful, maximum gradient infinity norm 0.004826. The 2-ms difference from J07 is not a controlled speed gain; this retains a different empirical noise law.

### K14

REJECT: worst-class epigraph risk improves H10 slightly to IoU 0.668234, but K13 is more accurate and faster, 0.0714 versus 0.0959 seconds, with higher small recall. BEETLE08 loses 3.77 points versus H10 and TIGER27 gains false green flecks. All optimizer flags are successful; feasibility alone does not certify KKT optimality or justify the less consistent anatomy.

### K15

REJECT: trace-norm shared colour contrast gives IoU 0.666388 at 0.0753 seconds, below H10 by 0.10364 points with 42 wins and 90 losses. 144/149 cases fail the proximal residual tolerance; maximum residual is 0.044922. Overviews retain false-fat structures and merge fine regions. Low rank does not improve the required class contrast.

## Expanded next-cycle inputs

All200 additional input sets are prepared and audited:40 each BCSS, TIGER, WSSS4LUAD, RINGS and OCELOT. The planned next corpus is349 ROIs, keeping all149 current images. The TIGER additions have40 source groups but many groups were already represented; non-overlapping windows do not make40 new patients. WSSS/RINGS have no verified pixel spacing and their physical metadata remain null. Feature extraction, unchanged RGB conversion generation, single-class/task adapters and a manifest-based review gate are still required before an expanded cycle can complete.
