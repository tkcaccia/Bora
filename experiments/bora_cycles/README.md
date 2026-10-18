# Twenty review-gated Bora improvement cycles

Current verified status: **eleven of twenty cycles complete**. Cycle11 has
469 actual all23-method overview reviews,25 planned plus2 posthoc native label
reviews,10,787 audited masks and a signed selection: **P11,N08,M04,J01,C17**.
All469 ROIs have verified physical sampling and meet the group/tile/size gates.
P11 leads selectable runtime-adjusted utility, but the B23 diagnostic control
still scores higher. P11 also fills real glass in several cases; it is an
unresolved native-support branch, not validated background removal.
P10/P12 are rejected for foreground geometry artifacts despite high scores.
These are not the five highest numerical utility ranks. The viewer exposes
all469 cases, transparent reference/coarse/final layers, physical scale bars,
scores, speed penalties and selection limitations.
See [CYCLE_ELEVEN_REVIEW.md](CYCLE_ELEVEN_REVIEW.md) for all20 decisions and
[CYCLE_ELEVEN_SELECTION.json](CYCLE_ELEVEN_SELECTION.json) for signed evidence.
[CYCLE_ELEVEN_EXECUTION.json](CYCLE_ELEVEN_EXECUTION.json) records source/input
hashes; the results-base `CURRENT_RUN.json` identifies the current state.
Nine cycles and the final all-five-fixed-conversion evaluation remain.

Cycle12 now has15 prospective implementations, three per signed survivor.
[CYCLE_TWELVE_PROPOSAL.md](CYCLE_TWELVE_PROPOSAL.md) gives every equation,
failure hypothesis and distinction from earlier mechanisms. Q01 explicitly
tests calibrated gray-glass removal; annotated pixels remain in scoring.
Q12/Q15 have compiled graph primitives; a port is not counted as an innovation.
The first24-case pilot was fully audited and actually reviewed, exposing an
empty-grid fallback artifact in Q12. Its repair requires a fresh pilot before
the469-case launch. See [CYCLE_TWELVE_PILOT_REVIEW.md](CYCLE_TWELVE_PILOT_REVIEW.md)
and CURRENT_RUN.json for observed limitations and live execution state.

Cycle ten has computed, scored, rendered and audited the same 349 frozen ROIs after its 13-image pilot,
357 checks and actual inspection of all pilot overviews plus three native
crops. Its prospective mathematics are in
[CYCLE_TEN_PROPOSAL.md](CYCLE_TEN_PROPOSAL.md); pilot behavior and finite-solver
limitations are in [CYCLE_TEN_PILOT_REVIEW.md](CYCLE_TEN_PILOT_REVIEW.md).
All349 full-run overviews and all15 planned native label crops have been actually
inspected. All7,678 mask/display checks,1,745 parent native
comparisons,1,745 parent accuracy comparisons and660 constant-input checks pass.
N07 remains a finite60-step approximation on319 informative cases; N14 has25
stationarity flags. The signed tenth selection is in CYCLE_TEN_SELECTION.json.
Read execution JSON and the signed selections
for newer status. Historical checkpoints later in this document describe
earlier stages, not the current completion count.

The user now requests a runtime penalty above one second per image. From
cycle5 the primary selection utility is the image-average of
`IoU - 0.05 * max(seconds - 1, 0)`. Raw IoU remains visible. The selected-five
viewer also shows this policy as a labelled posthoc comparison for cycles1–4,
without changing their signed selections. See [CYCLE_PROGRESS.md](CYCLE_PROGRESS.md).

The requested endpoint is **twenty completed cycles**, each testing twenty
algorithms on all frozen ROIs:149 in cycles1–7,349 in cycles8–10, and469
calibrated ROIs in cycle11's prospective scope.
A pilot or a completed prediction job is not
a completed cycle. Each cycle must be scored, rendered and actually inspected
before five survivors are selected. Subsequent cycles retain those five and
implement three distinct mathematical interventions per survivor, for fifteen
new implementations. Historical rejections remain recorded.

The protocol.json file preserves the original objective and evaluation rules.
Its historical 149-case scope is prospectively extended by benchmark_scope.py
and the active per-cycle PROTOCOL.json; do not rewrite signed earlier scopes.
The registry.py file gives twenty hypotheses, individual risks and explicit
implementation status. Their implementations are in methods.py; C++ is in
cpp/evidence/kernels.cpp. Numerical kernels and ports are not counted as
independent algorithmic innovations. C10, C17, C18 and C19 use specialized
compiled kernels; the other candidates can use compiled class scoring.

## Input and resolution policy

The active cycle11 input root is the immutable
`/mnt/sata_ssd/Bora-benchmark/results/bora-calibrated-tissues-inputs-2026-10-18-v1`.
Its PLAN SHA256 is
`3b23ea5d0f6d6e16679d6a2cfe216ea3934731ba06e0b902ca4204ee96ac3a8d`.
It contains132 BCSS,80 TIGER,40 OCELOT,10 BEETLE,7 CAMELYON16 fields,
and40 each IGNITE,PESO,GCSS,PUMA,GleasonXAI:469 ROIs in total.
All five fixed RGB conversions are frozen:09,10,15,19,equal09+19.
The primary comparison uses equal09+19 without changing its formula.
See the [benchmark navigation guide](../bora_extension/BENCHMARK_NAVIGATION.md).

Eligibility requires at least two coarse groups and two observed feature-selection
groups,16 usable coarse tiles, and real4×4 base encoder-field capacity. Physical
calibration also requires a native field of at least448µm on each axis.
Per-axis native and working µm/pixel, actual resize dimensions and upscaling
are recorded. Encoder windows are224 pixels at approximately0.5,1,2µm/pixel:
112,224,448µm fields, with actual rounded dimensions recorded. Most Bora
variants then analyse at1µm/pixel. Interpolation supplies no additional optical
detail, and objective magnification alone does not establish pixel spacing.
The physical and eligibility policies are versioned separately.

The80 historical WSSS/RINGS fields and40 acquired Histo-Seg fields are outside
the prospective benchmark because of size/group or unresolved-calibration
decisions. ORCA's32 eligible fields have not met the40-field acquisition quota.
They remain documented acquisition/diagnostic material, outside cycle11's
quality and runtime denominators. Historical349-image scores remain unchanged.

Fine reference masks are unavailable to the prediction functions. The runner
hashes all predictions before starting any scoring. Repeated selection uses
all active cases; this is an exploratory development benchmark, not independent
test-set or clinical performance.

Original known coarse tissue supplies semantic supervision. Predictions of
unknown tiles do not become semantic training labels. Unsupervised spatial
representation learning can observe the full detected tissue.

## White space, outliers and small regions

Frozen H&E-derived support remains the default. C20 explicitly tests a
different exterior-white exclusion, recording proposed removals. White
lumina/adipose are not automatically equivalent to glass; wrongly removing
annotated tissue reduces the same native IoU used for every candidate.

Annotation-free colour novelty generates connected region proposals with
their sizes and separate diagnostic runtime. These proposals have **unknown
semantic identity**. They do not establish new named biological classes.

The scorer includes every reference class, even if absent from coarse labels,
and every connected reference component. It reports completely missed regions,
correct-coverage recall at 50%, and physical area strata <=400, 400–2500 and
>2500 um². Reference0 pixels are always unscored. Empty strata are null.
The calibrated scope's broad-negative task adapter maps only photometrically
white prediction0 pixels to a declared negative category for scoring; non-white
unassigned pixels remain errors. OCELOT uses this prospective rule without
altering historical scores or saved masks. Non-cancer is not a stroma annotation.

## Execution

Compile on Chiamaka:

    g++ -O3 -std=c++17 -shared -fPIC -fopenmp \
      cpp/evidence/kernels.cpp -o cpp/evidence/libbora_evidence.so

Run the focused numerical, semantic and evaluation tests first:

    PYTHONPATH=src:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      python -m pytest -q tests/test_bora_cycle_foundation.py \
      tests/test_bora_cycle_candidates.py tests/test_bora_cycle_review.py

Use the existing Virchow2 environment providing OpenCV and SciPy.
Run at low priority with three processes and at most two numerical/OpenMP
threads each. Each candidate's time includes its own preparation, inference,
background policy and native projection. Shared disk loading, scoring,
compression and rendering are excluded. Novelty diagnostics have separate
timings. Do not give later algorithms free unrecorded preprocessing.
Virchow2 encoding and fixed RGB conversion are cached inputs, timed separately.
Reported candidate time is Bora processing time, not end-to-end pipeline latency.

    PYTHONPATH=src:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      nice -n 12 python -m experiments.bora_cycles.run \
      --source /path/to/virchow2-expanded-inputs-2026-10-17 \
      --root /path/to/bora-cycles-2026-10-17/cycle-01 \
      --workers 3 --repeats 1

Partial method/case scopes require the --pilot flag; pilots cannot complete a
cycle or select survivors. The pilot.sh script checks one case from each of
the four dataset groups. Changed source/protocol requires a fresh output
folder; executed source and compiled library hashes are archived.

The runner implements cycles one through eleven. The following are historical
checkpoints. Cycle one has 149 actual contact
sheet reviews, additional native crop reviews, and a signed five-survivor
selection. Cycle two retains C18/C17/C01/C03/C15 unchanged and implements
fifteen distinct children. Its mathematical changes and risks are described
in [CYCLE_TWO.md](CYCLE_TWO.md), with equations in registry.py. Later cycles
must be written from actual visual findings and measurements, then frozen
with five survivor IDs and fifteen child lineages. An automatic parameter
schedule does not constitute twenty innovation cycles.

Cycle two requires `--cycle 2 --previous /path/to/cycle-01`. Its runner checks
the previous 149 hashed reviews and signed selection, then verifies each
unchanged survivor's native mask against cycle one. The cycle_two.sh script
supports explicit `pilot` and `full` scopes. Compiled scoring is the default,
following exact cycle-one parity on all 3278 masks. New numerical/model tests
are in tests/test_bora_cycle_two.py.

Cycle two is now fully reviewed and selected: 149 actual all-method overview
reviews, six additional native-coordinate crop reviews, 3278 frozen masks
verified locally and 745 exact unchanged-survivor comparisons. The survivors
are D10, C17, C01, D08 and C15. [CYCLE_TWO_REVIEW.md](CYCLE_TWO_REVIEW.md)
explains the findings and limitations; CYCLE_TWO_SELECTION.json records all
twenty decisions and the accuracy/time/small-region Pareto evidence. Two of
the required twenty cycles are complete. Cycle three and later cycles still
require fifteen new implementations, full execution and actual review each.

## Review and selection

The gallery.html file exposes every image and native mask. Each contact.jpg
shows all methods for a case. VISUAL_MANIFEST.json identifies those sheets.
VISUAL_REVIEW.json begins empty: automated checks never fill it. The review.py
module records inspections only after images have actually been viewed.
Native zoom observations must be recorded separately.

The select function rejects pilots, missing cases, incomplete visual review,
missing tradeoffs, and survivors without all-case predictions. Every method
requires a written benefit/limitation or rejection reason. The validate_children
function enforces five parents with three distinct interventions each. All
numerical and visual evidence remains available, including failures.

At the user's updated request, open `selected.html` after every completed
cycle. This viewer shows only the five signed survivors, their aggregate and
per-image scores, and every case's original/reference/coarse/converted views
within that cycle's signed scope.
Masks overlay the H&E with adjustable opacity. Selection publishes it from
signed evidence without changing predictions or scores, and records its
hash and displayed methods in VIEWER_READY.json. The all-candidate gallery
and contact sheets remain available for the required internal review.
Only after all twenty cycles are complete should
the final all-conversion benchmark and complete history be presented as a
finished study.

Report export uses the same process limit as inference, retaining only small
contact thumbnails in memory instead of every native RGB mask. Indexed PNG
exports preserve exact class colours. A separate four-case publication check
verified all 88 native coloured masks and all four contact-sheet byte hashes
against the previous exporter. This engineering change is not a new candidate
and does not alter segmentation scores or measured candidate times.

The publish_only.py module can render existing predictions into a new folder,
without rerunning segmentation or overwriting signed visual-review evidence.

The publish_diagnostics.py `--reuse` option reuses colour-proposal maps only
when the frozen input plan, every current input file, case/RGB fingerprints,
proposal code and every cached diagnostic file pass hash checks. These maps
are independent of the candidate masks; they do not need to be recomputed or
copied for each cycle. Relative links preserve all diagnostic images, with a
DIAGNOSTICS_REUSE.json audit. This was verified across all 149 cycle-two cases.

Cycle three keeps the signed cycle-two survivors D10/C17/C01/D08/C15.
Its fifteen F-series implementations, equations and risks are described in
[CYCLE_THREE.md](CYCLE_THREE.md) and registry.py. F03 alone tests a new
component-shape glass policy; expert-labelled tissue it deletes still counts
against its score. The new C++ hull kernel is tested against NumPy and across
thread counts, and is not counted as a separate innovation. The runner requires
the complete cycle-two visual review and matching signed survivor lineage.
Cycle three cannot be called complete before all 149 outputs are reviewed.

Cycle three is now fully reviewed and selected: all 149 all-method contact
sheets were actually viewed, six additional native-coordinate crops were
recorded, and F01/C17/C01/D08/C15 were retained. The signed decisions for all
twenty candidates are in CYCLE_THREE_SELECTION.json; the measured gains,
regressions, white-gap failures and timing limitations are described in
[CYCLE_THREE_REVIEW.md](CYCLE_THREE_REVIEW.md). Three of twenty cycles are
complete. Earlier pending/execution snapshots above describe their historical
stage and do not supersede the signed selection.

Cycle four's fifteen G-series implementations are described before evaluation
in [CYCLE_FOUR.md](CYCLE_FOUR.md) and registry.py. They retain the signed
F01/C17/C01/D08/C15 parents unchanged. Run with `--cycle 4 --previous /path/to/cycle-03`;
cycle_four.sh accepts explicit pilot/full scope and an optional image-worker
count. Local fallback uses one image process and two numerical threads; its
timings are a different host stratum from previous Chiamaka runs. The new C++
exact categorical-count kernel is independently checked against exhaustive
outcomes and analytic-gradient tests. Implementation and passing tests do not
complete a cycle; all149 execution, scoring and actual visual review remain
required before selection.

Cycle five completed all execution, audit, visual-review and selection gates
under the new one-second runtime utility. H02/C17/H10/H04/C01 survived; the
slow F01 raw-accuracy leader was rejected. [CYCLE_FIVE.md](CYCLE_FIVE.md) and
[CYCLE_FIVE_REVIEW.md](CYCLE_FIVE_REVIEW.md) document the changes and outcomes.

Cycle six tested fifteen distinct children of those five. All 209 focused
checks passed locally and on Chiamaka, followed by the frozen eight-case pilot
and all 149 ROIs. The complete signed selection and reasons are in
CYCLE_SIX_SELECTION.json. Original H&E edge conductance improves appearance
fields; Gaussian-integrated hinge risk improves the fast discriminant branch.
Background deletion and nonlinear colour-vote artifacts remain documented
failures. Fourteen cycles and the final all-conversion comparison remain.

Cycle-seven interventions and numerical budgets are described prospectively in
[CYCLE_SEVEN.md](CYCLE_SEVEN.md) and seven_manifest.py. They retain the actual
signed J01/H02/C17/J07/H10 lineage. Implementation and numerical tests alone
do not complete cycle seven; all execution and actual review gates still apply.
