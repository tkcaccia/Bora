# Twenty review-gated Bora improvement cycles

Current verified status: **nine of twenty cycles complete**. Cycle nine has
349 actual all-method overview reviews, nine full-run native crop reviews,
7,678 audited masks and a signed selection: **J01, M04, K05, M11, C17**.
J01 remains the runtime-adjusted leader; the other four preserve distinct
mechanisms with explicit accuracy, speed and tissue-preservation tradeoffs.
See [CYCLE_NINE_REVIEW.md](CYCLE_NINE_REVIEW.md).

Cycle ten is running on the same 349 frozen ROIs after its 13-image pilot,
357 checks and actual inspection of all pilot overviews plus three native
crops. Its prospective mathematics are in
[CYCLE_TEN_PROPOSAL.md](CYCLE_TEN_PROPOSAL.md); pilot behavior and finite-solver
limitations are in [CYCLE_TEN_PILOT_REVIEW.md](CYCLE_TEN_PILOT_REVIEW.md).
No tenth selection is claimed. Read execution JSON and the signed selections
for newer status. Historical checkpoints later in this document describe
earlier stages, not the current completion count.

The user now requests a runtime penalty above one second per image. From
cycle5 the primary selection utility is the image-average of
`IoU - 0.05 * max(seconds - 1, 0)`. Raw IoU remains visible. The selected-five
viewer also shows this policy as a labelled posthoc comparison for cycles1–4,
without changing their signed selections. See [CYCLE_PROGRESS.md](CYCLE_PROGRESS.md).

The requested endpoint is **twenty completed cycles**, each testing twenty
algorithms on all frozen ROIs: 149 in cycles 1–7 and 349 from cycle 8 onward.
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

The active input root is the immutable
`/mnt/sata_ssd/Bora-benchmark/results/bora-expanded-349-inputs-2026-10-18-v1`.
It retains all 149 historical cases and adds 40 each BCSS, TIGER, WSSS4LUAD,
RINGS and OCELOT. All five RGB conversions and encoder arrays are frozen.
Every cycle uses fixed equal #9+#19 RGB for its primary comparison. See the
[benchmark navigation guide](../bora_extension/BENCHMARK_NAVIGATION.md).

Most calibrated methods analyse at 1 µm/pixel; historical C19 evaluates an
adaptive classifier at 0.5 µm/pixel. Actual per-axis spacing is saved. The 80
WSSS/RINGS additions lack verified MPP and remain pixel-only, with separate
pixel-area scores. Numerical coordinate units are not invented physical
spacing. Projection back to native pixels adds no optical information.

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
>2500 um². Annotation-0 pixels are always unscored. Empty strata are null.

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

    PYTHONPATH=src:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      nice -n 12 python -m experiments.bora_cycles.run \
      --source /path/to/virchow2-expanded-inputs-2026-10-17 \
      --root /path/to/bora-cycles-2026-10-17/cycle-01 \
      --workers 3 --repeats 1

Partial method/case scopes require the --pilot flag; pilots cannot complete a
cycle or select survivors. The pilot.sh script checks one case from each of
the four dataset groups. Changed source/protocol requires a fresh output
folder; executed source and compiled library hashes are archived.

The runner implements cycles one through ten. The following are historical
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
per-image scores, and all 149 original/reference/coarse/converted input views.
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
