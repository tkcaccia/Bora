# Twenty review-gated Bora improvement cycles

The requested endpoint is **twenty completed cycles**, each testing twenty
algorithms on all 149 frozen ROIs. A pilot or a completed prediction job is not
a completed cycle. Each cycle must be scored, rendered and actually inspected
before five survivors are selected. Subsequent cycles retain those five and
implement three distinct mathematical interventions per survivor, for fifteen
new implementations. Historical rejections remain recorded.

The protocol.json file preserves the complete objective and evaluation rules.
The registry.py file gives twenty hypotheses, individual risks and explicit
implementation status. Their implementations are in methods.py; C++ is in
cpp/evidence/kernels.cpp. Numerical kernels and ports are not counted as
independent algorithmic innovations. C10, C17, C18 and C19 use specialized
compiled kernels; the other candidates can use compiled class scoring.

## Input and resolution policy

Use the existing immutable virchow2-expanded-inputs-2026-10-17 directory.
All 149 cases and dataset groups are included, not only the nine previous
development cases. Every cycle uses fixed equal #9+#19 RGB for its primary
comparison. All five original RGB conversions remain unchanged.
No new Virchow2 inference is needed.

Most methods analyse at 1 um/pixel; C19 evaluates an adaptive classifier at
0.5 um/pixel. Actual per-axis analysis/native spacing is saved. Projection
back to native pixels adds no optical information.

Fine reference masks are unavailable to the prediction functions. The runner
hashes all predictions before starting any scoring. Repeated selection uses
all 149 cases; this is an exploratory development benchmark, not independent
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

The runner implements cycles one, two and three. Cycle one has 149 actual contact
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

Only after all twenty cycles are complete should the final all-conversion
benchmark, complete history and final viewer be presented as a finished study.

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
