# Expanded Bora benchmark: navigation and interpretation

**Cycle11 is signed on a frozen469-image calibrated cohort.** Eleven of20
improvement cycles are complete; the five survivors are P11,N08,M04,J01,C17.
All469 all23-method overview sheets and25 planned plus2 posthoc native label
sheets were actually reviewed. P11 leads candidate utility but B23 control
still scores higher; native-support glass filling remains unresolved.
The input root is `results/bora-calibrated-tissues-inputs-2026-10-18-v1`,
with PLAN SHA256
`3b23ea5d0f6d6e16679d6a2cfe216ea3934731ba06e0b902ca4204ee96ac3a8d`.
Its `ALL_INPUTS_FROZEN.json`, passing `COHORT_AUDIT.json` and
`SAMPLING_REVIEW.json` identify the eligible fields and physical windows.
The latest source of execution state is `CURRENT_RUN.json`, not an older
inventory count. Do not restart a live run or treat its pilot as a completed cycle.

The original349-image inventory is historical for cycles8–10. The size/group
gate admitted299; physical calibration subsequently clears269 of those. The
remaining9 WSSS4LUAD and21 RINGS fields have unresolved MPP and cannot enter
new inference. Do not launch from the unfiltered349 or the old299 plan.

The corrected new acquisition audit has200 calibrated fields:40 each from
IGNITE, PESO, GCSS, PUMA and GleasonXAI. The five-dataset review snapshot is
`results/bora-tissue-acquisition-2026-10-18-v5`; it also includes
Histo-Seg40 as acquisition inventory only because its MPP is unknown. ORCA currently
has32 calibrated disjoint fields from9 cases/cores, below the40-field quota.
These are ROI development images, not full-WSI or independent-patient tests.

Additional source roots are versioned per dataset:

| Dataset | Relative source-field root under `results/` |
| --- | --- |
| IGNITE | `bora-tissue-expansion-inputs-2026-10-18-v1/ignite` |
| PESO | `bora-tissue-expansion-inputs-2026-10-18-v2/peso` |
| GCSS | `bora-tissue-expansion-inputs-2026-10-18-v3/gcss` |
| GleasonXAI | `bora-tissue-expansion-inputs-2026-10-18-v4/gleasonxai` |
| PUMA | `bora-tissue-expansion-inputs-2026-10-18-v6/puma` |
| ORCA, quota not met | `bora-tissue-expansion-inputs-2026-10-18-v6/orca` |
| Histo-Seg, calibration pending | `bora-tissue-expansion-inputs-2026-10-18-v7/histoseg` |

The version6 move corrected accidentally nested output directories without
changing source pixels or plan hashes. An older Histo-Seg40 inventory had five
entirely white fields and was never encoded; version7 rejects blank and exact
duplicate fields. Keep the older inventories as diagnostics, not benchmark
inputs. See `benchmark/TISSUE_DATASET_EXPANSION.md` and the remote correction
receipt `results/bora-tissue-path-correction-2026-10-18-v1.json`.

`results/bora-tissue-features-160-2026-10-18-v1` contains the first four datasets'
480 frozen tensors and a passing full-corpus `FEATURE_AUDIT.json`.
`results/bora-puma-features-40-2026-10-18-v1` contains the separate120 PUMA
tensors with its own passing freeze and audit. All200 calibrated acquisitions
have been encoded and converted using the unchanged five formulae. These logical roots link to
computed caches in `/home/chiamaka/Bora-benchmark-compute` on NVMe. Provider
images and native image/reference fields remain on the SATA SSD. The fixed
RGB09/10/15/19/equal09+19 conversion formulae remain unchanged.

Cycle11 has15 new callable models plus the five parents, with three separately
labelled controls including a full-support ablation. Its17-case engineering
pilot passed mask, parent-parity and reporting audits; all17 overviews and17
native label crops were actually inspected. The full469-case run uses an
immutable source snapshot, three nice12 workers and two numerical threads per
worker. The completed full run is reviewed and signed as described above.
Cycle12's15 prospective children are implemented in `twelve_manifest.py`,
`cycle_twelve.py`, `twelve_appearance.py`, `twelve_support.py`, `twelve_regions.py`
and `native_twelve.py`; their equations and limitations are in
`CYCLE_TWELVE_PROPOSAL.md`. Read CURRENT_RUN.json for its engineering pilot;
it is not a completed cycle or a new signed selection. Candidate time excludes cached
Virchow2 encoding/conversion; it is not end-to-end pipeline latency.
The remaining historical349 navigation below documents signed earlier work;
its counts are not the current denominator.

## Locations

Chiamaka's data base is `/mnt/sata_ssd/Bora-benchmark`.

| Relative directory on Chiamaka | Contents |
| --- | --- |
| `raw` | Provider downloads and extracted source data; retain acquisition records and licenses |
| `prepared` | Provider/pair inventories, including663 OCELOT tissue pairs; not all are benchmark inputs |
| `results/bora-extension-inputs-2026-10-18-v1` |160 newly selected BCSS/TIGER/WSSS4LUAD/RINGS image/coarse/reference triples |
| `results/bora-ocelot-inputs-2026-10-18-v1` |40 selected OCELOT training tissue triples |
| `results/bora-extension-features-2026-10-18-v1` |600 frozen feature arrays across the200 additions, plus scale/coverage/calibration metadata |
| `results/bora-expanded-349-inputs-2026-10-18-v1` |Historical349-image input freeze for signed cycles8–10; all five fixed RGB conversions |
| `results/bora-eligibility-2026-10-18-v1` |Frozen score-blind decisions for all349, per-group tile counts, sampling review and included/diagnostic viewer |
| `results/bora-qualified-inputs-2026-10-18-v1` |Prospective299-image PLAN and input symlinks;50 excluded inputs separately under diagnostics/ |
| `results/bora-cycles-2026-10-17/cycle-08` |Signed, fully reviewed349-image cycle8 |
| `results/bora-cycles-2026-10-17/cycle-09` |Signed, fully reviewed349-image cycle9; survivors J01,M04,K05,M11,C17 |
| `results/bora-cycles-2026-10-17/cycle-10-pilot-v1` |Reviewed13-image engineering pilot; not a completed cycle |
| `results/bora-cycles-2026-10-17/cycle-10` |Signed349-image cycle10; survivors J01,M04,N08,N10,C17 |
| `results/bora-calibrated-tissues-inputs-2026-10-18-v1` |Active immutable469-image input freeze:132 BCSS,80 TIGER,40 OCELOT,10 BEETLE,7 CAMELYON16,40 each IGNITE/PESO/GCSS/PUMA/GleasonXAI |
| `results/bora-cycles-2026-10-17/cycle-11-pilot-v1` |Reviewed17-image engineering pilot; not a completed cycle |
| `results/bora-cycles-2026-10-17/cycle-11` |Signed469-case/23-method run; all audits and actual reviews complete; survivors P11,N08,M04,J01,C17; selected.html shows these five |
| `experiments/bora-extension-2026-10-18` |Immutable executed source snapshots, launch logs and acquisition/input audits |

Local outputs use `/Users/stefano/Documents/ChatGPT/Bora/benchmark/results`.
Remote cycle08/09/10 correspond to local `cycle-08-chiamaka`,
`cycle-09-chiamaka` and `cycle-10-chiamaka`. `CURRENT_RUN.json` in that local results base identifies
the active process, exact source snapshot, input freeze, last delivered viewer
and completed-cycle count. Read it before launching another job.

The historical 160/40 additions have local input triples in the separate
folders above. The new five-dataset expansion has two local example pairs per
dataset under `benchmark/examples/tissue-expansion-2026-10-18-v2`; its complete
200 source fields and encoder tensors remain on Chiamaka. Local combined-input
folders can contain metadata and review sheets without the full raster tree.
Do not infer a complete local dataset from the presence of a local plan.

The older acquisition code and examples are in the main local workspace at
`/Users/stefano/Documents/ChatGPT/Bora/benchmark`; its `DATASETS.md`,
`ANNOTATIONS.md`, `IMAGE_TYPES.md`, `resolution.json` and acquisition records
describe the original downloads. Active experimental source is in the Git
worktree `/Users/stefano/.codex/worktrees/gigatime-marker-wand/Bora`, with the
new acquisition/preparation code under `experiments/bora_extension`.

## Input files

Start with combined `PLAN.json`, `ALL_INPUTS_FROZEN.json`, then
`cases/<case>/INPUTS_FROZEN.json`. The plan records dataset, source identity,
ROI geometry, class names/palette, native and working pixel spacing, and
reference hash. The input manifest records individual file hashes.

Within a case:

* `original.native.png`: original dataset ROI pixel grid, including distributed
  image resampling where applicable; this does not imply native WSI optics.
* `original.png`: the working H&E raster.
* `reference.native.tif`: expert/provider reference remapped to compact class
  IDs on the native ROI grid. It is for post-prediction evaluation and review.
* `coarse.tif`: the frozen synthetic input tile mask used by Bora.
* `tissue.working.tif` and `tissue.native.tif`: image-derived support on each grid.
* `selection.json`: documented coarse training-class eligibility.
* `09/representation.png`, `10/representation.png`, `15/representation.png`,
  `19/representation.png`, `ensemble/representation.png`: fixed converted RGB.
  The primary cycles use equal09+19, named `ensemble`.
* `coarse.completed.tif`, where present: frozen annotation-free completion for
  exactly constant RGB. This fallback preserves existing coarse observations;
  it is not successful fine-boundary recovery.

For the historical 349-image freeze only, the plan SHA256 is
`7aff79d91f5bd8b301f939dc83bc7892eda388bf7f33e4476f6f2dc8cb9c8957`.
The all-input freeze SHA256 is
`52667d5a4e903b69e7666e8c1decfd74e4adfa604d99d14417f418b18ae0bcc9`.
There are6980 frozen files and1745 converted RGB rasters; all149 historical
input manifests are byte-identical to their earlier versions.

## Resolution, labels and quality scoring

Verified axis spacing is reported in micrometres per pixel, not inferred from
objective magnification. The 80 WSSS4LUAD/RINGS additions remain historical
pixel-only exploratory inputs; none enters the calibrated cycle11 scope.
Do not assign them 0.5µm/pixel or report physical areas. Histo-Seg's 40 acquired
fields also remain outside inference while MPP is unresolved. OCELOT uses the
provider's distributed tissue-image resized-axis spacing, not its cell patch
or underlying slide spacing. Each current input records native/working axes,
physical field size, actual resize and any upscaling. A 224-pixel Virchow2
window spans approximately 112,224,448µm at the three target scales
0.5,1,2µm/pixel; rounding and real complete-window counts are recorded.

Read each case's `class_names`, `palette` and raw-to-compact mapping rather than
assuming red always denotes tumor or green always denotes stroma. BCSS/TIGER
contain tissue compartments. WSSS4LUAD includes tumor, stroma and normal tissue.
RINGS is tumor versus outside tumor; OCELOT is cancer versus non-cancer tissue.
Neither negative target is an expert stromal annotation.

Compact reference0 is unscored annotation absence. Raw0 has dataset-specific
meaning: for RINGS it is a valid negative mapped to a positive compact ID.
All supported tissue, including unannotated tissue, is analyzed. Every positive
reference counts, including classes missing from the coarse mask and tumor
deleted by support/glass removal. Saved output0 means removed/unassigned
background. Historical RINGS scoring maps that output0 to its valid negative
target; deleted tumor still counts as an error. In the calibrated scope,
PESO and OCELOT declare broad negative categories: only photometrically white
prediction0 pixels map to that negative category for scoring. Non-white
unassigned pixels remain errors, as do deleted true epithelial/cancer pixels.
The saved masks are unchanged. This is our task adapter, not a provider metric.
Other tasks retain output0 as an error on positive references. White-looking
tissue is not automatically ignored. A broad negative is not a stroma annotation.

In the historical349-image inventory, thirty inputs have exactly constant converted RGB:11 WSSS4LUAD and19 RINGS.
Three WSSS inputs contain extra coarse labels outside eligible tile summaries.
Thus eligibility, classes anywhere in coarse labels, and reference classes are
different quantities. All methods/controls share one frozen coarse-completion
fallback; explicitly H&E-driven glass rules may additionally remove white.
These30 cases are excluded prospectively. They remain reproducible in the
signed historical cycles and visible in the separate diagnostics inventory.

Accuracy is equal-per-image native macro IoU. Primary selection utility is
`mean_i[macro_IoU_i - 0.05*max(seconds_i-1,0)]`. Apply the excess-time penalty
per image before averaging. Raw accuracy, elapsed time and penalty remain
separate. Shared encoder preparation is excluded from Bora runtime and must
be reported separately. Physical and pixel-based small-region statistics are
separate. Compare methods on the same current cohort and scoring policy.
Cycle11 checks unchanged native parent masks for269 shared fields; scores
retain the earlier ontology on229, while40 OCELOT cases are prospectively
rescored. A new cohort mean or changed scoring convention is not an algorithm gain.

These images are repeatedly reused development data. TIGER source-group reuse,
unverified RINGS patient identity and reused provider splits prevent claiming
an independent patient-level validation study. Source-coordinate overlap was
audited, and the selected non-TCGA TIGER additions exclude its BCSS-derived
subset. Do not treat source groups as automatically independent patients.

## Result files and workflow

`experiments/bora_cycles` contains candidate registries, mathematical manifests,
implementations, tests, run/review tools and execution reports. Cycle11 is
signed: read `CYCLE_ELEVEN_REVIEW.md` and `CYCLE_ELEVEN_SELECTION.json`. Its
implementation is in `eleven_manifest.py`, `cycle_eleven.py`, `eleven_appearance.py`,
`native_eleven.py` and `CYCLE_ELEVEN_EXECUTION.json`. The descriptive
`METHOD_CATALOGUE.json` indexes registered mechanisms and parent lineage;
an entry proves neither execution nor mathematical novelty. Source snapshots
in launch receipts identify the code actually used by each run.

Each result root has `PROTOCOL.json`, case prediction manifests, timed method
details and `ALL_PREDICTIONS_FROZEN.json` before fine-reference scoring starts.
Then `RESULTS.json` and each `SCORES.json` contain scores. Every method's
`native.tif` is the saved categorical prediction; `mask.png` preserves its
exact indices and palette. `contact.jpg` shows every method for actual review.

Integrity audits and numerical residuals do not constitute visual review.
`VISUAL_REVIEW.json` records actual hashed-sheet inspections. Native crops have
coordinates, source hashes and separate observations. `SMALL_REGION_ATLAS.json`
counts missed small reference components; rare-colour diagnostics propose
unresolved regions without assigning biological identities. Only a signed
`SELECTION.json` completes a cycle and authorizes its next15 child models.

At cycle end open `selected.html`, showing only the chosen five with all cases
selectable: original H&E, reference, input tile mask, fixed RGB and translucent
refinements. Keep source/input hashes unchanged, freeze predictions before
scoring, do not revive rejected RGB conversion04, and use the active protocol's
roster rather than older rejected strategies. Resource budget: at most three
nice12 image workers, two numerical/OpenMP threads each; leave desktop and
other users' jobs alone.


## Local mirrors from cycle10 onward

The Mac had about8 GiB free when cycle10 started. Complete result rasters stay
on Chiamaka. Local cycle10 and later mirrors contain contact sheets, detailed
label crops, JSON reports and actual review records; they are not full native
prediction archives. Nothing was deleted from earlier local results.

The viewer is served through an SSH forward at `http://127.0.0.1:64889`.
A remote result such as `results/bora-cycles-2026-10-17/cycle-10/selected.html`
appears at `/bora-cycles-2026-10-17/cycle-10/selected.html` on that local port.
This viewer reads every image, PNG mask and native TIFF from the canonical
remote result folder. Opening local mirrored HTML on port64888 does not supply
these omitted assets. Earlier64888 viewers remain unchanged.

Both server and SSH listener bind only127.0.0.1. The local results base has
`REMOTE_VIEWER.json` with the tunnel PID and port; Chiamaka has the HTTP server
receipt under `experiments/bora-extension-2026-10-18/result-viewer-64889.json`.
Check live processes and HTTP responses before restarting a tunnel or server.
A dropped connection is not proof that computation stopped. Do not re-run
predictions because a local mirror is intentionally incomplete.

`CURRENT_RUN.json` identifies the last actually delivered selected-five viewer.
A working server or exported crop does not select methods or establish an
actual inspection. After all349 overviews are reviewed and five are signed,
the report HTML and signed review/selection records are synchronized back to
Chiamaka, then its selected-five page is opened through the local forward.
