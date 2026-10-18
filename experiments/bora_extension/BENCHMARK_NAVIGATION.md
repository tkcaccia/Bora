# Expanded Bora benchmark: navigation and interpretation

The development corpus contains **349 ROI images**:132 BCSS,80 TIGER,40
WSSS4LUAD,40 RINGS tumor,40 OCELOT,10 BEETLE and7 CAMELYON16 windows. It retains
all149 original images and adds200,40 each from BCSS,TIGER,WSSS4LUAD,RINGS and
OCELOT. A source dataset may contain full WSIs; none of these349 analysis
images is a full WSI. This is a selected benchmark, not the complete contents
of every downloaded dataset.

Nine of the requested20 improvement cycles are signed. Cycle10 is currently
running on all349 cases; its13-image pilot and3 native label crops were
actually reviewed but do not complete a cycle. Check `CURRENT_RUN.json`,
execution reports and signed selection files for newer status.

## Locations

Chiamaka's data base is `/mnt/sata_ssd/Bora-benchmark`.

| Relative directory on Chiamaka | Contents |
| --- | --- |
| `raw` | Provider downloads and extracted source data; retain acquisition records and licenses |
| `prepared` | Provider/pair inventories, including663 OCELOT tissue pairs; not all are benchmark inputs |
| `results/bora-extension-inputs-2026-10-18-v1` |160 newly selected BCSS/TIGER/WSSS4LUAD/RINGS image/coarse/reference triples |
| `results/bora-ocelot-inputs-2026-10-18-v1` |40 selected OCELOT training tissue triples |
| `results/bora-extension-features-2026-10-18-v1` |600 frozen feature arrays across the200 additions, plus scale/coverage/calibration metadata |
| `results/bora-expanded-349-inputs-2026-10-18-v1` |Authoritative combined349-image input freeze; all five fixed RGB conversions |
| `results/bora-cycles-2026-10-17/cycle-08` |Signed, fully reviewed349-image cycle8 |
| `results/bora-cycles-2026-10-17/cycle-09` |Signed, fully reviewed349-image cycle9; survivors J01,M04,K05,M11,C17 |
| `results/bora-cycles-2026-10-17/cycle-10-pilot-v1` |Reviewed13-image engineering pilot; not a completed cycle |
| `results/bora-cycles-2026-10-17/cycle-10` |Running349-image cycle10; no visual selection yet |
| `experiments/bora-extension-2026-10-18` |Immutable executed source snapshots, launch logs and acquisition/input audits |

Local outputs use `/Users/stefano/Documents/ChatGPT/Bora/benchmark/results`.
Remote cycle08/09/10 correspond to local `cycle-08-chiamaka`,
`cycle-09-chiamaka` and `cycle-10-chiamaka`. `CURRENT_RUN.json` in that local results base identifies
the active process, exact source snapshot, input freeze, last delivered viewer
and completed-cycle count. Read it before launching another job.

All200 new original input triples have local copies in the separate160/40
folders above. The local combined349 input folder contains metadata and input
review sheets, not the full feature/raster tree. Feature arrays remain on
Chiamaka. Do not infer missing local data from the presence of a local plan.

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

The combined plan SHA256 is
`7aff79d91f5bd8b301f939dc83bc7892eda388bf7f33e4476f6f2dc8cb9c8957`.
The all-input freeze SHA256 is
`52667d5a4e903b69e7666e8c1decfd74e4adfa604d99d14417f418b18ae0bcc9`.
There are6980 frozen files and1745 converted RGB rasters; all149 historical
input manifests are byte-identical to their earlier versions.

## Resolution, labels and quality scoring

Verified axis spacing is reported in micrometres per pixel, not inferred from
objective magnification. All80 WSSS4LUAD/RINGS additions lack verified MPP and
remain pixel-only exploratory inputs. Do not silently assign0.5µm/pixel or
report physical areas for them. OCELOT uses the provider's distributed tissue
image resized-axis spacing, not its cell patch or underlying slide spacing.

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
background. RINGS scoring alone maps that output0 to its valid negative target;
deleted tumor still counts as an error. Other tasks retain output0 as an error
on positive references. White-looking tissue is not automatically ignored.

Thirty inputs have exactly constant converted RGB:11 WSSS4LUAD and19 RINGS.
Three WSSS inputs contain extra coarse labels outside eligible tile summaries.
Thus eligibility, classes anywhere in coarse labels, and reference classes are
different quantities. All methods/controls share one frozen coarse-completion
fallback; explicitly H&E-driven glass rules may additionally remove white.

Accuracy is equal-per-image native macro IoU. Primary selection utility is
`mean_i[macro_IoU_i - 0.05*max(seconds_i-1,0)]`. Apply the excess-time penalty
per image before averaging. Raw accuracy, elapsed time and penalty remain
separate. Shared encoder preparation is excluded from Bora runtime and must
be reported separately. Physical and pixel-based small-region statistics are
separate. Use original149 matched-cohort scores for cross-expansion comparisons.

These images are repeatedly reused development data. TIGER source-group reuse,
unverified RINGS patient identity and reused provider splits prevent claiming
an independent patient-level validation study. Source-coordinate overlap was
audited, and the selected non-TCGA TIGER additions exclude its BCSS-derived
subset. Do not treat source groups as automatically independent patients.

## Result files and workflow

`experiments/bora_cycles` contains candidate registries, mathematical manifests,
implementations, tests, run/review tools and execution reports. Cycle9 is
signed: read `CYCLE_NINE_REVIEW.md` and `CYCLE_NINE_SELECTION.json`. For active
cycle10, read `CYCLE_TEN_PROPOSAL.md`, `ten_manifest.py`, `cycle_ten.py`,
`CYCLE_TEN_PILOT_REVIEW.json` and `CYCLE_TEN_EXECUTION.json`.

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
