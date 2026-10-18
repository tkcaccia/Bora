# Additional inputs for the Bora improvement cycles

**Prospective update after cycle10:** the user excludes images with only one
group and requires sufficient real tiles. The score-blind
`eligibility.py` audit now accepts299/349 images:132 BCSS,80 TIGER,9 WSSS,
21 RINGS,40 OCELOT,10 BEETLE and7 CAMELYON16 windows. Fifty excluded inputs
remain in a separate diagnostics inventory, outside future scores/rankings.
See [the eligibility and sampling report](ELIGIBILITY_AND_SAMPLING.md),
`eligibility_policy.json` and `ELIGIBILITY_EXECUTION.json`. Earlier cycles and
the historical single-class descriptions below retain their original scope.
The remaining WSSS sampling mismatch still needs a separate geometry comparison
before cycle11 inference; this eligibility change is not a segmentation gain.

The user requested continuing the 20-cycle study and adding images alongside
the current 149. **200 additional image/coarse/reference triples are prepared
and pass a native-source/overlap audit.** Full encoder inference is also complete:
all600 arrays across200 additions pass the independent feature audit. Expanded
Bora cycle10 is signed after349 actual all-method overview inspections and15
native label crops. Ten of twenty cycles are complete. Its selected five are
J01,M04,N08,N10,C17. J01 remains the runtime-adjusted leader; the other four
retain distinct uncertainty,variance,background and palette mechanisms, with
explicit tradeoffs rather than a claim that they are the five highest numeric
utility ranks. See `../bora_cycles/CYCLE_TEN_REVIEW.md` for every decision.

| Dataset | Existing | Added | Frozen total | Added representation |
| --- | ---: | ---: | ---: | --- |
| BCSS |92|40|132|ROI window; documented spacing |
| TIGER |40|40|80|Non-TCGA ROI window; documented spacing |
| WSSS4LUAD |0|40|40|Native-pixel ROI window; spacing unverified |
| RINGS tumor |0|40|40|Native-pixel ROI window; spacing unverified |
| OCELOT |0|40|40|Tissue ROI; verified provider resized-axis spacing |
| BEETLE |10|0|10|Existing native WSI windows retained |
| CAMELYON16 |7|0|7|Existing native WSI windows retained |
| **Total** |**149**|**200**|**349**|All analysis images are ROIs, not full WSIs |

Cycle seven's 149-image protocol and predictions were frozen before this
request and remain unchanged. The349-image corpus is used prospectively from
cycle8, preserving149-image scores for matched historical comparisons.
Expanded prediction, scoring and review gates have executed. A running job
is not a completed cycle; every full all-method overview must actually be
inspected before five are signed.

## Sampling and source integrity

`plan.py` uses a fixed seed and existing annotated-class-entropy window
selection, never an algorithm score. BCSS adds 40 source cases outside the
current set. TIGER adds 40 new source ROIs across 40 source groups, with group
reuse relative to the current benchmark. Global source rectangles do not
overlap the current or newly selected windows. Non-TCGA selection excludes
the BCSS-derived TIGER subset. Source groups are not automatically verified
independent patients.

All 40 geometrically accepted WSSS4LUAD validation ROIs are represented. Its 80
problematic test-mask pairings remain excluded. RINGS adds 40 images across 32
filename-prefix groups of unverified patient meaning. Original 25 train / 15 test
tags are preserved. They will be used for development, so no independent
provider-test claim is allowed.

In the original cycles8–10 inventory, eight WSSS4LUAD and19 RINGS inputs have
one coarse-mask class; three further WSSS fields supply only one usable
feature-selection class. Those30 constant-input cases were included then,
with a documented fallback and no new-class refinement claim. They are now
excluded prospectively, as are the remaining small WSSS fields under the tile
rule. Historical fallback code remains available to reproduce signed results;
it is not permission to include single-group images in future benchmarks.

`audit_inputs.py` verifies source image/mask hashes, exact native RGB crops,
categorical remapping, working geometry and 90-pixel coarse majorities. It
checks no duplicate source RGB bytes against the 149 or across the 200 additions,
and checks TIGER global rectangles. All 200 pass. Historical plan SHA256:
`df1774258b8a1c43fdf0b845bbbbac930d82f8821f9a994e243512c222422392`.

## OCELOT

The original [OCELOT1.0.1 release](https://zenodo.org/records/8417503) is acquired
in full: 303,248,865 archive bytes, publisher MD5
`215230295b0440b4dc356519ef9ff644`, SHA256
`74f46b79e3c4076ca0012d403629ab1e1e412591faf79a910d4d8bdd92c47920`.
Its 663 tissue image/mask pairs exclude four under-annotated test cases from
older releases. Data are CC BY-NC4.0; retain the
[provider terms and attribution](https://lunit-io.github.io/research/ocelot_tc/).
No downloaded Python script was executed.

Only 40 training tissue pairs from 40 distinct TCGA case IDs enter the extension:
7 bladder, 7 endometrium, 6 head-and-neck, 7 kidney, 6 prostate, 7 stomach. Validation
and test pairs are inventoried but not selected. Cell point annotations are
not used as tissue masks. Tissue values1,2,255 mean non-cancer background,
cancer area and unknown. Unknown pixels remain unscored; non-cancer background
is not an expert stroma class. See the
[provider metadata/annotation description](https://lunit-io.github.io/research/ocelot_dataset/).

`prepare_ocelot.py` uses tissue `resized_mpp_x/y`, cross-checked against source
pixel bounds and source-slide axis spacing. It does not assign the cell-patch
or source-slide spacing to the distributed 1024px tissue image, nor assume
every image is exactly 0.8µm/pixel. Source axes, resized axes, native crops and
working sampling are preserved.

## Calibration and semantics

All 80 WSSS4LUAD/RINGS inputs preserve native pixels. Native/working MPP, field
size in micrometres and coarse physical size are null. Magnification does not
establish pixel calibration. Pixel-based exploratory comparisons are possible,
but calibrated Virchow2 sampling and physical-area/boundary metrics cannot be
claimed without evidence. Prediction adapters must distinguish numerical
pixel-scale settings from measured spacing, never quietly substitute 0.5µm/pixel.

WSSS4LUAD raw 0 is tumor, 1 stroma, 2 normal tissue, 3 excluded background. RINGS
raw 0 is a valid outside-tumor target, 1 tumor; this is not tumor/stroma ground
truth. Planning maps valid categories to positive compact IDs and reserves
compact 0 for source ignore. RINGS negatives can include glass/lumina:
task-specific foreground/background scoring must be explicit and retain errors
from deleted true tumor. The expanded scorer implements this task adapter prospectively from cycle8;
historical parent scores retain their original definitions exactly.

Fine references are used by planning and post-prediction scoring only. Every
tissue pixel, including unannotated tissue, needs encoder coverage. Conversion
mechanisms 09/10/15/19 and equal 09+19 stay fixed. Missing coarse classes count as
misses; rare colours have unknown biological identity. The one-second runtime
penalty remains in force.

## Storage and checks

Chiamaka base: `/mnt/sata_ssd/Bora-benchmark`.

* `results/bora-extension-inputs-2026-10-18-v1`:160 requested additions.
* `results/bora-ocelot-inputs-2026-10-18-v1`:40 OCELOT additions.
* `prepared/ocelot.jsonl`:663 original tissue-pair records.
* `raw/ocelot`:pinned archive and safe extraction.
* `experiments/bora-extension-2026-10-18`:executed source, launches, logs and
  `INPUT_AUDIT.json`.

Local result copies use the same folder names under
`/Users/stefano/Documents/ChatGPT/Bora/benchmark/results`.
`EXTENSION_EXECUTION.json` records actual counts, hashes and remaining work.
Eight focused extension checks pass locally and on Chiamaka: zero-label
semantics, null calibration, OCELOT resized axes, source overlap and archive
path safety. They do not replace the executed200-image native-source audit.

## Feature extraction stage

`features.py` reuses the pinned released Virchow2 checkpoint and original
encoder/normalization/token-stitching implementation. Every full working RGB
field is encoded at the base scale and factors2/4; neither fine annotations nor
tissue support gate the encoder. No Bora execution occurs in this stage.
Multiclass selected dimensions match the unchanged conversion's balanced mRMR
selector exactly. A single observed coarse class has no discriminative
statistic: it uses variance-only cache dimensions, explicitly marked as such;
its downstream class-probability RGB must be constant, with no reference-derived
extra classes.

The completed feature audit finds30 cases with one eligible training class:
11 WSSS and19 RINGS. The input-level counts above remain correct for classes
present anywhere in the coarse mask. In WSSS04/10/19, additional coarse labels
do not occur in eligible central tile summaries. Those three therefore also
require the explicit constant-conversion case; no extra class may be invented.

Small fields are reflect-padded at the right/bottom to the224px minimum window
before the existing14px patch padding. Original pixels are never resized for
this padding, and original token coordinates remain unchanged. Actual
resampling axes are recorded separately at each scale. Unknown MPP, window
field size and patch field size remain null for WSSS/RINGS; those runs are
exploratory native-pixel inference, not calibrated model sampling.

Each case has full coverage checks, source/cache hashes and a final
`FEATURES_FROZEN.json`; verified resumes do not re-encode. The job retains20GiB
free disk, uses two CPU threads and the existing GPU batch4/memory-fraction0.4
configuration at nice12. Five additional CPU checks cover null/actual-axis
calibration, exact unchanged feature selection, single-class degeneracy, full
field coverage without reading a poison reference file, and tampered-cache
rejection. Check execution records for actual pilot/full completion; source
availability alone is not evidence that inference finished.

`FEATURE_EXECUTION.json` records actual200-case/600-array completion, source/cache
and per-scale coverage checks, exact frozen code/checkpoint, the initial failed
model-directory launch, the corrected successful pilot, and tests. The full
feature and audit processes are terminal. Arrays remain on Chiamaka; local
copies contain metadata, while all200 original input triples are local.
Shared feature preparation averages1.753 seconds per new image, excluding model
loading. This is separate from Bora refinements' runtime and is not an
end-to-end subsecond claim. Cycle8 expanded scoring and review are complete;
cycle9 is signed against the identical frozen inputs; cycle10 is now running
all349 after its actually reviewed13-image pilot.

`task_adapter.py` prepares the expanded runner's unit and RINGS semantics
contract. Uncalibrated numerical coordinates are unit pixels, with physical
length/area keys relabelled to pixel units. RINGS scoring alone maps removed
background zero to the valid outside-tumor target; saved masks stay unchanged,
and deleted true tumor remains incorrect. Other datasets retain prediction
zero as an error on positive reference. Four checks cover calibration
contradictions, honest pixel reporting, deleted-tumor errors and all-tumor
crops. The cycle-eight runner executed these helpers and passed the full7678-mask
audit; cycles9–10 use the same unit and scoring contract.

## Completed conversion stage

`conversions.py` assembled the349-image input corpus and applied the exact
unchanged09/10/15/19/equal09+19 formulas to the200 additions. The independent
`audit_conversions.py` checks all6980 frozen files,1745 RGB rasters and149
byte-identical historical input manifests. Combined plan SHA256:
`7aff79d91f5bd8b301f939dc83bc7892eda388bf7f33e4476f6f2dc8cb9c8957`.
Chiamaka input root:
`/mnt/sata_ssd/Bora-benchmark/results/bora-expanded-349-inputs-2026-10-18-v1`.
The converter opens no fine-reference pixels and performs no Bora inference.
Original reference bytes are transferred and hash-checked as acquisition data.

Eight input sheets were actually inspected with H&E and all five conversions.
The constant RGB in single-eligible-class cases is a documented limitation,
not a reconstructed second tissue. Input inspection does not establish fine
mask accuracy or complete a cycle. `CONVERSION_EXECUTION.json` pins the successful
full audit, frozen source, pilot observations and real execution records.

The expanded scoring contract requires all349 images; a149-image subset cannot
complete cycle8. Accuracy is still equal per-image native macro IoU with the
same runtime penalty. Reports separate the original149/additional200 cohorts,
individual datasets, measured/unknown spacing and single/multiple eligible
coarse classes. Small-region summaries in micrometres and pixels are separate.
Cycle8 introduces the five signed parents plus fifteen models described in
`../bora_cycles/eight_manifest.py`; none changes the frozen RGB conversions.

The pilot's16 actual overview inspections exposed constant-RGB numerical class
artifacts in WSSS04/10. All30 degenerate inputs now use the same frozen completed-
coarse fallback for every candidate/control, with only independent H&E glass
handling retained. No refinement is claimed on constant evidence. A fresh
16-image pilot passes293 remote checks and the352-mask audit, including110
explicit fallback comparisons. Four native-label crops were actually inspected.
`../bora_cycles/CYCLE_EIGHT_EXECUTION.json` records the completed349-image
run, full audit and signed selection J01,L02,K05,L03,C17 after349 actually
viewed overviews and14 native crops. Cycle9 retains those exact five and adds
M01–M15; its327-check final pilot includes286 mask/display audits,65 parent
mask and65 parent-score comparisons,44 constant-input checks,13 actually
viewed overviews and3 native crops. Its full349-image run is in progress.
See `../bora_cycles/CYCLE_NINE_EXECUTION.json` for its current execution state.

The navigation guide in `../../benchmark/EXTENDED_BENCHMARK.md` distinguishes
raw downloads, frozen input triples, features, converted RGB and scored results.

The versioned [benchmark navigation guide](BENCHMARK_NAVIGATION.md) documents
the349-image layout, resolution, label semantics, input files and review workflow.
