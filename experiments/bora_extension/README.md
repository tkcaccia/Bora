# Additional inputs for the Bora improvement cycles

The user requested continuing the 20-cycle study and adding images alongside
the current 149. **200 additional image/coarse/reference triples are prepared
and pass a native-source/overlap audit.** This is input preparation, not
completed encoder inference, Bora prediction, expanded scores or a completed
cycle. Cycle seven is now signed after all149 actual overview inspections and
eight native label crops; seven of twenty cycles are complete.

| Dataset | Existing | Added | Planned total | Added representation |
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
request and remain unchanged. The 349-image corpus is intended for subsequent
cycles, preserving 149-image scores for historical comparisons. No cycle-eight
algorithm is designed before cycle-seven selection. The expanded prediction,
scoring and review gates must be implemented before an expanded cycle can
be called complete.

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

Eight WSSS4LUAD and 19 RINGS inputs have one observed coarse class. They remain
included; some contain fine classes missing from coarse. Single-coarse-class
and multiclass results must be shown separately alongside the overall image
average. Conversion adapters must record the degenerate class-constant case
and must not introduce reference-derived missing classes.

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
from deleted true tumor. The149-image scorer is not yet asserted to be a
correct adapter for every new task. No expanded score has been produced.

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
