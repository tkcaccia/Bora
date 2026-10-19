# Reuse feature contrast without a model download

The appearance tools operate on **already cached features on CPU**. They do not
open a GPU, load an encoder, download weights, or require Bora's compiled fastPLS
extension. The row bridge can be used by wsiTools without installing that native
extension. It needs the existing NumPy, SciPy, scikit-learn and threadpoolctl
dependencies. The spatial RGB9+19 API additionally uses OpenCV.

There are two different methods:

| Method | Inputs and use | Meaning of its colors |
|---|---|---|
| `RGB9Plus19Recipe` | Full 1280-channel spatial grids, coarse groups and the two physical scales; repeat exactly the fitted local field | Bora's original within-image anonymous-group evidence |
| `DiscriminativeAppearance` / Feature contrast | Cached feature rows with labelled training/reference examples; fit local groups or a shared semantic vocabulary | Display of retained class probabilities, not reconstructed H&E or biological marker intensities |

Feature contrast uses Bora's pairwise class-separation and redundancy-selection
principles. It **is not full RGB9+19**, has no multiscale consensus or spatial
quadrant completion, and is separate from the Neighbor context fastPLS tissue
classifier. Neither a pooled tile embedding nor one user click can substitute
for the full spatial/multiscale inputs required by RGB9+19.

## JSON/CSV bridge for an image viewer

Run the appearance-only command with an existing CPU Python environment:

```bash
PYTHONPATH=/path/to/Bora/src \
  /path/to/python -m bora.appearance_bridge /path/to/request.json
```

Keep the Python executable and Bora source directory configurable in a caller.
Use an environment containing the CPU dependencies listed above, and verify
that it imports the intended Bora checkout if another editable installation
exists. No dependency installation or inference happens in the bridge.

The CSV contains `id,x,y,label,split` plus the ordered feature columns. Rows may
be reference observations and new query observations in the same file.

* `feature_kind: virchow2-spatial` requires exactly **1280** ordered channels.
* `feature_kind: virchow2-pooled` requires the explicitly declared **2560**
  pooled representation. A 1280-channel patch-token mean is a different
  representation; do not relabel it pooled2560 or pad it with invented values.
* `encoder` records the checkpoint revision and lowercase SHA-256. The feature
  kind, column order and encoder dictionary must match exactly on reuse.
* Coordinates are native image **pixel centres**, never viewport/screen
  pixels. Supply both x/y columns and `coordinate_space: native_pixel_centres`,
  or omit both coordinates for a nonspatial table. Record crop origins,
  per-axis MPP and affine transforms in `provenance` in the calling project.
* Reference rows explicitly use `split: train` or `reference`. Fit requires at
  least 16 labelled rows, two classes, and two rows per class. `query`,
  `validation` and `test` rows are transformed but **never used for fitting**, even
  if they contain labels. Unknown split names fail. Labels must be nonempty.
* `mode: local` uses `label_kind: anonymous` for one project's group IDs (or
  `semantic` for an explicitly named local vocabulary). It does not establish
  a shared tissue meaning between projects.
* `mode: shared` requires `label_kind: semantic` and the same annotated class
  vocabulary across reference tissues. Use independent training annotations;
  a predicted class is not expert truth. Declare annotation/model provenance
  and patient/site splits in the project manifest. The bridge cannot establish
  their independence from CSV values alone.

This Python snippet builds a complete request without hand-writing 1280 names.
It assumes an existing compatible CSV; it creates no features:

```python
import json
from pathlib import Path

request = {
    "format": "bora-appearance-request-v1",
    "mode": "shared",
    "feature_kind": "virchow2-spatial",
    "feature_table": "reference-and-query.csv",
    "feature_columns": [f"f{i:04d}" for i in range(1280)],
    "label_column": "label", "split_column": "split",
    "id_column": "id", "x_column": "x", "y_column": "y",
    "coordinate_space": "native_pixel_centres",
    "label_kind": "semantic",
    "classes": {"1": "tumor", "2": "stroma"},
    "encoder": {
        "revision": "3158645804b69e3f3bc4439d4116edddf0840a72",
        "sha256": "8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b"
    },
    "provenance": {"annotation_source": "expert training mask", "project": "my-study"},
    "output_prefix": "results/contrast"
}
Path("request.json").write_text(json.dumps(request, indent=2))
```

`classes` must name exactly the represented training labels. Labels are string
IDs in the table contract; their fitted lexicographic order is recorded. A
palette can be supplied as `{"1": [220, 40, 70], "2": [40, 180, 70]}`.
An inline `features` row matrix is also supported instead of `feature_table`;
use aligned `labels`, `splits`, `ids`, and optional `coordinates: {x: [...], y:
[...]}` arrays. Keep requests bounded to the active reference/query region.

Successful commands print one JSON result on stdout and create:

* `contrast.recipe.json`: checksum-protected, fitted selection/scaler/logistic
  coefficients and provenance; **loading never refits** and never executes
  pickle. This contains derived training statistics, not raw training rows.
* `contrast.scores.csv`: `id,x,y,predicted_label,probability:<class>...` in the
  recorded class order. These numeric probabilities are the stable evidence.
* `contrast.rgb.csv`: `id,x,y,r,g,b`. Colors are display only.
* `contrast.report.json`: method/scope, feature kind, fit row count, output
  paths, checksums, and whether a frozen recipe was loaded.

Errors are printed to stderr with exit status 2. Existing output files are not
overwritten. A checksum detects corruption; it is not a cryptographic signature
or a proof that the training data were appropriate.

For another compatible image, set `recipe_path` to the saved recipe, point the
request to its query CSV, and use a new `output_prefix`. Labels and split columns
are unnecessary when applying an existing recipe. Nothing is refitted. Changing
only `palette` changes the display colors while preserving the probability
vectors. A viewer should use those vectors for feature comparisons, rather than
Euclidean distances between arbitrary display colors.

## Fit and replay the full local RGB9+19 formula

```python
from bora.rgb_recipe import RGB9Plus19Recipe

# Existing arrays: tokens[ceil(H/14), ceil(W/14), 1280], coarse[H,W],
# palette[positive_group_id-1, 3]. Each variant has values and shape_yx,
# plus selected_dimensions if it stores exactly 64 selected channels.
recipe = RGB9Plus19Recipe.fit(tokens, coarse, palette, variants,
                             provenance={"case": "calibrated-field-A"})
recipe.save("field-A.rgb-recipe.json")

loaded = RGB9Plus19Recipe.load("field-A.rgb-recipe.json")
probability = loaded.predict_proba(tokens, variants)
rgb, details = loaded.transform(tokens, variants)
```

The artifact retains selected64 channels, persistent32 positions, all three
scalers, both balanced logistic heads and all four Ridge completion heads, as
well as source hashes, shapes, settings and producer versions. Transform runs
linear predictions only. The original `convert_rgb()` convenience API still
fits a new local recipe, then transforms; use the fitted API for repeated work.

The recipe is deliberately bound to its original token field, coarse mask and
scale arrays. Spatial quadrant routing and anonymous classes are local. A
different field needs an explicit new local fit; it must not be called frozen
cross-tissue transfer. Its probability columns retain anonymous group IDs.

To reuse in `bora prepare`/`bora refine`, use `rgb_recipe` in the case JSON
instead of `evidence_rgb`, while retaining the base tokens and variants:

```json
{"rgb_recipe": "field-A.rgb-recipe.json"}
```

This is one additional field in a complete calibrated case, not a complete case
manifest on its own. Do not provide both `rgb_recipe` and `evidence_rgb`. A coarse
mask or scale-cache mismatch fails. The already documented homogeneous-window
palette fallback remains an explicit WSI-only path; it is not a fitted RGB9+19
recipe or evidence of recovered fine boundaries.

## Comparison and limits

For shared transfer, freeze encoder/preprocessing, feature semantics and order,
training-only selected dimensions/scaler/readout, class vocabulary and palette.
Keep the target's patient/site out of training and method selection. Retain
unannotated tissue as unannotated, and audit background exclusion independently.
Do not infer comparable marker intensities from local RGB colors.

The initial tests use deterministic synthetic cached arrays: complete-raster
parity against the old RGB9+19 implementation for two/three groups, load/replay
with fitting disabled, palette-invariant probabilities, frozen transfer,
train/test separation, bad feature contracts and CSV round-trip. These validate
software behavior, not tissue accuracy. Real cached H&E files alone are not
spatial features; a `NOVELTY.npz` distance/region raster is not an embedding cache.
No encoder or GPU is run by these tests. Evaluate scientific accuracy on
independent, calibrated, annotation-compatible cases before reporting transfer.
