# fastPLS correction audit

Bora 0.2.1 vendors the MIT-licensed fastPLS [core commit
`265108b`](https://github.com/tkcaccia/fastPLS/commit/265108b6aa250c2534f98ec89037bb5cd001dca0)
and synchronized [Python-binding commit
`b4ea261`](https://github.com/tkcaccia/fastPLS-py/commit/b4ea261902244bfb84ad933a236d364d19f905ee).
The two changed vendored headers match the upstream core byte for byte:

| Header | SHA256 |
| --- | --- |
| `core/matrix.hpp` | `3b6c01f91f2507a200ff397b7acb87d1c2db094651fec8b31f9484ff1296fe42` |
| `core/simpls.hpp` | `901874fa69159d28d70bdf87a4ee8e27c469620c9f517080981eff9e6210ef85` |

The SIMPLS change retains cross-block score correlations in the Gram matrix
used by LDA. The matrix change rejects allocation-size multiplication overflow.
The Python API and the remaining vendored source did not change. All 30
vendored-file hashes in `vendor/PROVENANCE.json` were verified after the update.

## Effect on the reported Bora cohort

The original manuscript model contains 967 training groups with 2560 float64
predictors and 25 represented tissue classes. Refit with 24 components under
the corrected core, it gave the same 967 training predictions as the previous
core. For the 116 saved evaluation fields, we compared the corrected head's
predictions against every saved pre-correction class assignment. All **399
whole-group predictions were identical** (zero changed fields). Bora geometry,
RGB conversion, polygon export and ground truth were not recomputed in this
audit. Because each field has the same saved geometry and each group the same
class, its published final mask and quality score are unaffected by this core
correction. This check is specific to the recorded cohort and is not a general
equivalence claim for other data. The source artifact and saved-result hashes
are recorded in [`vendor/FASTPLS_CORRECTION_AUDIT.json`](../vendor/FASTPLS_CORRECTION_AUDIT.json).

The regression test `test_corrected_fastpls_lda_handles_wide_24_component_fit`
also exercises the upstream 40×2560 float64 failure case directly, requesting
24 SIMPLS–LDA components. The build and tests are CPU-only; no encoder, GPU or
HPC work was run for this update.

## Artifact compatibility

Bora's NPZ model contains training descriptors and labels and refits the native
head when loaded. It now records both the binding and core revisions, and it
rejects artifacts pinned to the older implementation rather than silently
changing their predictions. Recreate an artifact with `bora train` using the
same training manifest, or explicitly refit from a retained training descriptor
matrix and labels before saving a new artifact.

For an older Bora NPZ artifact that you created and trust, the latter path is:

```python
import json
import numpy as np
from bora.classifier import NeighborContextClassifier

with np.load("old-model.npz", allow_pickle=False) as old:
    if set(old.files) != {"x", "y", "metadata"}:
        raise ValueError("Unexpected model artifact")
    metadata = json.loads(str(old["metadata"].item()))
    if metadata.get("format") != "bora-neighbor-context-v1":
        raise ValueError("Unexpected model format")
    x, y = old["x"], old["y"]

NeighborContextClassifier().fit(x, y, metadata["classes"]).save("corrected-model.npz")
```

Keep the old artifact for provenance; use a new output name. This explicit
operation refits the classifier under the corrected core, so verify any new
predictions on the target cohort before comparing results across versions.
