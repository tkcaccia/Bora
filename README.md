# Bora

**Bora's current tissue-classification method is the eight-class supervised
Virchow2 ridge model.** It takes a complete, aligned 1,280-channel Virchow2
spatial-token grid from an H&E image, selects 203 channels within the fitted
model, and projects class scores to a pixel-level mask. The retained labels are
tumor, stroma, necrosis, epithelium, blood, vessel, adipose, and immune
infiltrate. ROI output is a class-mask PNG; the whole-slide route also produces
a full-resolution TIFF and native-coordinate GeoJSON. The corrected PEsO
two-class head is separate because its stroma label is an H&E-derived **weak
proxy**, not expert annotation.
The deployed 203-channel fit is retained for reproducibility. A later
training-only comparison that removed known cross-source TCGA patient overlap
favored an all-channel ridge control; channel selection is therefore not
established as an accuracy improvement.

The installable package exposes `bora fit`, `bora predict`, `bora evaluate`,
and `bora verify`. Start with the [reproduce-one-result guide](docs/reproduce-one-result.md)
for a SHA256-pinned BCSS ROI, or read the [supervised command
contract](docs/supervised-commands.md). Encoder inference and image calibration
are upstream: these commands consume existing feature caches and run on CPU.
The frozen [validation code release](docs/VALIDATION_FREEZE_2026-11-12.md)
records source identities and limits. The 87-image Benchmark 1 gallery is an
inspected development set, not an independent external test. The original
training-selection folds were later found to overlap on known TCGA patient
IDs across source datasets; their selection scores do not establish
patient-disjoint validation. Whole-slide masks have no independent dense
whole-slide expert reference.

## Install

Use Python 3.10–3.13 and a C++17 compiler. Linux needs BLAS/LAPACK development
libraries (`libopenblas-dev liblapack-dev` on Ubuntu); macOS uses Accelerate.
The MIT-licensed fastPLS core is vendored at audited revisions, with notices
in `vendor/`. The optional whole-slide reader requires libvips.

```bash
git clone https://github.com/tkcaccia/Bora.git
cd Bora
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,wsi]'
bora --help
pytest -q
```

## Current eight-class workflow

1. Prepare a calibrated H&E image, its measured coordinate transform and a
   complete 1,280-channel Virchow2 feature grid. Keep encoder checkpoint,
   preprocessing, grid alignment and SHA256 provenance with the cache.
2. Freeze the eight-class crosswalk and training split. Fit the classwise
   top-32-feature ridge head on training tokens with `bora fit`; the union in
   the accepted model contains 203 channels. Never tune on evaluation masks.
3. Use `bora predict` with a checksum-bound manifest. ROI inference can restrict
   predictions to declared dataset classes for the historical source-conditioned
   display or leave all eight classes eligible for unrestricted scoring. It smooths contours and
   reassigns small islands. Whole-slide inference uses reviewed Bora tissue
   support and a training-frozen class-presence rule before physical cleanup
   and vector export.
4. Use `bora verify` on the model and outputs. Use `bora evaluate` only with
   masks registered in the same frame; report class overlap separately from
   class-agnostic boundary and region geometry.

See [supervised commands](docs/supervised-commands.md) for JSON schemas,
class restrictions, the PEsO variant, geometry thresholds and WSI inputs. New
outputs require a fresh directory; historical results are never overwritten.
The public viewer is in [Bora-Analysis](https://github.com/tkcaccia/Bora-Analysis).

## History and compatibility

- **KODAMA–Bora group route:** The original route clusters admitted Virchow2
  tiles, refines anonymous groups, and optionally names whole groups with a
  separately trained Neighbor-context fastPLS classifier. Group IDs and
  tissue names are different quantities. The frozen KODAMA 0.99.7/R 4.6
  runtime must not be replaced with system KODAMA 3.3. The `prepare`, `train`,
  `refine`, and `kodama-*` commands remain available. See the
  [field workflow](docs/WORKFLOW.md) and [region identity](docs/REGION_IDENTITY.md).
- **Colorectal whole-slide route (0.2.2):** Anonymous tile groups feed one
  slide-wide RGB9+19 response fit, H&E refinement, a global mask, and compact
  GeoJSON. Its historical 400 µm² cleanup and 1.5 µm smoothing belong to that
  frozen result, not the current supervised whole-slide settings. The
  `wsi-fit`, `wsi-project`, `wsi`, and `wsi-crop` commands reproduce it. See the
  [frozen colorectal workflow](docs/FROZEN_COLORECTAL_WSI.md) and
  [profile](src/bora/frozen_colorectal.json).
- **Earlier field/PCA32 compatibility APIs:** The three-group PCA32 and
  field-level commands remain for historical reproduction. Their settings do
  not override the eight-class method.
- **Release lineage:** 0.2.3 added Chiamaka-local full-channel and tissue
  support fixes; 0.2.4 added MIT distribution metadata; 0.3.0 adds the
  supervised commands while retaining historical routes. Source hashes and
  benchmark qualifications are in the [validation freeze](docs/VALIDATION_FREEZE_2026-11-12.md).

The installed 0.3.0 route reproduced the frozen case-001 and case-069 ROI
masks exactly and matched an approved public ovarian slide in all 218,365,952
full-resolution mask pixels, raw score values and native polygon features.
Those are regression checks, not tissue-accuracy estimates or a claim that
every public slide was reprocessed by the package. The historical finalizer
and its receipts remain frozen. Installation and import do not launch a GPU,
background search, or benchmark campaign.

## License

Bora software is released under the [MIT License](LICENSE). Vendored fastPLS
notices and Stefano Cacciatore's copyright remain in `vendor/` and
`src/bora/_vendor/`. This software license does not grant rights to third-party
images, annotations, encoder checkpoints, benchmark media or manuscript text.
