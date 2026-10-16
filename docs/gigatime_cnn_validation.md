# GigaTIME CNN: DAPI audit and corrected Bora pilot

16 October 2026. **The original GigaTIME CNN replaces Flash for this experiment.**
The CNN implementation passes checks against released paired H&E/mIF examples,
and its nuclear localization on the selected TIGER ROI is substantially clearer.
The earlier Flash outputs are rejected for target-image DAPI quality and remain
archived. They must not support manuscript claims about marker-driven Bora.

## What was corrected

The original pilot used an incorrect interpretation of physical sampling. The
authors clarify that each registered 128 µm field is 556 pixels, then resized
to 512 pixels before inference in 256-pixel windows. Thus the intended model
sampling is **0.25 µm/pixel**, not 128/556 µm/pixel and not the native TIGER
0.5 µm/pixel. [Author clarification](https://github.com/prov-gigatime/GigaTIME/issues/4).

Channel 0 is DAPI, as confirmed by the authors; no channel permutation or signal
inversion is applied. RGB/ImageNet normalization and the official CNN
architecture are used. All 212 state tensors load strictly (9,164,055
parameters), with no missing or unexpected weights. The pinned model revision
is `a6556030bdb8071857559e12b921620aa1bbac5b`.
[Channel order](https://github.com/prov-gigatime/GigaTIME/issues/8),
[released model](https://huggingface.co/prov-gigatime/GigaTIME).

Alternative Flash class-token/normalization hypotheses, input RGB/BGR changes,
stain normalization, four physical scales and three fixed native BCSS crops
were investigated and retained. None justified calling the target Flash DAPI
fixed. Macenko normalization improved an H&E stain-agreement proxy but did not
improve the unused paired-reference results, so **production uses unchanged RGB**.
The CNN was selected from a matched disjoint-window comparison: on 45 unused
reference patches, CNN DAPI AUROC was 0.9202 versus Flash 0.9157, and 8-pixel
window correlation was 0.6661 versus 0.6441.

## Reference check of the actual production workflow

The authors' public archive contains 50 paired examples. The first five
lexicographically sorted H&E files were used for diagnostic development; the
remaining 45 were used for confirmation. Patient-level independence of these
patches is unknown, so these are **45 patches, not 45 patients**.

The production check uses the actual overlap-add inference routine, with
256-pixel windows, stride 128, Hann blending and saved float16 maps. Reference
fields are resized from 556 to 512, and mIF-derived nuclear masks are resized
with nearest-neighbor interpolation to the same frame. Mean per-patch results:

| DAPI measure | CNN production workflow |
|---|---:|
| Pixel AUROC | 0.9221 |
| Average precision | 0.6328 |
| Pixel Pearson correlation | 0.5925 |
| Pearson over 8-pixel windows | 0.6701 |
| Dice at activation threshold 0.5 | 0.6132 |
| Mean activation inside reference nuclei | 0.7330 |
| Mean activation outside reference nuclei | 0.0970 |

DAPI localizes to nuclei substantially better than in the rejected pilot, but
false positives, missed nuclei and broad boundaries remain. The numerical
results do not establish perfect cell segmentation. Fixed reference examples
are selected by filename rather than by a favorable score.

All 23 channels were also compared with the released binary references,
restricted to released cell support. This second diagnostic uses binary DAPI
signal rather than the all-nuclei mask above, so its correlation differs. It
does not reproduce every cell-convex-hull relabeling and target-smoothing step
of the publication. An explicitly exploratory eligibility screen requires
mean 8-pixel-window Pearson >=0.30 in at least 20 variable reference patches.
DAPI is visualized but is excluded from protein-feature ranking; TRITC and Cy5
are background channels. **CD20, Tryptase, Actin-D and Transgelin fail the
screen and are excluded from refinement.** Six displayed channels are DAPI,
CK, CD3, CD4, CD68 and Ki67. Complete channel diagnostics are saved as JSON.

## Target image and biological limits

TIGER 114S is an expert-annotated **breast tissue ROI**, not a WSI. Native
1224 × 1166 pixels at provider-reported approximately 0.5 µm/pixel correspond
to approximately 612 × 583 µm. Inference uses 2448 × 2332 pixels at 0.25 µm/pixel;
upsampling does not recover optical detail. The microscope objective,
numerical aperture and optical resolving power remain unknown. Predictions
return to the native frame; Bora's separate 2× working grid is approximately
1 µm/pixel. Expert masks are unchanged.

The TIGER ROI has **no measured DAPI or protein-stain truth**. Nuclear alignment
can be inspected in H&E, but agreement with H&E is not independent validation of
every predicted protein. CK and other channels must remain labelled virtual
predictions, and their absolute activation is not a measured concentration.
The reference results support a checked research implementation, not a claim
that all markers are biologically correct in this breast sample.

## Corrected tissue-boundary pilot

The same coarse masks, RGB baseline and local-wand parameters are used.
Channels are selected from coarse tile summaries, after the reference
eligibility screen; the expert tissue mask enters only after predictions are
fixed. The coarse masks themselves were derived from the expert reference by
`tiling-mask-cpp`: this is controlled boundary reconstruction, not independent
tissue classification. Three tile widths are degradations of one ROI, not
three independent cases. Boundary F1 uses a 4 µm tolerance.

| Tile width | Appearance | Macro IoU | Macro Dice | Boundary F1 | ASSD (µm) |
|---:|---|---:|---:|---:|---:|
| 56 | Coarse input | 0.7819 | 0.8742 | 0.1940 | 12.7766 |
| 56 | RGB Bora | 0.8150 | 0.8954 | 0.3042 | 11.3719 |
| 56 | CNN markers + Bora | 0.7926 | 0.8810 | 0.2512 | 11.8231 |
| 90 | Coarse input | 0.7220 | 0.8341 | 0.1463 | 16.3139 |
| 90 | RGB Bora | 0.7476 | 0.8515 | 0.2037 | 15.8557 |
| 90 | CNN markers + Bora | 0.7448 | 0.8493 | 0.2370 | 13.7574 |
| 216 | Coarse input | 0.5184 | 0.6486 | 0.1555 | 24.2772 |
| 216 | RGB Bora | 0.5515 | 0.6845 | 0.1794 | 19.5947 |
| 216 | CNN markers + Bora | 0.5547 | 0.6890 | 0.2096 | 18.7049 |

At 90 pixels, boundary F1 improves from 0.2037 to 0.2370 and ASSD decreases
from 15.86 to 13.76 µm compared with RGB Bora, while macro IoU is slightly
lower (0.7448 versus 0.7476). At 216 pixels, boundary F1, ASSD and overlap
improve modestly. At 56 pixels RGB Bora is better. These mixed, one-image
results support further testing rather than an overall superiority claim.

## Deliverables and verification

- `dapi_paired_reference_check.png`: fixed unused examples, reference nuclei,
  CNN DAPI and overlays.
- `dapi_correction_zoom.png`: the same TIGER nuclear crop, rejected Flash
  output and replacement CNN DAPI/CK.
- `overlays_tile_90.png`: original H&E, expert mask, tiling-mask input and
  RGB/CNN-marker Bora masks at 35% opacity.
- `marker_overlays_0.250000.png`: DAPI and five eligible protein channels
  over unchanged H&E.
- `results.json`, `scores.csv`, `production_dapi_validation.json`,
  `production_channel_validation.json`: full metrics and provenance.
- `inference_0.250000/`: dense native/model-grid maps on Chiamaka only.

Results on Chiamaka:
`/mnt/sata_ssd/Bora-benchmark/results/gigatime-cnn-pilot-2026-10-16/`.
Local figures, masks and metadata:
`/Users/stefano/Documents/ChatGPT/Bora/benchmark/results/gigatime-cnn-pilot-2026-10-16/`.
All diagnostic alternatives:
`/mnt/sata_ssd/Bora-benchmark/results/gigatime-flash-audit-2026-10-16/`.

[Code and reproducibility guide](../experiments/gigatime_flash/README.md).
The existing environment of other projects was not modified. All weights,
raw data and dense maps remain outside Git. Fifteen focused tests pass,
including unchanged RGB behavior, feature-driven boundaries, physical frame
preservation, adapter arithmetic and invalidation of stale inference caches.
The actual production reference check and overlay figures were also reviewed.
