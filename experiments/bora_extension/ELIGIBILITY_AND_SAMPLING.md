# Prospective multigroup eligibility and sampling review

The user requested excluding images with only one group, imposing a minimum
number of real tiles, and investigating WSSS4LUAD/new-dataset resolution.
The new eligibility audit accepts **299 of the349 existing development ROIs**.
It does not run an encoder or Bora, read fine reference mask values, or consult
algorithm predictions, accuracy or timing. Eligibility rules are frozen in
`eligibility_policy.json`; all original inputs and signed cycles1–10 remain
unchanged. A changed denominator is not an improvement in segmentation.

| Dataset | Original inventory | Prospective benchmark | Diagnostic exclusions |
| --- | ---: | ---: | ---: |
| BCSS |132|132|0|
| TIGER |80|80|0|
| WSSS4LUAD |40|9|31|
| RINGS tumour |40|21|19|
| OCELOT |40|40|0|
| BEETLE |10|10|0|
| CAMELYON16 windows |7|7|0|
| **Total** |**349**|**299**|**50**|

## Exactly what qualifies

An image must have at least two positive compact categories in its original
input tiled mask **and** at least two categories in the actual coarse tile
observations used for feature selection. Compact0 is ignore and does not count
as a group. RINGS outside-tumour is a valid compact category, including true
negative tissue/glass, and is not an expert stroma annotation.

Both real, unpadded image axes must span at least four complete224px encoder
fields: a4×4 block,896px along each axis, at the frozen base sampling. Count
geometric capacity, not overlapping224px windows, repeated14px tokens or
reflection padding. Capacity does not establish statistical independence.
The decision is anchored to the frozen image hash and geometry; arbitrary
upsampling cannot rescue an excluded image. A later sampling change needs a
new geometry audit retaining this original eligibility anchor.

At least16 original usable labelled90px coarse tiles are also required. The
audit calls the converter's actual `coarse_tile_observations` function, rather
than approximating that count. A tile needs at least80% of its full side on
both axes, at least95% one positive category and real token centres in its
central half. Counts for every group are saved. There is no additional
per-group sample cutoff: a rare group with one usable tile remains included.
All reference-positive pixels, including fine classes absent from coarse
input, remain errors when missed. The fine reference is never used to remove
hard pixels or qualify a case in this audit.

The4×4 and16-tile limits are explicit engineering choices for spatial context
and per-image fitting, not established biological/statistical sample-size
requirements. They were not tuned against scores. The cohort remains repeatedly
reused development data, not independent validation.

There are27 single-group coarse masks:8 WSSS and19 RINGS. Three additional WSSS
images contain more than one mask category but provide only one eligible
feature-selection category. Thus30 images fail the usable-group requirement.
All31 small WSSS images fail the real-field requirement;28 also have fewer
than16 usable coarse tiles. Eleven of these31 also fail the usable-group rule.
The union is50 exclusions. All299 accepted images have at least two input and
observed groups, adequate real field geometry and at least16 usable tiles.

## What the resolution investigation establishes

The [original WSSS4LUAD paper](https://arxiv.org/html/2204.06455v2) describes
released patches at10× magnification, including9 large and31 small validation
fields. Its source scanner spacing must not be assigned directly to every
distributed10× patch. Per-file pixel calibration remains unverified in the
downloaded mirror, so physical lengths/areas stay null.

[Virchow2's provider model card](https://huggingface.co/paige-ai/Virchow2/blob/main/README.md)
documents224px encoder inputs and training at0.25,0.5,1 and2µm/px
(nominal40×,20×,10× and5×). Native10× WSSS input is therefore not intrinsically
outside its documented magnifications. However, our unchanged factor1/2/4
image pyramid supplies nominal10×/5×/2.5× contexts. The widest2.5× context is
below the documented training magnifications. This is a scale mismatch to
investigate, not proof that it alone caused the observed score difference.

Across the40 WSSS cases,3 base fields,30 half-scale fields and31 quarter-scale
fields need reflection to reach224px on at least one axis. Those contexts
cannot be counted as independent real tissue. For example, WSSS01 has370×419
real pixels: capacity for one complete base field. Its quarter-scale field is
only92×105 pixels, with132/119px right/bottom minimum padding. Reflection fills
an encoder input but contributes no new tissue information.

All9 accepted WSSS fields are1280×1280 native-pixel crops and have complete
real windows at all three scales. Their nominal2.5× context issue remains
unresolved; passing the tile/group filter does not certify microscopy sampling.
Before cycle11 inference, freeze a separate geometry comparison on these9
fields, preserving input tile labels, RGB conversion formulae, original image
identity and native reference geometry. Resampling must be reported as
resampling and cannot create optical detail or qualifying tile count. A new
geometry version needs a matched unchanged-parent baseline before new methods
can claim improvement.

The other309 images have no minimum-window padding at any of the three scales.
BCSS,TIGER,OCELOT,BEETLE and CAMELYON16 have recorded per-axis spacing;
base/half/quarter sampling is approximately0.5/1/2µm/px, with actual rounding
retained. RINGS has adequate native-pixel field size but unverified distributed
MPP, so it remains explicitly exploratory in pixel units. Its
[publisher release](https://data.mendeley.com/datasets/h8bdwrtnr5/1) does not
establish per-file calibration. Magnification alone is insufficient to infer it.

OCELOT uses the provider's **resized tissue-patch axes**, cross-checked against
source-coordinate widths/heights and slide axes, rather than source scanner
or cell-patch MPP. The40 selected fields retain that metadata. See the
[provider dataset description](https://lunit-io.github.io/research/ocelot_dataset/)
and [original1.0.1 release](https://zenodo.org/records/8417503).

Encoder window224px, spatial token14px, coarse supervision90px, and Bora's
analysis grid are different quantities. At0.5µm/px, they correspond to112µm,
7µm and45µm for the first three. Unknown-MPP inputs use pixels, so the same
numerical constants do not establish matched physical tissue scale.

## Saved evidence and use

Chiamaka report: `/mnt/sata_ssd/Bora-benchmark/results/bora-eligibility-2026-10-18-v1`.
`ELIGIBILITY.json` contains every decision, original input hashes, usable tile
coordinates/counts and per-group counts. `POLICY.json` and `PROTOCOL.json` were
written before case inspection. `SAMPLING_REVIEW.json` checks1047 cached scale
metadata records against the frozen declarations and resized geometry. It does
not rehash feature array bytes; earlier feature audits did that.

Prospective input view:
`/mnt/sata_ssd/Bora-benchmark/results/bora-qualified-inputs-2026-10-18-v1`.
Its `PLAN.json` and `cases/` contain only299 accepted inputs. The50 excluded
inputs are symlinked separately under `diagnostics/`; they cannot enter its
quality/runtime denominator or ranking. No source data are duplicated/deleted.
All original349 inputs remain available at their original immutable path.
The runner scope gate rejects the old unfiltered349-image source from cycle11
onward, verifies the eligibility manifest, and rejects any included row that
fails the frozen group/tile rules.

The inventory viewer has separate included/diagnostic modes. It displays
original H&E, reference, input tiled mask, fixed converted RGB and the existing
five cycle10 masks. These are historical outputs for inspection, not a new
cycle or new score. Cycle11 is still unexecuted. Ten of twenty cycles are
complete; the sampling comparison remains the next preflight step.
