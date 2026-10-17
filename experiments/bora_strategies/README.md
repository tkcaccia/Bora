# Expanded retained-strategy Bora benchmark

Active scope: B02, B03, B05, B08, B09, B10, B12 and B19, with RGB conversions
#9, #10, #15, #19 and equal-evidence 9+19. Strategy numbers and RGB conversion
numbers are separate. Rejected implementations have been removed. Historical
results and their saved runtime sources remain archives, not active modules.
The production Bora package is unchanged by benchmark strategy removal.

B19 revision 2 votes across B02/B05/B09/B10; ties use appearance. This replaces
its historical dependency on rejected B01 and B06. Seven other retained solvers
and all five retained conversions keep their previous parameters.

## Expanded experiment

Chiamaka outputs:

- `/mnt/sata_ssd/Bora-benchmark/results/virchow2-expanded-inputs-2026-10-17/`
- `/mnt/sata_ssd/Bora-benchmark/results/bora-retained-expanded-2026-10-17/`

The input plan retains 49 previously inspected cases and adds 24 TIGER and
76 BCSS groups: 149 ROIs total. TIGER is non-TCGA and BCSS is TCGA. New groups
are disjoint from the previous plan. Available BEETLE and eligible CAMELYON
sources were already represented, so new-only results cover two datasets.
The full corpus covers four datasets. There are 5,960 final masks.

```bash
python -m experiments.virchow2_multi.plan --data-root DATA --root INPUTS \
  --previous OLD_MULTICASE --quotas tiger=40 bcss=92 camelyon16-subset=7 beetle=10 --seed 8811
python -m experiments.bora_strategies.prepare --root INPUTS \
  --previous OLD_MULTICASE --model-dir CHECKPOINT
python -m experiments.bora_strategies.run --source INPUTS --root OUTPUTS --workers 2
python -m experiments.bora_strategies.score --source INPUTS --root OUTPUTS --workers 2
python -m experiments.bora_strategies.publish --source INPUTS --root OUTPUTS
python -m experiments.bora_strategies.audit --source INPUTS --root OUTPUTS \
  --previous-results OLD_TWENTY_STRATEGY_OUTPUTS
```

Freeze all retained RGB inputs, then every prediction, before scoring fine
references. Selection and fitting use original reference-derived coarse labels
only. Encode the entire RGB field. Remove independent RGB-detected background
after segmentation. Score original annotation-valid pixels, without restricting
them to detected tissue. Thus unannotated tissue is predicted but unscored, and
annotated tissue falsely removed by the detector remains penalized.

The tile control is the completed common input projected to native coordinates
and clipped by the same background support. Aggregate scores average conversions
within case, cases within dataset, then equal datasets. Report new-only results
separately. This is controlled refinement with reference-derived per-image
supervision, not independent classifier validation. Unknown optical resolution
and objective magnification are not inferred from pixel spacing.

`gallery.html` contains synchronized semi-transparent original/reference/tile/
final masks, all five conversion images, native TIFF/PNG downloads, class legends,
actual native/working/analysis/display sampling, scores and cohort filters.
`all_outputs.html` includes every case/conversion comparison sheet.
`AUDIT.json` verifies tissue coverage, background clipping, fine scoring support,
label palettes, hashes, group separation, and pixel-identical previous RGBs and
seven unchanged solvers on the original cases. Revised B19 is excluded from that
historical-output parity requirement.
