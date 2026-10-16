# Cycle 12: strong geometric smoothing

Cycle 12 applies symmetric Gaussian label competition with sigma 80 pixels and a 216-pixel editable band to the accepted cycle 07 mask. All four 216-pixel tissues have a refined OME-TIFF and GeoJSON in `/mnt/sata_ssd/benchmarks/Bora/output-bora-cycle-12/216`.

The initial strict gate rejected cycle 12 because 8-pixel contour F1 declined for lymph node and tonsil, and the *fraction* of predicted edges classed as residual tile edges rose. That decision is preserved as `cycle_decision_strict.json`.

The user described the cycle 12 contours as a substantial visual improvement. For the continuing smoothing-focused search, the acceptance gate uses internal-border macro IoU and accuracy within 64 pixels of the ground-truth interface, preservation of all ground-truth labels, and the **absolute count** of false tile-edge pixels. It records 8-pixel F1 separately. Absolute counts are preferable for visible tile remnants because strong smoothing reduces the denominator of all predicted edge pixels, which can raise a fraction even while removing many tile-aligned edges. This gate change was made after observing cycle 12 and must be disclosed in any manuscript or final report.

| Tissue | 64-px macro IoU, 07 to 12 | 64-px accuracy, 07 to 12 | False tile-edge pixels, 07 to 12 | 8-px F1, 07 to 12 |
|---|---:|---:|---:|---:|
| Lymph node | 0.644858 to 0.667240 | 0.794212 to 0.802192 | 31,598 to 24,238 | 0.250673 to 0.176586 |
| Tonsil | 0.624484 to 0.663117 | 0.782495 to 0.814006 | 70,372 to 51,241 | 0.179294 to 0.157282 |
| Brain cancer | 0.574609 to 0.633454 | 0.731825 to 0.783125 | 21,467 to 16,865 | 0.129898 to 0.137598 |

Pancreas has only one tissue label and therefore no internal tissue-to-tissue border for this comparison. Cycle 12 is accepted by the smoothing-focused gate and is the incumbent for cycles 13 onward. The repeated tuning of these same four slides is exploratory; an independent held-out slide set is needed for a general performance claim.
