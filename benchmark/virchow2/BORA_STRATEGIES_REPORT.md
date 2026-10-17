# Twenty Bora inference strategies on the frozen 49-ROI benchmark

Best observed individual pipeline: **9 + 19 RGB with B09 geodesic competition**,
native macro IoU **0.759756**, versus **0.758606** for B01 on that same input.
Boundary F1 at 8 µm is **0.509499 versus 0.495495**. The paired descriptive IoU
difference interval is **[-0.002191, +0.004640]**, with 27 improved cases, 21
worse and one tied. This is a post-comparison candidate, not held-out evidence
of general superiority. The aggregate strategy ranking below averages all seven
inputs and therefore answers a different question.

Highest measured aggregate macro IoU: **B01 Current balanced wand**, 0.755587; change versus current B01 +0.000000.

All 49 ROIs from TIGER (16), BCSS (16), CAMELYON16 (7) and BEETLE (10) are evaluated with the same seven retained RGB conversions: 5, 8, 9, 10, 15, 19, and 9 + 19. The 20 strategies produce **6,860 final native masks**. RGB conversion #4 remains rejected. Strategy IDs B01–B20 are separate from RGB conversion IDs. No unfavorable result is discarded.

## Distinct hypotheses

| Strategy | Mechanism |
|---|---|
| B01 Current balanced wand | Local appearance competition in the existing boundary band, median9, ds2; exact previous baseline. |
| B02 Whole-field appearance classification | Class-conditional Gaussian appearance learned from original coarse interiors relabels the entire field; sparse tile-center anchors. |
| B03 Multiscale posterior agreement | Fit class appearance independently after 1, 4 and 12 µm averaging; combine posteriors geometrically over the whole field. |
| B04 Region snapping | Partition converted RGB into SLIC regions; assign complete regions from original coarse-interior votes, with appearance for unlabelled regions. |
| B05 Seeded watershed | Flood the converted-image gradient from sparse original tile-center seeds; all pixels participate. |
| B06 Graph random-walker propagation | Screened harmonic propagation on the converted-image region adjacency graph, with coarse anchors and a weak appearance source. |
| B07 Alpha-expansion graph cut | Optimize a multiclass appearance/Potts energy through graph-cut expansion moves on image regions. |
| B08 Finite-range CRF | Mean-field posterior inference with contrast-weighted messages at three physical neighborhood radii. |
| B09 Geodesic competition | Each coarse class expands along minimum cumulative appearance/edge travel cost; competing geodesic distances select the label. |
| B10 Convex total-variation partition | Primal-dual optimization of a simplex-valued class field with appearance fidelity and isotropic TV regularity. |
| B11 Vector region active contour | Discrete vector Chan–Vese-style region competition jointly updates class appearance means and curvature costs. |
| B12 Hierarchical region merging | Agglomerate adjacent image regions by adaptive mean-color distance while preventing conflicting coarse-anchor mergers. |
| B13 Spectral architecture partition | Partition the normalized image-region graph spectrally; assign classes to global appearance clusters using coarse supervision. |
| B14 Digital topology-preserving moves | Allow full-field unary-driven label changes only at digital simple points for donor and recipient classes. |
| B15 Object-context region classifier | Use original H&E nuclei/lumen/context features plus converted evidence to classify whole H&E regions from coarse interiors. |
| B16 Area-constrained transport | Entropy-regularized pixel-to-class transport matches soft class masses to completed coarse tissue proportions. |
| B17 Nonlocal patch voting | Retrieve similar multiscale converted-image patches from original coarse interiors and vote their labels, including distant tissue regions. |
| B18 Elastic mask deformation | Optimize one smooth displacement field shared by all initial classes against image appearance; geometry, rather than independent labels, is the variable. |
| B19 Independent-solver consensus | Combine whole-field appearance, graph propagation, TV partition and current-wand decisions; resolve tied votes by appearance. |
| B20 Coarse-supervised dilated CNN | Fit a small per-image network on original coarse interiors using converted RGB and H&E, with class-balanced patch sampling and no fine-label access. |

These are experimental implementations of distinct inference ideas, not twenty proven methods or twenty claims of methodological novelty. Some share an appearance source or graph representation; their inference variables and objectives differ. B08 uses finite-range local mean-field messages, not a fully connected dense-CRF implementation. B11 is discrete vector region competition inspired by Chan–Vese, not an exact reproduction of the continuous level-set solver. B18 has a smooth bounded displacement and does not guarantee diffeomorphic invertibility.

## Fixed inputs, supervision and geometry

Exact SHA-verified previous converted RGBs and original coarse masks are reused. Supervised fits use original coarse-interior labels only. Completed coarse labels initialize geometry or supply a coarse area prior for specified strategies; they are not treated as new supervised training annotations. B11 updates appearance from its evolving region assignments as part of its unsupervised energy optimization. Fine-reference pixels are absent from prediction code. Each scoring pass follows a complete prediction freeze and hash verification.

B01 copies each previous current native mask pixel for pixel. Its analysis TIFF is a resampled copy for the shared B19 consensus grid; its scored native mask is the exact prior output. New solvers resample each axis to approximately **1 µm/px** using recorded image sampling, then project categorical predictions to the exact native reference coordinates by nearest neighbor. Actual analysis shape and per-axis spacing are saved per conversion. Region target spacing is 15 µm, capped at 4,096 regions. B17 retrieves descriptors at approximately 4 µm spacing; B18 optimizes flow internally on a fourfold coarser grid before interpolating the displacement. All morphology and contextual distances use physical spacing where specified. These are ROIs rather than full-WSI inference; objective magnification and optical resolving power remain unknown.

Sparse original tile-center anchors, rather than complete fixed class interiors, guide most new solvers. Region-based methods may fix whole regions that contain anchors; elastic deformation retains its shared geometry instead of forcibly resetting anchor pixels. This enables whole-region revisions. The gallery reports what fraction of tissue outside B01’s editable band is relabelled.

Every detected tissue pixel receives a positive label, including unannotated tissue. The independent previous native RGB tissue mask removes background only after segmentation. The original annotation-valid scoring support is unchanged and never intersected with detected tissue. Therefore unannotated tissue is displayed but unscored, while annotated tissue removed by the detector remains penalized. All original reference classes remain in macro scoring, including those missing from coarse supervision. CAMELYON non-tumor reference is automatic tissue support, not expert stroma.

## Ranking

Primary score: mean native macro IoU over the seven conversions within each case, then cases within each dataset, then equal weight over four datasets. Per-conversion rankings are separate. A 3,000-resample paired case bootstrap within dataset keeps all seven correlated conversions together. The intervals are descriptive on reused cases. Boundary F1 at 8 µm is secondary; scores at 4 and 16 µm and per-class IoU/Dice are also retained.

| Rank | Strategy | Native macro IoU | Change vs B01 | Paired descriptive 95% interval | Boundary F1 at 8 µm |
|---|---|---:|---:|---|---:|
| 1 | B01 Current balanced wand | 0.755587 | +0.000000 | [+0.000000, +0.000000] | 0.492152 |
| 2 | B09 Geodesic competition | 0.755335 | -0.000253 | [-0.003799, +0.003561] | 0.505071 |
| 3 | B18 Elastic mask deformation | 0.751282 | -0.004306 | [-0.008671, +0.000350] | 0.485808 |
| 4 | B14 Digital topology-preserving moves | 0.750099 | -0.005488 | [-0.007962, -0.003184] | 0.484117 |
| 5 | B03 Multiscale posterior agreement | 0.742830 | -0.012757 | [-0.022027, -0.003221] | 0.501879 |
| 6 | B06 Graph random-walker propagation | 0.742471 | -0.013116 | [-0.020439, -0.006298] | 0.509444 |
| 7 | B19 Independent-solver consensus | 0.740272 | -0.015315 | [-0.025862, -0.004569] | 0.512432 |
| 8 | B10 Convex total-variation partition | 0.739954 | -0.015634 | [-0.026250, -0.004725] | 0.509478 |
| 9 | B05 Seeded watershed | 0.739607 | -0.015980 | [-0.022385, -0.009157] | 0.487530 |
| 10 | B07 Alpha-expansion graph cut | 0.739338 | -0.016249 | [-0.024654, -0.007961] | 0.506648 |
| 11 | B08 Finite-range CRF | 0.738197 | -0.017391 | [-0.028110, -0.006316] | 0.510506 |
| 12 | B17 Nonlocal patch voting | 0.737336 | -0.018252 | [-0.026524, -0.009813] | 0.474108 |
| 13 | B02 Whole-field appearance classification | 0.736551 | -0.019037 | [-0.030061, -0.007607] | 0.508399 |
| 14 | B12 Hierarchical region merging | 0.735903 | -0.019684 | [-0.028096, -0.011773] | 0.499971 |
| 15 | B15 Object-context region classifier | 0.728019 | -0.027569 | [-0.035376, -0.020355] | 0.452033 |
| 16 | B04 Region snapping | 0.724658 | -0.030929 | [-0.037462, -0.024598] | 0.444116 |
| 17 | B11 Vector region active contour | 0.723891 | -0.031697 | [-0.040540, -0.023290] | 0.378746 |
| 18 | B16 Area-constrained transport | 0.692073 | -0.063514 | [-0.085058, -0.043782] | 0.440110 |
| 19 | B20 Coarse-supervised dilated CNN | 0.687369 | -0.068218 | [-0.089344, -0.049390] | 0.399662 |
| 20 | B13 Spectral architecture partition | 0.598579 | -0.157008 | [-0.191815, -0.123382] | 0.253847 |

## Per-conversion comparison

| RGB input | Best measured strategy | IoU | Change vs current B01 |
|---|---|---:|---:|
| 05 | B01 Current balanced wand | 0.753246 | +0.000000 |
| 08 | B01 Current balanced wand | 0.753534 | +0.000000 |
| 09 | B01 Current balanced wand | 0.758203 | +0.000000 |
| 10 | B09 Geodesic competition | 0.757851 | +0.002149 |
| 15 | B01 Current balanced wand | 0.753404 | +0.000000 |
| 19 | B09 Geodesic competition | 0.757374 | +0.000957 |
| ensemble | B09 Geodesic competition | 0.759756 | +0.001150 |

The prespecified expert-tissue subset excludes CAMELYON's automatically generated non-tumor reference. On the three remaining datasets, B09 mean IoU is 0.726149, versus B01 0.724743. B09's means are slightly higher on BCSS, TIGER and BEETLE, but lower on CAMELYON. These are small, descriptive differences on reused cases; the primary four-dataset interval for the B09–B01 IoU difference includes zero. B09 has a higher observed boundary F1 at 8 µm across all four dataset means.

## Interpretation and reproducibility

The same cases and coarse labels were previously inspected, and coarse labels are reference-derived. This is exploratory controlled refinement, not independent held-out classification or clinical validation. Coarse interiors are therefore usually already correct; this design does not independently test recovery from realistic whole-region classification errors. Patient grouping is incompletely verified, so case-bootstrap intervals are descriptive. The best measured strategy is selected after examining these cases; it needs a locked independent test cohort before a generalization claim. The score does not establish that unannotated predictions are correct. Whole-field alternatives may mislabel large regions or lose classes; all such outputs remain visible and receive their full macro-score penalties. B14 topology constraints apply before the common background clipping, which can itself change topology.

The original smoothing ablation is preserved separately under virchow2-smoothing-2026-10-17. Removing median9 altered only about 0.23% of tissue for RGB #9 on tiger_10. Across seven conversions and four equally weighted datasets, current median9 macro IoU was 0.755587 versus 0.755049 without the label filter, while boundary F1 at 8 µm was 0.492152 versus 0.497364. Thus the ablation showed a small overlap/boundary tradeoff rather than universal harm from smoothing. The expanded study changes inference mechanisms rather than merely retuning that filter.

All parameters are fixed in PROTOCOL.json. A geometry review found that B17 originally expanded [::step] samples using generic resize centers, which introduced an approximately half-sample offset. A synthetic linear-coordinate identity test exposed the issue. B17 was regenerated for all 343 pairs with interpolation at the actual sampled pixel coordinates. The initial predictions, scores and code are archived under bora-strategies-initial-grid-2026-10-17. The other 19 outputs were retained byte for byte, with unchanged function ASTs verified. Their scores were reused only after native-mask hashes, reference-plan hashes, spacing and evaluator source matched. Final B17 scores were recomputed after the corrected complete prediction set was frozen. CORRECTION_LINEAGE.json records this revision; it was a coordinate correction, not a score-driven parameter search.

The CNN is a small per-image coarse-only fit with fixed seed, 80 class-balanced patches, original H&E and converted RGB inputs, dilations 1/4/8 and no fine-mask training. Its stage losses and parameter count are saved. B19 combines B01/B02/B06/B10 votes and resolves ties by appearance. Fit code, seeds, input hashes, package versions and frozen output hashes are retained; no pretrained model or model download is introduced.

Stage times in details.json exclude shared preparation and may reuse cached region graphs. B01 time measures copying an existing baseline, not its original inference. These times are not standalone runtime rankings. CPU processing uses three low-priority workers with two computational threads each on Chiamaka. Initial unscored timing runs are preserved separately.

## Files and views

- gallery.html: same original/reference/tile/converted input above all 20 masks, synchronized crop, zoom and opacity, exact magenta differences from B01, native TIFFs, scores and per-class details.
- all_outputs.html: all 49 cases × seven conversions, every 20-strategy comparison sheet. Click into the native-mask viewer for detail.
- cases/<case>/<conversion>/<strategy>/: native class TIFF, analysis-grid TIFF, lossless indexed class PNG, exact difference map, display previews and diagnostics. Display previews are nearest-neighbor resamples capped at 1,536 pixels; native PNG/TIFF files preserve full resolution.
- results.json / scores.csv: all 6,860 native scores and aggregate/per-conversion rankings.
- PROTOCOL.json / FROZEN.json / ALL_PREDICTIONS_FROZEN.json / AUDIT.json: parameters, provenance, inputs, prediction hashes and invariant checks.
- Keep preceding full-tissue and multi-image folders alongside these outputs for the original RGB, annotation masks and actual RGB conversions.

## Methodological sources

- [scikit-image segmentation API](https://scikit-image.org/docs/0.25.x/api/skimage.segmentation.html): SLIC and seeded watershed.
- [Boykov, Veksler and Zabih: graph-cut expansion moves](https://www.cs.cornell.edu/rdz/Papers/BVZ-pami01-final.pdf).
- [Shi and Malik: normalized graph partitioning](https://people.eecs.berkeley.edu/~malik/papers/SM-ncut.pdf).
- [Chan, Sandberg and Vese: vector-valued region active contours](https://www.math.ucla.edu/~lvese/PAPERS/JVCIR2000.pdf).
- [Krähenbühl and Koltun: Gaussian-potential CRF inference](https://proceedings.neurips.cc/paper/4296-efficient-inference-in-fully-connected-crfs-with-gaussian-edge-potentials.pdf); B08 is explicitly a simpler finite-range approximation.
