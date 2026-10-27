# Audit of Bora mask to GeoJSON export

This audit covers the current region-of-interest export in `src/bora/pipeline.py` and the separate whole-slide export in `src/bora/wsi_export.py`. It evaluates whether vectorization preserves the *predicted* tissue map. It does not measure agreement with expert annotations.

## What the ROI exporter does

1. Connected tissue regions and enclosed background holes smaller than 20 native pixels are merged or filled. Exterior glass stays background.
2. The cleaned label mask is polygonized at native pixel edges. Components smaller than 20 pixels are excluded, and the initial coverage is simplified with a two-pixel tolerance while keeping shared borders coincident.
3. Shared arcs are converted to physical coordinates using both pixel-spacing axes. Junctions are fixed; each arc is simplified once with a requested tolerance of at most 3 µm; faces are assigned to the predicted class with the largest overlap.
4. Candidate geometry must preserve every predicted class, have valid nonoverlapping polygons, have at least 0.97 intersection-over-union for each class against the *preceding polygon coverage*, change at most 1.5% of predicted tissue area, and use fewer contour segments. Tolerances of 3, 2, 1, 0.5 and 0.25 µm are tried in order. The previous coverage is retained if none passes.
5. Continuous polygon parts and enclosed holes below 20 native pixel² are removed or filled. The final GeoJSON is rasterized; that raster is saved as `mask.ome.tif`. A cleanup repair makes the rasterized polygons agree exactly with the finalized raster.

The 20-pixel rule is strict: a part with area exactly 20 is retained. The segment counter walks every polygon ring, so a shared border contributes to both neighboring class counts. It is suitable for before/after ratios, not a count of unique arcs.

## Material defect found and corrected

The previous ROI call supplied `min_area=500` to the first polygon converter although the preceding and subsequent cleanup used 20 pixels. A 100-pixel class-2 island inside class 1 was therefore omitted before the 0.97 geometry check. Rasterizing the resulting polygon coverage produced **100 background pixels and no class-2 pixels**. The geometry check could not detect this loss because it compared against the already-filtered coverage. The pipeline now supplies `min_area=20`; the same 100-pixel island is retained exactly, and the exported GeoJSON rasterizes to the input mask. Additional 20-pixel rectangles of several aspect ratios were retained in a CPU check. A regression test protects the 100-pixel case.

This is a change to the saved mask as well as its GeoJSON: the pipeline derives `mask.ome.tif` from the final vectors. It is not a cosmetic simplification change.

## Saved development evidence and its limit

The earlier 116 ROI exports used the **superseded 500-pixel prefilter**. Their saved diagnostics show how much their accepted shared-boundary fit simplified the preceding polygon coverage:

| Source | Images | Ring segments before → after | Reduction | Minimum per-class IoU across images | Largest changed-tissue fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| BCSS | 42 | 485,666 → 215,923 | 55.5% | 0.9701 | 1.19% |
| TIGER | 26 | 85,700 → 37,096 | 56.7% | 0.9700 | 1.28% |
| IGNITE | 24 | 68,935 → 30,797 | 55.3% | 0.9717 | 1.39% |
| GCSS | 24 | 73,980 → 43,067 | 41.8% | 0.9701 | 1.42% |
| All | 116 | 714,281 → 326,883 | 54.2% | 0.9700 | 1.42% |

No case fell back to the unsimplified coverage. Polygon parts declined from 4,134 to 4,079. These figures describe the **legacy output only**. The IoU comparator was the already-filtered predicted coverage, not the pre-export mask or an expert mask. Consequently, these numbers cannot establish preservation of all 20–499-pixel regions or the performance of the corrected exporter. ROI benchmark masks, segmentation scores, vector counts and runtimes must be regenerated before they are reported as results for the corrected release.

The whole-slide route is different. It polygonizes each of 228 native windows without the ROI 500-pixel prefilter, uses a 0.75-native-pixel coverage simplification with guards and local 20-pixel cleanup, and exported 2,886 features with 11,107,958 ring segments. The recorded vector pass took 1,407.4 seconds on the 75,250 × 48,740-pixel colorectal slide. Its recorded zero raster/vector mismatch verifies the two saved outputs agree; it is not proof that vectorization left the upstream prediction unchanged. The whole-slide result is from one partially annotated slide and does not measure general export performance.

## Required evaluation after re-export

For each image, retain the **pre-export predicted native mask** independently of the GeoJSON. Compare the final raster against it by class: component retention for areas 20–499 pixels, class IoU, changed tissue and background area, and border displacement in micrometres. Audit validity, interclass overlap, holes and parts below 20 pixels, polygon/raster equality, segment reduction, GeoJSON bytes, peak memory and export time. Report distributions by dataset and spacing, plus the worst cases; inspect images where a class disappears or a border crosses an expert region. For the whole-slide route, report the same checks per window and in slide coordinates, including seam continuity. Expert segmentation accuracy must be scored separately from these export-fidelity checks.

Sources: `src/bora/pipeline.py`, `src/bora/multicpu_geojson.py`, `src/bora/compact_geojson.py`, `src/bora/mask_cleanup.py`, `src/bora/wsi_export.py`, `tests/test_pipeline.py`, and saved development records in [Bora-Analysis](https://github.com/tkcaccia/Bora-Analysis/tree/main/benchmark/cmpb_evidence) (`vector_fidelity.json` and `wsi_scale.json`).
