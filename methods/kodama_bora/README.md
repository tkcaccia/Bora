# KODAMA–Bora and fastPLS group naming

`run_wsi_kodama.R` is the corrected, pinned R 4.6/KODAMA 0.99.7
reference for H&E-only tile grouping. It was exercised on a training pilot.
Bora's Python core under `src/bora/` performs group-based feature selection,
H&E-guided projection, physical contour cleanup, and registered raster/GeoJSON
export. The existing 13 public WSI group results are pinned outputs and retain
their original group IDs and colors.

`verify_trained_model.py` and `stage_wsi_prediction.py` are the executed
research sources for the separate fastPLS tissue-name bridge. The classifier
uses expert targets from admitted Benchmark 1 training images and does not
alter KODAMA group geometry. A published group name is a prediction, not
whole-slide expert validation. The fitted fastPLS artifact is identified by
SHA256 in the validation freeze record and remains in the local data store.

The attempted fresh eight-class KODAMA/group-majority refit was rejected by
its training-only rare-class viability gate. This release does not claim a
new eight-class KODAMA–fastPLS validation or make its unsuccessful candidate
a final model. New evaluations must bind their own input manifest and
source-group split before opening reference masks.
