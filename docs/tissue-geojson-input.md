# Tissue-area GeoJSON input for whole-slide Bora

The `bora wsi` case manifest may include `quality_control` with format
`bora-background-only-support-v1`. Its `tissue_geojson` points to a GeoJSON
FeatureCollection whose polygon coordinates are **pixels in the complete
calibrated H&E image** (`coordinate_system: calibrated_image_pixel_xy`). Each
feature marks tissue. Polygon holes and areas outside the features are
background. A reviewed GeoJSON is an input to Bora's final spatial support;
the output receipt records its checksum.

```json
{
  "quality_control": {
    "format": "bora-background-only-support-v1",
    "mode": "background-only",
    "tissue_geojson": {"path": "/path/tissue-area.working.geojson", "sha256": "..."},
    "mask": {"path": "/path/tissue-support-mask.tif", "sha256": "..."},
    "summary": {"path": "/path/BACKGROUND_ONLY_QC.json", "sha256": "..."}
  }
}
```

The binary mask is a reusable 5 µm/pixel raster cache of the GeoJSON. Bora
binds all three files to the calibrated image, rasterizes the GeoJSON and
requires exact pixel agreement with the cache before inference. It then
resamples the verified support to the calibrated image for whole-slide
prediction. The same cached support supplies exact per-tile background area
for the strict clustering threshold of **less than 10%**. Virchow2's 1,280
raw channels remain available independently of that tile threshold.

For the 30 October whole-slide benchmark, the tissue geometry excludes
external glass, large enclosed white voids and user-identified scanner areas,
while retaining small enclosed white adipose spaces. GrandQC predictions are
retained for diagnosis but do not enter tile admission or final support.
