# Train and use the field-level compatibility API

For the selected full-slide segmentation route, use the
[frozen colorectal workflow](FROZEN_COLORECTAL_WSI.md). The field-level case
format below remains supported for earlier results and tissue-model training.

Bora separates **anonymous region geometry** from **tissue identity**. At
inference it reads a calibrated H&E image, tile-cluster labels, full Virchow2
spatial tokens and a fitted Neighbor context fastPLS model. It refines the
clusters into complete tissue groups, predicts one tissue class for each group,
and writes a classified native-resolution TIFF plus GeoJSON. A disconnected
group has one shared identity. Input cluster numbers need not match the tissue
class numbers used by the model.

## 1. Make a calibrated case

An image and every derived input must cover the same physical field. The
working H&E is uint8 RGB at approximately 0.5 µm/px. Virchow2 features are
the full 1280-channel spatial-token grid from the pinned checkpoint and have
shape `ceil(height/14) × ceil(width/14) × 1280`; padded edge tokens retain
their original grid coordinates. Tissue support is a binary image on the
working grid. If native output has a different resolution, supply its own
binary support and actual per-axis native spacing. Bora checks the physical
field of view before processing.

The cluster map can be an aligned categorical TIFF in `coarse`, or a 2D NumPy
grid of tile assignments in `tile_labels`. For a grid, `tile_size_yx` gives
height and width in **working-image pixels**. The first tile begins at `(0,0)`;
the last row/column may be partial. If clustering used another tile origin,
write the expanded raster explicitly. IDs must be integers `1..255` and zero
means unassigned. At least 16 usable tiles and two groups are required.

```json
{
  "image": "he-working.png",
  "tile_labels": "clusters.npy",
  "tile_size_yx": [90, 90],
  "tissue": "tissue-working.tif",
  "tokens": "virchow2-base.npy",
  "mpp_xy": [0.5, 0.5],
  "native_mpp_xy": [0.25, 0.25],
  "native_tissue": "tissue-native.tif",
  "evidence_rgb": "fixed-conversion.png",
  "encoder_revision": "3158645804b69e3f3bc4439d4116edddf0840a72",
  "encoder_sha256": "8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b"
}
```

All paths are relative to this case JSON. `evidence_rgb` is an optional
previously computed fixed conversion; omitting it makes Bora perform the
conversion from the base tokens and declared 1/2-µm feature variants and
palette described in the [README](../README.md). Bora records when a cached
RGB is supplied and does not pretend it recomputed that cache. The original
H&E and the converted evidence play distinct roles: H&E guides appearance
regularization, while the converted RGB supplies class evidence. Pixel
spacing is sampling, not optical resolving power.

## 2. Prepare each annotated training image

For every independent training image, create a case JSON with its own image,
cluster labels and features, then run:

```bash
bora prepare training-image-A.json --output prepared/training-image-A
```

`prepared/training-image-A/groups.ome.tif` contains anonymous refined groups.
`descriptors.npz` contains one 2560-value Virchow2 descriptor per whole group:
1280 pooled appearance values and 1280 contrasts against physical neighbors.
The expert annotation has **not** been read at this point.

## 3. Build a fastPLS tissue model from an expert mask

An expert TIFF must be a two-dimensional integer semantic mask aligned to the
**native** `groups.ome.tif` grid. Map every used source code to a common model
class ID. Any code not mapped or explicitly ignored is rejected. Zero is
ignored by default; if zero is a real tissue code, map it explicitly. Multiple
training cases may use different raw codes for the same named model class.

```json
{
  "format": "bora-training-v1",
  "classes": {"1": "tumor", "2": "stroma"},
  "cases": [
    {
      "descriptors": "prepared/training-image-A/descriptors.npz",
      "groups": "prepared/training-image-A/groups.ome.tif",
      "annotation": "expert-A.tif",
      "annotation_values": {"10": 1, "20": 2},
      "source_group": "patient-A",
      "split": "train",
      "min_coverage": 0.5,
      "min_purity": 0.8
    }
  ]
}
```

For each refined group, coverage is the proportion of its native pixels with
an expert tissue label. Purity is the dominant model class's proportion among
those annotated pixels. Groups below either declared threshold are excluded;
the exclusion count and reasons are reported. This prevents a partly annotated
or mixed group being silently treated as a clean example. Each source image
must name its patient or highest available independent `source_group`, and all
cases in this manifest must be `split: train`. Keep target patients out of
training and model selection.

## 4. Or use expert GeoJSON

GeoJSON must use ROI-local **native image pixel-edge coordinates**, with a
`FeatureCollection` of non-overlapping valid polygons. Each feature needs a
class property. By default Bora reads `properties.value`; a dotted
`annotation_field` can select a nested field such as
`properties.classification.name`. The manifest states the coordinate system
explicitly, maps source property values to model class IDs, and uses the same
coverage/purity rules as TIFF masks:

```json
{
  "descriptors": "prepared/training-image-B/descriptors.npz",
  "groups": "prepared/training-image-B/groups.ome.tif",
  "annotation": "expert-B.geojson",
  "annotation_coordinate_space": "native_image_pixel_edges",
  "annotation_field": "classification.name",
  "annotation_values": {"Tumor": 1, "Stroma": 2},
  "source_group": "patient-B",
  "split": "train"
}
```

This object is a second entry in the manifest's `cases` array. A GeoJSON
feature with `properties.classification.name = "Tumor"` maps to model class
`1`. GeoJSON in whole-slide coordinates must be translated to the case crop
before training; Bora rejects geometry outside its native image. A QuPath
export with a different property layout can set another `annotation_field`.

If expert targets have already been adjudicated for every whole group, the
earlier `labels: {"1": 1, "2": 2}` form remains valid **instead of**
`annotation`/`groups`. It must never be included together with an annotation.

## 5. Train and refine a different image

```bash
bora train training.json --output my-tissue-model.npz
bora refine target-image.json --model my-tissue-model.npz --output results/target-image
```

`target-image.json` contains the same predictor fields as a training case,
but no expert mask, class target or GeoJSON reference. Bora rejects unknown
case fields. The model file records its class vocabulary, fit settings and
training hashes. It contains training descriptors and labels, not executable
pickle; share it only when source-data terms permit. Loading deterministically
refits the native fastPLS head once, so keep one loaded model for a batch.

The result directory contains `groups.ome.tif`, `converted-rgb.png`,
`classified-before-export.ome.tif`, final `mask.ome.tif`, `mask.geojson` and
`report.json`. The GeoJSON features contain `predicted_tissue_id` and
`predicted_tissue` in their properties. The final TIFF is rasterized from the
final polygons. Reported timings exclude upstream Virchow2 extraction and,
when supplied, creation of cached RGB evidence.

The manuscript's 349 training source images, source IDs, native crops,
provider links and acquisition instructions are documented in
[Bora-Analysis](https://github.com/tkcaccia/Bora-Analysis/blob/main/docs/MANUSCRIPT_MODEL_DATA.md).

For CPU-only viewer integration and repeat use of fitted evidence, see
[Appearance recipes](APPEARANCE_RECIPES.md). This covers the full geometry-bound
RGB9+19 fit/replay API and the separately named row-wise Feature contrast
bridge. Shared semantic Feature contrast is an explicit training-only frozen
mapping; it is not the Neighbor context classifier or local RGB9+19.
