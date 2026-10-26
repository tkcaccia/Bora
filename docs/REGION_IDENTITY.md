# Tissue identity for accepted tissue contours

The three-group contour route produces anonymous KODAMA-group annotations.
`classify-regions` adds names from an independently trained Neighbor-context
fastPLS model without recalculating those boundaries. It does not train on the
target slide or claim that a transferred model is biologically validated there.
No pretrained model weights or target-slide data are distributed with Bora.

```bash
bora classify-regions accepted.geojson registration.json \
  --tokens virchow2-full1280.npy --model neighbor-context.npz \
  --output named-tissue
```

The token array is a floating `Y × X × 1280` NumPy `.npy` file. It must contain
the unselected spatial tokens from the model's declared Virchow2 encoder, on
the padded 14-pixel working grid: each spatial dimension is the corresponding
working image dimension divided by 14, rounded up. The caller is responsible
for using a checkpoint and sampling calibration compatible with the trained
model. Tokens cannot be reconstructed from a three-channel RGB image or a
32-dimensional PCA field.

The GeoJSON must be a nonempty FeatureCollection of valid Polygon or
MultiPolygon features in native image pixel-edge coordinates. Its positive
integer `properties.value` identifies the **original anonymous group**, not a
connected component or predicted tissue class. Disconnected pieces share one
pooled descriptor and one identity. Sparse group IDs are accepted. Use
`--group-field source_group_id` when the original group ID is stored elsewhere;
dotted property paths are supported. The command accepts at most 255 groups.
Different groups must not overlap at native pixel centres.

The registration file binds both files and declares their shared physical field:

```json
{
  "format": "bora-region-registration-v1",
  "source_frame_id": "specimen/native-image",
  "native_shape_yx": [448, 672],
  "working_shape_yx": [224, 336],
  "native_mpp_xy": [0.25, 0.25],
  "pixel_aligned": true,
  "registration_method": "native_same_grid",
  "image_sha256": "64 lowercase hexadecimal characters",
  "contours_sha256": "64 lowercase hexadecimal characters",
  "tokens_sha256": "64 lowercase hexadecimal characters"
}
```

Image dimensions describe the same origin, axes and complete physical field;
this is the caller's registration assertion, not a registration algorithm.
The image hash identifies the native H&E image but the command does not read
the image itself. If the GeoJSON also declares `native_shape_yx`,
`source_image_sha256` or `source_frame_id`, those values must match. Token and
contour file hashes are checked before inference, and all four input files,
including model and registration, are checked again before publication.

Native pixel-centre overlap determines each token's weight in its source group.
The 1280-channel area-weighted mean is rounded to float32, as in the original
`bora.groups.group_means`. It is concatenated with its contrast to neighbouring
group means, weighted by shared boundary length in micrometres. Interfaces
crossing stripe boundaries are counted once. Isolated groups have zero contrast.
The frozen model is loaded once; loading reconstructs the deterministic native
fastPLS head from its saved training descriptors and targets.

The default 256 MiB working-allocation guard bounds the main descriptor, stripe
and token-block arrays. It automatically reduces stripes below the requested
`--stripe-rows` maximum when needed. This is an allocation estimate rather than
a process memory limit: GeoJSON, the model, library internals and operating-system
page cache add memory. Set `--maximum-working-mib` explicitly for a large group
count; failure leaves the requested output unpublished.

The new output directory contains:

- `tissue-annotations.geojson`: the same ordered geometry objects, with tissue
  names in `name` and `classification.name`; original IDs/names remain in
  `source_group_id` and `source_group_name`.
- `descriptors.npz`: native-overlap descriptors, source IDs, native pixel mass,
  physical neighbour graph, tissue predictions, LDA ranking and PLS responses.
- `report.json`: `bora-region-identity-v1` completion receipt, input/output hashes,
  frame, calibration, group results and geometry-only hash. It states
  `geometry_unchanged: true`, `biological_accuracy_qualified: false` and
  `responses_are_calibrated_probabilities: false`.

PLS class-indicator responses and LDA rank are not calibrated probabilities.
External tissue ground truth is still required to establish accuracy. A tissue
name does not make a group ground truth or replace a user-edited annotation.

Python users can call `bora.region_identity.classify_regions` with the same
five input/output paths. `describe_region_polygons` exposes the streamed
descriptor calculation for parity checks against the original group API.
