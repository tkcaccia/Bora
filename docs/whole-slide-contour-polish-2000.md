# Whole-slide contour polish and 2000 µm² geometry cleanup

This is the current whole-slide benchmark rule, superseding the historical
400 and 1000 µm² rules for new results. Apply it to the **complete calibrated slide**
before generating any diagnostic crop or publishing a result. A region or
enclosed unoccupied hole with physical area **strictly below 2000 µm²** is
reassigned; exactly 2000 µm² is retained. Calculate pixel area from measured
X and Y spacing, not an assumed nominal 0.5 µm/pixel.

The `bora wsi-polished` function accepts `minimum_area_um2` and
`contour_sigma_um`; its command-line options are `--minimum-area-um2` and
`--contour-sigma-um` (alias `--contour-smoothing-um`). Both must be positive
finite numbers. Defaults are 2000 µm² and 6 µm, which remain the required
settings for the current whole-slide benchmark. An alternate run uses a fresh
output directory and records its chosen values in the input identity, contour
receipt, geometry audit and final result. The preliminary repair is capped at
the selected final minimum so a lower requested threshold cannot be erased
by the historical 400 µm² preparation step.

```bash
bora wsi-polished case.json --output result-new \
  --minimum-area-um2 2000 --contour-sigma-um 6
```

1. Keep the original KODAMA groups, all 1280 raw Virchow2 channels, and the
   reviewed tissue-area GeoJSON unchanged. Use the continuous Bora response
   with no hard square tile-label prior. On the complete slide, run the
   H&E-guided diffusion and the existing 1.5 µm one-hot contour step.
2. Apply the historical 400 µm² raster repair once to give the contour polish
   a stable input. Save its checksum. Smooth the positive class indicators
   with a 6 µm Gaussian in physical X/Y units and select the strongest class
   at each *already assigned* pixel. Preserve zero-class pixels exactly,
   including exterior glass and large enclosed voids; do not expand beyond
   the reviewed tissue support. Use overlapping blocks and verify exact
   equality across computation seams.
3. On the whole polished raster, repeatedly reassign every 8-connected
   positive component below 2000 µm² to the most common adjacent class in its
   8-neighborhood. An isolated small island becomes background. Fill enclosed
   unoccupied background holes below 2000 µm²; keep exterior glass and large
   voids. Repeat until no new small component is exposed, with a finite
   maximum number of passes.
4. Vectorize the completed raster and fit shared borders once. Remove any
   continuous polygon part below 2000 µm² and fill any unoccupied enclosed
   vector hole below 2000 µm². Repeat after raster/vector repair and after
   conversion to native image coordinates. Preserve class topology, prevent
   overlaps, and verify final raster/GeoJSON agreement.
5. Save a checksum-bound receipt recording physical spacing, thresholds,
   smoothing width, changed pixels, small-region counts, seam checks, minimum
   final areas and hashes. Review the full-slide H&E overlay and selected
   crops. A crop is an exact subset of the global result and receives no
   separate cleanup. Publish only a result that passes these checks and has
   no visible 224-pixel tile contours.

The physical area threshold concerns individual connected geometries, not
the appearance of a downscaled whole-slide JPEG. At the F0065 preview scale,
a 2000 µm² circular region is about 3.7 display pixels across, so visible
small regions can remain when their true area exceeds the threshold.
