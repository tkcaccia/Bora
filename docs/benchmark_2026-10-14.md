# Four-tissue tile-size benchmark (14 October 2026)

## Inputs and outputs

- Input H&E images and original GeoJSON annotations:
  `/mnt/sata_ssd/benchmarks/Bora/input`.
- Fresh C++ tiling-mask outputs:
  `/mnt/sata_ssd/benchmarks/Bora/output-tilling-mask/<sample>`.
- Bora outputs by tile size:
  `/mnt/sata_ssd/benchmarks/Bora/output-bora/{56,90,216}/<sample>`.
- Final PDF destination:
  `/home/chiamaka/Bora/output/pdf/Bora_four_tissue_tile_size_benchmark.pdf`.
- Native-resolution GeoJSON rasterizations and scores:
  `/home/chiamaka/Bora/output/benchmark_geojson_eval/{56,90,216}/<sample>`.

The four samples are pancreas, lymph node, tonsil and brain cancer. The current
pancreas annotation has only one foreground class (blood vessels), unlike the
earlier handoff dataset. All data for this benchmark were regenerated from the
current `input` GeoJSON files; older metrics must not be reused.

The Bora checkout was at `00d9cc5d3d06cc4d55a5ec2e71676d7f7f52c92e`
and the local tiling-mask checkout at
`c67c756d97f1eb6616fa2c53e30ce266c9526905`, both with uncommitted
changes. Those hashes alone do not reproduce the runs; preserve the dirty
source files and per-sample logs when archiving the experiment.

## Methods

The local modified `tiling-mask-cpp` executable was run once per tissue with
repeated `--tile-size 56 --tile-size 90 --tile-size 216`, four raster/output
workers and `--border-error-percent 0`. It produced four full-resolution
`mask_pixel_cpp.tif` reference rasters, 12 full-resolution `mask_tile_grid_*`
masks and 12 compact `mask_tile_index_grid_*` masks. The full-resolution masks
were checked to match the respective image dimensions.

The local `scripts/run_bora_from_tile_grids.py` runs native-resolution
annealed competition plus marker-controlled watershed with 2,048-pixel
streaming blocks, pyramidal OME-TIFF output and accurate-profile GeoJSON. It
scores the coarse and refined TIFFs against `mask_pixel_cpp.tif`. One batch is
run for each tile size; all three are resumable.

`scripts/score_bora_geojson.py` rasterizes the final GeoJSON at native
resolution and scores that deliverable against the same reference raster. It
also measures TIFF-to-GeoJSON agreement. `scripts/make_bora_visual_examples.py`
selects illustrative improvement and regression crops, and
`scripts/build_bora_benchmark_report.py` builds the final PDF.

The accuracy report uses macro IoU and macro Dice over foreground classes,
foreground IoU without class identity, and pixel accuracy including background.
Bora runtime includes watershed refinement and GeoJSON export. Tiling-mask
runtime is shared across the three tile sizes; GeoJSON rasterization and
scoring are a separate audit step.

## Reproduce the final audit and report

After all 12 Bora runs complete, run from `/home/chiamaka/Bora`:

```bash
PYTHONPATH=src /home/chiamaka/miniconda3/envs/medsam/bin/python \
  scripts/score_bora_geojson.py \
  --bora-root /mnt/sata_ssd/benchmarks/Bora/output-bora \
  --tiling-root /mnt/sata_ssd/benchmarks/Bora/output-tilling-mask \
  --evaluation-root /home/chiamaka/Bora/output/benchmark_geojson_eval

PYTHONPATH=src /home/chiamaka/miniconda3/envs/medsam/bin/python \
  scripts/make_bora_visual_examples.py \
  --input /mnt/sata_ssd/benchmarks/Bora/input \
  --tiling-root /mnt/sata_ssd/benchmarks/Bora/output-tilling-mask \
  --evaluation-root /home/chiamaka/Bora/output/benchmark_geojson_eval \
  --output /home/chiamaka/Bora/tmp/pdfs/bora_examples

/home/chiamaka/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  scripts/build_bora_benchmark_report.py \
  --bora-root /mnt/sata_ssd/benchmarks/Bora/output-bora \
  --tiling-root /mnt/sata_ssd/benchmarks/Bora/output-tilling-mask \
  --evaluation-root /home/chiamaka/Bora/output/benchmark_geojson_eval \
  --examples /home/chiamaka/Bora/tmp/pdfs/bora_examples \
  --output /home/chiamaka/Bora/output/pdf/Bora_four_tissue_tile_size_benchmark.pdf
```

## Status at handoff

The tiling-mask stage is complete. The three Bora batches were started on
14 October 2026; check `output-bora/{56,90,216}/<sample>/bora.log`,
`comparison.json`, and batch `summary.json` for their current state. Do not
assume completion from the presence of a partly written OME-TIFF alone.
After all 12 `comparison.json` files and final GeoJSONs exist, run the vector
scorer, make the visual examples, build the PDF, render each PDF page and
inspect layout before delivery. Do not report the older six-tissue results as
these results.
