# Bora

Bora refines coarse tile-level tissue labels into pixel-level masks. It accepts
an OME-TIFF image and aligned TIFF label mask, then writes a refined OME-TIFF
and GeoJSON. Label IDs are preserved without a fixed segment-count limit.

The default chain first applies CellPhenotyper's deterministic annealed-wand
multiclass competition near interfaces (64-pixel native radius, 4x working
downsample), then marker-controlled watershed at native image resolution. For
every positive label, Bora creates a protected interior core and editable
boundary band. MedSAM can be appended as an optional box-prompted stage.
Overlaps are resolved by confidence.

The design is adapted from the `medsam-refine` step in
[`tkcaccia/CellPhenotyper`](https://github.com/tkcaccia/CellPhenotyper). The
exact annealed-wand v2 and post-refinement GeoJSON source snapshots are
retained under `reference/CellPhenotyper/`, with provenance documented
alongside them. CellPhenotyper's converter remains available with
`--geojson-backend cellphenotyper`. The default `multicpu` backend uses four CPU workers,
coverage-aware simplification (so shared label boundaries remain coincident),
and one dissolved feature per label. The `accurate`, `balanced`, and `fast`
profiles expose measured fidelity/complexity tradeoffs; `accurate` is the
default and polygonizes native resolution in four memory-bounded strips.
`balanced` retains native polygonization with stronger simplification, while
`fast` uses pyramid level 3. CUDA is optional. The `cuda` backend uses CuPy for
connected-component labeling and area filtering, then uses Rasterio/Shapely on
CPU for polygonization and topology operations. It records the accelerator and
device in the run report and falls back safely to `multicpu` unless
`--no-geojson-cuda-fallback` is supplied.

### Validated GeoJSON profiles

The profiles were measured by rasterizing the result back across all
1,733,311,788 level-0 pixels of the supplied Bora mask. Each emits three
top-level features—one per observed foreground label.

| Profile | Pixel accuracy | Macro label IoU | Polygon parts | Size | Conversion |
|---|---:|---:|---:|---:|---:|
| accurate (default) | 99.909% | 0.9981 | 2,267 | 39.9 MB | 54.0 s |
| balanced | 99.807% | 0.9960 | 2,267 | 17.8 MB | 49.0 s |
| fast | 98.617%* | 0.9713* | 1,927 | 10.5 MB | 5.0 s |

`*` The fast profile was evaluated at pyramid level 2; accurate and balanced
were exhaustively evaluated at level 0.

The CUDA backend was exercised on the four-label lymph-node output at pyramid
level 3: it labeled 922 components on an RTX 5060 Ti, removed 505 components
below the configured 10,000-pixel level-0 area, and emitted four valid class
features in 7.8 seconds. GPU startup and transfer overhead can make small masks
slower than the CPU backend; CUDA is intended for large component-filtering
workloads rather than as an unconditional speedup.

## Install and run

```bash
python -m pip install -e '.[test]'
bora refine tissue.ome.tif coarse-mask.tif \
  --output refined-mask.ome.tif --geojson refined-mask.geojson \
  --backend watershed
pytest
```

For CUDA-assisted mask conversion on CUDA 13, install
`python -m pip install -e '.[cuda-geojson]'` and select
`--geojson-backend cuda`. CUDA 12 installations should install the matching
`cupy-cuda12x` package instead of the CUDA 13 extra.

Use `--watershed-resolution accelerated --watershed-max-side 256` for an
explicitly approximate watershed run. Native resolution is the default.
Annealed competition can be tuned with `--wand-*` options or disabled with
`--no-annealed-wand`. `pathsegmentor` is reserved until an explicit mapping
from numeric labels to text prompts is supplied; Bora fails clearly rather
than silently substituting another model.

Pyramidal, tiled OME-TIFF output is enabled by default. Pyramid construction
uses lossless `SIMPLE` downsampling for categorical labels and requires
`bioformats2raw` and `raw2ometiff` on `PATH`. Use `--no-pyramid` only when a
flat intermediate TIFF is explicitly required.

For MedSAM, install `.[medsam]`, provide the official repository if necessary,
and run with `--backend medsam --checkpoint /path/medsam_vit_b.pth`.

PathSegmentor is available as a text-prompted alternative. Install its official
environment (including its pinned custom Detectron2 build), clone the repository,
download `model_state_dict.pt`, and
provide a JSON mapping for every non-zero mask label:

```json
{
  "1": "tissue-level tumor in breast pathology",
  "2": "tissue-level stroma in breast pathology"
}
```

```bash
bora refine image.ome.tif mask.tif --output refined.ome.tif \
  --geojson refined.geojson --backend pathsegmentor \
  --repo-dir /path/to/PathSegmentor --checkpoint /path/to/model_state_dict.pt \
  --label-map labels.json --device cuda
```

The spelling `--backend pathsegmentator` is accepted as an alias. PathSegmentor
operates on 1024-pixel pathology patches; Bora restores predictions to native
resolution and restricts changes to the configured boundary envelope while
preserving eroded label cores.

The adapter has been checkpoint-tested with official PathSegmentor commit
`e67763ab835b3ef184cb6791eb2cd059ec8b9f9a`, an RTX 5060 Ti, and CUDA 13.0.
Modern PyTorch requires replacing `value.type()` with `value.scalar_type()` in
the two `AT_DISPATCH_FLOATING_TYPES` calls in PathSegmentor's
`ms_deform_attn_cuda.cu` before compiling the extension. Single-process
inference does not use MPI; on systems without a system MPI library, make the
repository's `mpi4py` import optional.
Full installation and compilation instructions are in
[`docs/pathsegmentor_install.md`](docs/pathsegmentor_install.md); the exact
modern-PyTorch compatibility patch is retained under `compat/`.

The six-tissue, 56-pixel technical benchmark draft is available as Markdown,
DOCX, and PDF under `docs/` and `output/pdf/`. A separate
[`Medical Image Analysis` manuscript draft](docs/Bora_manuscript_Medical_Image_Analysis_draft.docx)
and [highlights](docs/Bora_highlights_Medical_Image_Analysis.docx) follow that
journal's submission format. The latter draft still requires verified results,
author declarations, and other submission materials. Neither draft includes
the ongoing regenerated multi-size benchmark.

## Multi-image GeoJSON benchmark pipeline

The `benchmark` command runs the complete experiment for every matched TIFF
and GeoJSON pair in a directory:

1. `tiling-mask` rasterizes the original GeoJSON and produces a majority-vote
   tile grid. Its `mask_pixel.tif` is retained as the pixel-level reference.
2. Bora uses a full-resolution categorical tile mask directly, or expands a
   compact tile index grid to the original dimensions without interpolation,
   then refines that coarse mask against the original image.
3. Both the unrefined baseline and Bora result are compared with the reference.
   Per-class and aggregate Dice, IoU, and accuracy are written to
   `comparison.json`, `summary.json`, and `summary.csv`.

Install both repositories into the same environment:

```bash
git clone https://github.com/tkcaccia/tiling-mask.git
python -m pip install -e ./tiling-mask
python -m pip install -e ./Bora
```

Image and annotation names are paired by sample stem. The CellPhenotyper
suffix `_standard_grown_mask_smooth_class.geojson` is removed automatically,
so `sample.btf` pairs with
`sample_standard_grown_mask_smooth_class.geojson`.

```bash
bora benchmark /mnt/sata_ssd/benchmarks/Bora \
  --output /mnt/sata_ssd/benchmarks/Bora_results \
  --tile-size 224 --backend watershed --workers 4
```

If images and GeoJSON files live in different directories, add
`--image-dir /path/to/images`; matching is still performed by sample stem.

The default is native-resolution watershed with annealed-wand competition and
pyramidal output. Runs are resumable by default. Use `--no-resume` to rebuild
completed stages, `--coordinates world` for geospatial GeoJSON, or
`--class-property classification.name` to select an explicit class field.
MedSAM and PathSegmentor accept the same `--backend`, `--checkpoint`,
`--repo-dir`, and `--device` options as `bora refine`. Extra refine options can
be repeated as `--bora-arg=--option` (and `--bora-arg=value` where needed).
For the native implementation, pass `--tiling-mask-backend cpp` together with
`--tiling-mask-command /path/to/tiling-mask-cpp`. The native backend supports
pixel-coordinate GeoJSON and its standard automatic QuPath class properties.

## 216-pixel image-guided boundary study

The 216-pixel experiment reuses the existing full-resolution
`mask_tile_grid_216x216_cpp.tif` files. Bora's `adaptive-wand` backend uses
seeded GrabCut for tissue-background edges and a balanced local-color wand
for every interface between foreground tissue classes. The wand estimates
each class's nearby LAB distribution from protected interior pixels, lets
adjacent classes compete symmetrically within a tile-scale band, and removes
detached class islands. MedSAM and PathSegmentor are not used here.

```bash
PYTHONPATH=src python scripts/run_adaptive_boundary_benchmark.py \
  --images /mnt/sata_ssd/benchmarks/Bora/input \
  --tiles /mnt/sata_ssd/benchmarks/Bora/output-tilling-mask \
  --output /mnt/sata_ssd/benchmarks/Bora/output-bora-cycle-06 \
  --tile-size 216 --block-size 2048
```

Every sample directory contains `bora_refined.geojson`, a pyramidal OME-TIFF,
and a comparison JSON. The GeoJSON carries the original tiling-mask class
names and colors as wsiTools annotation metadata. A new experimental cycle
gets a new numbered output folder. Cycles 01-05 were removed at the user's
request after visual review showed unchanged tile steps at internal borders.

Accuracy is evaluated from the final GeoJSON rasterized at native resolution.
The primary gate compares only fixed 64-pixel bands around ground-truth or
original tile-mask borders. Internal tissue-class borders and the
tissue-background border are reported separately, along with 8-pixel
boundary F1 and the number of pixels that switch between foreground classes.
Pancreas has one tissue class, so only its exterior border is scored. A candidate is accepted only
when border macro IoU and border pixel accuracy do not fall for any tissue,
at least one measure rises, and all ground-truth classes remain represented.
The whole-mask scores remain available in each `comparison.json` as secondary
diagnostics. Use `scripts/score_bora_geojson.py`,
`scripts/score_216_borders.py`, and `scripts/evaluate_216_cycle.py` to
reproduce the decision from saved outputs.

## Experimental C++ triangular-mesh refinement

The newer contour stage regularizes a categorical mask on a triangular mesh,
then optionally rounds the mesh corners. It edits only internal tissue-class
boundaries; external tissue/background contours remain the responsibility of
the preceding image-guided Bora stage. Mesh support grows in locally ambiguous
color regions and stays narrower where nearby tissue classes have distinct
RGB distributions. The kernel uses OpenMP, and the streaming wrapper processes
WSI blocks in multiple worker processes.

```bash
cmake -S cpp -B cpp/build
cmake --build cpp/build -j 4
PYTHONPATH=src python scripts/run_tri_mesh_cycle.py \
  --source-root /path/to/pre-strong-bora-output \
  --output /path/to/numbered-cycle \
  --tiling-root /path/to/regenerated-tile-masks \
  --images /path/to/original-images \
  --adaptive-by-tile-size --mesh-spacing 64 --support-radius 112 \
  --band-radius 216 --color-weight 0.2 --post-smooth-radius 16
```

The regenerated four-slide, three-size benchmark and the planned 200 numbered
cycles are documented in [the experiment protocol](docs/tri_mesh_200_cycle_protocol.md).
Every full cycle must export 12 GeoJSONs and is scored on internal-border
macro IoU, accuracy, 8/16/32-pixel contour F1, ASSD, HD95, false tile-edge
pixels, retained classes, and measured processing time. The current input
GeoJSONs differ from the older 216-pixel benchmark annotations, so their
scores are kept in a separate versioned benchmark directory.

Dataset acquisition code, provenance, annotation definitions and access status
for the histopathology benchmark are available in [`benchmark/`](benchmark/).
The acquisition workflow inventories public image/mask pairs and does not run
Bora. Dataset files and local examples remain outside Git.

## License

No license has yet been selected.
