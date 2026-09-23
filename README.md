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

The Medical Image Analysis manuscript is available as Markdown, DOCX, and PDF
under `docs/` and `output/pdf/`. Numerical findings remain explicit
placeholders until a prespecified benchmark is run.

## License

No license has yet been selected.
