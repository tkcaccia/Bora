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
`fast` uses pyramid level 3. CUDA is not required (and was unavailable on the
validation host). All settings are exposed as `--geojson-*` options.

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
were exhaustively evaluated at level 0. CUDA was unavailable on the validation
host, so the implemented optimized backend is CPU-only.

## Install and run

```bash
python -m pip install -e '.[test]'
bora refine tissue.ome.tif coarse-mask.tif \
  --output refined-mask.ome.tif --geojson refined-mask.geojson \
  --backend watershed
pytest
```

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

The Medical Image Analysis manuscript is available as Markdown, DOCX, and PDF
under `docs/` and `output/pdf/`. Numerical findings remain explicit
placeholders until a prespecified benchmark is run.

## License

No license has yet been selected.
