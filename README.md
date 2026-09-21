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
[`tkcaccia/CellPhenotyper`](https://github.com/tkcaccia/CellPhenotyper). That
The exact annealed-wand v2 and GeoJSON source snapshots are retained under
`reference/CellPhenotyper/`, with provenance documented alongside them.

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
