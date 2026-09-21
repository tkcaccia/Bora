# Bora

Bora refines coarse tile-level tissue labels into pixel-level masks. It accepts
an OME-TIFF image and aligned TIFF label mask, then writes a refined OME-TIFF
and GeoJSON. Label IDs are preserved without a fixed segment-count limit.

For every positive label, Bora creates a protected interior core and editable
boundary band. A watershed image-gradient proposal refines the border; MedSAM
can instead provide box-prompted proposals. Overlaps are resolved by confidence.

The design is adapted from the `medsam-refine` step in
[`tkcaccia/CellPhenotyper`](https://github.com/tkcaccia/CellPhenotyper). That
source contains no method literally named “Wald”; Bora interprets this as the
watershed/border-band stage.

## Install and run

```bash
python -m pip install -e '.[test]'
bora refine tissue.ome.tif coarse-mask.tif \
  --output refined-mask.ome.tif --geojson refined-mask.geojson \
  --backend watershed
pytest
```

For MedSAM, install `.[medsam]`, provide the official repository if necessary,
and run with `--backend medsam --checkpoint /path/medsam_vit_b.pth`.

The manuscript draft is in `docs/manuscript.md`; numerical findings remain
explicit placeholders until a prespecified benchmark is run.

## License

No license has yet been selected.
