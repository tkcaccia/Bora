# CellPhenotyper reference sources

These files were copied from:

- Repository: https://github.com/tkcaccia/CellPhenotyper
- Commit: `a60797730accef0b35a7c1172c6e2d8d50a42070`
- Commit date: 2026-08-31

Included files:

- `bin/medsam_border_refine.py`
- `bin/refine_grown_tissue_medsam.py`
- `bin/mask_to_geojson.py`
- `bin/annealed_wand_boundary.py`

The annealed-wand and updated GeoJSON files were copied from the later run
snapshot `uni2_resolution_comparison_20260909/project`, where the boundary
competition identifies itself as schema
`cellphenotyper.annealed_wand_boundary.v2`.

`bin/mask_to_geojson.py` was refreshed from the `main` branch of
`tkcaccia/CellPhenotyper` on 2026-09-21. Its Git blob SHA is
`fd2dd40b227b85e542bf3da1d602a7eac5123990`; the production package contains
an unchanged copy at `src/bora/cellphenotyper_mask_to_geojson.py`.

## License status

At the referenced commit, the CellPhenotyper repository does not contain a
`LICENSE`, `COPYING`, or `NOTICE` file. Therefore, no open-source license grant
could be copied or inferred. Copyright remains with the original author(s), and
these files are retained as provenance-marked reference material. Before
redistribution or adaptation in Bora, obtain an explicit license or permission
from the CellPhenotyper copyright holder.

MedSAM itself is a separate upstream project and its code/checkpoint may have
independent licensing terms. Bora must document and comply with those terms if
it distributes or downloads MedSAM assets.
