# Full WSI and ROI classification

Classification describes the **actual acquired image**, not the dataset title,
number of source slides, annotation extent, objective magnification or image size.
It is separate from whether a supplied mask is suitable for evaluation.

- **WSI:** complete original scanned slide, retaining its native whole-slide
  extent and pyramid. A sparse annotation on a complete WSI does not make the
  image an ROI.
- **ROI:** a cropped tissue region, tile, patch or microscopy field. Small
  cell-scale patches also fall in this category. A crop taken from a WSI
  remains an ROI even when saved at native resolution.

These categories do not imply independent patients or slides. In particular,
the CAMELYON example ROIs and their full parent WSIs come from the same slides.
An image's file extension or pixel dimensions cannot establish that it is a WSI.

## Acquired image categories on Chiamaka

Status on 15 October 2026; root `/mnt/sata_ssd/Bora-benchmark`.

| Acquired dataset / representation | Category | Number of image references | Explanation |
|---|---|---:|---|
| BCSS | ROI | 151 | Breast tissue regions; no complete source slides acquired |
| TIGER ROI archive | ROI | 2,030 | Regions from 195 source WSIs; 195 is not an acquired full-WSI count |
| CAMELYON16 original TIFFs | **WSI** | **3** | Complete `tumor_058`, `tumor_091`, `tumor_107`, with original full-size masks and XML |
| CAMELYON16 example windows | ROI | 3 | Native 4096 × 4096 level-0 crops with recorded coordinates and parent WSI paths |
| SICAPv2 release 1 | ROI | 18,783 | 512 × 512 patches; the 155-slide table does not contain complete WSI images |
| GlaS | ROI | 165 | Cropped colorectal gland images |
| CRAG candidate mirror | ROI | 213 | Cropped gland images; original processing/provenance remains unverified |
| SegPath native subset | ROI | 24 | Three 984 × 984 patches for each of eight binary targets |
| WSSS4LUAD valid dense-mask inventory | ROI | 40 | Accepted validation image/mask pairs |
| WSSS4LUAD weak training images | ROI | 10,091 | Image-level labels only; original image bytes remain embedded in the downloaded Parquet shard |
| WSSS4LUAD quarantined test images | ROI | 80 | Downloaded images with missing or incompatible masks; excluded from the prepared paired inventory |
| RINGS | ROI | 1,500 | The same RGB fields have separate gland and tumor masks: 3,000 annotation-target records, not 3,000 unique RGB images |
| Processed CoCaHis VISTA subset | ROI | 302 | Provider-processed 1024 × 1024 patches; full parent-slide provenance unresolved |

There are **3 full WSI image references and 33,382 ROI image references** in the
combined classification inventory. This totals **33,385 distinct source image
references** across the representations above. It is not a count of independent
biological samples or a global content-hash deduplication.

The classification CSV has **34,885 records** because RINGS contributes a row
for each annotation target. Of these, 24,711 records are prepared image/mask
pairs (23,211 distinct RGB paths); three describe complete source WSIs; 10,091
describe weak training images; and 80 retain the quarantined test-image status.
Archive copies, recovery prefixes, annotation-mask images and copied examples
are not counted again as benchmark input images.

PAIP 2019 is intended as a **full WSI** dataset, but no PAIP images have been
acquired pending provider approval. Canonical CRAG is also pending. Neither
contributes images to these acquired-image counts.

## Per-image metadata and local examples

The canonical classification files on Chiamaka are:

```text
/mnt/sata_ssd/Bora-benchmark/metadata/image-types.csv
/mnt/sata_ssd/Bora-benchmark/metadata/image-types-summary.json
```

They are also copied to the local examples folder:

```text
/Users/stefano/Documents/ChatGPT/Bora/benchmark/examples/image-types.csv
/Users/stefano/Documents/ChatGPT/Bora/benchmark/examples/image-types-summary.json
```

Each record contains `image_type` (`WSI` or `ROI`), `image_subtype`, the
classification basis, annotation status and source file or Parquet identity.
Parent-slide identity and crop coordinates are retained where established.
An unknown parent image type is stated explicitly; a cropped field is not
assumed to have a complete, available source WSI. Mask files are annotation
assets rather than additional RGB benchmark images.

**All 54 locally copied example images are ROIs.** The example viewer, index
table and each `01-metadata.json` / `02-metadata.json` / `03-metadata.json`
display the category. CAMELYON examples identify the parent WSI path and crop
coordinates; `camelyon16-subset/whole-slide-information.json` separately
describes the three complete WSIs, which remain on Chiamaka at:

```text
/mnt/sata_ssd/Bora-benchmark/raw/camelyon16-subset/images/
```

The resolution/annotation CSV `metadata/image-information.csv` (copied locally
as `all-image-information.csv`) also adds image-type fields to its 24,711
prepared annotation-pair records. The broader `image-types.csv` includes the
complete WSIs and images outside that paired inventory.

## Reproduce without running Bora

From the repository root, with the acquisition environment active:

```bash
python -m benchmark.image_types --examples /mnt/sata_ssd/Bora-benchmark/examples
python -m benchmark.examples --destination /mnt/sata_ssd/Bora-benchmark/examples --count 3
python -m benchmark.describe --examples /mnt/sata_ssd/Bora-benchmark/examples
```

The standalone classification command reads existing inventories only. The
description command also regenerates the classification exports. Neither
rewrites original image/mask bytes or the hashed acquisition inventories.
Classification rules are in [image_types.py](image_types.py); source and
acquisition details remain in [REQUESTED_DATASETS.md](REQUESTED_DATASETS.md)
and [ADDITIONAL_DATASETS.md](ADDITIONAL_DATASETS.md).
The [dated classification receipt](image-types-2026-10-15.json) records the
verified counts, export hashes and checks for this update.
