# Image resolution and annotation interpretation

Updated 15 October 2026. These files describe supplied annotations; no Bora
inference, new tissue annotation or benchmark scoring was performed.

## Resolution of the acquired images

| Dataset | Distributed pixel spacing | Magnification / scanner information | Evidence and limitation |
|---|---|---|---|
| BCSS mirror | 0.25 µm/pixel | Mirror reports 40× equivalent | Pinned mirror card; not independently calibrated per PNG |
| TIGER ROIs | Approximately 0.5 µm/pixel | Multiple source scanners; no single optical magnification assigned here | Provider explicitly reports this spacing for ROI images and masks |
| GlaS | 0.620 µm/pixel | 20× equivalent; Zeiss MIRAX MIDI | Publication: native scans at 0.465 µm/pixel were rescaled before tile extraction; acquired BMP files do not embed reliable calibration |
| CRAG candidate | **Unknown for these acquired files** | Original CRAG: 0.55 µm/pixel, 20×, Omnyx VL120 | Original publication supplies the nominal value; undocumented mirror processing prevents assigning it as verified file spacing |
| PAIP 2019 | Pending acquisition | Provider reports 20×, Aperio AT2 | Obtain spacing from approved SVS metadata; magnification alone is insufficient |

Sources: [pinned BCSS mirror card](https://huggingface.co/datasets/MedOtter/BCSS/tree/502d4a3fbc77dbaca6f4664c19e2379ff077d418),
[TIGER provider](https://tiger.grand-challenge.org/Data/),
[GlaS challenge paper, Materials](https://arxiv.org/abs/1603.00275),
[MILD-Net paper, §3.1](https://arxiv.org/abs/1806.01963),
[PAIP provider](https://paip2019.grand-challenge.org/Dataset/).
The machine-readable source and confidence fields are in [resolution.json](resolution.json).

Here, resolution means image sampling in micrometres per pixel. It is distinct
from optical resolving power, objective magnification and screen zoom. Field
of view is calculated as pixel dimensions × reported pixel spacing. A value
reported for a dataset is not an independent calibration of a downloaded file.
No physical area or field of view is assigned to the unverified CRAG mirror.

## Are the masks annotated?

Yes. BCSS and TIGER supply **semantic tissue masks**: an integer at each pixel
indicates a tissue compartment. BCSS was produced through structured
crowdsourcing. TIGER includes pathologist-annotated RUMC/JB cases and adapted
BCSS/NuCLS annotations for TCGA cases. These are supplied reference labels,
not outputs created by Bora.

GlaS supplies **gland instance masks** drawn by an expert pathologist. Zero
denotes non-gland; positive integers distinguish individual gland objects.
For example, gland 1 and gland 2 are two objects, not two tissue classes. The
same applies to the intended original CRAG dataset; the acquired candidate's
annotation fidelity remains unverified.

## Which areas are tumor or stroma?

For BCSS and TIGER, inspect the pixel values using their respective code maps:

| Pixel value | BCSS meaning | TIGER meaning |
|---:|---|---|
| 0 | Outside annotated ROI; ignore | Excluded; ignore |
| 1 | Tumor | Invasive tumor |
| 2 | Stroma | Tumor-associated stroma |
| 3 | Lymphocytic infiltrate | In-situ tumor |
| 4 | Necrosis or debris | Healthy glands |
| 5 | Glandular secretions | Necrosis outside in-situ tumor |
| 6 | Blood | Inflamed stroma |
| 7 | Excluded region | Rest |
| 20 | DCIS | Not a TIGER class |

BCSS has additional codes through 21, listed in [labels.json](labels.json).
The complete [original BCSS code map](https://github.com/PathologyDataScience/BCSS/blob/3c49d554497dd896cd02ffe382503ef4ce28b1c1/meta/gtruth_codes.tsv)
and [TIGER definitions](https://tiger.grand-challenge.org/Data/) are the sources.
Values beyond 2 are not interchangeable between the datasets. The proposed
BCSS ignore set is 0, 7 and 15; all original pixel values are retained.

**GlaS and CRAG do not provide a general pixel-wise tumor–stroma map.** A
non-gland pixel cannot automatically be called stroma. GlaS image-level
benign/malignant grades are recorded separately where supplied; a grade does
not establish a malignancy class for each pixel or each individual gland.
PAIP, once acquired, will distinguish whole tumor and viable tumor using
separate masks, rather than the BCSS/TIGER tissue ontology.

## Per-image files and examples

All 24,711 currently prepared image/mask annotation-target records are described in
`/mnt/sata_ssd/Bora-benchmark/metadata/image-information.csv`. Each row records
WSI/ROI type, pixel dimensions, reported spacing, confidence/source, annotation provenance
and the names of the mask values present. The original acquisition inventories
and their recorded SHA256 identities are preserved; the description layer
adds subsequently researched resolution evidence without rewriting them.

[IMAGE_TYPES.md](IMAGE_TYPES.md) distinguishes complete scanned slides from
ROIs/patches and documents the broader `metadata/image-types.csv` inventory,
which includes the three complete CAMELYON16 WSIs and images outside the paired
inventory. All 54 local examples are ROIs; a native crop is not a complete WSI.

The local examples contain `01-metadata.json` (and corresponding 02/03 files)
with the original source ID, image/mask hashes, resolution evidence, tissue
legend, native pixel counts and area fractions. Gland files list instance IDs.
The `viewer.html` file shows supplied labels over the original RGB image,
with an opacity slider, legend and hover lookup. Masks are sampled with nearest
neighbours for display; pixel counts use the complete original masks. Original
RGBs and masks are unchanged. Viewer colours are visualization choices, not
the provider's original colour convention and not predictions.

TIGER `03-image.png` has been replaced by a **1224 × 1166** tissue ROI from case
114S, with invasive tumor and tumor-associated stroma among its annotated
classes. Its paired mask was replaced at the same time. At the provider's
approximate 0.5 µm/pixel spacing it spans approximately **612 × 583 µm**.
TIGER examples now require at least 1000 pixels on both sides, preventing
cell-scale NuCLS crops from being selected again. This selection rule affects
only convenience examples; all downloaded ROIs remain available.

To regenerate descriptions and the local viewer without running Bora:

```bash
python -m benchmark.examples --destination /mnt/sata_ssd/Bora-benchmark/examples --count 3 --replace-changed
python -m benchmark.describe --examples /mnt/sata_ssd/Bora-benchmark/examples
```

## Additional datasets from the PathSegmentor paper

See [ADDITIONAL_DATASETS.md](ADDITIONAL_DATASETS.md) for source-specific details.
WSSS4LUAD uses **0 = tumor**, 1 = stroma, 2 = normal and **3 = ignored white
background**. Only 40 geometrically valid validation pairs enter the inventory.
RINGS supplies separate binary gland and tumor masks; a tumor-negative pixel
is not a stroma annotation. The CoCaHis VISTA subset has binary masks whose
foreground meaning still needs reconciliation. None of these new releases has
verified distributed-image micrometres/pixel; the catalog preserves reported
magnifications and source-slide information without inventing calibration.

## Current eight-dataset acquisition

[REQUESTED_DATASETS.md](REQUESTED_DATASETS.md) records original SICAPv2 and
bounded CAMELYON16/SegPath acquisitions. CAMELYON spacing is retained per axis
from each WSI, including native level-0 crop coordinates. SegPath spacing is
publication-reported at 0.220818 micrometres/pixel, with independent binary
targets derived from immunofluorescence. SICAPv2 raw JPEG mask intensities are
not interpreted as Gleason class IDs, and its overlay is disabled pending a
verified decoding rule or lossless annotations.
