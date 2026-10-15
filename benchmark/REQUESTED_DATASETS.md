# Status of the eight requested datasets

Checked and acquired on 15 October 2026. Storage root on Chiamaka:
`/mnt/sata_ssd/Bora-benchmark`. Bora has not been run.

## Recommended roles and acquisition status

| Dataset | Source size | Annotation | Best use for Bora | Priority | Actual acquisition |
|---|---|---|---|---|---|
| [BCSS](https://github.com/PathologyDataScience/BCSS) | 151 breast cancer ROIs, 22 encoded values | Multiclass tissue masks | Tumor, stroma, necrosis and immune-region boundaries | Very high | All 151 pairs from the pinned normalized mirror |
| [TIGER](https://tiger.grand-challenge.org/Data/) | 195 source WSIs with annotated ROIs | Seven tissue classes plus excluded pixels | Multiclass refinement and boundary preservation | Very high | Complete ROI archive: 2,030 pairs; not all full WSIs |
| [PAIP 2019](https://paip2019.grand-challenge.org/Dataset/) | 50 annotated HCC training WSIs | Separate whole-tumor and viable-tumor masks | Large-scale tumor borders | Very high | Pending signed and approved provider agreement |
| [CAMELYON16](https://registry.opendata.aws/camelyon/) | Original challenge approximately 400 WSIs; current AWS release 399 | Expert-derived tumor polygons and raster masks | Metastasis margins and WSI scale | High | Three complete positive training WSIs, original masks and XML; explicit convenience subset |
| [SICAPv2](https://data.mendeley.com/datasets/9xxm58dvs3/1) | Original archive: 18,783 paired 512px patches from 155 slides / 95 patient IDs | JPEG annotation masks and image/slide Gleason tables | Exploratory grade interfaces after encoding review | High | Complete original publisher archive; mask decoding and boundary fidelity unresolved |
| [GlaS](https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/) | 165 colorectal images | Gland instance masks | Fine gland boundaries | Medium | All 165 image/mask pairs from the pinned mirror |
| [CRAG](https://warwick.ac.uk/fac/cross_fac/tia/data/mildnet/) | Intended original v2: 213 colorectal images | Gland instance masks | Irregular glands and topology | Medium | 213-pair candidate mirror staged; canonical release and provenance still pending |
| [SegPath](https://dakomura.github.io/SegPath/) | Eight original binary target datasets, 158,687 CSV-listed patches | Registered immunofluorescence-derived masks | Additional fine-scale target boundaries | Medium | Bounded native subset from each of eight targets; complete source CSVs and licenses retained |

Full-release names and acquired subsets must be distinguished in a manuscript.
SICAPv2's public archive contains patches, not original full-resolution WSIs.
The three CAMELYON slides are not a representative cohort or a blind test set.
SegPath's inspection examples do not constitute a statistically powered benchmark.
[IMAGE_TYPES.md](IMAGE_TYPES.md) explicitly categorizes each acquired representation
as full WSI or ROI: only the three complete CAMELYON16 TIFFs are WSIs; every
local example is an ROI, including the native CAMELYON windows.
[DATASETS.md](DATASETS.md) retains the broader study-design notes and
[ADDITIONAL_DATASETS.md](ADDITIONAL_DATASETS.md) covers the paper-based additions.

## CAMELYON16: verified native geometry and resolution

The [AWS open-data registry](https://registry.opendata.aws/camelyon/) identifies
the public source bucket `camelyon-dataset` in `us-west-2`, managed by Radboud
University Medical Center, under CC0. The complete current CAMELYON16 prefix
contains 399 WSIs, approximately **752.5 GB of WSI images** and **762.5 GB of
all files**. It does not fit the roughly 70 GiB free SSD space available at the
start of this addition; the acquisition preserves the existing 20 GiB reserve.

The native subset comprises `tumor_058`, `tumor_091` and `tumor_107`. Selection
uses the smallest eligible positive slide in each filename range 001-070,
071-100 and 101-111. Slides listed by the provider as not exhaustively annotated
were excluded. This is a storage-conscious convenience selection; numerical
ranges are not verified patient groups or a substitute for sampling by hospital
and lesion type. Both original training partitions and positive-slide selection
are explicit. No full-release or balanced-cohort claim is made.

Each complete source WSI, corresponding original categorical TIFF mask and tumor
XML is retained. Published MD5 checksums were verified, and SHA256 is recorded.
The original mask and WSI share exactly the same level-0 dimensions. Separate
X/Y micrometres-per-pixel values are read from each WSI through OpenSlide; values
are retained without forcing isotropy or assuming a uniform 0.25 micrometres/pixel.

Native example crops are 4096 x 4096 pixels at level 0, centred on an existing
tumor polygon boundary and required to contain both tumor and non-tumor tissue.
No rescaling, stain normalization, model prediction or new contour rasterization
is performed. Crop coordinates and original file hashes are retained in
`metadata/camelyon16-subset-wsi.json`. These crops are for local inspection and
inventory validation, not a prespecified future evaluation sampling scheme.

The [provider README](https://camelyon-dataset.s3.us-west-2.amazonaws.com/CAMELYON16/README.md)
defines the actual distributed mask values:

| Value | Meaning | Provenance / proposed handling |
|---|---|---|
| 0 | Background or unassigned region | Exclude from tissue comparisons |
| 1 | Normal tissue | Automatically detected tissue; not expert stroma ground truth |
| 2 | Tumor | Rasterized from expert annotation polygons |

The XML includes exclusion holes within tumor regions; the supplied masks
preserve the provider rasterization. The current release's normal-tissue mask
must not be described as an independent expert annotation of normal tissue.

## SICAPv2: complete original archive, unresolved pixel encoding

The [original publisher archive](https://data.mendeley.com/datasets/9xxm58dvs3/1),
DOI 10.17632/9xxm58dvs3.1, is licensed CC BY 4.0. Its size is 2,174,534,398 bytes.
This is Mendeley release **1**, matching the URL in the requested list.
The author's [profile](https://jusiro.github.io/) links a separate Mendeley
release 2, which has not been substituted for or validated as this release.
The publisher supplies SHA256
`de0826cbf8f1cdf86ac46562a541288841824952bbc2b348a0bad576cbec29ae`,
which was checked before extraction. This replaces dependence on an unrelated
uploader's undocumented mirror for this dataset.

Actual contents are 18,783 RGB JPEG patches and 18,783 matching grayscale JPEG
annotation images, all 512 x 512, together with provider spreadsheet tables.
The archive README states **10x** distributed magnification with 50% patch
overlap. Higher magnification values in other catalogs do not calibrate these
files. JPEG density fields are not microscopy calibration, so actual
micrometres/pixel remains unknown.

The README defines conceptual labels as 0 non-cancerous, 1 Gleason 3, 2 Gleason 4,
3 Gleason 5. **The decoded JPEG values are not those four categorical codes.**
For example, the first supplied mask has a dominant positive intensity of 200
with additional ringing intensities. Those values are retained exactly.
No rounding, thresholding, guessed intensity mapping or mask reconstruction has
been applied. A lossless mask source or a documented provider decoding rule is
needed before quantitative pixel-boundary evaluation. The viewer consequently
disables the interpreted segmentation overlay for SICAPv2 and provides the raw
mask file and image-level table label separately. Non-cancerous also must not be
renamed stroma.

The original `wsi_labels.xlsx` table maps 155 slides to 95 patient IDs. The primary
provider classification tables contain 9,959 training and 2,122 test patches,
with no shared patient IDs in those two tables. Another 6,702 acquired patch
pairs are outside those primary classification tables and remain explicitly
unassigned. The four supplied validation folds and cribriform tables are
retained unchanged, with table hashes and extracted metadata. This does not
assign the remaining patches to a new evaluation split.

## SegPath: eight native binary targets, bounded download

Original release links are taken directly from the
[author's dataset page](https://dakomura.github.io/SegPath/). All eight compressed
TAR archives together total **305,525,589,629 bytes**, before extraction. Full
releases are recorded in `sources.lock.json` for acquisition on suitable storage.
The selected subset retains three complete nonempty/nonfull H&E/mask pairs per
target from distinct source slide IDs in archive order. That selection is for
inspection and deliberately contains visible target boundaries; it cannot be
used to estimate an unbiased prevalence or population performance.

| Target | Marker | Original Zenodo record |
|---|---|---|
| Epithelium | panCK | [7412731](https://zenodo.org/records/7412731) |
| Smooth muscle / myofibroblasts | alpha-SMA | [7412732](https://zenodo.org/records/7412732) |
| Red blood cells | CD235a | [7412580](https://zenodo.org/records/7412580) |
| Leukocytes | CD45RB | [7412739](https://zenodo.org/records/7412739) |
| Lymphocytes | CD3/CD20 | [7412529](https://zenodo.org/records/7412529) |
| Endothelium | ERG | [7412512](https://zenodo.org/records/7412512) |
| Plasma cells | MIST1 | [7412500](https://zenodo.org/records/7412500) |
| Myeloid cells | MNDA | [7412690](https://zenodo.org/records/7412690) |

Original H&E PNGs are opaque RGBA, not altered into newly encoded RGB files.
Each image and binary mask is 984 x 984 pixels; 0 means outside that target and
1 means target. Separate target cohorts cannot be merged into one multiclass
image. Epithelium is not a malignancy label, and alpha-SMA is not an exhaustive
annotation of all stroma. The masks derive from immunofluorescence registration,
thresholding and provider postprocessing, with nuclear segmentation for some
cell types; they are not all manual expert tracings.

The [original publication](https://pmc.ncbi.nlm.nih.gov/articles/PMC9982301/)
reports a Hamamatsu NanoZoomer S60 at 40x and **0.220818 micrometres/pixel**.
The distributed 984px tiles correspond to 1024px patches with 20-pixel borders
removed; the acquisition does not rescale them. This is publication-reported
calibration, not independent optical measurement of each PNG.

The complete provider CSVs list 158,687 source patches. The 24 selected examples
comprise 22 training patches, one validation patch and one test patch; this is
not a held-out test cohort.

All original per-target CSVs and licenses are downloaded in full and verified
against the Zenodo MD5s. Supplied train/validation/test membership is preserved.
The data are CC BY-NC-SA 4.0. A TMA slide ID is not automatically a patient ID;
this subset does not independently reconstruct the provider's TMA-spot/patient
mapping or guarantee cross-target patient independence.

Only a bounded prefix of each original TAR is transferred. Complete native
PNG payloads are retained without modification, and source prefix bytes,
HTTP byte ranges, source record, source archive size/checksum and selected-file
SHA256 hashes are recorded. The **complete TAR MD5 cannot be verified from a
partial download**, and the report explicitly records that limitation. The
retained `.part` prefixes are provenance/recovery material, not complete usable
TAR archives. Interrupted transfers can replay the retained prefix and append
an exact resumed byte range. No upstream executable code is run.

## Commands

```bash
python -m pip install -r benchmark/requirements.txt numpy Pillow pytest
python -m benchmark.download sicapv2 --workers 4 --extract
python -m benchmark.prepare_requested sicapv2

python -m benchmark.download camelyon16-subset --workers 4
python -m benchmark.prepare_requested camelyon16-subset

python -m benchmark.segpath_subset --pairs-per-target 3 --max-prefix-gib 2
python -m benchmark.prepare_requested segpath-subset

python -m benchmark.examples --destination /mnt/sata_ssd/Bora-benchmark/examples --count 3
python -m benchmark.describe --examples /mnt/sata_ssd/Bora-benchmark/examples
python -m pytest -q benchmark/tests
```

These commands download, inspect supplied annotations and copy examples. They
do not run Bora. Results remain separate from acquisition metadata. PAIP and
canonical CRAG are still pending their previously documented access/provenance
conditions; neither is reported as acquired here.
