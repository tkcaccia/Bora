# Additional datasets identified from the attached PathSegmentor paper

Acquisition date: 15 October 2026. Data are on Chiamaka at
`/mnt/sata_ssd/Bora-benchmark`. This work downloads and inventories data only;
Bora, PathSegmentor and VISTA-PATH were not run. No evaluation scores or coarse
input labels were generated.

## Selection and actual availability

The supplied paper, *Segment Anything in Pathology Images with Natural Language*,
arXiv:2506.20988v2 (19 August 2025), lists 21 source datasets in Appendix A,
Table A1 (printed page 44). The best additions for refining tissue regions are
**WSSS4LUAD**, **RINGS**, **CoCaHis** and **DigestPath2019**. They add lung,
prostate and colorectal metastasis tissue, complementing the existing breast
and colorectal gland data.

| Source dataset | Release acquired | Usable annotation scope | Status |
|---|---|---|---|
| WSSS4LUAD | Complete pinned MedOtter/WSSS4LUAD-v2 mirror, 3 Parquet files, 1,632,102,034 bytes | 40 validation image/mask pairs; 3 tissue classes plus excluded white background | Validation pairs pass native geometry checks; test masks quarantined |
| RINGS | Original publisher TRAIN.zip and TEST.zip, 5,279,334,842 bytes total | 1,500 RGBs, each with a manual gland mask and a separate manual tumor mask; released 1,000 train / 500 test split | Complete original release downloaded |
| CoCaHis | Complete author-released VISTA-PATH **processed training subset**, 2 TARs, 434,042,880 bytes | 302 paired 1024 x 1024 patches with binary masks | Exploratory staging; original CoCaHis not acquired; semantic mapping, license and physical calibration require reconciliation |
| DigestPath2019 | None | Colon tissue segmentation is relevant to tumor borders | Provider requires a signed database-use agreement; no agreement was signed |

The available CoCaHis derivative must not be described as acquisition of all 82
original images, seven annotators, 19 patients, or an independent test cohort.
The original provider server was unreachable from both the Mac and Chiamaka.
The derivative was released by the VISTA-PATH authors, with source-dataset terms
retained, and explicitly contains training data. No upstream processing script
was executed.

## WSSS4LUAD: tissue compartments and pairing audit

The [original challenge publication](https://arxiv.org/abs/2204.06455) describes
87 source slides (67 GDPH, 20 TCGA), 10,091 training patches with **image-level**
labels, 40 densely annotated validation ROIs and 80 test ROIs. The source slides
are separated between challenge partitions. Slide/patient assignments for the
validation and test ROIs are not available in the acquired mirror, so ROI IDs
must not be treated as independent patient IDs.

Pinned mirror: [MedOtter/WSSS4LUAD-v2](https://huggingface.co/datasets/MedOtter/WSSS4LUAD-v2/tree/aa035a16f230622a17f51bc6b5180209a04bec7b).
No stain normalization or geometric processing was applied during acquisition.

| Native mask value | Meaning | Include in tissue accuracy? |
|---|---|---|
| 0 | Tumor epithelium | Yes |
| 1 | Tumor-associated stroma | Yes |
| 2 | Normal tissue | Yes |
| 3 | White background, including alveolar spaces | No |

**Zero means tumor, unlike BCSS and TIGER.** Preserve this ontology rather than
applying one global background rule. Annotation was assisted by model proposals,
manually refined by pathologists and confirmed by a review board; describe it as
pathologist-reviewed, model-assisted ground truth rather than entirely de novo
manual tracing.

Actual audit of every row in the pinned release:

- Training: all 10,091 patches have image-level presence labels and no dense mask.
  They remain in the downloaded Parquet, with a separate image-label inventory;
  they are not included as segmentation ground truth.
- Validation: all 40 masks match the RGB dimensions and the separately supplied
  background masks. There are nine large ROIs with both dimensions at least
  1,000 pixels. Three large tumor/stroma examples are retained locally.
- Test: 10 masks are absent and the other 70 disagree with their RGB dimensions.
  Their background masks match the RGB sizes, indicating a dense-mask pairing
  problem. All 80 test ROIs are excluded from the paired inventory. The exact
  received bytes, source IDs, dimensions and hashes are retained in the audit.
  Masks were not resized or reassigned to make them fit.

The accepted ROIs range from 217 to 4,074 pixels in width and 206 to 4,074 in
height. These are the native distributed dimensions, not PathSeg's processed
patches. File-size agreement is necessary but cannot establish expert annotation
fidelity; a visual review is still appropriate before reporting results.

Microscopy: the challenge paper reports distributed patches at **10x**. It
reports GDPH source scanning at **40x, 0.2517 micrometres/pixel**, using Leica GT
450. That is source-slide spacing, not verified spacing of the downsampled
mirror patches, and TCGA acquisition is heterogeneous. No micrometres/pixel value
is assigned to these PNGs until the distributed-image calibration is confirmed.
Physical boundary distances should therefore not be reported yet; pixel-based
measurements must identify this limitation.

## RINGS: original gland boundaries and cancer tissue

The [original publisher release](https://data.mendeley.com/datasets/h8bdwrtnr5/1)
is version 1, DOI **10.17632/h8bdwrtnr5.1**, released under **CC BY 4.0**.
The associated [Salvi et al. publication](https://doi.org/10.1016/j.artmed.2021.102076)
is *A hybrid deep learning approach for gland segmentation in prostate
histopathological images*, Artificial Intelligence in Medicine 115 (2021),
102076.

Each native 1500 x 1500 RGB PNG has two corresponding one-bit PNG masks:

- `MANUAL GLANDS`: 0 outside the gland target; 1 inside a gland region.
- `MANUAL TUMOR`: 0 outside the tumor target; 1 inside cancer tissue.

They are **separate binary tasks**, not mutually exclusive classes. A negative
tumor pixel is not automatically stroma. The supplied gland mask has no instance
IDs; this is not an instance-label release even though the original publication
describes annotation of individual gland contours. No connected components,
synthetic class maps or new masks were created.

The two targets are indexed separately as `rings-glands.jsonl` and
`rings-tumor.jsonl`. They refer to the same 1,500 RGBs: **3,000 target pairs do not
mean 3,000 independent images**. Retain the published train/test split. Filename
prefixes are recorded only as within-partition source identifiers; their patient
meaning is unverified, and they must not be used to claim a patient-independent
resplit. Mac resource-fork files in the original ZIPs are retained as archive
contents but never treated as images.

The attached paper lists RINGS at **100x** in Table A1. The PNGs inspected have no
embedded pixel-spacing calibration; the distinction between total and objective
magnification is unresolved here. **Actual micrometres/pixel remains unknown**;
100x cannot be converted into a physical pixel size without additional metadata.
This is valuable for gland and tumor morphology, while calibrated distance
metrics await verified spacing.

## CoCaHis: explicit distinction between original and derivative

The [original CoCaHis resource](https://cocahis.irb.hr/) accompanies Sitnik et al.,
[*A dataset and a methodology for intraoperative computer-aided diagnosis of a
metastatic colon cancer in a liver*](https://doi.org/10.1016/j.bspc.2020.102402).
It contains 82 H&E frozen-section images from 19 patients, multiple expert
annotations and alternative stain variants. Original HDF5 download information
and provider checksums are recorded in `sources.lock.json`; its host was
unreachable during this acquisition.

The acquired release is instead the **302-patch training subset** from
[zhihuanglab/VISTA-PATH_data](https://huggingface.co/datasets/zhihuanglab/VISTA-PATH_data/tree/42d723a4d7dbd38bb605f2aaa8e32cb7353fe83e/CoCaHis),
pinned at commit `42d723a4d7dbd38bb605f2aaa8e32cb7353fe83e`. The release describes
RGB PNG patches with uint8 categorical masks and 0 as background/unannotated.
All 302 acquired masks contain values 0 and 1 and match their 1024 x 1024 RGBs.
The data card does not resolve the precise foreground definition, annotator
aggregation, original stain variant, patient mapping or resampling history for
this subset. Foreground is therefore displayed as **supplied foreground,
semantic mapping unverified**, rather than asserting an unverified tumor/stroma
label. This dataset is staged for exploration, not yet approved for a manuscript
benchmark. It is not a held-out test set for VISTA-PATH.

The attached Table A1 reports original CoCaHis as 40x; that does not calibrate
these processed patches. Micrometres/pixel is unknown. No new resizing was
performed during acquisition. The original dataset license must be checked
before analysis or redistribution; the derivative card explicitly preserves
each constituent dataset's original terms.

## Disposition of all 21 paper datasets

This table records source-dataset relevance, not new evaluation results. The
paper's anatomical taxonomy and counts must not be mistaken for the original
providers' label ontologies or cohort sizes.

| Dataset in Table A1 | Boundary task | Disposition |
|---|---|---|
| BCSS | Multiclass tissue | Already acquired; keep original 22-code catalog distinct from PathSeg's taxonomy |
| CoCaHis | Metastatic tumor in liver | New processed subset staged; original host unavailable |
| CoNIC2022 | Nuclei and cell types | Optional cell-scale extension; outside primary tissue-region benchmark |
| CRAG | Gland boundaries | Existing candidate remains separate from canonical CRAG |
| DigestPath2019 | Colon tissue/tumor margins | High-priority next addition, pending required data-use agreement |
| GlaS | Gland boundaries | Already acquired |
| Janowczyk | Nucleus segmentation for the source subset used here | Optional cell-scale extension; do not confuse with the author's other tasks |
| MIDOG | Mitotic figures and hard negatives | Detection/cell-scale task, lower priority for tissue-region boundaries |
| MoNuSAC | Nuclear instances and types | Optional cell-scale extension |
| NuCLS | Breast nuclear classes | Optional cell-scale extension |
| PanNuke | Nuclear instances and types across organs | Optional cell-scale extension; PathSeg's 85 hierarchical labels are not 85 original nuclear types |
| RINGS | Prostate glands and cancer tissue | New complete original release acquired |
| SegPath | Immunofluorescence-derived cell/tissue masks | Subsequent bounded native subsets; see REQUESTED_DATASETS.md |
| SICAPv2 | Prostate Gleason regions | Original archive acquired subsequently; JPEG annotation encoding unresolved; see REQUESTED_DATASETS.md |
| TNBC | Breast nuclei/cell types | Optional cell-scale extension |
| WSSS4LUAD | Lung tumor/stroma/normal tissue | New mirror acquired; use 40 validated dense validation pairs only |
| CPM17 | Nucleus segmentation | Paper's external set; optional cell-scale extension |
| CPM15 | Nucleus segmentation | Paper's external set; optional cell-scale extension |
| Kumar | Multi-organ nuclei | Paper's external set; optional cell-scale extension |
| Lizard | Colorectal nuclei and types | Paper's external set; optional cell-scale extension |
| CoNSeP | Colorectal nuclei and types | Paper's external set; optional cell-scale extension |

## How these data can be described in a manuscript

Report the **acquired release, unique images, annotation targets, original
partitions and exclusions**, not just dataset names. The attached paper's
Section 4.1 (printed pages 26-27) describes magnification harmonization, patching
and resizing to 1024 x 1024. For example, Table A1's 38,227 RINGS images and 2,863
WSSS4LUAD images are PathSeg processing counts, not the native cohorts acquired
here. Our workflow preserves every source image/mask's received bytes and uses
native distributed pixel geometry.

A suitable acquisition description is:

> Additional source datasets were selected from the PathSeg source catalog for
> relevance to tissue and gland boundary refinement. The original RINGS release
> supplied 1,500 prostate images with paired gland and tumor annotations. A
> pinned WSSS4LUAD mirror supplied 40 geometrically consistent annotated
> validation ROIs; 80 test ROIs were excluded because their dense masks were
> missing or dimensionally inconsistent. A separately identified processed
> CoCaHis training subset was staged for exploratory inspection and excluded
> from the primary benchmark until semantic and calibration provenance could
> be resolved. No segmentation model was executed during data acquisition.

Additional reporting requirements before any future evaluation:

- Separate primary tissue-compartment results from gland-boundary and cell-scale
  tasks. Keep supplied annotations immutable and distinguish them from future
  coarse inputs or model predictions.
- Use each dataset's class ontology and ignore regions. Report the tissue types
  actually present per image, not a global assumption that zero is background.
- Report physical distances only for verified pixel spacing. Keep magnification,
  source-slide spacing and distributed-image spacing separate.
- Preserve released partitions, resolve patient/slide mappings before resplitting,
  and report dataset overlap. Existing TIGER/BCSS reuse remains relevant.
- Record training exposure for any future backbone or baseline. CoCaHis, RINGS
  and WSSS4LUAD are **internal datasets** in PathSegmentor, and the downloaded
  CoCaHis derivative is VISTA-PATH training data. They are not novel external
  tests for those models. Bora has not been run or trained in this acquisition.
- Describe WSSS4LUAD validation use as validation; an already exposed validation
  release should not silently become a claimed blind test cohort.

## Reproduce these additions

From the Bora repository root in the existing benchmark environment:

```bash
python -m benchmark.download wsss4luad
python -m benchmark.download rings --workers 4 --extract
python -m benchmark.download cocahis-vista --extract

python -m benchmark.prepare_additional wsss4luad
python -m benchmark.prepare_additional rings
python -m benchmark.prepare_additional cocahis-vista
python -m benchmark.audit_additional
python -m benchmark.examples --destination /mnt/sata_ssd/Bora-benchmark/examples --count 3
python -m benchmark.describe --examples /mnt/sata_ssd/Bora-benchmark/examples
python -m pytest -q benchmark/tests
```

Only explicitly listed VISTA-PATH shards are fetched; this does not download its
237 GiB aggregate collection. Safe TAR extraction permits paired image/mask
shards while rejecting links, traversal and collisions. Multi-ZIP datasets use
separate archive directories. All acquired files and valid image/mask pairs
receive SHA256 hashes. Publisher/LFS checksums are distinguished from hashes
observed after download. `additional-acquisition-2026-10-15.json` records the
actual release and inventory checksums; per-image metadata and class legends
also appear in `metadata/image-information.csv` and the local examples viewer.

A byte-level audit found 1,842 unique new annotated RGBs and no identical RGB
bytes crossing their released partitions. This does not establish patient or
slide independence. RINGS contributes 1,500 of those RGBs with two annotation
targets each; the staged CoCaHis subset contributes 302 and WSSS4LUAD contributes
40. At that acquisition snapshot the paired inventory contained 4,401 distinct
image paths and 5,901 annotation targets, including existing candidate data.
Subsequent SICAPv2 and CAMELYON16/SegPath acquisitions are recorded in
[REQUESTED_DATASETS.md](REQUESTED_DATASETS.md).

The source PDF is retained locally by the user and is not redistributed in Git.
The original paper code's acquisition instructions were read as source
information; they were not treated as authorization to run models or sign data
agreements.
