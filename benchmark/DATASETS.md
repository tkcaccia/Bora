# Dataset report for a future Bora benchmark

Prepared 15 October 2026. This is an acquisition and study-design report.
No Bora execution, accuracy estimate or comparative performance claim has
been made. Counts measured from the acquired files are in
[ACQUISITION.md](ACQUISITION.md); cohort sizes below describe the source datasets.

## Recommended roles

| Dataset | Source cohort / annotations | Proposed role | Acquisition scope |
|---|---|---|---|
| BCSS | 151 breast-cancer ROIs; 22 encoded mask values including ignored regions | Primary multiclass tissue interfaces | Full pinned, normalized RGB mirror |
| TIGER WSIROIS | 195 source slides; seven tissue classes in annotated ROIs | Multiclass validation with source-overlap control | Full ROI-level archive, not all WSIs |
| PAIP 2019 | 50 annotated HCC training WSIs; separate whole-tumor and viable-tumor masks | Large-scale liver tumor boundaries | Pending provider-approved delivery |
| GlaS | 165 colorectal images with gland instance masks | Supplementary gland-boundary assessment | Full pinned mirror |
| CRAG | 213 colorectal images with gland instance annotations in the source release | Supplementary irregular gland structures | Canonical release pending; candidate mirror staged separately |
| CAMELYON16 | 400 lymph-node WSIs; tumor contours and binary masks for annotated positive training slides | Optional WSI tumor-margin extension | Three original positive training WSIs; see REQUESTED_DATASETS.md |
| SICAPv2 | Prostate histology WSIs with global Gleason scores and patch-level grades | Optional grade-interface research after annotation review | Complete original publisher patch archive; raw JPEG mask encoding unresolved |
| SegPath | Eight cell-type segmentation tasks with immunofluorescence-derived masks | Optional cell-scale extension | Bounded native subset of all eight targets; see REQUESTED_DATASETS.md |

The primary question is refinement of existing tissue labels. Gland instance
separation, cell-type delineation and coarse patch grading address different
targets and should be reported separately. BCSS and non-overlapping TIGER ROIs
are the most direct initial tissue-level datasets. PAIP adds a distinct organ
and whole-slide scale after access approval.

## BCSS

[Original repository](https://github.com/PathologyDataScience/BCSS),
[original code map](https://github.com/PathologyDataScience/BCSS/blob/3c49d554497dd896cd02ffe382503ef4ce28b1c1/meta/gtruth_codes.tsv),
[Amgad et al., 2019](https://doi.org/10.1093/bioinformatics/btz083).
BCSS uses structured crowdsourcing; describe its annotation provenance
accurately rather than calling every pixel an independent expert annotation.
The dataset is CC0; the upstream download code has a separate MIT license.

Code 0 denotes outside-ROI pixels. Code 7 is explicitly excluded tissue and
code 15 is undetermined. Preserve all raw codes; future evaluators should
explicitly state whether 7 and 15 are ignored in addition to 0. This inventory
records the proposed ignore set `[0, 7, 15]` without calculating any scores.
Tumor, stroma, lymphocytic infiltrate and necrosis are codes 1–4. The full
22-code table is retained in `labels.json`. Rare categories must not be silently
merged or counted as background.

The acquired [MedOtter mirror](https://huggingface.co/datasets/MedOtter/BCSS/tree/502d4a3fbc77dbaca6f4664c19e2379ff077d418)
contains 151 image/mask pairs in ten Parquet shards. Its card reports
color-normalized RGB and 0.25 µm/pixel, rebuilt from another mirror. This is a
mirror preprocessing claim, not a fresh measurement of scanner calibration.
Appearance preprocessing can affect boundary refinement. A future raw-RGB
replication should use the original provider's download workflow and report
the exact RGB source. The mirror's `train` partition is a storage partition,
not an official held-out test set. Preserve TCGA patient identifiers.

## TIGER

[Provider data description](https://tiger.grand-challenge.org/Data/),
[ROI archive DOI](https://doi.org/10.5281/zenodo.6014422),
[provider license](https://zenodo.org/records/6014422/files/license.txt?download=1).
The WSIROIS source comprises 151 TCGA slides, 26 RUMC slides and 18 JB slides.
The downloaded ROI archive is much smaller than downloading every full WSI.
Masks encode invasive tumor, tumor-associated stroma, in-situ tumor, healthy
glands, necrosis, inflamed stroma and rest as values 1–7; 0 is excluded.
Annotations use the provider's 0.5 µm/pixel coordinate convention.

The archive contains a BCSS tissue subset and smaller tissue/cell ROIs. TCGA
material derives from BCSS and NuCLS: this is not an independent replication
of BCSS. For external validation after BCSS development, use non-TCGA cases
and retain slide groups. Do not assume non-TCGA slide IDs are verified patient
IDs. Nested ROIs within a slide must remain in the same analysis group.
Annotations and RUMC/JB images are CC BY-NC 4.0; TCGA images retain the original
TCGA rights, as stated in the included license. Seven-class masks should not
be assumed equivalent to the BCSS 22-code ontology.

## PAIP 2019

[Dataset description](https://paip2019.grand-challenge.org/Dataset/),
[participation/access](https://paip2019.grand-challenge.org/),
[current PAIP access portal](https://wisepaip.org/).
There are 50 annotated training slides, 10 validation slides and 40 test
slides. The challenge does not release the latter groups' ground truth in its
original protocol. Training data include SVS images, XML annotations and
binary TIF masks for whole tumor and viable tumor. The official ground truth
is the supplied TIF masks, not a new conversion of the XML. Negative pen
regions are excluded. Whole tumor and viable tumor are related but different
targets and require separate inventories and outcomes.

Registration and a signed, approved Data Use and Confidentiality Agreement
precede provider delivery. No agreement was signed and no PAIP data or examples
were obtained in this task. After delivery, verify image/mask pyramid levels,
coordinate alignment and pixel spacing before extracting windows. Estimate
storage from the approved file manifest before acquiring full WSIs.

## GlaS

[Original challenge site](https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/),
[challenge paper](https://arxiv.org/abs/1603.00275),
[acquired Kaggle mirror, version 1](https://www.kaggle.com/datasets/sani84/glasmiccai2015-gland-segmentation/versions/1).
There are 85 training images, 60 Test A images and 20 Test B images. Masks
contain gland instances; mask zero is a true non-gland class rather than
outside-ROI uncertainty. Preserve instance IDs. A later binary gland/non-gland
assay would lose instance separation information and must be named explicitly.
Do not treat the two test sets as one interchangeable difficulty stratum.

The original Warwick page redirected to sign-in during acquisition. The
mirror supplies BMP images, masks and `Grade.csv`. It lists its license as
"Other (specified in description)"; no broader redistribution permission is
inferred here. Cite the original challenge and record the mirror version.
Its patient table permits an overlap audit. The acquired official image
splits share patient IDs, so retaining those splits alone does not establish
patient-independent evaluation. Exact observed overlap is recorded in the
acquisition report. The challenge paper reports distributed spacing of
0.620 µm/pixel after rescaling native 0.465 µm/pixel scans. This dataset-level
evidence is now recorded in the description layer; no reliable calibration is
embedded in the acquired BMPs. See [ANNOTATIONS.md](ANNOTATIONS.md).

## CRAG

[Original data page](https://warwick.ac.uk/fac/cross_fac/tia/data/mildnet/),
[Graham et al., MILD-Net](https://arxiv.org/abs/1806.01963),
[OpenDataLab entry](https://opendatalab.com/OpenDataLab/CRAG).
The intended v2 reference has 173 training images and 40 test images. Its task
is gland instance segmentation. Older distributions may contain extra test
annotation files; the canonical importer requires exactly paired 173/40
images and masks. The original Warwick page currently requires sign-in.

An accessible [candidate mirror](https://www.kaggle.com/datasets/hoinhnphm/crag-dataset/versions/1)
is saved under `raw/crag-candidate`, separately from `raw/crag`. The uploader
provides no description and lists an unknown license. Pairing and file hashes
can be checked, but these do not establish original mask fidelity, official
splits or unchanged instance identities. Candidate examples are labelled as
such. Do not describe this candidate as a verified original CRAG release in
a manuscript. Obtain a provider-approved release and reconcile its metadata
before using the images for reported results.
The original CRAG paper reports 0.55 µm/pixel at 20× with an Omnyx VL120;
this nominal value is kept separately from the unknown actual spacing of
the candidate mirror. Gland instance IDs do not encode tumor/stroma classes.

## Optional extensions

[CAMELYON16](https://camelyon16.grand-challenge.org/Data/) contains 270 training
and 130 test WSIs. Positive training slides have tumor contour XML and binary
masks. The original challenge emphasizes metastasis detection; a segmentation
boundary study would need its own specified sampling, handling of negatives
and metrics. Three complete positive training WSIs are now acquired as an explicit convenience subset;
see [REQUESTED_DATASETS.md](REQUESTED_DATASETS.md).

[SICAPv2, version 1](https://doi.org/10.17632/9xxm58dvs3.1) is described by its
provider as WSI global Gleason scores plus patch-level grades and is CC BY 4.0.
The original publisher archive is now acquired. Its raw JPEG mask encoding
and boundary fidelity remain unresolved. Audit its actual annotation granularity before calling it an expert
pixel-boundary benchmark. Coarse grade labels cannot automatically serve as
high-resolution boundary ground truth.

[SegPath](https://github.com/dakomura/SegPath) publishes eight cell-type tasks.
The [author's data site](https://dakomura.github.io/SegPath/) describes paired
984×984 H&E images and binary masks derived using immunofluorescence. These
are not interchangeable with manually drawn tissue-region boundaries. The
data site states CC BY-NC-SA 4.0. Its size and different annotation mechanism
make it a supplementary task. A bounded native subset is now acquired for
each of its eight targets; see [REQUESTED_DATASETS.md](REQUESTED_DATASETS.md).

## Information to retain for the manuscript

Report the source release, mirror and preprocessing, acquisition date, archive
and per-file hashes, selected patient/slide/ROI IDs, physical spacing, label
mapping, ignored pixels, official versus newly assigned splits, cross-dataset
overlap, unavailable data, and all exclusions with reasons. Keep development
and final evaluation groups separate at the patient level where identifiers
are available. Original instance masks must remain available even if a future
analysis uses a semantic gland union.

For a later refinement study, distinguish controlled degradation of reference
masks from refinement of predictions produced by a separate tissue classifier.
The former measures boundary-recovery behavior under a specified perturbation;
it does not demonstrate clinical or end-to-end segmentation performance.
Prespecify coarsening scale, grid placement, valid annotation region, overlap
metrics and boundary-distance tolerances. Report physical units only when
pixel spacing has been verified. Compare the unchanged input labels with the
complete Bora pipeline and relevant stage ablations using identical inputs.
These are recommendations for future work; none were executed here.
