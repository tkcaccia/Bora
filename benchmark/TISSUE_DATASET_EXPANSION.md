# Additional tissue-region datasets

This acquisition extends the Bora development benchmark with **40 eligible
native-source ROI images per included dataset**. It does not run Virchow2,
change the RGB conversions, run Bora, or create new accuracy scores. The
machine-readable acquisition manifest records actual completed counts. Source
patients, slides, TMA cores and ROIs are different units and are reported
separately; forty crops are not automatically forty independent patients.

## Physical calibration requirement (latest user instruction)

**New Virchow2 inference and Bora benchmark runs require traceable native and
working µm/pixel axes.** Magnification alone and pixel dimensions are
insufficient. Unknown-calibration images may remain in the acquisition inventory
but are excluded from new model inference and benchmark scores. A metadata audit
of the299 size/group-qualified existing fields clears269;9 WSSS4LUAD and21 RINGS
fields remain calibration-pending. Earlier signed scores remain historical.

The current encoder uses224×224px inputs at approximately0.5,1 and2µm/px:
physical window widths112,224 and448µm. Calculate exact per-axis values from
actual resize dimensions, including source pixels per window,14px token size,
112px stride and90px coarse supervision tiles. Record source scanner/objective
when reported; optical resolving power remains unknown unless actually supplied.
The hard checks are in `calibration.py` and `CALIBRATION_POLICY.json`, called
before GPU loading and before admitting inputs to cycle11 onward. Resampling is
explicit: interpolation can match a physical field but cannot restore missing
optical detail or establish an acquired microscope magnification.

## Acquired and audited fields

The first acquisition audit contains160 physically calibrated fields:40 each
from IGNITE, PESO, GCSS and GleasonXAI. Original file hashes, paired native and
working grids, all fine classes, observed coarse groups and physical window
sizes were checked. This is an acquisition inventory; a combined inference
cohort and new segmentation scores have not yet been frozen.

The local review page is
[the acquisition viewer](http://127.0.0.1:64889/bora-tissue-acquisition-2026-10-18-v1/index.html).
Each original/reference/coarse preview has a100µm scale bar. The viewer states
the actual native/working axes and any interpolation. Two aligned native
image/mask examples per completed dataset are copied under
`benchmark/examples/tissue-expansion-2026-10-18-v1/`, with individual JSON
provenance and separately named working grids. Source data remain outside Git.

## Sources being acquired

|Dataset|Annotation task|Released source / analysis unit|Native sampling|
|---|---|---|---|
|[IGNITE v2](https://zenodo.org/records/17735903)|NSCLC H&E tissue compartments, including tumor epithelium, reactive epithelium, stroma, inflammation and necrosis|H&E ROI / ROI; prefer one per provider patient ID|Provider0.5µm/px|
|[GCSS](https://zenodo.org/records/8176354)|Gastric tumor, lymphoid stroma, desmoplastic stroma, muscle and necrosis|TCGA-STAD slide-derived ROI / ROI; one per selected TCGA case|Source GDC SVS calibration × original crop bounds / distributed PNG dimensions, independently for each axis|
|[PESO test release v1.1](https://zenodo.org/records/5137717)|Epithelial versus non-epithelial tissue; no gland-instance task|2500px JPEG ROI and WSI-sized categorical mask / ROI; one per40 source slides|Provider ROI/XML0.48µm/px; source GT grid0.24µm/px|
|[PUMA training tissue annotations](https://zenodo.org/records/15050523)|Melanoma tumor, stroma, vessels, epidermis and necrosis; nuclei files unused|5120px true-context ROI /2048px centre crop with expert labels only in its1024px centre|Exact TIFF axis tags, approximately0.2263µm/px, rather than rounded0.22|
|[ORCA author validation set](https://sites.google.com/unibas.it/orca/download)|Carcinoma versus non-carcinoma tissue, with non-tissue ignored|4500px/2250px circular source cores / ROI; TCGA case from filename|Preserve native pixels until exact source spacing is traceable; never infer MPP solely from20x/40x|
|[Histo-Seg v2](https://data.mendeley.com/datasets/vccj8mp2cg/2)|Skin layers, tissue compartments and carcinomas,11 positive classes; glands are one semantic tissue type|Original large JPEG slide mosaics /40 non-overlapping native ROIs; exact smaller slide count disclosed|Provider20x acquisition; exact MPP unverified, so pixels only|
|[GleasonXAI](https://doi.org/10.6084/m9.figshare.27301845.v1)|Architectural Gleason-pattern3/4/5 tissue regions, not gland instances|Original TissueArray TMA core / ROI; one per selected core|Author calibration code reports0.5455µm/px for original TissueArray images|

All images used by the algorithm are ROIs, including fields extracted from
WSIs. This is not an evaluation on complete WSIs. IGNITE uses training fields;
PUMA uses training tissue annotations; PESO test and ORCA validation fields
become **development data**, with their provider split tags retained. No
independent provider-test claim is made. TCGA-derived datasets can share cases;
source-group reconciliation and grouping are required before any held-out
experiment or combined-cohort inference.

## Admission rules

Require two distinct positive tissue/task groups in the coarse input and in
its actual feature-selection observations. Ignore/unannotated/glass is not a
second group. Require at least16 usable90px coarse observations, and at least
four complete224px fields on each axis in both the actual native source and
working image. A small image cannot qualify through padding, overlap, rotation
or upsampling. These are engineering eligibility cutoffs, not a statistical
sample-size validation.

For a catch-all negative target, require a stained-tissue observation so glass
alone cannot qualify as another tissue type. Do not apply that rule to explicit
expert pale-tissue classes such as fat. Preserve every fine reference class,
including classes absent from the coarse supervision. See
[`TISSUE_ACQUISITION_POLICY.json`](../experiments/bora_extension/TISSUE_ACQUISITION_POLICY.json)
and the existing tile-observation policy.

Acquisition selects boundary-containing, annotation-diverse fields without
access to algorithm predictions, quality scores or runtime scores. Fine masks
are used here to create controlled coarse inputs and retain expert references;
prediction must receive only the image and coarse input. This is a controlled
refinement benchmark, not population-random sampling.

## Geometry and annotation conventions

- **PESO:** the JPEG/XML grid and the source mask grid differ by exactly2.
  Read the5000×5000 GT window at twice the XML coordinates, then project labels
  at pixel centres using `[1::2,1::2]`. Reading the coordinates at1:1 selects
  unannotated parts of the GT slide. The corrected overlays were inspected.
  Non-epithelial is a broad negative target, including lumina; it is not an
  expert stroma class. Only original released test GT is used, rather than the
  U-Net-generated training annotations.
- **GCSS:** the distributed ROI was downsampled relative to its original TCGA
  crop. Assigning source-slide MPP directly to the PNG would be wrong. Wider
  native fields are used where necessary to retain896 original pixels before
  any0.5µm/px working-grid resampling. Labels−1,0,6 are outside the five target
  compartments and remain ignored.
- **IGNITE:** use H&E tissue segmentation only. Code0 is unknown, code1 is
  glass/background and codes2–16 are tissue compartments. Context is not
  necessarily included in the annotation-extent metadata; verify actual paired
  image/mask dimensions.
- **PUMA:** use the real context TIFF, never padded or enlarged1024px originals.
  Its central1024px RGBA pixels must exactly equal the separately released ROI.
  Rasterize only tissue polygons, preserve holes, and record vector-to-pixel
  and overlap conventions. A single original polygon extends to−1px; the fixed
 1024px canvas clips excursions of at most1px, with no coordinate rescaling.
 Larger excursions fail rather than silently changing the frame.
 Unannotated context remains reference0, while all
  its tissue is available for future encoder inference.
- **ORCA:** the original paper specifies black non-tissue, grey non-carcinoma
  tissue and white carcinoma. Released PNGs contain grey129/130 and antialiased
  intermediate levels. The explicit nearest-semantic-intensity conversion uses
  cutpoints65 and193, with source levels and conversion receipts retained.
  Boundary pixels are retained rather than erased. A grey shade is not another
  tissue class.
- **Histo-Seg:** use the exact RGB palette in the source article's Table1,
  including background0, keratin, epidermis, papillary/reticular dermis,
  hypodermis, glands, follicles, inflammation, BCC, SCC and IEC. Stream native
  mosaics through libvips; never load a gigapixel image into a NumPy array.
- **GleasonXAI:** retain normalized original expert polygons. Rasterize per
  provider annotator ID in grade order3,4,5, matching the author's frame order.
  The local hard reference uses the unique plurality among covering annotation
  IDs; ties and unannotated regions are ignored. This consensus convention is
  ours, not an unchanged provider-supplied hard mask. Provider annotation IDs
  are not asserted to be independent people.

Pixel sampling in µm/pixel, objective magnification and optical resolving power
are distinct. Exact native/working axes, field of view and source coordinates
are recorded when available. Optical resolving power is not inferred. Unknown
MPP remains null and cannot support physical-area or physical-distance claims.
Negative-target adapters must preserve their task semantics in future scoring;
non-epithelial/non-carcinoma is not automatically stroma. Actual background
support is an inference/output step, not a new annotated tissue class.

## Other sources reviewed

|Source|Disposition / reason|
|---|---|
|[PAIP2019](https://paip2019.grand-challenge.org/Dataset/), later PAIP challenges|Relevant tumor-region annotations; provider approval/data-use agreements are required. No agreement was signed or access application submitted.|
|[AGGC2022](https://aggc22.grand-challenge.org/AGGC22/)|Relevant prostate regions; provider application/delivery process, not immediately downloadable here.|
|[DROID](https://datahub.aida.scilifelab.se/10.23698/aida/drco)|Relevant tissue compartments; controlled AIDA research access.|
|[DigestPath](https://digestpath2019.grand-challenge.org/)|Relevant colon tumor masks; original access conditions require separate agreement. Do not substitute model predictions as expert GT.|
|[Queensland skin](https://doi.org/10.14264/8be4bd0)|Strong candidate:290 hand-annotated tissue mosaics, native0.67µm/px; original host currently returns403 from this environment. Downsampled versions need their actual multiplied MPP. Not counted as acquired.|
|[CATCH](https://github.com/DeepMicroscopy/CanineCutaneousTumors)|Relevant canine tissue compartments, a separate cross-species stratum. Native IDC DICOM acquisition and annotation-to-slide mapping still need an adapter; a4.05µm/px mirror is not native0.25µm/px data. Not counted as acquired.|
|[BACH](https://zenodo.org/records/3632035)|The WSI segmentation subset is relevant; microscopy classification images are not eligible. Separate WSI/annotation adapter and source CC BY-NC-ND notices must be retained. Not counted as acquired in this batch.|
|[HunCRC](https://doi.org/10.7937/tcia.9cjf-0127)|Native WSI/region source is relevant; classification-only small-patch exports do not substitute for it. Native multipart access and dense-mask pairing not yet reconciled.|
|[ADP/HistoSegNet](https://www.dsp.utoronto.ca/projects/ADP/projects/index.php)|Check actual pixel-annotated subset, source permissions and real field size; image-level labels alone cannot be used as masks. Not counted.|
|[PanopTILs](https://sites.google.com/view/panoptils/)|Native1024px fields at0.25µm/px are too small after the frozen0.5µm/px base sampling; also check BCSS/NuCLS source overlap.|
|SICAPv2, LUAD-HistoSeg, EBHI-SEG|Available small/derivative fields do not satisfy the current genuine-field sampling requirement; do not enlarge them to qualify.|
|GlaS, CRAG|Gland-instance tasks are excluded from this tissue-region expansion.|
|PanNuke, MoNuSeg, CoNIC, cell-only SegPath tasks|Cell/nucleus tasks are excluded.|
|NCT-CRC, Kather texture sets, BRACS and classification-only collections|Image labels are not pixel-level tissue-region ground truth.|
|[Cross-cancer TCGA tumor ROIs](https://zenodo.org/records/18668580)|Released images do not include the required expert multiclass pixel masks; related model predictions are not expert GT.|
|[Tissues with Issues](https://zenodo.org/records/21649203)|Tissue-versus-glass task supplies only one actual tissue group, so fails the two-tissue-group rule.|

These are explicit acquisition dispositions, not a claim that every public
histopathology dataset has been exhausted. Relevant pending sources stay in the
catalog rather than being misreported as downloaded benchmark images.

## Reproduction and file layout

Use the existing benchmark virtual environment with NumPy, Pillow, OpenCV,
tifffile, OpenSlide, gdown and libvips/pyvips. Limit numerical/OpenMP/libvips
threads to2, use low-priority workers, and retain20GiB free on the data SSD.
No source-provided executable scripts or model weights are run.

From the repository root, the entry point is
[`download_additional_tissues.py`](download_additional_tissues.py):

```bash
python benchmark/download_additional_tissues.py download-primary --data /mnt/sata_ssd/Bora-benchmark
python benchmark/download_additional_tissues.py inventory --data /mnt/sata_ssd/Bora-benchmark
python benchmark/download_additional_tissues.py download-extra --data /mnt/sata_ssd/Bora-benchmark
python benchmark/download_additional_tissues.py orca-download --data /mnt/sata_ssd/Bora-benchmark
python benchmark/download_additional_tissues.py prepare --dataset ignite --data /mnt/sata_ssd/Bora-benchmark --out /path/to/fresh/inputs
```

Repeat preparation for each dataset in a fresh output location. The actual
execution manifest lists the versioned acquisition roots used on Chiamaka;
these paths are not interchangeable with the prior299-case encoded cohort.
The viewer currently resolves those recorded execution roots. Preparation
outputs `PLAN.json` only after40 cases qualify; a shortfall gets an explicit
`QUOTA_SHORTFALL.json`, never synthetic or duplicated replacements.

Each case contains `original.native.png`, working `original.png`, categorical
`reference.native.tif`, controlled `coarse.tif` and `INPUTS_ACQUIRED.json` with
class maps, ignore values, coordinates, source IDs, source/file hashes and
eligibility observations. Original archives/native files are under
`/mnt/sata_ssd/Bora-benchmark/raw`; acquisition receipts are under `metadata`.
The report/viewer includes preview JPEGs only; these are not categorical metric
masks. Large datasets use source byte-range caches with retained block hashes,
selected ZIP-member CRC checks and selected file hashes. A range subset never
claims that the publisher's complete-archive checksum was verified.

Before another Bora cycle: finish the independent acquisition audit, reconcile
shared source cases and sampling, freeze a combined cohort, encode all tissue
including unannotated context, and rerun matched parents on the identical
cohort. Changing the image denominator cannot be reported as an algorithm gain.
The signed earlier cycle artifacts remain unchanged.
