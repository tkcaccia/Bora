# Bora: scalable boundary-constrained refinement of tile-level tissue labels into pixel-accurate segmentations

**Manuscript type:** Original Research Article  
**Target journal:** *Medical Image Analysis*  
**Status:** protocol-stage draft; all bracketed fields require completion before submission

**Authors:** [AUTHOR 1], [AUTHOR 2], [AUTHOR 3]  
**Affiliations:** [AFFILIATIONS]  
**Corresponding author:** [NAME, POSTAL ADDRESS, EMAIL]  
**ORCID identifiers:** [ORCIDS]

## Highlights

- Bora converts coarse, multi-label tissue masks into full-resolution pixel masks.
- Refinement is confined to an editable boundary band while label cores are protected.
- Overlapping border tiles enable processing independent of the number of input labels.
- Annealed label competition and native-resolution watershed refine boundaries sequentially.
- Raster OME-TIFF and vector GeoJSON outputs support interoperable downstream analysis.

## Abstract

Tile-level classifiers are computationally attractive for gigapixel histopathology, but their block-like boundaries can distort morphometry, spatial relationships, and downstream region-based analysis. We present **Bora**, a boundary-constrained algorithm that converts a labelled tile-level mask into a full-resolution pixel-level tissue segmentation. Bora accepts a tissue image in pyramidal OME-TIFF format and a spatially aligned TIFF label mask containing an arbitrary number of segment identifiers. Its refinement chain first applies deterministic annealed-wand competition: a bounded multiclass Potts model operating only near interfaces between foreground labels. It then applies marker-controlled watershed on the native-resolution image gradient while preserving eroded label cores. MedSAM can provide a subsequent box-prompted refinement stage; PathSegmentor is reserved for experiments with an explicit label-to-text mapping. Overlapping predictions are reconciled by confidence, followed by topology-conscious cleanup and restoration of protected cores. Outputs comprise a tiled, losslessly compressed pyramidal OME-TIFF label map and a coordinate-matched GeoJSON representation.

We will evaluate Bora on [NUMBER] slides from [COHORTS, ORGANS, STAINS AND INSTITUTIONS], using pixel-level expert annotations created under [ANNOTATION PROTOCOL]. Primary endpoints are boundary intersection-over-union and normalized surface Dice; secondary endpoints include Dice similarity coefficient, 95th-percentile Hausdorff distance, label retention, topology errors, runtime, and peak memory. Comparisons will include the unrefined tile mask, annealed-wand competition alone, native watershed alone, their sequential combination, and the combination followed by MedSAM or PathSegmentor. **No performance results are reported in this draft because the benchmark has not yet been run.**

**Keywords:** computational pathology; semantic segmentation; boundary refinement; whole-slide imaging; MedSAM; PathSegmentor; OME-TIFF; GeoJSON

## 1. Introduction

Semantic segmentation is central to quantitative pathology because regional measurements depend on where tissue compartments begin and end. Whole-slide images are, however, commonly analysed as grids of tiles: a classifier assigns one class to each tile and the class map is projected back into slide coordinates. This design reduces computational and annotation costs but quantizes boundaries at the tile stride. The resulting error is concentrated along interfaces, precisely where invasion fronts, tumour–stroma contacts, gland contours, and tissue exclusion zones are measured. Region-overlap metrics can obscure such errors for large objects, motivating explicit boundary-sensitive evaluation [1].

Promptable foundation models provide a potential route from coarse masks to local, image-supported contours. The Segment Anything Model (SAM) introduced a promptable segmentation architecture trained at large scale [2]. Direct use on medical images is not uniformly reliable, particularly for weak or ambiguous boundaries [3]. MedSAM adapts SAM to medical imagery using more than one million image–mask pairs across multiple modalities and disease contexts [4]. Pathology-specific approaches have subsequently targeted the appearance and scale distribution of histology. PathoSAM focuses on interactive and automatic nuclei segmentation [5], while PathSegmentor uses natural-language prompts to delineate histological structures across spatial scales [6]. These models are promising refinement engines, yet applying them indiscriminately across a gigapixel image is expensive and risks altering already reliable regions.

We hypothesize that coarse tissue labels can serve as spatial priors and that model inference need only occur near their boundaries. This leads to a constrained formulation: preserve an eroded core of every input segment, construct an editable band around its boundary, process only image tiles intersecting that band, and arbitrate conflicts among label-specific predictions. The approach is adapted from the MedSAM refinement stage of CellPhenotyper [7], where large masks are handled by downsampled morphology and full-resolution processing is restricted to overlapping cluster-border tiles. Bora generalizes this step into a standalone, format-aware tool that accepts any number of mask labels and emits both raster and vector products.

The intended contributions are:

1. a scalable coarse-to-fine formulation that limits foundation-model inference to label boundaries;
2. a protected-core and constrained-fusion mechanism designed to preserve semantic identity across an arbitrary number of input labels;
3. a reproducible sequence of annealed-wand multiclass competition, native-resolution watershed, and optional promptable-model refinement; and
4. an evaluation protocol emphasizing boundary fidelity, failure safety, computational cost, and whole-slide interoperability.

## 2. Materials and methods

### 2.1 Study design and data

This study will use [RETROSPECTIVE/PROSPECTIVE] digitized tissue specimens from [INSTITUTIONS]. The development, internal test, and external test cohorts will contain [N], [N], and [N] slides, respectively. Include a patient-level flow diagram reporting exclusions, scanner models, objective magnification, micrometres per pixel, tissue sites, stains, diagnoses, and class prevalence. Splits must be performed at patient level; slides or tiles from one patient must not cross splits. If parameters are selected on a validation set, the test set must remain locked until final analysis.

The reference annotation protocol is [DETAILS]. At least [N] pathologists will independently trace [ALL/STRATIFIED] boundary regions at full resolution using [SOFTWARE]. Disagreement will be resolved by [CONSENSUS/ADJUDICATION]. Report annotator expertise, blinding, annotation zoom, and whether coarse input masks were hidden. A subset of [N] slides will be re-annotated to quantify inter- and intra-observer variability.

**Required before submission:** populate a dataset table; provide a class ontology; document whether labels denote mutually exclusive tissue classes or distinct instances; and explain how holes, disconnected components, touching labels, and uncertain regions are represented.

### 2.2 Inputs and coordinate model

Bora receives (i) an OME-TIFF tissue image and (ii) a TIFF or OME-TIFF integer mask. The image and mask must share the same level-0 spatial reference or provide sufficient metadata for an exact resampling transform. Image axes, channel mapping, physical pixel sizes, pyramid level, and affine transform are recorded at ingestion. Nearest-neighbour interpolation is mandatory for categorical masks. Label 0 denotes background by default; every positive integer is retained as a distinct segment or class identifier. The implementation must choose an output integer type from the maximum observed label rather than imposing an 8- or 16-bit label-count ceiling.

OME-TIFF is used to retain microscopy metadata and permit tiled/pyramidal access [8]. The refined mask is written in the image's level-0 coordinate system with lossless compression and matching physical calibration. GeoJSON geometries use pixel coordinates unless a physical-coordinate option is selected; the coordinate reference and origin convention are declared in a top-level metadata object and sidecar provenance record. Because GeoJSON assumes planar coordinates but its formal CRS semantics are geographic, consumers must not interpret pathology pixel coordinates as longitude and latitude [9].

### 2.3 Overview of Bora

Let the image domain be \(\Omega\), the input labels be \(L_0:\Omega\rightarrow\{0,\ldots,K\}\), and the final labels be \(L^*\). Bora operates independently for each observed label \(k\), without allocating arrays proportional to an assumed maximum class count:

1. identify connected components and spatial extents of \(L_0=k\);
2. build a protected core \(C_k\) and outer envelope \(E_k\);
3. run annealed-wand competition along interfaces between foreground labels;
4. apply marker-controlled watershed to the native-resolution image gradient;
5. schedule overlapping image tiles intersecting the editable region \(B_k\);
6. optionally infer a probability map with MedSAM or PathSegmentor;
7. fuse overlapping and competing label predictions; and
8. apply constrained cleanup, pyramid construction, polygonization, and provenance recording.

Only \(B=\bigcup_k B_k\) is editable. Outside \(B\), \(L^*=L_0\); within protected cores, \(L^*(x)=k\) for \(x\in C_k\). These invariants should be enforced in code and tested automatically.

### 2.4 Protected cores and editable boundary bands

For label support \(S_k=\{x:L_0(x)=k\}\), the nominal core and envelope are

\[
C_k = \operatorname{erode}(S_k,r_c)\cup Q_k, \qquad
E_k = \operatorname{dilate}(S_k,r_o),
\]

where \(Q_k\) is an optional trusted seed region, \(r_c\) is the core erosion radius, and \(r_o\) is the outward search radius. The editable band is initially \(B_k=E_k\setminus C_k\). If erosion removes a small component completely, its trusted seeds or original component are retained according to [SMALL-COMPONENT POLICY]. Radii are specified in micrometres and converted to pixels using OME metadata; pixel defaults are allowed only when calibration is absent and must be reported.

For very large masks, distance transforms and morphology may be computed at a controlled downsampling factor before conservative upsampling. Full-resolution morphology is used for final constraint enforcement. White-background exclusion is estimated by [METHOD AND THRESHOLD], but never overrides protected cores or trusted seeds.

### 2.5 Annealed-wand boundary competition

The term *wand* denotes constrained, frontier-driven label growth rather than a Wald statistical test. Bora implements CellPhenotyper schema `cellphenotyper.annealed_wand_boundary.v2`. The default wrapper reproduces the CellPhenotyper working grid: BOX-filter the image by a factor of four, stride-subsample the labels and tissue support, scale the 64-pixel native boundary radius to the working grid, run competition, and project labels back by nearest-neighbour indexing while preserving the full-resolution foreground footprint. Let \(q_k(x)\) denote the mean-field probability of label \(k\) at pixel \(x\). Six appearance channels are constructed from robustly standardized CIE Lab and optical-density RGB features. A per-label prototype is the median feature vector of its non-editable interior. The data term is the mean squared distance to this prototype, clipped at 100. Pairwise Potts weights between four- or eight-connected neighbours are

\[
w_{xy}=d_{xy}\exp[-\beta\lVert f(x)-f(y)\rVert_2^2/C],
\]

where \(d_{xy}=1\) for axial neighbours and \(1/\sqrt{2}\) for diagonal neighbours, \(C\) is the feature-channel count, and \(\beta\) controls edge sensitivity. At temperature \(T\), local probabilities follow

\[
q_k(x)\propto\exp\{- [\lambda_d D_k(x)+\lambda_s\sum_yw_{xy}(1-q_k(y))]/T\}.
\]

Temperature decreases geometrically from 2.0 to 0.05 over 16 iterations. A label may compete at a pixel only when it is the current label or occurs in its local connected frontier. Updates are limited to a dilated internal boundary band; protected pixels, background, and the foreground footprint are invariant. The lowest hard-energy state observed during cooling is retained, guaranteeing that the reported discrete energy does not exceed its initial value. CellPhenotyper uses a 64-pixel full-resolution boundary radius and four-fold working downsampling; Bora exposes both quantities and records changed pixels and energy diagnostics.

### 2.6 Native-resolution watershed

The annealed result is passed to marker-controlled watershed at native resolution. For each label, an eroded protected core provides foreground markers and the complement of its dilated outer envelope provides background markers. The elevation surface is the Sobel magnitude of mean normalized RGB intensity. Watershed is solved within each haloed WSI block with compactness 0.001; only the editable band between core and envelope may change. Outputs from competing labels and overlapping tiles are selected by confidence, while core pixels receive immutable priority. An explicitly labelled accelerated mode may downsample the watershed analysis grid and restore it by nearest-neighbour projection, but all principal experiments use the native-resolution mode.

### 2.7 Border tiling and prompts

For each label, connected border regions are covered by square tiles of [TILE SIZE] pixels with [OVERLAP] pixels of overlap. Tiles containing no editable pixels are skipped. Border-aware scheduling is deterministic, and tile coordinates are saved for reproducibility and restart. Padding uses [REFLECT/CONSTANT] mode and predictions are cropped back to the valid image extent.

For MedSAM, each coarse connected component inside a tile is converted to a bounding-box prompt expanded by [MARGIN] pixels and clipped to the tile. [POSITIVE/NEGATIVE POINT] prompts derived from the protected core and neighbouring labels are used if supported by the selected implementation. RGB conversion, intensity normalization, resizing, and restoration to native resolution follow the published MedSAM preprocessing [4] and are recorded in the run manifest.

For PathSegmentor, the text prompt is obtained from a required label-to-name mapping, e.g. label 3 to “tumour epithelium.” Synonyms and prompt templates are fixed before evaluation. Labels lacking a valid semantic name fall back to MedSAM or are left unchanged; they must not be silently assigned a generic prompt. PathSegmentor's 2026 publication and model version should be frozen by commit/checkpoint hash [6].

### 2.8 Fusion and label preservation

Each backend prediction is masked by \(E_k\) and scored within \(\tilde B_k\). Overlapping tiles for one label are blended using [WEIGHTED WINDOW/MAX/MEAN] fusion. At pixels proposed by multiple labels, the winner is

\[
L^*(x)=\arg\max_k\{p_k(x)+\lambda\,q_k(x)\},
\]

where \(p_k\) is model confidence and \(q_k\) is a coarse-prior term based on distance to \(S_k\). Ties are resolved deterministically by [RULE]. Protected-core assignments override model predictions. A final pass removes components below [AREA], fills holes below [AREA], smooths contours at radius [RADIUS], and retains only components connected to their input support or trusted seeds. Cleanup must not merge two identifiers. Every removed, added, and conflicted pixel count is written to a QC summary.

### 2.9 Raster and vector output

The primary raster output is `[sample]_refined.ome.tif`, containing the level-0 integer label map and eight successively downsampled levels for the current whole-slide configuration. Bora writes tiled pyramidal OME-TIFF with lossless LZW compression and `SIMPLE` nearest-neighbour downsampling so categorical label IDs are not interpolated. The companion GeoJSON is generated before pyramid conversion from the authoritative level-0 raster. Each feature records `label`, CellPhenotyper-compatible `value` and `classification`, and `area_px`. Coordinates follow level-0 pixel edges. Large rasters are polygonized in bounded tiles with a one-pixel halo; geometries are clipped to non-overlapping tile interiors, repaired when invalid, and optionally simplified while preserving topology. The copied CellPhenotyper vectorizer remains the reference for future provenance-linked vector output. Rasterizing the GeoJSON must reproduce the mask within a prespecified tolerance of [TOLERANCE] pixels.

### 2.10 Comparators and ablations

The prespecified comparison is:

- input tile mask (no refinement);
- morphology-only cleanup;
- annealed-wand competition alone;
- native-resolution watershed alone;
- annealed-wand followed by native-resolution watershed;
- direct full-region MedSAM;
- Bora–MedSAM;
- direct PathSegmentor, where a valid class prompt exists;
- Bora–PathSegmentor; and
- [ADDITIONAL SPECIALIST BASELINE].

Ablations remove annealed competition, native watershed, protected-core enforcement, overlap, confidence-aware fusion, background exclusion, and topology cleanup. Further ablations vary annealing temperature, Potts smoothness, edge sensitivity, connectivity, boundary radius, watershed compactness, border-band width, and tile overlap. The same preprocessing, hardware, and reference annotations are used for all eligible methods.

### 2.11 Endpoints and statistical analysis

The two primary metrics are Boundary IoU at a physical tolerance of [D] µm [1] and normalized surface Dice at [D] µm. Secondary metrics are per-class Dice, Jaccard index, average symmetric surface distance, HD95, false-positive and false-negative boundary displacement, component count error, label disappearance rate, changed-core pixel count, raster–vector round-trip error, wall time, tiles processed, peak RAM, and peak accelerator memory. Metrics are computed per slide and per class; absent-class handling is specified before analysis.

Primary comparisons use paired [WILCOXON/SIGN-FLIP/PERMUTATION] tests at slide level with effect sizes and 95% bootstrap confidence intervals clustered by patient. Multiplicity across two primary endpoints and [N] methods is controlled using [HOLM PROCEDURE]. Scanner, institution, stain, tissue class, baseline boundary quality, segment size, and label count define prespecified subgroup analyses. Report distributions and individual-slide points, not only means. A failure is any crash, empty output for a present label, core-invariant violation, invalid geometry, or runtime above [LIMIT]; failures remain in the intention-to-process analysis using the prespecified worst-score rule.

### 2.12 Implementation and reproducibility

Bora is implemented in [PYTHON VERSION] using [LIBRARIES AND VERSIONS]. Experiments will freeze the Bora commit, CellPhenotyper source commit, model checkpoint checksum, and container digest. Random seeds and deterministic settings are [DETAILS]. Hardware is [CPU, RAM, GPU]. Source code will be released at `https://github.com/tkcaccia/Bora` under [LICENSE], and an archival DOI will be created at [ZENODO DOI]. Data and annotations will be shared at [REPOSITORY] subject to [ACCESS CONDITIONS]. A synthetic, redistributable example should be included for continuous integration.

## 3. Results

### 3.1 Cohort and processing characteristics

**Placeholder—do not infer values.** Report the participant/slide flow, image dimensions, physical resolution, tissue classes, segment counts, editable-band fraction, and reference-annotation coverage. Table 1 should distinguish development, internal test, and external test data.

### 3.2 Primary boundary-refinement performance

**Placeholder.** Report Boundary IoU and normalized surface Dice for every comparator with paired differences, confidence intervals, adjusted *p* values, and failure counts. State explicitly whether both primary hypotheses were supported. Figure 2 should show representative successes and failures selected by a prespecified rule rather than visual preference.

### 3.3 Region overlap, topology, and label safety

**Placeholder.** Report Dice/IoU alongside component errors, label loss, core-invariant violations, and changes in segment adjacency. A boundary improvement should not be described as clinically meaningful unless the magnitude is related to annotator variability or a downstream measurement.

### 3.4 Generalization and subgroup analyses

**Placeholder.** Report external-site results and stratification by scanner, stain, tissue type, segment size, baseline error, and number of labels. Clearly identify exploratory analyses.

### 3.5 Ablation and computational performance

**Placeholder.** Quantify the incremental value of annealed-wand competition, native watershed, core protection, overlap, and each promptable backend. Report end-to-end runtime, throughput, peak memory, processed-area fraction, and scaling with image area, boundary length, and label count.

## 4. Discussion

This study is designed to determine whether pixel-level tissue boundaries can be recovered from coarse tile labels by restricting promptable-model inference to a narrow, explicitly governed search region. Bora does not replace the upstream semantic classifier: it assumes that tile labels are broadly correct and uses them to define identity, prompts, and immutable interiors. This distinction is important. Unconstrained re-segmentation may correct boundaries but can also erase rare labels, exchange adjacent classes, or hallucinate tissue in slide background. The protected-core invariant and auditable competition rule are intended to make such failures detectable.

The proposed evaluation emphasizes boundary metrics because region overlap is relatively insensitive to contour errors in large tissue compartments [1]. Dice remains useful for comparability but is insufficient as the sole endpoint. Conversely, a sharp boundary score alone cannot establish semantic correctness. The benchmark therefore combines boundary fidelity, class retention, topology, vector consistency, failure frequency, and computational resource use.

MedSAM offers spatially prompted, medical-domain segmentation [4], whereas PathSegmentor introduces pathology-specific text prompting [6]. Their comparison tests whether explicit class semantics improve boundary selection when neighbouring tissues have similar morphology. Any PathSegmentor advantage must be interpreted with awareness of possible overlap between its training sources and evaluation datasets; public-dataset provenance and leakage checks are required. Results should also separate labels with natural semantic names from arbitrary clusters, for which text prompting is not well defined.

Anticipated limitations include dependence on the upstream mask, imperfect calibration metadata, ambiguity at mixed or transitional tissue interfaces, model sensitivity to stain and scanner variation, and potential fragmentation during tiled raster-to-vector conversion. Boundary-only refinement cannot recover a wholly missed region far outside the outer envelope and may preserve an incorrect interior by design. Annealed-wand competition changes label identity only where foreground labels already meet and therefore cannot recover missing foreground. Native watershed is sensitive to weak gradients and stain artefacts. All radii should be evaluated in physical rather than purely pixel units.

If validated, Bora could provide a practical bridge between inexpensive tile classification and geometry-aware downstream analysis. Clinical or biological utility, however, requires a separate task-specific study; improved image metrics alone do not demonstrate diagnostic benefit.

## 5. Conclusion

Bora is a proposed boundary-constrained framework for converting multi-label tile masks into full-resolution tissue segmentations and matched vector geometries. Its central design is to spend computation at uncertain borders while preserving confident interiors and semantic identifiers. Conclusions about accuracy, generalization, or efficiency are deferred until the prespecified internal and external benchmarks are completed.

## Declarations

### Ethics approval and consent to participate

[INSTITUTIONAL REVIEW BOARD NAME, APPROVAL NUMBER, DATE, CONSENT OR WAIVER, AND DATA-DEIDENTIFICATION DETAILS]. For public datasets, list their licenses and original ethics statements.

### Consent for publication

[NOT APPLICABLE / DETAILS].

### Data availability

[DATASETS, ACCESSION NUMBERS, DATA USE RESTRICTIONS, AND REQUEST PROCEDURE].

### Code availability

Code is planned for release at <https://github.com/tkcaccia/Bora>. Before submission, provide the release tag, commit hash, software license, model-weight licenses, container digest, example data, and archival DOI.

### Competing interests

[AUTHORS MUST DECLARE ALL FINANCIAL AND NON-FINANCIAL COMPETING INTERESTS].

### Funding

[FUNDER, GRANT NUMBER]. The funder had [ROLE/NO ROLE] in study design, data collection, analysis, interpretation, manuscript preparation, and the decision to submit.

### Author contributions (CRediT)

[AUTHOR]: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing—original draft, Writing—review & editing, Visualization, Supervision, Project administration, Funding acquisition. [ASSIGN EACH ROLE ACCURATELY].

### Declaration of generative AI use

[COMPLETE ACCORDING TO THE JOURNAL POLICY IN FORCE AT SUBMISSION. State the tool, purpose, and human verification; authors retain responsibility for the work.]

## Proposed figures and tables

1. **Figure 1:** Bora workflow: coarse labels → annealed-wand competition → native watershed/protected cores → optional prompts → constrained fusion → pyramidal OME-TIFF and GeoJSON.
2. **Figure 2:** Prespecified representative internal and external cases with image, coarse mask, reference, comparator outputs, and signed boundary-error maps.
3. **Figure 3:** Paired primary endpoint distributions and per-class effects with patient-clustered confidence intervals.
4. **Figure 4:** Accuracy–efficiency trade-off and scaling with boundary length and label count.
5. **Figure 5:** Failure taxonomy and uncertainty/QC signals.
6. **Table 1:** Cohort and acquisition characteristics.
7. **Table 2:** Primary and secondary segmentation endpoints.
8. **Table 3:** Ablations, runtime, memory, and failure rates.

## References

1. Cheng B, Girshick R, Dollár P, Berg AC, Kirillov A. Boundary IoU: improving object-centric image segmentation evaluation. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2021:15334–15342. <https://doi.org/10.1109/CVPR46437.2021.01508>.
2. Kirillov A, Mintun E, Ravi N, et al. Segment Anything. *Proceedings of the IEEE/CVF International Conference on Computer Vision*. 2023:3992–4003. <https://doi.org/10.1109/ICCV51070.2023.00371>.
3. Mazurowski MA, Dong H, Gu H, Yang J, Konz N, Zhang Y. Segment Anything Model for medical image analysis: an experimental study. *Medical Image Analysis*. 2023;89:102918. <https://doi.org/10.1016/j.media.2023.102918>.
4. Ma J, He Y, Li F, et al. Segment anything in medical images. *Nature Communications*. 2024;15:654. <https://doi.org/10.1038/s41467-024-44824-z>.
5. Archit A, Nair S, Khalid N, et al. PathoSAM: a foundation model for interactive and automatic segmentation of histopathology images. *Proceedings of Machine Learning Research*. [VERIFY FINAL VOLUME, PAGES, YEAR, AND DOI; current accessible manuscript: <https://openreview.net/pdf?id=EWGV97ESaP>].
6. Chen Z, Hou J, Lin L, et al. Segment anything in pathology images with natural language. *Nature Computational Science*. 2026. <https://doi.org/10.1038/s43588-026-01042-5>.
7. Caccia T, et al. CellPhenotyper [computer software]. GitHub. [ADD VERSION, ARCHIVE DOI, CONTRIBUTORS, AND YEAR]. <https://github.com/tkcaccia/CellPhenotyper>.
8. Goldberg IG, Allan C, Burel J-M, et al. The Open Microscopy Environment (OME) Data Model and XML file: open tools for informatics and quantitative analysis in biological imaging. *Genome Biology*. 2005;6:R47. <https://doi.org/10.1186/gb-2005-6-5-r47>.
9. Butler H, Daly M, Doyle A, Gillies S, Hagen S, Schaub T. The GeoJSON Format. RFC 7946. Internet Engineering Task Force; 2016. <https://doi.org/10.17487/RFC7946>.

## Pre-submission completion checklist

- Freeze and archive the annealed-wand v2 and native-watershed implementation commit.
- Freeze the research question, endpoints, tolerance distances, and statistical plan before testing.
- Populate all bracketed fields; remove protocol language once results exist.
- Add a complete dataset/provenance table and a leakage assessment.
- Run all baselines and ablations on locked patient-level splits.
- Report confidence intervals, adjusted tests, failures, and compute consumption.
- Verify every reference against the final publisher record.
- Release an auditable code version and model/checkpoint manifests.
- Confirm licences for CellPhenotyper-derived code, MedSAM, PathSegmentor, and model weights.
- Follow the *Medical Image Analysis* author guide current at submission; Elsevier describes typical original articles as approximately 3,000–6,000 words with 3–5 figures and 30–50 references, but the journal’s live requirements take precedence.
