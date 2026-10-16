# Boundary refinement improves tissue extent but can erode semantic fidelity in gigapixel pathology maps

**Manuscript type:** Article
**Target journal:** *Nature Communications*
**Status:** complete technical benchmark draft; author, ethics and release metadata remain to be supplied

**Authors:** [AUTHOR 1], [AUTHOR 2], [AUTHOR 3]  
**Affiliations:** [AFFILIATIONS]  
**Corresponding author:** [NAME, POSTAL ADDRESS, EMAIL]  
**ORCID identifiers:** [ORCIDS]

## Abstract

Tile-wise tissue classification scales to gigapixel histology but replaces irregular biological interfaces with a block grid. We developed **Bora**, a boundary-constrained framework that expands categorical tile grids to native resolution and refines only their uncertain borders while preserving label cores. Bora combines deterministic annealed multiclass competition with marker-controlled watershed, writes pyramidal OME-TIFF and GeoJSON outputs, and supports MedSAM or PathSegmentor as optional refinement engines. We evaluated the default chain in a controlled reconstruction benchmark spanning six human Visium HD tissue images, 11.46 billion pixels and 37 foreground classes. Pixel-level masks rasterized directly from the original GeoJSON annotations served as references; 56-pixel majority-vote grids were the only label input to Bora. Refinement improved binary foreground Dice in all six tissues (mean change, +0.00042) but reduced pixel accuracy in all six (mean, -0.00153), macro Dice in five of six (mean, -0.00740) and macro intersection-over-union in five of six (mean, -0.01160). Rare classes were disproportionately affected. These results reveal a central limitation of image-driven post-processing: histological edges can recover tissue extent yet need not coincide with semantic boundaries inherited from molecular or regional annotations. Bora therefore provides both a scalable implementation and an auditable test bed for conservative refinement. The findings motivate separate treatment of tissue-background and inter-class interfaces, confidence-gated updates and class-adaptive displacement constraints.

**Keywords:** computational pathology; semantic segmentation; boundary refinement; whole-slide imaging; spatial transcriptomics; OME-TIFF; GeoJSON

## 1. Introduction

Semantic segmentation is central to quantitative pathology because regional measurements depend on where tissue compartments begin and end. Whole-slide images are, however, commonly analysed as grids of tiles: a classifier assigns one class to each tile and the class map is projected back into slide coordinates. This design reduces computational and annotation costs but quantizes boundaries at the tile stride. The resulting error is concentrated along interfaces, precisely where invasion fronts, tumour–stroma contacts, gland contours, and tissue exclusion zones are measured. Region-overlap metrics can obscure such errors for large objects, motivating explicit boundary-sensitive evaluation [1].

Promptable foundation models provide a potential route from coarse masks to local, image-supported contours. The Segment Anything Model (SAM) introduced a promptable segmentation architecture trained at large scale [2]. Direct use on medical images is not uniformly reliable, particularly for weak or ambiguous boundaries [3]. MedSAM adapts SAM to medical imagery using more than one million image–mask pairs across multiple modalities and disease contexts [4]. Pathology-specific approaches have subsequently targeted the appearance and scale distribution of histology. PathoSAM focuses on interactive and automatic nuclei segmentation [5], while PathSegmentor uses natural-language prompts to delineate histological structures across spatial scales [6]. These models are promising refinement engines, yet applying them indiscriminately across a gigapixel image is expensive and risks altering already reliable regions.

We hypothesize that coarse tissue labels can serve as spatial priors and that model inference need only occur near their boundaries. This leads to a constrained formulation: preserve an eroded core of every input segment, construct an editable band around its boundary, process only image tiles intersecting that band, and arbitrate conflicts among label-specific predictions. The approach is adapted from the MedSAM refinement stage of CellPhenotyper [7], where large masks are handled by downsampled morphology and full-resolution processing is restricted to overlapping cluster-border tiles. Bora generalizes this step into a standalone, format-aware tool that accepts any number of mask labels and emits both raster and vector products.

The contributions of this work are:

1. a scalable coarse-to-fine formulation that limits foundation-model inference to label boundaries;
2. a protected-core and constrained-fusion mechanism designed to preserve semantic identity across an arbitrary number of input labels;
3. a reproducible sequence of annealed-wand multiclass competition, native-resolution watershed, and optional promptable-model refinement; and
4. a controlled, whole-slide benchmark that exposes a reproducible trade-off between foreground recovery and multiclass semantic preservation.

## 2. Results

### 2.1 A controlled reconstruction benchmark across 11.46 billion pixels

We assembled six human Visium HD tissue images representing pancreas, tissue-array, lymph-node, tonsil, colon-cancer and lung-cancer specimens. Images ranged from 1.22 to 3.53 gigapixels and together comprised 11.46 billion native-resolution pixels. For each specimen, the original class-labelled GeoJSON was rasterized at native resolution to form the reference. Majority voting in non-overlapping 56 × 56-pixel tiles then produced the deliberately quantized categorical input. This design isolates the information lost through tiling: the reference and coarse masks share an ontology and coordinate system, and no registration or class-remapping error is introduced.

The benchmark is a reconstruction experiment, not an independent clinical-validation cohort. The reference geometries were not newly traced by pathologists and the six images should not be interpreted as independent patient-level evidence. Their role is to stress-test whether image-driven refinement can reverse a known tiling transformation at whole-slide scale.

### 2.2 Refinement consistently improves foreground extent but not semantic overlap

All six specimens completed the full reference comparison. The unrefined majority-vote masks already achieved pixel accuracies from 0.9721 to 0.9937. Bora increased binary foreground Dice in every specimen, with changes from +0.00005 to +0.00089 (mean, +0.00042). This consistent direction indicates that image-supported refinement better aligned the union of all tissue classes with the foreground footprint.

The same updates were detrimental when label identity was considered. Pixel accuracy decreased in every specimen (mean change, -0.00153), as did micro Dice (mean, -0.00296). Macro Dice decreased in five of six specimens (mean, -0.00740), and macro IoU decreased in five of six (mean, -0.01160). Colon cancer showed the largest loss in macro IoU (-0.01864), closely followed by lung cancer (-0.01855), while pancreas was the only specimen with a small macro-IoU improvement (+0.00059). Thus, an improvement in the binary tissue outline did not imply improved multiclass segmentation.

### 2.3 Small semantic regions are vulnerable to boundary competition

The largest class-specific errors occurred in relatively small regions. In colon cancer, a class supported by 5.33 million reference pixels had an IoU of 0.6304 after refinement; in lung cancer, a 17.76-million-pixel class reached 0.6496; in tonsil, a 2.64-million-pixel class reached 0.7255; and in lymph node, a 6.56-million-pixel class reached 0.6963. These failures were dominated by false-positive expansion into adjacent classes rather than complete disappearance. Large foreground compartments, by contrast, retained high overlap and dominated the binary foreground score.

This size dependence follows from symmetric boundary motion: a displacement of fixed width represents a much larger fraction of a small region than of a large one. It also exposes a limitation in the default confidence arbitration, which does not calibrate its acceptance threshold by class prevalence or component geometry.

### 2.4 Whole-slide execution is feasible but computationally intensive

The default native-resolution chain processed all six images using memory-bounded 2,048-pixel blocks. Depending on image area and boundary density, 174–632 blocks per specimen required refinement, while homogeneous blocks were copied unchanged. Refinement time ranged from 5,314 to 29,169 seconds per image and totalled 64,421 seconds (17.9 h). The slowest tissue-array image was both the largest specimen and the one with the largest editable workload. GeoJSON conversion required 17–73 seconds per specimen and emitted one top-level feature per foreground label while retaining disconnected polygon parts.

The tiling-mask C++ stage was substantially faster than refinement; for the 1.53-gigapixel pancreas image it rasterized the reference and generated the tile grid in 7.56 seconds. These measurements show that bounded-memory processing makes gigapixel execution possible, but the present watershed implementation remains the dominant computational cost.

### 2.5 The observed failure mode motivates conservative refinement

The directionally opposed binary and multiclass results indicate that the default objective is misaligned with the semantic benchmark. Histological gradients provide useful evidence at the boundary between tissue and glass background. Interfaces between transcriptomic clusters or regional annotations, however, may have weak, displaced or absent counterparts in RGB morphology. Allowing the same image-driven rule to move both interface types therefore improves foreground extent while exchanging labels internally.

These findings define three immediate design requirements: tissue-background and inter-class boundaries should be handled separately; a proposed change should be accepted only when its evidence exceeds a calibrated confidence threshold; and maximum displacement and area change should depend on component size. These are testable algorithmic consequences of the benchmark rather than post hoc explanations of individual images.

## 3. Methods

### 3.1 Study design and data

We performed a controlled degradation-and-reconstruction study using six human tissue images and their matched class-labelled GeoJSON files. Each GeoJSON was rasterized by `tiling-mask-cpp` in image-pixel coordinates. The resulting native mask was retained unchanged as the reference. A categorical tile grid was produced by majority vote over 56 × 56-pixel regions, with background participating in the vote and deterministic lowest-ID tie breaking. The grid was expanded back to native dimensions by nearest-neighbour repetition and supplied to Bora together with the original image. The reference mask was never supplied to the refinement algorithm.

The tissues were pancreas (33,007 × 46,391 pixels), tissue array (79,157 × 44,649), lymph node (30,966 × 39,421), tonsil (29,473 × 41,910), colon cancer (53,020 × 43,603) and lung cancer (54,639 × 29,814). Together they contained 11.46 billion pixels. Labels were mutually exclusive categorical regions with zero reserved for background. This technical benchmark uses existing computational annotations and does not establish diagnostic performance or inter-observer agreement.

### 3.2 Inputs and coordinate model

Bora receives (i) an OME-TIFF tissue image and (ii) a TIFF or OME-TIFF integer mask. The image and mask must share the same level-0 spatial reference or provide sufficient metadata for an exact resampling transform. Image axes, channel mapping, physical pixel sizes, pyramid level, and affine transform are recorded at ingestion. Nearest-neighbour interpolation is mandatory for categorical masks. Label 0 denotes background by default; every positive integer is retained as a distinct segment or class identifier. The implementation must choose an output integer type from the maximum observed label rather than imposing an 8- or 16-bit label-count ceiling.

OME-TIFF is used to retain microscopy metadata and permit tiled/pyramidal access [8]. The refined mask is written in the image's level-0 coordinate system with lossless compression and matching physical calibration. GeoJSON geometries use pixel coordinates unless a physical-coordinate option is selected; the coordinate reference and origin convention are declared in a top-level metadata object and sidecar provenance record. Because GeoJSON assumes planar coordinates but its formal CRS semantics are geographic, consumers must not interpret pathology pixel coordinates as longitude and latitude [9].

### 3.3 Overview of Bora

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

### 3.4 Protected cores and editable boundary bands

For label support \(S_k=\{x:L_0(x)=k\}\), the nominal core and envelope are

\[
C_k = \operatorname{erode}(S_k,r_c)\cup Q_k, \qquad
E_k = \operatorname{dilate}(S_k,r_o),
\]

where \(Q_k\) is an optional trusted seed region, \(r_c\) is the core erosion radius, and \(r_o\) is the outward search radius. The editable band is initially \(B_k=E_k\setminus C_k\). If erosion removes a component completely, its original support becomes the core so that the label cannot disappear solely because it is smaller than the erosion diameter. The present benchmark used pixel radii because harmonized physical calibration was not available; calibrated studies should specify these quantities in micrometres.

For very large masks, Bora streams haloed blocks and performs final constraint enforcement at full resolution. Homogeneous blocks are copied without refinement. The halo equals the largest active morphology or competition radius plus four pixels, preventing ordinary boundary operations from being clipped at block edges.

### 3.5 Annealed-wand boundary competition

The term *wand* denotes constrained, frontier-driven label growth rather than a Wald statistical test. Bora implements CellPhenotyper schema `cellphenotyper.annealed_wand_boundary.v2`. The default wrapper reproduces the CellPhenotyper working grid: BOX-filter the image by a factor of four, stride-subsample the labels and tissue support, scale the 64-pixel native boundary radius to the working grid, run competition, and project labels back by nearest-neighbour indexing while preserving the full-resolution foreground footprint. Let \(q_k(x)\) denote the mean-field probability of label \(k\) at pixel \(x\). Six appearance channels are constructed from robustly standardized CIE Lab and optical-density RGB features. A per-label prototype is the median feature vector of its non-editable interior. The data term is the mean squared distance to this prototype, clipped at 100. Pairwise Potts weights between four- or eight-connected neighbours are

\[
w_{xy}=d_{xy}\exp[-\beta\lVert f(x)-f(y)\rVert_2^2/C],
\]

where \(d_{xy}=1\) for axial neighbours and \(1/\sqrt{2}\) for diagonal neighbours, \(C\) is the feature-channel count, and \(\beta\) controls edge sensitivity. At temperature \(T\), local probabilities follow

\[
q_k(x)\propto\exp\{- [\lambda_d D_k(x)+\lambda_s\sum_yw_{xy}(1-q_k(y))]/T\}.
\]

Temperature decreases geometrically from 2.0 to 0.05 over 16 iterations. A label may compete at a pixel only when it is the current label or occurs in its local connected frontier. Updates are limited to a dilated internal boundary band; protected pixels, background, and the foreground footprint are invariant. The lowest hard-energy state observed during cooling is retained, guaranteeing that the reported discrete energy does not exceed its initial value. CellPhenotyper uses a 64-pixel full-resolution boundary radius and four-fold working downsampling; Bora exposes both quantities and records changed pixels and energy diagnostics.

### 3.6 Native-resolution watershed

The annealed result is passed to marker-controlled watershed at native resolution. For each label, an eroded protected core provides foreground markers and the complement of its dilated outer envelope provides background markers. The elevation surface is the Sobel magnitude of mean normalized RGB intensity. Watershed is solved within each haloed WSI block with compactness 0.001; only the editable band between core and envelope may change. Outputs from competing labels and overlapping tiles are selected by confidence, while core pixels receive immutable priority. An explicitly labelled accelerated mode may downsample the watershed analysis grid and restore it by nearest-neighbour projection, but all principal experiments use the native-resolution mode.

### 3.7 Border tiling and prompts

For each label, connected border regions are covered by 2,048-pixel square tiles with 256 pixels of overlap. Tiles containing no editable pixels are skipped. Border-aware scheduling is deterministic. At image margins, arrays are cropped to their valid extent rather than padded.

For MedSAM, each coarse connected component inside a tile is converted to a bounding-box prompt expanded by four pixels and clipped to the tile. RGB conversion, intensity normalization, resizing, and restoration to native resolution follow the published MedSAM preprocessing [4] and are recorded in the run manifest. MedSAM was not activated in the benchmark reported here.

For PathSegmentor, the text prompt is obtained from a required label-to-name mapping, e.g. label 3 to “tumour epithelium.” Synonyms and prompt templates are fixed before evaluation. Labels lacking a valid semantic name fall back to MedSAM or are left unchanged; they must not be silently assigned a generic prompt. PathSegmentor's 2026 publication and model version should be frozen by commit/checkpoint hash [6].

### 3.8 Fusion and label preservation

Each backend prediction is masked by \(E_k\) and scored within \(\tilde B_k\). Overlapping tiles retain the proposal with maximum confidence. At pixels proposed by multiple labels, the winner is

\[
L^*(x)=\arg\max_k p_k(x),
\]

where \(p_k\) is backend confidence after masking by the label-specific envelope. Strict greater-than comparison leaves the earlier deterministic label assignment unchanged on exact confidence ties. Protected-core assignments override model predictions. The benchmark used a 32-pixel minimum-area threshold and binary closing with a two-pixel disk. Only components connected to their input support are retained, and cleanup never merges two identifiers.

### 3.9 Raster and vector output

The primary raster output is `[sample]_refined.ome.tif`, containing the level-0 integer label map and successively downsampled levels. Bora writes tiled pyramidal OME-TIFF with lossless LZW compression and `SIMPLE` nearest-neighbour downsampling so categorical label IDs are not interpolated. The companion GeoJSON is generated after pyramid conversion. The retained CellPhenotyper converter (Git blob `fd2dd40`) is available as a reference backend. Bora's default four-CPU backend polygonizes a selected categorical pyramid level, filters small components, simplifies the complete polygon coverage jointly so shared class boundaries remain coincident, dissolves components per label in parallel, and maps coordinates to level 0. It emits one top-level feature per observed foreground label while retaining disconnected polygon parts and holes. Prespecified accurate, balanced and fast profiles control pyramid level, area filtering and simplification.

### 3.10 Comparators and ablations

The present experiment compares the expanded majority-vote baseline with the default Bora chain (annealed-wand competition followed by native-resolution watershed). The software also supports the following prespecified future ablations and model comparisons:

- input tile mask (no refinement);
- morphology-only cleanup;
- annealed-wand competition alone;
- native-resolution watershed alone;
- annealed-wand followed by native-resolution watershed;
- direct full-region MedSAM;
- Bora–MedSAM;
- direct PathSegmentor, where a valid class prompt exists;
- Bora–PathSegmentor.

Ablations remove annealed competition, native watershed, protected-core enforcement, overlap, confidence-aware fusion, background exclusion, and topology cleanup. Further ablations vary annealing temperature, Potts smoothness, edge sensitivity, connectivity, boundary radius, watershed compactness, border-band width, and tile overlap. The same preprocessing, hardware, and reference annotations are used for all eligible methods.

### 3.11 Endpoints and statistical analysis

We computed pixel accuracy, binary foreground Dice and IoU, and per-class Dice and IoU by streaming aligned raster windows. Macro metrics assign equal weight to every foreground label; micro metrics pool class-specific true-positive, false-positive and false-negative counts. Background is included in pixel accuracy but excluded from multiclass Dice and IoU. Changes are reported as Bora minus the unrefined expanded tile mask.

Because this technical series contains six images without patient-level sampling metadata, we report individual specimens, ranges and descriptive means rather than inferential *P* values. Confidence intervals or claims of population generalization would be inappropriate. A subsequent locked evaluation should use patient-level splits, boundary IoU and normalized surface Dice at a prespecified physical tolerance, paired effect sizes and patient-clustered bootstrap confidence intervals.

### 3.12 Implementation and reproducibility

Bora is implemented in Python 3.10 using NumPy, SciPy, scikit-image, tifffile, Rasterio, Shapely, Pillow and pyvips. Reference rasterization and tile aggregation used the OpenMP-enabled C++ implementation of `tiling-mask`. The default run used four output workers, 2,048-pixel streaming blocks, a 64-pixel annealed boundary radius evaluated at four-fold downsampling, 16-pixel core erosion, 24-pixel outer dilation and native-resolution watershed. Categorical pyramids used nearest-neighbour (`SIMPLE`) downsampling and lossless LZW compression. Source code is maintained at <https://github.com/tkcaccia/Bora>; a versioned release, licence and archival DOI must be supplied before submission.

## 4. Discussion

This study asked whether pixel-level tissue boundaries can be recovered from coarse tile labels by restricting image-driven inference to a narrow, explicitly governed search region. The answer depends on what constitutes a boundary. Bora improved the union of foreground classes in every fully scored image, yet usually reduced multiclass overlap. This divergence is not a contradiction: the visible tissue edge and the semantic interface between two annotated regions are different estimation problems.

Bora does not replace the upstream semantic classifier. It assumes that tile labels are broadly correct and uses them to define identity and protected interiors. The present results show that these safeguards are necessary but not sufficient. A protected core prevents wholesale class erasure, while millions of individually plausible border updates can still produce a measurable loss in class IoU. The tissue-array result is particularly informative: foreground Dice increased while macro IoU fell by more than 0.016, demonstrating why a binary metric alone is inadequate for multiclass refinement.

The evaluation emphasizes both foreground and class-aware metrics because region overlap is relatively insensitive to contour errors in large tissue compartments [1]. Dice remains useful for comparability but is insufficient as the sole endpoint. Conversely, a sharp boundary score alone cannot establish semantic correctness. Future validation should combine boundary fidelity, class retention, topology, vector consistency, failure frequency and computational resource use.

MedSAM offers spatially prompted, medical-domain segmentation [4], whereas PathSegmentor introduces pathology-specific text prompting [6]. Neither model was included in the reported default-chain benchmark. Their future comparison should test whether explicit class semantics improve boundary selection when neighbouring tissues have similar morphology. Any advantage must be interpreted with awareness of overlap between training sources and evaluation datasets. Labels with natural semantic names must also be separated from arbitrary clusters, for which text prompting is not well defined.

Important limitations constrain interpretation. First, the reference masks were computationally rasterized from existing GeoJSON rather than independently traced by blinded pathologists. Second, the six images do not constitute a patient-level cohort and cannot support clinical generalization or formal population inference. Third, the comparison isolates a 56-pixel tiling transformation but does not model upstream classification errors. Fourth, the current evaluation lacks boundary-distance metrics, topology analysis and stage-wise ablation, so it cannot attribute error separately to annealed competition and watershed. Finally, refinement depends on the upstream mask and cannot recover a wholly missed region outside its search envelope. Native watershed is also sensitive to weak gradients and stain artefacts.

The most direct improvement is to partition the editable set into external and internal interfaces. Image-driven watershed can be retained for tissue-background borders, where RGB evidence is usually meaningful. Internal class interfaces should use a conservative geometric prior unless an image model produces calibrated evidence above a threshold. A graph formulation could combine a strong unary penalty for changing the tile label, a weaker image-edge term, class-specific area constraints and a maximum displacement of substantially less than half the 56-pixel tile width. Rare labels should receive narrower bands and higher overwrite thresholds. These changes should be selected on development tissues and evaluated once on held-out specimens to avoid optimizing against the test GeoJSON.

Bora already provides a practical engineering bridge between inexpensive tile classification and geometry-aware downstream analysis. Its scientific value is equally diagnostic: it makes the failure mode measurable rather than hiding it behind a foreground score. Clinical or biological utility requires a separate task-specific study; improved image metrics alone would not demonstrate diagnostic benefit.

## 5. Conclusion

Bora converts multiclass tile grids into full-resolution, pyramidal raster masks and matched vector geometries using bounded-memory boundary processing. Across a controlled gigapixel benchmark, the default chain consistently improved binary tissue extent but usually reduced multiclass semantic overlap, with the largest relative errors in small regions. The result establishes that foreground recovery and semantic preservation must be optimized separately. Conservative, confidence-gated and class-adaptive refinement is therefore not merely an implementation preference but the principal requirement for the next generation of coarse-to-fine pathology segmentation.

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

1. **Figure 1:** Bora workflow: GeoJSON-derived reference → majority-vote tile mask → annealed competition → constrained watershed → pyramidal OME-TIFF and GeoJSON.
2. **Figure 2:** Paired changes in foreground Dice, macro IoU and pixel accuracy for all specimens, with zero-change reference lines.
3. **Figure 3:** Representative success and failure cases showing RGB image, coarse grid, reference, Bora output and signed class-error maps.
4. **Figure 4:** Class-wise IoU change against reference class area, highlighting vulnerability of small regions.
5. **Table 1:** Image dimensions, class counts, processed blocks and runtime.
6. **Table 2:** Baseline, Bora and paired changes for all reported overlap metrics.
7. **Extended Data Table 1:** Stage-wise ablations and conservative-refinement parameter sweep.

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
- Add complete dataset provenance, licences and a training-data leakage assessment.
- Run all baselines and ablations on locked patient-level splits.
- Report confidence intervals, adjusted tests, failures, and compute consumption.
- Verify every reference against the final publisher record.
- Release an auditable code version and model/checkpoint manifests.
- Confirm licences for CellPhenotyper-derived code, MedSAM, PathSegmentor, and model weights.
- Complete the Nature Portfolio reporting summary and follow the live target-journal requirements at submission.
