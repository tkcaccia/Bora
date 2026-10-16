# Bora for image guided refinement of tile level tissue masks in digital pathology

**Working article draft for Medical Image Analysis, 16 October 2026.** The study contains selected regions from three datasets; the pilot and descriptive extensions are not prevalence estimates for the full collections.

**Authors, in requested order:** Martin Ocharo¹,³; Dalia Ahmed¹,²; Moussa Kassim Mohammed¹,³; Awonke Fuzile³; Aluwani Nemutavhani³; Enya Steyn³; Dupe Ojo¹; Silvano Piazza⁴; Dinesh Gupta⁵; Stefano Cacciatore¹.

¹ Bioinformatics Unit, International Centre for Genetic Engineering and Biotechnology, Cape Town, South Africa. ² Histopathology and Cytology Department, Faculty of Medical Laboratory Science, Omdurman Ahlia University, Sudan. ³ University of Cape Town, Cape Town, South Africa. ⁴ Computational Biology Group, International Centre for Genetic Engineering and Biotechnology, Trieste, Italy. ⁵ Translational Bioinformatics Group, International Centre for Genetic Engineering and Biotechnology, New Delhi, India.

**Affiliation verification:** The ICGEB directory supports affiliations ¹, ⁴ and ⁵. A 2026 publication supports Dalia Ahmed's two affiliations. Enya Steyn has a University of Cape Town record. University of Cape Town affiliations for Martin Ocharo and Moussa Kassim Mohammed were supplied by the author and should be confirmed before submission; the entries for Awonke Fuzile and Aluwani Nemutavhani follow the requested fallback and also need confirmation. The spelling of Moussa Kassim Mohammed needs confirmation against the ICGEB directory, which lists “Moussa Kassim.” Corresponding author, emails, ORCIDs and CRediT roles require author confirmation.

## Abstract

Tile-derived tissue maps have grid-shaped boundaries that obscure pathological compartments. Bora refines a supplied categorical tile mask with local haematoxylin and eosin colour evidence, protects reliable class interiors, and exports native-coordinate raster and GeoJSON regions. We analyzed 63 distinct annotated regions of interest (ROIs) from BCSS and TIGER breast tissue and WSSS4LUAD lung tissue. A three-ROI pilot crossed tile widths of 56, 90 and 216 pixels with requested border error rates of 0%, 1% and 5%. Macro intersection over union (IoU) improved in 24 of 27 pilot conditions; class-pair internal boundary F1 within 8 pixels improved in 21 of 24 conditions with defined scores. At zero error, 35 additional WSSS4LUAD ROIs showed median macro IoU changes of −0.0031, +0.0040 and +0.0029 by ascending tile width, indicating mixed region-overlap effects despite generally better boundary proximity. In ten further TIGER source groups, median macro IoU gains were +0.0235, +0.0284 and +0.0254; eight further BCSS source groups improved both macro IoU and class-pair F1 at every tested width. All 54 GeoJSON outputs in the two new breast extensions rasterized exactly to their refined masks and repeated conversion was byte identical. A separate tissue-region analysis used three native CAMELYON16 tumour windows and three RINGS prostate tumour-mask pairs. These are controlled reconstructions from reference-derived coarse masks, not tests of independently predicted feature clusters or complete whole-slide throughput.

**Keywords:** computational pathology; tissue segmentation; boundary refinement; tile mask; GeoJSON; reproducibility

## Abbreviations

| Abbreviation | Meaning |
|---|---|
| ASSD | Average symmetric surface distance |
| CIELAB | CIE L*a*b* colour space |
| F1 | Harmonic mean of precision and recall |
| H&E | Haematoxylin and eosin |
| HD95 | 95th-percentile Hausdorff distance |
| IoU | Intersection over union |
| JSONL | JSON Lines |
| OME-TIFF | Open Microscopy Environment Tagged Image File Format |
| RGB | Red, green and blue |
| ROI | Region of interest |
| SHA256 | Secure Hash Algorithm 256-bit |
| TCGA | The Cancer Genome Atlas |
| WSI | Whole-slide image |

BCSS, TIGER and WSSS4LUAD are dataset names defined in Table 1.

## 1 Introduction

Pixel-level tissue maps are needed to quantify the spatial arrangement of pathological compartments, yet exhaustive whole-slide boundary annotation is costly. MedSAM (Ma et al., 2024) and PathSegmentor (Chen et al., 2026) have expanded segmentation from spatial and language prompts, respectively. Their outputs depend on a specified target, whereas tile-based analyses can already contain a categorical map whose principal error is geometric: class boundaries inherit the square tile grid. A method that refines all interfaces in that existing map addresses a distinct downstream task.

Tile-based analysis remains practical for characterizing large whole-slide images. UNI (Chen et al., 2024), UNI2 (Mahmood Lab, 2025) and Virchow2 (Zimmermann et al., 2024) learn tile representations through self-supervised pretraining without pixel-level segmentation masks. Their pretrained features can be clustered on a new slide without a target-specific dense training set or a separate spatial prompt for every region. Tile embeddings can also be reused for alternative groupings, which is useful when surveying large whole-slide images. This advantage concerns exploratory tissue grouping, not automatic semantic segmentation: cluster identifiers require independent interpretation or validation before they can be named as tissue classes. Moreover, each projected tile receives a single label even when it contains several tissue compartments. Its borders follow the sampling grid rather than the contours visible in the haematoxylin and eosin (H&E) image, distorting both tissue–background and internal tissue interfaces. MedSAM and PathSegmentor can be applied to whole-slide regions, but a cluster ID alone supplies neither MedSAM's spatial target nor PathSegmentor's semantic description; neither tool is designed specifically to adjust every boundary of a supplied multiclass tile map while retaining its labels.

The methodological gap is the absence of a direct route from an existing multiclass tile map to tissue contours at native image resolution. Such a route should use the tile labels as prior information, revise the interfaces between every pair of adjacent tissue groups, and preserve valid regions while reducing grid-shaped edges and small artifacts. The same refined geometry should be available as a raster and as vector regions for downstream analysis. Image appearance and spatial coherence must be balanced while retaining the supplied categorical information.

Bora addresses this gap by combining the aligned tile mask with the original H&E image. The mask supplies the existing tissue-group assignments, while local image appearance guides changes to internal class interfaces and the tissue–background contour. Bora then exports a native-size categorical mask and GeoJSON regions. This design can turn a coarse map already produced during tile-based analysis into a more spatially precise representation without collecting a new dense training set, drawing a prompt for each region, or fitting a task-specific pixel classifier. It also separates the geometric task of locating boundaries from the semantic task of naming clusters, so the original group identities remain traceable through refinement. Bora cannot name those groups or reconstruct a class that has disappeared completely from the tile input.

We evaluate whether these proposed benefits are realized in measured region overlap, class-pair boundary agreement, class retention, GeoJSON reproducibility and processing time. A three-region pilot from BCSS, TIGER and WSSS4LUAD tests 56, 90 and 216 pixel tiles with requested error rates of 0%, 1% and 5%; descriptive extensions increase the study to 63 distinct regions under narrower settings. The same reference masks generate the coarse inputs and provide the scores, so the experiments measure recovery from controlled tile quantization rather than performance on independently predicted feature clusters. Direct comparisons with MedSAM and PathSegmentor remain to be tested.

## 2 Related work

Related methods differ in how they obtain tissue identity and where they locate a boundary. This distinction matters because Bora receives class assignments rather than inferring the names of tissue compartments from the image alone.

### 2.1 Tissue segmentation in digital pathology

Pixel classifiers and foundation-model-derived dense predictors learn semantic classes from annotated examples (Lv et al., 2026). Representative-tile sampling has also supported whole-slide diagnosis (Barker et al., 2016), but a slide-level classifier does not locate the interfaces between tissue compartments. Tile embeddings from UNI (Chen et al., 2024), UNI2 (Mahmood Lab, 2025) and Virchow2 (Zimmermann et al., 2024) support self-supervised tissue grouping across large slides, but a cluster ID is not itself a validated tissue class. When group labels are projected to the image, the tile footprint quantizes every interface.

### 2.2 Boundary refinement and classic image processing

Active contours (Chan and Vese, 2001), interactive graph cuts (Boykov and Jolly, 2001), GrabCut (Rother et al., 2004), conditional random fields (Krähenbühl and Koltun, 2011) and watershed offer complementary ways to combine image evidence and spatial coherence. For a multiclass tile map, the refinement must also preserve reliable class interiors and avoid replacing one annotation with another merely to improve edge smoothness.

### 2.3 Foundation models for segmentation refinement

MedSAM (Ma et al., 2024) can delineate a prompted target without training on labels from the new image collection, but it still needs a target prompt. PathSegmentor (Chen et al., 2026) uses language descriptions for pathology structures. Both can be deployed region by region on large images; neither directly accepts arbitrary tile-cluster identifiers as a fully specified multiclass contour-refinement problem. A fair end-to-end comparison would need common category definitions, prompt generation, image coverage and timing.

### 2.4 Limitations of existing approaches

Existing prompt-based and dense prediction methods do not by themselves solve the practical task of geometrically revising an already assigned, potentially unnamed multiclass tile map. Classical boundary methods can improve edge placement but may erase narrow classes or introduce artifacts without class-aware constraints. Stain variation across centres can also change image appearance (Hetz et al., 2024), which may affect a refinement cost based on local colour. Bora treats the supplied labels as a prior and refines their interfaces using local colour evidence; semantic validation remains an independent requirement.

## 3 Methods

Figure 1 shows the input sources, refinement operations and outputs in their order of use. A prospective feature-clustering route based on UNI2 (Mahmood Lab, 2025) or Virchow2 (Zimmermann et al., 2024) representations and the evaluated reference-derived tiling route using tiling-mask converge on the same categorical-mask input. The original H&E image supplies both image evidence for Bora and, in the prospective route, tile features. The dense benchmark annotation generates coarse test inputs and supplies reference scores; it is never provided to the refinement algorithm. Bora handles internal interfaces between positive tissue classes separately from the tissue–background contour, which uses seeded GrabCut (Rother et al., 2004), before producing a raster and GeoJSON.

**Figure 1.** Landscape Bora workflow and study design. The left branch shows a proposed feature-clustering route in which UNI2 or Virchow2 tile features yield tissue-group labels; this route was not evaluated. The right branch shows the controlled benchmark in which tiling-mask quantizes a dense reference at 56, 90 or 216 pixels and requests 0%, 1% or 5% border error. The common Bora path validates the aligned inputs, refines internal tissue interfaces by local colour competition and the external tissue contour by seeded GrabCut, then applies categorical cleanup. The native-size raster is polygonized to GeoJSON. Solid black arrows show image and mask flow; the dashed black arrow carries the reference only to evaluation. H&E, haematoxylin and eosin; RGB, red-green-blue; CIELAB, CIE L*a*b* colour space; OME-TIFF, Open Microscopy Environment Tagged Image File Format. Figure file: `/home/chiamaka/Bora/docs/figures/bora_workflow.png`.

### 3.1 Overview of Bora

Bora accepts an H&E image and an aligned categorical tile mask, edits interfaces between adjacent groups, and writes a native-coordinate raster and GeoJSON. The evaluated path uses local class competition and a separate tissue/background operation. The feature-clustering path in Figure 1 is prospective and was not evaluated in this study.

### 3.2 Coarse segmentation and label representation

Each observed valid source class is mapped to a positive consecutive integer while dataset-specific ignored pixels map to zero. The map from source values to Bora values is recorded for every case. Inputs must have identical width and height; the labels are categorical and are never interpolated as continuous intensities. Reference-derived tiling is used only to construct controlled benchmark inputs (Section 4.2).

### 3.3 Multiclass boundary refinement

The internal method identifies interfaces between two positive labels and permits changes within a radius of approximately 0.28 times the source tile width at the analysis scale. For each class, reliable interior pixels at least 0.28 tile widths from its boundary estimate spatially weighted means and variances in smoothed CIELAB colour. The local Gaussian kernel scale is 0.70 tile widths. Each candidate class receives the mean of three squared standardized colour residuals plus 1.5 times the squared normalized distance outside its original region. Candidates farther than the search radius cannot claim a pixel. The lowest cost class wins within the editable band. The use of image appearance and spatial coherence is related to graph cuts (Boykov and Jolly, 2001) and conditional random fields (Krähenbühl and Koltun, 2011); the local cost and class protection described here are Bora-specific.

### 3.4 Protected-core and watershed refinement

Bora receives the original RGB tissue image and a categorical mask expanded to identical dimensions. The tested backend is `adaptive-wand`; the source tile width is passed explicitly. Haloed blocks allow bounded processing, and central regions are written to the output. The RGB image and mask are downsampled by a factor of two for boundary inference, then the categorical result is expanded by nearest neighbour to native pixels. Thus the output dimensions are native, while this backend's boundary decisions are made at half resolution. The halo is at least 48 pixels and otherwise scales with tile width. Protected class interiors anchor the optimization.

A 9 pixel categorical median filter and seed connectivity check remove isolated assignments. The external tissue/background contour is handled separately by seeded GrabCut (Rother et al., 2004) with a foreground core fraction of 0.37 tile widths, a 5 pixel median filter and small component cleanup. The evaluated adaptive-wand path does not run marker-controlled watershed; watershed is a separate Bora backend used only in a comparator. The method cannot recover a class that tiling has removed completely.

### 3.5 Whole-slide processing and GeoJSON generation

The refined label plane is written as a lossless OME TIFF at native pixel coordinates. For this pilot the GeoJSON converter reads level zero, polygonizes connected pixels by integer label, repairs invalid geometry when needed, and unions components per class. Coverage aware simplification is available in Bora; it is disabled here so conversion fidelity can be measured without deliberate geometric approximation. Components are retained at zero minimum area. Every feature includes its integer `value` and categorical classification. A second conversion uses identical settings and an independent output file. We compare SHA256 of the two byte streams and rasterize each GeoJSON in the original pixel grid; exact byte equality and pixel agreement are separately reported. Default production profiles may simplify and filter shapes, so the pilot's exact round trip result should not be generalized to those profiles.

The pilot runner fixes the three sample IDs and seed, records the inventory row and class remapping, and resumes completed conditions without changing source files. The extension scripts record their selection rules and source identifiers. Companion scripts produce result CSVs, plots and the class-pair boundary audit. The reference masks, original images and acquisition hashes remain unmodified. Output provenance files record the Bora Git HEAD, a dirty-worktree flag, SHA256 values for the relevant local source files and tiling-mask executable, package versions and CPU model. Benchmark scripts are published in Bora-extra, but the runs predate that repository split; exact reproduction requires a tagged snapshot of the run-time code and executable together with the still-local derived outputs.

Haloed blocks and window reads provide a bounded-memory route for whole-slide inputs. The present ROI timing experiment does not establish complete-WSI throughput; the independent full-slide engineering benchmark is kept separate until its results and resource measurements are finalized.

## 4 Experimental setup

The experiments evaluate reconstruction from reference-derived tile masks. All acquisition, pairing, ontology and resolution claims follow the frozen source inventories; no dense reference is passed to Bora during refinement.

### 4.1 Datasets and reference annotations

The evaluation comprises **three primary benchmark datasets and 63 distinct annotated image/mask ROIs**: 12 BCSS breast ROIs from the dataset described by Amgad et al. (2019) and acquired through the MedOtter (2026a) mirror, 15 TIGER breast ROIs (TIGER Grand Challenge, 2022), and 36 WSSS4LUAD lung ROIs from the challenge described by Han et al. (2022) and acquired through the MedOtter (2026b) mirror. Each dataset contributes one ROI to the initial pilot; the descriptive extensions add 11 BCSS, 14 TIGER and 35 WSSS4LUAD ROIs. These are image counts, not numbers of tile-size/error-rate runs. The ten newly added TIGER and eight newly added BCSS ROIs come from distinct unused source groups and were tested at three widths with zero requested error. We report the 63 images as analyzed or evaluated, not as an independent blind test cohort. The 36 WSSS4LUAD images belong to the source validation split, while the acquired BCSS and TIGER inventories have no provider-assigned held-out split.

**Table 1.** Size of each complete acquired dataset, source partitions and numbers of images evaluated with Bora. Dataset size counts source images or ROIs, not repeated tile settings or independent patients. “Evaluated” counts unique annotated images.

| Benchmark | Retrieved repository or release | Source partitions and annotation status | Entire acquired dataset | Bora evaluated |
|---|---|---|---:|---:|
| BCSS breast tissue | MedOtter/BCSS, pinned Hugging Face mirror | Dense ROI pairs; unsplit | 151 ROIs | 12 |
| TIGER breast tissue | TIGER WSIROIS, Zenodo 6014422 | ROI pairs from 195 source WSIs; unsplit | 2,030 ROIs | 15 |
| WSSS4LUAD lung tissue | MedOtter/WSSS4LUAD-v2, pinned Hugging Face mirror | 10,091 weakly labelled train patches; 40 dense validation pairs; 80 test ROIs excluded | 10,211 images | 36 validation ROIs |

BCSS image and mask files were retrieved from the MedOtter/BCSS Hugging Face mirror (https://huggingface.co/datasets/MedOtter/BCSS; revision `502d4a3fbc77dbaca6f4664c19e2379ff077d418`), rather than directly from the upstream BCSS project (MedOtter, 2026a). The mirror describes its RGB images as colour normalized. TIGER ROI images and annotations were retrieved as `roi-level-annotations.zip` from Zenodo record 6014422 (https://doi.org/10.5281/zenodo.6014422; TIGER Grand Challenge, 2022). WSSS4LUAD files were retrieved from the MedOtter/WSSS4LUAD-v2 Hugging Face mirror (https://huggingface.co/datasets/MedOtter/WSSS4LUAD-v2; revision `aa035a16f230622a17f51bc6b5180209a04bec7b`; MedOtter, 2026b). Acquisition was recorded on 15 October 2026. The acquisition code stores file hashes, image/mask paths, source IDs, label definitions and split information in the benchmark's `prepared/*.jsonl` inventories. The WSSS4LUAD training patches have image-level labels rather than dense segmentation masks and were not used to train or evaluate Bora. Its 80 received test ROIs were excluded because 10 dense masks were absent and 70 disagreed with their RGB dimensions. Four of the 40 geometrically valid validation ROIs had only one observed valid tissue class and were excluded from the multiclass analysis.

Two further sources supply tissue-region masks under distinct endpoints. RINGS prostate tumour images and `MANUAL TUMOR` annotations were retrieved from the publisher's Mendeley Data version 1 (https://doi.org/10.17632/h8bdwrtnr5.1; Salvi et al., 2021). The negative RINGS tumour mask is not an annotated stroma class.

Three CAMELYON16 complete slides, tumour masks and XML annotations for cases `tumor_058`, `tumor_091` and `tumor_107` were retrieved through the Registry of Open Data on AWS (https://registry.opendata.aws/camelyon/; Radboud University Medical Center, 2026). Bora was evaluated on 4096 × 4096 native windows from these slides.

CAMELYON16 normal tissue was automatically detected, not expert classified. Neither small exploratory subset is pooled with the primary multiclass ROI analysis in Table 1, and the window tests are not complete-slide runs.

The benchmark uses immutable acquisition inventories and associated resolution and label metadata. The BCSS pilot is a 1222 × 1036 breast ROI with tumour, stroma and necrosis or debris; the mirror reports 0.25 µm per pixel, equivalent to approximately 40× sampling (Amgad et al., 2019). The TIGER pilot is a 1224 × 1166 breast ROI from case 114S with invasive tumour, tumour-associated stroma, healthy glands, inflamed stroma and “rest”; the source reports approximately 0.5 µm per pixel (TIGER Grand Challenge, 2022). It is outside the TIGER TCGA/BCSS subset. The WSSS4LUAD pilot is a 248 × 206 lung validation ROI with tumour and stroma; its distributed pixel spacing is unverified (Han et al., 2022). We do not assign it a physical resolution merely from a reported nominal magnification.

The acquired BCSS masks (MedOtter, 2026a) designate values 0, 7 and 15 as ignored, and TIGER masks (TIGER Grand Challenge, 2022) designate value 0 as ignored. WSSS4LUAD masks (MedOtter, 2026b) use 0 for tumour and 3 for ignored white background. Before tiling, each ROI's observed valid class values are mapped to positive consecutive identifiers and ignored pixels to zero. The mapping is saved beside outputs. This is essential because the tiling backend treats zero as background. Scoring uses only pixels with a valid reference class; boundary scoring concerns interfaces between two positive tissue classes. Data relationships and masks come from JSONL inventories, never filename inference.

**Table 5.** Separate tissue-region experiments. Each source contributes three distinct images tested at 56, 90 and 216 pixels with zero requested border error. These endpoints are not pooled with the primary multiclass cohort.

| Source | Acquired release | Tissue-region endpoint | Images | Conditions |
|---|---|---|---:|---:|
| RINGS | Mendeley publisher release | Binary prostate tumour | 3 | 9 |
| CAMELYON16 | Original AWS slides and annotations | Tumour versus automatically detected normal in a native window | 3 | 9 |

Figure 4 compares the three primary tissue-mask datasets at matching image coordinates. The supplementary tissue atlas shows the RINGS tumour and CAMELYON16 window comparisons using the same reference–tile–Bora column order and semi-transparent masks.

### 4.2 Generation of coarse segmentation masks

The C++ tiling-mask tool (https://github.com/tkcaccia/tiling-mask) forms one categorical value per non-overlapping square tile by majority vote over the dense reference. Tile widths are 56, 90 and 216 pixels; partial edge tiles are retained. For each width we request border error rates of 0%, 1% and 5%, using seed 20261016. The tool swaps selected tiles only where neighboring positive classes make a tile eligible; the actual eligible and swapped counts are recorded and reported. Compact tile grids are expanded by nearest neighbour to exactly match the ROI before Bora runs. Tiling-mask supplies the controlled coarse inputs for the refinement experiment.

For the separate tissue-region experiments, three inventoried RINGS tumour-mask pairs and three native CAMELYON16 windows were tiled at 56, 90 and 216 pixels with zero requested error. CAMELYON16 uses tumour and automatically detected normal tissue; unassigned pixels are ignored. The RINGS task remains binary, with all non-tumour pixels treated as outside rather than stroma. These tasks are analyzed independently of the primary multiclass cohort.

### 4.3 Baseline methods and ablation studies

After the three ROI pilot, we applied the frozen adaptive-wand settings to WSSS4LUAD validation pairs, from the challenge described by Han et al. (2022) and acquired through the MedOtter (2026b) mirror, with at least two observed valid classes at zero requested tile error and all three tile widths. The original pilot ROI is excluded from extension summaries. Source slide and patient linkage is unavailable, so this is an ROI level descriptive extension rather than an independent patient test. An initial breast extension uses three additional BCSS ROIs (Amgad et al., 2019) from distinct TCGA case identifiers and four TIGER `tissue-cells` ROIs (TIGER Grand Challenge, 2022) from distinct non-TCGA source groups, limited to images of at most three million pixels and a 56 pixel tile width. A further ten inventory-ordered eligible, previously unused non-TCGA TIGER source groups were selected using the same size and class rules and processed at all three widths with zero requested error. Eight further BCSS source groups with at least two observed valid classes and at most five million pixels were processed at the same three widths and zero requested error. Selection and source identifiers are recorded before summarization. We also compare the pilot outputs with fixed 5 and 9 pixel categorical median filters and the separate Bora marker controlled watershed backend without annealed competition.

The expanded tile map is the primary matched baseline. Median filtering and the separate watershed backend are limited comparators, not a factorial ablation of every adaptive-wand stage. Protected-core, local-colour and external-contour ablations have not yet been completed; no stage-specific contribution is inferred from the existing comparison.

### 4.4 Evaluation metrics

We compare both the expanded tile baseline and Bora raster with the native reference. Following the distinction among segmentation error types described by Taha and Hanbury (2015), we report valid pixel accuracy, macro IoU over reference classes, internal boundary F1 at 8, 16 and 32 pixel tolerance, average symmetric surface distance, 95th percentile Hausdorff distance and class retention. For class-pair boundary matching, a boundary is indexed by its unordered pair of positive tissue labels. Each predicted boundary pixel can match only a reference boundary of the same class pair. Horizontal and vertical adjacency are considered only when both neighboring reference pixels are valid, applying the ignore mask symmetrically. Pairwise precision and recall pool matched and unmatched boundary pixels across class pairs; an absent predicted or reference pair contributes zero matched pixels. Surface distances are summarized only for class pairs present in both maps, and missing pairs are listed separately. An 8 pixel tolerance corresponds to 2 µm at the BCSS spacing reported by Amgad et al. (2019) and approximately 4 µm at the TIGER spacing (TIGER Grand Challenge, 2022); WSSS4LUAD physical spacing is unverified (Han et al., 2022). Pixel tolerances therefore do not represent a common physical tolerance across datasets.

For the separate binary RINGS tumour endpoint we additionally report foreground IoU, Dice, pixel accuracy, external target-boundary F1 at 8, 16 and 32 pixels, average symmetric surface distance, 95th-percentile Hausdorff distance, and the number of coarse masks with no positive tumour region. A missing predicted boundary has an undefined F1 when no comparison is possible. These binary scores are not averaged with the primary class-pair internal-boundary scores. The three CAMELYON16 windows use the class-pair scoring protocol.

### 4.5 Computational environment

Wall time is recorded for image/mask preparation, tiling all three sizes at each requested error rate, Bora refinement with vectorization, and independent repeat vectorization. Bora's own conversion report isolates the GeoJSON time. Input and output byte counts, commands, label maps and per condition JSON reports are retained. Preparation, grid expansion and scoring timings were measured on a later repeat pass for some records and are marked as such. The workstation has a 13th generation Intel Core i7-13700 CPU with 24 logical CPUs; the tiler used four workers and GeoJSON conversion used two in the main runs. Current timings are single ROI measurements on a shared machine without warm-up repetitions, peak memory capture or controlled worker scaling; they are not whole slide throughput estimates.

## 5 Results

All reported numerical results below derive from completed ROI experiments. Conditions within one image share the same annotation and are not statistically independent.

### 5.1 Segmentation accuracy and boundary fidelity

At zero requested tile error, macro IoU improved in eight of nine dataset–tile combinations and was unchanged for WSSS4LUAD at 216 pixels (Fig. 2). BCSS macro IoU changed from 0.727 to 0.794, 0.658 to 0.746 and 0.515 to 0.552 at 56, 90 and 216 pixels respectively. TIGER changed from 0.782 to 0.815, 0.722 to 0.748 and 0.518 to 0.550. WSSS4LUAD changed from 0.845 to 0.868, 0.721 to 0.735 and 0.386 to 0.386. At 56 pixels, class pair aware internal boundary F1 within 8 pixels rose from 0.302 to 0.499 on BCSS, 0.169 to 0.275 on TIGER, and 0.500 to 0.635 on WSSS4LUAD. Across 27 conditions, mean macro IoU gain was 0.034, with 24 gains and three unchanged conditions. Among 24 conditions with defined class pair F1, mean gain was 0.094, with 21 gains and three losses. At 90 pixels the WSSS4LUAD boundary F1 declined from 0.179 to 0.152. At 216 pixels its coarse mask has no interface between positive tissue classes, making an internal boundary F1 undefined rather than zero.

**Figure 2.** Macro intersection over union (IoU) for the selected region of interest from each dataset. Solid lines are Bora; dashed lines are the expanded tile baseline. The different colours are separate error rates. Figure file: `/mnt/sata_ssd/Bora-benchmark/results/bora-pilot-independent-2026-10-16/pilot-macro-iou.png`.

The 5% requested perturbation swapped 15, 7 and 1 BCSS tiles at widths 56, 90 and 216, and 12, 5 and 1 TIGER tiles. Relative to the corresponding zero-error coarse masks, changed pixel fractions were 3.59%, 4.48% and 3.69% for BCSS and 2.60%, 2.84% and 1.30% for TIGER. It swapped no WSSS4LUAD tiles because this small field had at most seven eligible internal border tiles; all three measured changed-pixel fractions were zero. Identical WSSS4LUAD curves therefore do not demonstrate error robustness. At 216 pixels the WSSS4LUAD tile baseline and Bora both retained only one of two reference classes; the missing class explains the absent internal boundary and unchanged IoU.

At 56 pixels and 0% requested error, BCSS class IoU (Amgad et al., 2019) changed from 0.609 to 0.713 for tumour, 0.679 to 0.758 for stroma, and 0.894 to 0.912 for necrosis or debris (Fig. 3). TIGER class IoU (TIGER Grand Challenge, 2022) changed from 0.902 to 0.922 for invasive tumour, 0.751 to 0.783 for tumour-associated stroma, 0.625 to 0.665 for healthy glands, 0.772 to 0.828 for inflamed stroma, and 0.860 to 0.878 for “rest.” WSSS4LUAD class IoU (Han et al., 2022) changed from 0.896 to 0.909 for tumour and 0.793 to 0.827 for stroma. These are dataset-specific annotation classes: TIGER “rest” is not equivalent to BCSS necrosis, and WSSS4LUAD zero encodes tumour rather than background in the supplied source mask.

**Figure 3.** Class intersection over union (IoU) at 56 pixels and 0% requested error, grouped by source dataset and its own annotation meanings. Figure file: `/mnt/sata_ssd/Bora-benchmark/results/bora-pilot-independent-2026-10-16/pilot-class-iou.png`.

**Figure 4.** Registered tissue-mask comparison. BCSS breast, TIGER breast and WSSS4LUAD lung occupy separate rows; within each row, the same H&E coordinates appear under the original reference annotation (left), 56 × 56 pixel majority-vote tile mask (centre) and Bora-refined mask (right). Categorical tissue regions are shown as 42%-opacity colour fills, with a row-specific label key. The black bar indicates one tile width; all panels include multiple tiles. Zero-error inputs are shown. The small WSSS4LUAD ROI is displayed in full, while boundary-rich registered crops are used for the larger breast ROIs. Colours encode dataset-specific labels and are not pooled across rows. Figure file: `/home/chiamaka/Bora/docs/figures/multi_tissue_mask_comparison.png`.

Of 40 acquired WSSS4LUAD validation pairs (MedOtter, 2026b), four contain only one observed valid class and are excluded from this multiclass analysis. The initial pilot ROI is excluded from the extension summary, leaving 35 ROIs at each width. Their parent slides and patients cannot be identified from the release. Median macro IoU fell by 0.0031 at 56 pixels, with 20 decreases and 15 increases. It rose by 0.0040 at 90 pixels and 0.0029 at 216 pixels. Median class pair boundary F1 improved where both masks had an interface, but several ROIs worsened and many 216 pixel masks had no defined internal score (Table 2; Fig. 5). These ROI distributions show that the favorable three ROI pilot does not generalize uniformly, especially at the smallest tile width.

**Table 2.** WSSS4LUAD validation ROI extension, excluding the pilot ROI. Changes are Bora minus the matched tile baseline. F1 counts include only defined class pair scores. Times include refinement and unsimplified GeoJSON conversion.

| Tile px | ROIs | Median dIoU | IoU +/-/= | F1 n | Median dF1 | F1 +/-/= | Median s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 56 | 35 | -0.0031 | 15/20/0 | 31 | +0.0248 | 24/7/0 | 1.82 |
| 90 | 35 | +0.0040 | 20/15/0 | 29 | +0.0330 | 26/2/1 | 1.88 |
| 216 | 35 | +0.0029 | 19/8/8 | 20 | +0.0329 | 17/2/1 | 1.74 |

No new reference class disappeared during Bora refinement in these 35 ROIs: at 56, 90 and 216 pixels, the respective coarse masks had already lost a class in 4, 5 and 17 ROIs, and the refined masks retained the same class counts.

**Figure 5.** Region-of-interest changes in macro intersection over union (IoU) and class-pair boundary F1 for the WSSS4LUAD extension. Each point is one ROI and red diamonds show medians. F1 is the harmonic mean of boundary precision and recall; it is omitted when no interface score is defined. Figure file: `/mnt/sata_ssd/Bora-benchmark/results/bora-wsss-validation-2026-10-16/reviewer_cohort/delta_distribution.png`.

At 56 pixels and zero requested error, a separate seven-group breast extension showed macro IoU increases in all three BCSS and all four TIGER source groups. Median changes were +0.0574 and +0.0187, respectively; median class pair F1 changes were +0.1143 and +0.0686. Two BCSS inputs had lost a rare reference class during tiling; Bora preserved their coarse mask class counts but did not recover the missing class. All seven unsimplified GeoJSON outputs exactly recovered their rasters. This extension was selected by image size and group identity, so it is descriptive rather than a prevalence estimate for either dataset.

A further ten non-TCGA TIGER source groups contributed 30 completed zero-error conditions at 56, 90 and 216 pixels (Table 3). Median macro IoU gains were +0.0235, +0.0284 and +0.0254, with gains in 9/10, 10/10 and 9/10 groups. Class-pair internal boundary F1 within 8 pixels improved in all ten groups at each width, with median gains +0.0854, +0.0795 and +0.0676. The group selection was inventory-ordered, restricted to `tissue-cells` ROIs with at least two observed valid classes, dimensions at least 1000 × 1000 pixels and area at most three million pixels, and excluded previously studied groups. Thus these results support a larger descriptive breast analysis but do not represent random sampling of the full TIGER collection.

**Table 3.** New TIGER source-group expansion at zero requested border error. Changes are Bora minus the paired coarse baseline; internal F1 uses a class-pair match and 8 pixel tolerance. The time includes refinement and GeoJSON conversion.

| Tile px | Groups | Median dIoU | IoU improved | Median dF1 | F1 improved | Median time s | GeoJSON round trip |
|---:|---:|---:|---:|---:|---:|---:|---|
| 56 | 10 | +0.0235 | 9/10 | +0.0854 | 10/10 | 2.02 | 10/10 exact |
| 90 | 10 | +0.0284 | 10/10 | +0.0795 | 10/10 | 2.17 | 10/10 exact |
| 216 | 10 | +0.0254 | 9/10 | +0.0676 | 10/10 | 3.08 | 10/10 exact |

Eight previously unused BCSS source groups contributed 24 completed zero-error conditions (Table 4). Each case had at least two observed valid tissue classes and no more than five million pixels. Median macro IoU gains were +0.0173, +0.0203 and +0.0184 at widths 56, 90 and 216; all eight groups improved at every width. Median class-pair 8-pixel F1 gains were +0.0502, +0.0581 and +0.0612, again positive in every case. Bora retained exactly the class IDs present in every matched coarse input. These cases were chosen by inventory order and size, so selection bias remains possible.

**Table 4.** New BCSS source-group expansion at zero requested border error. Changes are Bora minus the paired coarse baseline; internal F1 uses a class-pair match and 8 pixel tolerance. Time includes refinement and GeoJSON conversion.

| Tile px | Groups | Median dIoU | IoU improved | Median dF1 | F1 improved | Median time s | GeoJSON round trip |
|---:|---:|---:|---:|---:|---:|---:|---|
| 56 | 8 | +0.0173 | 8/8 | +0.0502 | 8/8 | 3.04 | 8/8 exact |
| 90 | 8 | +0.0203 | 8/8 | +0.0581 | 8/8 | 3.59 | 8/8 exact |
| 216 | 8 | +0.0184 | 8/8 | +0.0612 | 8/8 | 6.57 | 8/8 exact |

Surface-distance summaries add a qualification to these gains. Median changes in average symmetric surface distance at 56, 90 and 216 pixels were −2.19, −5.26 and −14.16 pixels in the new TIGER groups and −1.94, −5.22 and −22.03 pixels in the new BCSS groups. Median 95th-percentile Hausdorff distance changed by +0.50, −10.55 and −23.65 pixels in TIGER and +0.01, −10.18 and −31.40 pixels in BCSS. Thus the typical contour moved closer at every width, but the more extreme boundary discrepancy did not improve at 56 pixels. Distances concern observed internal class pairs and are in pixels, not a common physical unit across datasets.

**Figure 6.** Source-group changes in macro intersection over union and class-pair internal boundary F1 within 8 pixels for the new BCSS and TIGER extensions at zero requested border error. Thin lines connect the three widths for one source group; heavy lines show dataset medians. The horizontal dashed line marks no change. Each group contributes one image, so connected points are repeated conditions rather than independent samples. Figure file: `/home/chiamaka/Bora/docs/figures/breast_extension_changes.png`.

### 5.2 Comparison with alternative methods

A fixed 5 pixel median filter on the expanded tile mask had mean macro IoU gain of 0.0001 and mean class pair F1 change of -0.0016 across the 27 conditions; a 9 pixel filter had corresponding changes of 0.0002 and -0.0030. In a nine-condition zero-error comparison, Bora exceeded Bora's separate marker controlled watershed backend without annealed competition on macro IoU in eight conditions and tied in one. The watershed comparator reached class pair F1 0.203 on the WSSS4LUAD 90 pixel pilot, above Bora's 0.152, although its macro IoU was 0.693 versus Bora's 0.735. These comparisons show an accuracy tradeoff. The median filters are weak smoothing comparators; an independent graph cut or active contour baseline remains necessary.

### 5.3 Contributions of individual refinement stages

A full stage ablation is not available. The available median-filter comparison isolates simple categorical smoothing, and the watershed comparison evaluates a different backend rather than removal of one component from adaptive-wand. Consequently, the separate effects of local colour competition, protected cores, GrabCut and cleanup cannot yet be quantified.

### 5.4 Whole-slide scalability and computational efficiency

The pilot measured mean Bora subprocess wall times, including vectorization, of 2.44 s for BCSS, 2.70 s for TIGER and 1.57 s for WSSS4LUAD across nine conditions per ROI. Median times in the 35-ROI lung extension were 1.82, 1.88 and 1.74 s at 56, 90 and 216 pixels; nine larger ROIs required medians of 4.53, 3.91 and 5.87 s. These are shared-workstation ROI measurements. The ten-group TIGER expansion had median wall times of 2.02, 2.17 and 3.08 s; the eight-group BCSS expansion had medians of 3.04, 3.59 and 6.57 s at 56, 90 and 216 pixels. Complete-WSI wall time, peak memory and worker scaling have not yet been measured under this study protocol.

### 5.5 GeoJSON fidelity and output complexity

All 27 pilot conditions yielded a GeoJSON file. With full resolution polygonization, no area filtering and no simplification, rasterization of the vector output agreed with the refined mask at every pixel in all 27 conditions. The GeoJSON therefore has the same pixel-level scores against ground truth as its refined raster for these conditions. Independent repeated conversion produced identical GeoJSON bytes in all 27. For a separate audit of Bora's default `accurate` profile (2 pixel simplification and 500 pixel² minimum area), raster agreement was 99.741% for BCSS, 99.833% for TIGER and 99.861% for WSSS4LUAD at 56 pixels and zero requested error. Feature counts were unchanged in these three examples, while file sizes fell from 146 to 65 kB, 100 to 49 kB and 3.1 to 1.8 kB respectively. All 30 previously audited GeoJSON files contained valid, nonempty geometries without positive area overlap between different classes. Hole topology and viewer ingestion still require a separate audit. This profile is not pixel exact. The 30 TIGER and 24 BCSS expansion outputs each had exact GeoJSON-to-raster agreement and matched byte-identical repeat conversions. No simplified-profile fidelity claim is made for those extensions.

All 105 unsimplified WSSS4LUAD GeoJSON outputs exactly recovered their refined rasters and repeated conversions were byte identical. Among nine larger lung ROIs, median converter time was approximately 1.1–1.2 s across the three widths. This measurement does not establish complete whole-slide conversion throughput.

### 5.6 Additional tissue-region endpoints

The RINGS prostate tumour-mask subset contributed nine binary tissue-region conditions over three images. At 56 and 90 pixels, median foreground IoU improved by +0.0130 and +0.0684, with corresponding external boundary F1 gains of +0.0445 and +0.1648 within 8 pixels (Table 6). At 216 pixels, majority-vote tiling removed the tumour class from two of three inputs; the median IoU change was therefore zero. These small, selected subsets do not establish prostate-level generalization. All nine unsimplified GeoJSONs exactly recovered the refined raster and repeated conversion was byte identical.

The three 4096 × 4096 native CAMELYON16 windows contributed nine two-class conditions. Median macro IoU gains were +0.00285, +0.00698 and +0.01009 at 56, 90 and 216 pixels; class-pair 8-pixel F1 gains were +0.0757, +0.1216 and +0.1027. The comparison contrasts expert-derived tumour with automatically detected normal tissue and is not a fully expert-defined tissue ontology. All nine GeoJSON conversions were pixel exact and byte reproducible. The supplementary atlas shows the reference, tile and Bora maps as semi-transparent colour fills on registered H&E image areas.

**Table 6.** Separate tissue-region results. Each source contributes three images per tile width; values are median Bora-minus-tile changes. RINGS reports foreground IoU and external tumour-boundary F1; CAMELYON16 reports macro IoU and class-pair tumour/auto-normal boundary F1. These distinct endpoints are not pooled. Wall time includes refinement and vectorization. “Lost” counts RINGS inputs without a tumour class after tiling.

| Source | Tile px | Images | Median dIoU | Median dF1-8 | Lost | Median wall s | GeoJSON |
|---|---:|---:|---:|---:|---:|---:|---|
| RINGS tumour | 56 | 3 | +0.0130 | +0.0445 | 0/3 | 1.40 | exact |
| RINGS tumour | 90 | 3 | +0.0684 | +0.1648 | 0/3 | 1.60 | exact |
| RINGS tumour | 216 | 3 | 0.0000 | +0.1197 | 2/3 | 1.19 | exact |
| CAMELYON16 | 56 | 3 | +0.00285 | +0.0757 | – | 4.48 | exact |
| CAMELYON16 | 90 | 3 | +0.00698 | +0.1216 | – | 4.93 | exact |
| CAMELYON16 | 216 | 3 | +0.01009 | +0.1027 | – | 10.53 | exact |

## 6 Discussion

The primary multiclass study shows that spatial quantization can be partly corrected at interfaces between several breast and lung tissue classes (Figs. 2–4). The expanded lung set shows a mixed response: median macro IoU at 56 pixels declined even though median class-pair boundary F1 improved (Table 2; Fig. 5). A contour may move closer to a reference edge while shifting enough tissue area to lower region overlap. At 216 pixels, tiling had already removed a reference class in 17 of 35 lung extension ROIs, beyond what a method anchored to surviving classes can restore. Figure 4 shows how the same local method replaces tile steps with morphology-following contours in representative breast and lung image regions; it is a visual comparison, not a substitute for cohort-level scores.

The TIGER pilot and breast extensions use non-TCGA TIGER cases, but the source groups were selected by eligibility and inventory order rather than random sampling. Conditions on one ROI are correlated, and the 27-condition pilot average has no patient-level confidence interval. WSSS4LUAD parent slide and patient linkage are unavailable, so its 35 ROI distribution cannot support patient-level uncertainty estimates. The coarse inputs were generated from the same annotations used for scoring. This design measures controlled recovery after tile quantization, not performance on independently predicted Virchow2 or UNI2 clusters. The additional prostate and lymph-node tumour-region experiments use distinct endpoints and only three selected images each, so they cannot establish organ-level generalization. This study has no verified prostate multiclass mask, controlled 10×/20×/40× series or complete whole-slide timing. CAMELYON16 has three acquired complete slides, but only native windows were processed here; these results cannot be pooled with the multiclass ROI endpoint without a common annotation protocol.

Implementation exposed three linked limitations. Zero is ignored in the acquired BCSS masks (MedOtter, 2026a) and TIGER masks (TIGER Grand Challenge, 2022), but denotes tumour in WSSS4LUAD (MedOtter, 2026b); an incorrect remapping can therefore turn tissue into background. Some acquired pairs lack dense masks or disagree in dimensions, while RINGS tumour-negative pixels do not provide a validated stroma label. Tile expansion creates right-angle steps, while local colour competition can create isolated assignments or erase narrow tissue regions. Protected interiors and seed connectivity limit those errors but cannot restore a class absent from the tile input. The tested backend infers contours at half resolution, and compressed masks require windowed reads for bounded memory. Finally, polygon simplification and component filtering change rasterized pixels as well as file size. Block seams, complete WSI resource use, CPU scaling, hole topology and viewer ingestion remain unaudited; shared-workstation timings do not isolate these costs.

These findings argue for an evaluation that retains the image, source mask, class map, tile width, seed, requested and actual swapped-tile counts, executable hash and vectorization settings. Each refined mask should be scored against its matched tile baseline using class-aware internal boundary F1 and surface distance alongside macro IoU, class retention and changed-pixel fraction; a boundary score should remain undefined when no valid class pair survives tiling. Results should be stratified by dataset, tissue compartment and measured micrometres per pixel, with case- or patient-disjoint uncertainty estimates where provenance permits. Outside this benchmark, users should verify class inventories, holes and viewer ingestion and measure GeoJSON raster round-trip error when using a simplified profile. Settings should be frozen after development-set selection. Future comparisons should test MedSAM (Ma et al., 2024), PathSegmentor (Chen et al., 2026), graph cuts (Boykov and Jolly, 2001) and learned dense predictors (Lv et al., 2026) on common images, class definitions, prompts, hardware and end-to-end timing.

The next experiment should use complete WSIs with coarse masks generated independently from UNI2 (Mahmood Lab, 2025) or Virchow2 (Zimmermann et al., 2024) features, fixing cluster-to-tissue interpretation without test annotations so Bora's geometric contribution can be isolated. It should include case-disjoint slides, verified physical spacing and mask encoding, and full-slide wall time, peak memory, CPU use, worker scaling and block-seam measurements. Image resolution affects pathology classification (Neary-Zajiczek et al., 2023), and magnification sampling affects foundation-model representations (Möllers et al., 2026); neither result establishes a resolution optimum for Bora's boundary task. A future Bora study should therefore stratify boundary scores by measured pixel spacing and tile width in micrometres. Method development could compare stain-normalized or texture-aware costs, multiscale edge evidence and topology-aware treatment of narrow regions against graph cuts (Boykov and Jolly, 2001), active contours (Chan and Vese, 2001) and dense conditional random fields (Krähenbühl and Koltun, 2011), accepting changes on held-out boundary and overlap metrics rather than smoothness alone.

Expression-based clusters in spatial transcriptomics (Ståhl et al., 2016) offer a separate possible application. SPADE (Redekop et al., 2026) demonstrates that paired histology and spatial transcriptomics can inform image representations, but it does not evaluate refinement of a supplied categorical spot map. After accurate registration of a Visium spot grid (10x Genomics, n.d.-a) or Visium HD bin grid (10x Genomics, n.d.-b) to its H&E image, cluster assignments could be represented as a coarse categorical map. Bora could then be evaluated for morphology-guided refinement of interfaces supported by the image; conversely, Bora-refined histology regions could provide spatial context for interpreting expression clusters. Such analysis would require an explicit rule for mapping spots or bins to image pixels and an independent assessment of registration error and biological agreement. Refining a histological boundary cannot increase molecular resolution or establish the identity of an expression cluster. No spatial transcriptomics data were analyzed in the present study.

## 7 Conclusion

Bora improved region and class-pair boundary scores across a larger descriptive breast ROI set, while the broader lung ROI extension showed mixed region overlap and generally improved boundary proximity. The separate prostate tumour experiment showed that majority-vote tiling can remove a tissue-region class at coarse resolution; Bora cannot recover a class absent from its input. Unsimplified GeoJSON conversion was byte reproducible and pixel exact; the default accurate profile was not pixel exact in three audited cases. These controlled reconstruction results justify further testing with independently generated cluster masks and complete whole slides, but do not establish general clinical segmentation performance.

## Data and code availability

The Bora source is at https://github.com/tkcaccia/Bora, the acquisition and benchmark scripts are at https://github.com/tkcaccia/Bora-extra, and the tiling-mask source is at https://github.com/tkcaccia/tiling-mask. Pilot acquisition identities, commands, outputs and aggregate results are retained in a local research corpus. A reproducible archival release requires tagged code versions, environment and executable hashes, dataset identifiers and permitted derived outputs. Redistribution rights for each source dataset must be checked separately. No patient-level data were added to Git.

## References

10x Genomics, n.d.-a. Visium Spatial Gene Expression for Fresh Frozen: product documentation. 10x Genomics. https://www.10xgenomics.com/support/spatial-gene-expression-fresh-frozen (accessed 16 October 2026).

10x Genomics, n.d.-b. Visium HD Spatial Applications Performance, technical note CG000686, Rev. B. 10x Genomics. https://cdn.10xgenomics.com/image/upload/v1758134784/support-documents/CG000686_VisiumHDSpatialApplicationsPerformance_TechnicalNote_RevB.pdf (accessed 16 October 2026).

Amgad, M., Elfandy, H., Hussein, H., et al., 2019. Structured crowdsourcing enables convolutional segmentation of histology images. Bioinformatics 35, 3461–3467. https://doi.org/10.1093/bioinformatics/btz083.

Barker, J., Hoogi, A., Depeursinge, A., Rubin, D.L., 2016. Automated classification of brain tumor type in whole-slide digital pathology images using local representative tiles. Medical Image Analysis 30, 60–71. https://doi.org/10.1016/j.media.2015.12.002.

Boykov, Y.Y., Jolly, M.-P., 2001. Interactive graph cuts for optimal boundary and region segmentation of objects in N-D images. Proceedings of the IEEE International Conference on Computer Vision, 105–112. https://doi.org/10.1109/ICCV.2001.937505.

Chan, T.F., Vese, L.A., 2001. Active contours without edges. IEEE Transactions on Image Processing 10, 266–277. https://doi.org/10.1109/83.902291.

Chen, R.J., Ding, T., Lu, M.Y., et al., 2024. Towards a general-purpose foundation model for computational pathology. Nature Medicine 30, 850–862. https://doi.org/10.1038/s41591-024-02857-3.

Chen, Z., Hou, J., Lin, L., et al., 2026. Segment anything in pathology images with natural language. Nature Computational Science 6, 1024–1038. https://doi.org/10.1038/s43588-026-01042-5.

Han, C., Pan, X., Yan, L., et al., 2022. WSSS4LUAD: Grand Challenge on Weakly-supervised Tissue Semantic Segmentation for Lung Adenocarcinoma. arXiv preprint arXiv:2204.06455. https://arxiv.org/abs/2204.06455.

Hetz, M.J., Bucher, T.-C., Brinker, T.J., 2024. Multi-domain stain normalization for digital pathology: A cycle-consistent adversarial network for whole slide images. Medical Image Analysis 94, 103149. https://doi.org/10.1016/j.media.2024.103149.

Krähenbühl, P., Koltun, V., 2011. Efficient inference in fully connected CRFs with Gaussian edge potentials. Advances in Neural Information Processing Systems 24, 109–117. https://proceedings.neurips.cc/paper/4296-efficient-inference-in-fully-connected-crfs-with-gaussian-edge-potentials.

Lv, J., Zhu, Y., Okada, S., et al., 2026. Computationally efficient pathology segmentation using knowledge distillation from foundation models. arXiv preprint arXiv:2609.03947v1. https://arxiv.org/abs/2609.03947.

Ma, J., He, Y., Li, F., et al., 2024. Segment anything in medical images. Nature Communications 15, 654. https://doi.org/10.1038/s41467-024-44824-z.

Mahmood Lab, 2025. UNI2-h model card. Hugging Face. https://huggingface.co/MahmoodLab/UNI2-h (accessed 16 October 2026).

MedOtter, 2026a. BCSS pinned dataset mirror, revision 502d4a3fbc77dbaca6f4664c19e2379ff077d418. Hugging Face. https://huggingface.co/datasets/MedOtter/BCSS/tree/502d4a3fbc77dbaca6f4664c19e2379ff077d418 (accessed 16 October 2026).

MedOtter, 2026b. WSSS4LUAD-v2 pinned dataset mirror, revision aa035a16f230622a17f51bc6b5180209a04bec7b. Hugging Face. https://huggingface.co/datasets/MedOtter/WSSS4LUAD-v2/tree/aa035a16f230622a17f51bc6b5180209a04bec7b (accessed 16 October 2026).

Möllers, A., Hense, J., Schulz, F., Milbich, T., Alber, M., Ruff, L., 2026. Mind the gap: Continuous magnification sampling for pathology foundation models. Medical Image Analysis, article 104327, advance online publication. https://doi.org/10.1016/j.media.2026.104327.

Neary-Zajiczek, L., Beresna, L., Razavi, B., Pawar, V., Shaw, M., Stoyanov, D., 2023. Minimum resolution requirements of digital pathology images for accurate classification. Medical Image Analysis 89, 102891. https://doi.org/10.1016/j.media.2023.102891.

Radboud University Medical Center, 2026. CAncer MEtastases in LYmph nOdes challeNge (CAMELYON) dataset. Registry of Open Data on AWS. https://registry.opendata.aws/camelyon/ (accessed 16 October 2026).

Redekop, E., Pleasure, M., Wang, Z., et al., 2026. SPADE: Spatial transcriptomics and pathology alignment using a mixture of data experts for an expressive latent space. Medical Image Analysis 113, 104197. https://doi.org/10.1016/j.media.2026.104197.

Rother, C., Kolmogorov, V., Blake, A., 2004. GrabCut: interactive foreground extraction using iterated graph cuts. ACM Transactions on Graphics 23, 309–314. https://doi.org/10.1145/1015706.1015720.

Salvi, M., Bosco, M., Molinaro, L., et al., 2021. RINGS algorithm dataset. Mendeley Data, version 1. https://doi.org/10.17632/h8bdwrtnr5.1.

Ståhl, P.L., Salmén, F., Vickovic, S., et al., 2016. Visualization and analysis of gene expression in tissue sections by spatial transcriptomics. Science 353, 78–82. https://doi.org/10.1126/science.aaf2403.

Taha, A.A., Hanbury, A., 2015. Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool. BMC Medical Imaging 15, 29. https://doi.org/10.1186/s12880-015-0068-x.

TIGER Grand Challenge, 2022. TIGER WSIROIS data and tissue annotation description. Zenodo, record 6014422. https://doi.org/10.5281/zenodo.6014422 (accessed 16 October 2026).

Wei, Y., He, Y., Wu, L., et al., 2026. Free lunch in medical image foundation model pre-training via randomized synthesis and disentanglement. arXiv preprint arXiv:2602.12317v1. https://arxiv.org/abs/2602.12317.

Zimmermann, E., Vorontsov, E., Viret, J., et al., 2024. Virchow2: scaling self-supervised mixed magnification models in pathology. arXiv preprint arXiv:2408.00738v3. https://arxiv.org/abs/2408.00738.

## Submission items pending author input

Confirm every author's name, current affiliation, contribution and consent to authorship; choose the corresponding author; supply ORCIDs, acknowledgements, funding, conflict of interest, ethics/data use declarations and biographies. Prepare the journal's separate highlights, graphical abstract and Statement of Significance after the full benchmark is complete. Do not submit this pilot as a final dataset level performance claim.
