# Twenty distinct hypotheses for Virchow2-assisted Bora

Experimental protocol, 16 October 2026. Execution status and actual results are
recorded in the output `COMPLETE.json`, `AUDIT.json` and `REPORT.md`. The previous design
crossed four selectors with five pixel decoders. That design was rejected and
its runner removed. Selector comparisons are technical controls, not separate
ideas in this catalogue. No model decoder comparison will be counted toward
the twenty ideas below. Established components are being combined into testable
hypotheses; this document does not claim that all twenty are new algorithms.

The scientific question is whether retaining the class-discriminative part of
Virchow2's representation improves the location of tissue interfaces. Visual
resemblance to H&E is a separate outcome. A false-colour feature image, a
retrieved image, a hybrid using original pixels and an optimized inverse image
must be labelled accordingly.

## Proposed experiments

| ID | Idea | Intervention and hypothesis | Output and information used |
|---|---|---|---|
| 01 | Selective feature inversion | Freeze Virchow2 and optimize RGB pixels from noise to match selected spatial features, with a weak image regularizer. Test whether these features alone contain recognizable tissue architecture. | Approximate generated H&E; selected feature targets only. Multiple noise starts measure ambiguity; no original-pixel residual. |
| 02 | Minimal visual evidence | Learn the smallest spatial mask on the source image that preserves each class's discriminative feature score when the rest is replaced by neutral stain. Test which physical structures are necessary for class evidence. | Evidence image and spatial mask; original pixels explicitly retained in the selected regions. Mask is not a tissue annotation. |
| 03 | Nearest-class counterfactual | Find a small RGB edit that changes a tile's feature-based prediction to its hardest competing class. Map the edits rather than treating the changed image as real tissue. Test which structures distinguish difficult pairs. | Counterfactual image plus edit-magnitude map; uses original initialization and pixel-change penalty. Never presented as recovered anatomy. |
| 04 | Shared and class-specific concepts | Learn a sparse concept dictionary from selected spatial features, separate concepts shared by all groups from class-specific combinations, and display only the distinguishing combinations. Test whether a mixed latent coordinate becomes interpretable after factorization. | Concept RGB image and separate concept maps; feature-only representation, not regenerated H&E. Components are not biological markers. |
| 05 | Worst-class discriminative geometry | Learn a compact three-dimensional representation that maximizes the weakest pairwise class separation, with equal weight for rare classes. Test a purpose-built colour geometry instead of variance-ranked PCs. | Feature RGB; coarse class labels and selected features only. Report every class's held-out recall and margin. |
| 06 | Boundary-specific feature experts | Select evidence separately for each adjacent tissue pair and switch the representation at that pair's interface. Test whether tumor–stroma and gland–stroma boundaries require different evidence. | Pair-specific feature images and one merged Bora mask. Interface ownership is fixed from coarse labels to prevent conflicting updates. |
| 07 | Remove the colour-explained subspace | Predict selected features from stain/brightness summaries and retain the unexplained residual directions. Test whether morphology can distinguish classes after easy colour shortcuts are removed. | Residual-feature RGB; original colour is used only to estimate the nuisance subspace. Compare lost class evidence as well as boundary quality. |
| 08 | Stain-perturbation invariance | Re-encode morphology-preserving stain perturbations and keep class-discriminative directions whose evidence is stable across them. Test robustness under colour change rather than only removing linear colour dependence. | Stable-feature RGB and sensitivity map. Requires additional genuine Virchow2 inference; no claim of measured stain invariance before it runs. |
| 09 | Scale-persistent class evidence | Encode at 0.5, 1 and 2 µm/px and retain class evidence that persists across physical scales. Test whether stable tissue identity is more useful than fine texture alone. | Consensus feature RGB; original sampling and each resampled input's effective µm/px recorded. Coarser inputs add context, not optical resolution. |
| 10 | Context chooses, detail places | Use a larger physical field of view to select plausible tissue classes, then let fine-scale selected features position the boundary. Test complementary semantic context and local geometry rather than requiring scale invariance. | Hierarchical feature image and native-frame boundary proposal. Requires multi-scale inference; class context and boundary evidence remain separately inspectable. |
| 11 | Context-disagreement gating | Compare the same spatial location across overlapping encoder windows. Use selected-feature evidence only where context agrees; use ordinary H&E Bora evidence elsewhere. Test whether context instability predicts harmful refinements. | Context-variation map and hybrid Bora result; original RGB is an explicit fallback. Variation is not calibrated uncertainty and single-view locations cannot establish agreement. |
| 12 | Geodesic evidence transport | Build a graph of spatial neighbours weighted by selected-feature similarity, then propagate coarse interior class evidence along low-cost paths. Test whether tissue-following connectivity handles curved regions better than Euclidean distance. | Class-evidence RGB and probability maps; feature-space graph, coarse interior seeds. No original-pixel synthesis. |
| 13 | Feature discontinuity as a barrier | Use abrupt changes in selected features as barriers that stop a boundary moving across a semantic interface, rather than predicting each pixel's class independently. Test boundary evidence directly. | Barrier image and constrained Bora mask; selected features plus coarse labels. Include an ordinary H&E-gradient control. |
| 14 | Semantic features, native superpixels | Partition H&E into small edge-following regions, pool selected features into those regions and move class evidence over their adjacency graph. Test whether native image geometry can restore boundary precision missing from the token grid. | Region feature RGB and refined mask; original H&E explicitly supplies high-resolution edges. Detail is not claimed to have been recovered from Virchow2 alone. |
| 15 | Tissue architecture from object relations | Derive candidate nuclei and lumens from H&E, associate them with nearby selected-feature evidence, and describe their density, neighbourhoods and enclosure patterns. Test gland architecture and cell arrangement rather than isolated token appearance. | Object-relation maps and Bora input; uses original RGB for candidate objects. Object extraction failures must be visible, and no unvalidated object map is called ground truth. |
| 16 | Soft tissue unmixing | Express each token as a mixture of class prototypes, using soft transport constraints so dominant classes cannot absorb small glands or inflamed stroma. Test whether a continuous mixture models interfaces better than a hard class decision. | Fraction maps and mixture RGB; coarse class counts supply only a weak area prior. Report sensitivity to incorrect coarse class proportions. |
| 17 | Topology-aware feature refinement | Add penalties for unsupported fragments, narrow bridges and closing gland lumens to a selected-feature boundary objective. Test whether tissue topology helps when local appearance is ambiguous. | Refined mask with an explicit topology-change map; uses feature evidence and coarse constraints. Do not force the coarse mask's topology to be correct. |
| 18 | Feature-matched tissue quilting | Retrieve real small patches from other spatial blocks using selected-feature similarity, then solve overlap seams to form a mosaic. Test whether the selected representation retrieves the right morphology without training an RGB decoder. | Retrieved H&E mosaic with donor-coordinate map. Exclude the target block and a context buffer from donors; self retrieval cannot count as reconstruction. |
| 19 | Predictable latent structure | Mask selected feature tokens, predict them from surrounding tokens and retain the spatially predictable component. Test whether coherent tissue architecture survives while isolated unstable activations disappear. | Completed-feature RGB plus residual map. Train/evaluate on separate spatial blocks and show whether small real structures are oversmoothed. |
| 20 | Feature-guided detail reconstruction | Decompose original H&E into coarse structure and wavelet detail; retain or attenuate detail according to discriminative feature interfaces. Test whether preserving real fine structure while suppressing irrelevant texture gives Bora a better input. | Hybrid H&E image; original pixels supply detail, selected features control where it is retained. This is not a feature-only inverse. |

## Initial priorities

The most directly relevant hypotheses for Bora are 06 (different evidence for
each boundary), 10 (context and detail), 13 (semantic barriers), 14 (native image
geometry) and 20 (retain relevant detail). Experiment 01 is the direct test of
the proposed Virchow2 reversal. Its visual fidelity must be assessed honestly;
Virchow2 is an encoder and provides no native RGB decoder.

Each idea gets one primary preset fixed before fine-mask evaluation. Alternative
seeds, feature counts and regularization settings are diagnostics within that
idea, not extra numbered experiments. An idea that fails is recorded as failed,
not silently replaced with a similar decoder to reach twenty results.

## Shared data and resolution

Use the same TIGER breast tissue **ROI**, not a full WSI:
`tissue-cells--114S_[33188, 13944, 34412, 15110]`, 1224 × 1166 pixels.
The provider-reported sampling is approximately 0.5 µm/px, corresponding to
612 × 583 µm. Objective magnification, numerical aperture and optical resolving
power have not been established. Pixel sampling is not an optical-resolution
measurement.

Native Virchow2 inference uses overlapping 224 × 224 windows, about 112 × 112 µm.
Each 14 × 14 token samples about 7 × 7 µm. Native-resolution output files do not
imply native-pixel semantic information. Experiments using H&E to supply finer
geometry must disclose that source. Dense spatial tokens are retained; a single
pooled tile vector cannot localize an internal tissue boundary.

The checkpoint is `paige-ai/Virchow2`, revision
`3158645804b69e3f3bc4439d4116edddf0840a72`, weights SHA-256
`8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b`.
Model files and derived images remain outside Git.

## Selection and fair assessment

Select features from the interiors of the existing 90-pixel coarse tiles.
Balance all five nonzero classes: invasive tumor, tumor-associated stroma,
healthy glands, inflamed stroma and rest. Include low activation as evidence;
features strongly expressed in only one common class cannot define the whole
representation. Report selected dimensions/directions and their contribution
to each class and each difficult class pair. Latent dimensions are not markers.

Feature selection and scaling belong inside each training fold for reported
held-out discrimination. Spatially grouped tiles provide the observational
units, not millions of correlated pixels. One ROI and overlapping encoder
contexts do not establish patient-level generalization. The coarse tile mask
was derived from expert reference labels; this is controlled refinement from
coarse annotation, not an independently predicted tissue classifier benchmark.

Freeze outputs before loading the fine reference for scoring. Compare against
the identical coarse mask and ordinary H&E Bora with fixed settings. Use all
classes' overlap, boundary accuracy and distances in µm, with special attention
to the weakest class. Reconstruction fidelity and attractive colours are not
evidence of improved segmentation.

For every successful idea show the original H&E, actual input representation,
semi-transparent coarse mask, fine reference overlay and Bora result overlay,
with fixed colours, scale bars and identical full-field/crop coordinates. Keep
reconstruction, feature map, retrieved mosaic and hybrid labels visible. Show
failures and uncertainty alongside successful images.

Primary source: [Virchow2 model card](https://huggingface.co/paige-ai/Virchow2).

## Reproduce the experiment

Run from the repository root in a dedicated environment, with `PYTHONPATH=src`.
Install this directory's requirements and the repository's ordinary dependencies.
The source folder must contain the unchanged TIGER ROI `image.original.png`,
`tile_90/coarse.tif`, metadata `results.json`, and, for final evaluation only,
`expert_reference.compact.tif`. No GigaTIME feature or marker output is read.
The directory name of the earlier pilot is historical; only its original H&E,
coarse labels, metadata and expert reference are used here.
Use a fresh, empty output directory for a new image or changed preset. The
historical per-window caches support resuming these fixed inputs; they are not
a general cache that validates arbitrary changes of source images or settings.

The released Virchow2 weights require approved access. Download the pinned
`config.json` and `model.safetensors` through the Hugging Face CLI. Do not put
tokens or checkpoint files in Git.

```bash
export PYTHONPATH=src
# Set SOURCE, RUN and WEIGHTS to the source, output and pinned checkpoint folders.
python -m experiments.virchow2_inverse.extract --source "$SOURCE" --model-dir "$WEIGHTS" --output "$RUN/features" --device cuda --batch-size 2
python -m experiments.virchow2_inverse.prepare --source "$SOURCE" --root "$RUN"
python -m experiments.virchow2_inverse.audit_encoder --source "$SOURCE" --root "$RUN" --model-dir "$WEIGHTS"
python -m experiments.virchow2_inverse.gpu_first_run --source "$SOURCE" --root "$RUN" --model-dir "$WEIGHTS" --steps 32
python -m experiments.virchow2_inverse.gpu_ideas --source "$SOURCE" --root "$RUN" --model-dir "$WEIGHTS" --steps 64 --modes 2
python -m experiments.virchow2_inverse.cpu_ideas --source "$SOURCE" --root "$RUN"
python -m experiments.virchow2_inverse.finalize --source "$SOURCE" --root "$RUN"
python -m experiments.virchow2_inverse.review_outputs --root "$RUN"
python -m experiments.virchow2_inverse.audit_results --root "$RUN"
```

`gpu_first_run.py` preserves the exact first implementation used for ideas 01
and 03 and the initial, restrictive mask trial. `gpu_ideas.py` implements the
logged mask-objective revision, a smaller mask learning rate and best-iterate
retention. These are recorded diagnostics inside idea 02, not additional ideas.
Ideas 01 and 03 use 32 steps/window; the primary idea 02 uses 64. Budgets are
exploratory, with no claim of converged or globally optimal solutions.

CUDA memory is bounded to 40% per process and CPU thread pools to two. Do not run
two gradient jobs at once. GPU stages checkpoint per-window outputs so failures
can be diagnosed. The full feature extraction and all numerical intermediates
stay in the output directory. The gallery and image panels are viewable offline.
