# Relevance of Lv et al. (2026) to Bora

Source: Jiaqi Lv et al., *Computationally Efficient Pathology Segmentation using Knowledge Distillation from Foundation Models*, arXiv:2609.03947v1, 3 September 2026. https://arxiv.org/pdf/2609.03947

## What the paper does

The authors train a Virchow2-based pixel segmentation teacher, including a LoRA-adapted version. The teacher produces soft pixel predictions and image embeddings for unlabeled patches. Response and feature distillation transfer these signals to SegFormerB5 or EfficientV2M-UNet students. The students are then fine-tuned against pixel-level annotations. Evaluation covers PUMA, IGNITE, BEETLE, and blood vessel segmentation. Their principal accuracy metric is Dice, and their efficiency benchmark is patch throughput on L4 and RTX3060 GPUs. The EUNet-M student reaches about ten times the throughput of a Virchow2-based network on the RTX3060 under their 448-by-448-patch benchmark.

## Overlap and distinction

Both projects seek accurate and efficient pixel-level tissue segmentation of H&E images, including multiple tissue classes. Both draw motivation from pathology foundation models and expensive whole-slide inference.

Bora's technical question is different: given an existing categorical UNI2-clustered tile mask and the original slide, can it adjust **every tissue-to-tissue interface** at native pixel resolution while retaining the supplied regions? It is currently an image-guided, training-free refinement and vector-export pipeline. Lv et al. instead learn a pixel segmentation model from supervised labels and teacher-generated pseudo-labels. Their inference does not take an existing tile segmentation as a constraint. The paper neither tests a tile-boundary correction method nor reports a tissue-to-tissue contour metric; Dice over regions can remain high when a boundary retains visible square tile steps.

The paper therefore overlaps in application and in the broad foundation-model setting, but it does not appear to anticipate Bora's specific input-output problem or current boundary-competition algorithm. This is a technical comparison, not a patent novelty opinion.

## Useful implications for Bora

1. A trained student's soft class probabilities could become an image-derived unary cost within Bora's symmetric editable boundary band. The fixed class cores and label-preservation rules would still come from the input mask. This is a future model-assisted comparison, not part of the current training-free cycles.
2. Compare full-slide processing time and peak memory, including vector export, in addition to accuracy. The paper's patch-per-second GPU results are not directly comparable to Bora's full-slide wall time.
3. Keep class IoU/Dice for context but choose candidates using tissue-to-tissue border-band accuracy and IoU, narrow-tolerance contour F1, label retention, and false tile-imprint measures. This directly measures Bora's central claim and addresses a limitation the paper acknowledges for Dice-only evaluation.
4. Four Visium slides are suitable for algorithm debugging but repeated selection on their ground truths risks overfitting. Reserve different slides for a held-out evaluation before making general performance claims.

## Paper limitations relevant here

The authors acknowledge dependence on task-matched unlabeled data, transfer of teacher errors, and use of Dice without downstream utility measures. Their throughput experiment measures 448-by-448-patch GPU inference and omits full-slide stitching, input/output, and GeoJSON export. Those results establish an efficiency advantage for their students under their benchmark, not a direct speed ranking against Bora.
