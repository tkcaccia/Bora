# Eight-class supervised Virchow2 ridge head

These are copies of the executed Chiamaka sources for the training-only model search, final fit, 87-image development evaluation, contour finishing, black-and-white response maps and comparison figures. The data, frozen manifests and fitted model weights are retained separately under their access terms; a checkout alone cannot reproduce the images.

The prespecified search compared four CPU heads on 139,736 sampled tokens from the 397 retained training images, with two source-group-disjoint folds. Each candidate received all 1,280 Virchow2 channels before feature selection. The selected head takes 32 standardized contrast features per class, unions the indices across eight broad classes and fits class-weighted multiclass ridge regression with alpha 10. The final union has 203 unique channels. The frozen rule chose the fastest candidate within 0.005 case macro IoU of the best four-dataset training score.

The model was fitted on all 397 training images and evaluated once on 87 previously inspected development images. Mean expert-present case macro IoU was 0.701618 across all 87 and 0.615325 as an equal-dataset mean across the four multiclass datasets. These are development results, not independent-test performance. The 19 PEsO cases have one scored class each. The former KODAMA–Bora spatial masks were not recoverable for a new broad-label group refit; that route is separate from this supervised head.

The executable scripts contain local Chiamaka paths to frozen input receipts. Run them only after supplying verified images, masks, full-channel features and those receipts. The public gallery contains outputs, not the expert source masks or encoder weights.
