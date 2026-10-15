import numpy as np

from benchmark.describe import legend_rows
from benchmark.examples import select_examples


def test_tiger_examples_replace_cell_crops_with_tumor_stroma_tissue_region():
    def row(name, size, cohort, group, labels):
        return {"id": name, "dataset": "tiger", "width": size, "height": size,
                "cohort": cohort, "group_id": group, "raw_labels": labels}
    inputs = [row("large-bcss", 1400, "TCGA", "A", [1, 2]),
              row("first-independent", 1200, "non-TCGA", "B", [4, 7]),
              row("cell-crop", 129, "TCGA", "C", [2]),
              row("second-independent", 1224, "non-TCGA", "D", [0, 1, 2, 6])]
    selected = select_examples(inputs, 3)
    assert [r["id"] for r in selected] == ["large-bcss", "first-independent", "second-independent"]
    assert all(min(r["width"], r["height"]) >= 1000 for r in selected)


def test_legend_counts_native_pixels_and_preserves_distinct_tissue_ontologies():
    labels = {"bcss": {"labels": {"0": "outside_roi", "1": "tumor", "2": "stroma"}}}
    mask = np.array([[0, 1, 1], [2, 2, 2]], dtype=np.uint8)
    legend = legend_rows(mask, {"dataset": "bcss", "ignore_labels": [0]}, labels)
    assert [r["pixels"] for r in legend] == [1, 2, 3]
    assert legend[0]["ignored"] and legend[0]["percent_of_included_pixels"] is None
    assert legend[1]["name"] == "tumor" and legend[1]["percent_of_included_pixels"] == 40
    assert legend[2]["name"] == "stroma" and legend[2]["percent_of_included_pixels"] == 60


def test_wsss_zero_is_tumor_and_mismatched_masks_are_rejected():
    from benchmark.prepare_additional import validate_wsss_pair
    labels = {"wsss4luad": {"labels": {"0": "tumor", "1": "stroma", "3": "background"}}}
    mask = np.array([[0, 0, 1], [1, 3, 3]], dtype=np.uint8)
    assert validate_wsss_pair((3, 2), mask) is None
    assert validate_wsss_pair((2, 3), mask) == "image_mask_shape_mismatch"
    legend = legend_rows(mask, {"dataset": "wsss4luad", "ignore_labels": [3]}, labels)
    assert legend[0]["name"] == "tumor" and not legend[0]["ignored"]
    assert legend[0]["percent_of_included_pixels"] == 50
    assert legend[2]["ignored"]
