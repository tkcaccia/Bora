import json

import pytest

from benchmark.image_types import classification_records, classify_image, export_classifications, summarize


def test_camelyon_full_slide_and_its_native_window_have_different_types():
    slide = {"dataset": "camelyon16-subset", "width": 97792, "height": 221184,
             "image_levels": [[97792, 221184], [48896, 110592]]}
    assert classify_image(slide, whole_slide=True)["image_type"] == "WSI"
    crop = {"dataset": "camelyon16-subset", "group_id": "tumor_058", "crop_x": 48384,
            "crop_y": 107075, "crop_level": 0, "width": 4096, "height": 4096}
    info = classify_image(crop)
    assert info["image_type"] == "ROI"
    assert info["parent_image"].endswith("/tumor_058.tif")
    assert info["parent_slide_id"] == "tumor_058"
    with pytest.raises(ValueError, match="complete-slide"):
        classify_image({**crop, "image_levels": [[4096, 4096]]}, whole_slide=True)
    with pytest.raises(ValueError, match="geometry"):
        classify_image({**slide, "width": 10}, whole_slide=True)


@pytest.mark.parametrize("dataset", ["tiger", "sicapv2", "bcss", "glas", "crag-candidate", "wsss4luad",
                                     "rings-glands", "rings-tumor", "cocahis-vista", "segpath-panck"])
def test_release_extent_takes_precedence_over_size_and_wsi_source_title(dataset):
    # Even a large tiled TIFF cannot become a full WSI through a size heuristic.
    row = {"dataset": dataset, "width": 100000, "height": 100000, "image": "labelled-WSI.tif"}
    assert classify_image(row)["image_type"] == "ROI"
    with pytest.raises(ValueError, match="complete-slide"):
        classify_image(row, whole_slide=True)


def test_unknown_release_is_not_silently_classified():
    with pytest.raises(ValueError, match="No verified"):
        classify_image({"dataset": "new-dataset", "width": 512, "height": 512})


def test_inventory_covers_weak_and_quarantined_images_without_double_counting_targets(tmp_path):
    prepared, metadata = tmp_path / "prepared", tmp_path / "metadata"
    prepared.mkdir()
    metadata.mkdir()
    for dataset in ("rings-glands", "rings-tumor"):
        row = {"dataset": dataset, "id": "sample", "image": "rings/one.png", "mask": f"{dataset}/one.png"}
        (prepared / f"{dataset}.jsonl").write_text(json.dumps(row) + "\n")
    paired = {"dataset": "wsss4luad", "id": "validation--one", "image": "wsss/one.png"}
    (prepared / "wsss4luad.jsonl").write_text(json.dumps(paired) + "\n")
    weak = {"id": "train-one", "split": "train", "source_shard": "raw/train.parquet"}
    (metadata / "wsss4luad-image-labels.jsonl").write_text(json.dumps(weak) + "\n")
    (metadata / "wsss4luad-mask-audit.json").write_text(json.dumps([
        {"id": "one", "image": "wsss/one.png", "status": "accepted"},
        {"id": "two", "image": "wsss/two.png", "status": "missing_dense_mask"},
    ]))
    before = {p.name: p.read_bytes() for p in prepared.iterdir()}
    rows = list(classification_records(tmp_path))
    summary = summarize(rows)
    assert summary["records"] == 5
    assert summary["unique_image_references_by_type"] == {"ROI": 4}
    assert summary["records_by_scope"] == {
        "image_level_only": 1, "prepared_annotation_pair": 3, "quarantined_dense_pair": 1}
    assert rows[-1]["annotation_status"] == "missing_dense_mask"
    assert export_classifications(tmp_path) == summary
    assert {p.name: p.read_bytes() for p in prepared.iterdir()} == before
