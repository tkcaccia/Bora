import json

import numpy as np
import pytest
import tifffile

from bora.annotations import read_annotation, group_targets, _class_map
from bora.classifier import NeighborContextClassifier
from bora.cli import train_file


def test_train_from_native_semantic_mask_excludes_mixed_group(tmp_path):
    rng = np.random.default_rng(42)
    ids = np.arange(1, 41, dtype=np.int64)
    x = rng.normal(size=(40, 2560)).astype(np.float64)
    x[:20, :12] += 2
    groups = np.broadcast_to(ids, (20, 40)).astype(np.uint16).copy()
    annotation = np.where(groups <= 20, 11, 12).astype(np.uint8)
    annotation[:10, 39] = 11  # No single reliable class for the last whole group.
    tifffile.imwrite(tmp_path / "groups.tif", groups)
    tifffile.imwrite(tmp_path / "expert.tif", annotation)
    np.savez_compressed(tmp_path / "descriptors.npz", x=x, group_ids=ids,
        area_pixels=np.full(40, 20),
        metadata=json.dumps({"format": "bora-neighbor-context-descriptors-v1"}))
    manifest = {"format": "bora-training-v1", "classes": {"1": "tumor", "2": "stroma"},
        "cases": [{"descriptors": "descriptors.npz", "groups": "groups.tif",
                   "annotation": "expert.tif", "annotation_values": {"11": 1, "12": 2},
                   "source_group": "patient-A", "split": "train"}]}
    (tmp_path / "training.json").write_text(json.dumps(manifest))
    report = train_file(tmp_path / "training.json", tmp_path / "model.npz")
    assert report["training_groups"] == 39 and report["excluded_groups"] == 1
    model = NeighborContextClassifier.load(tmp_path / "model.npz")
    assert len(model.y_) == 39 and set(model.y_) == {1, 2}
    with np.load(tmp_path / "model.npz", allow_pickle=False) as saved:
        provenance = json.loads(str(saved["metadata"].item()))["training_provenance"]
    assert provenance["cases"][0]["excluded_groups"][0]["reason"] == "mixed_tissue"
    assert provenance["cases"][0]["annotation_sha256"]


def test_geojson_annotation_uses_explicit_native_coordinates_and_class_mapping(tmp_path):
    collection = {"type": "FeatureCollection", "coordinate_space": "native_image_pixel_edges",
                  "native_shape_yx": [8, 10], "features": [
                      {"type": "Feature", "properties": {"classification": {"name": "Tumor"}},
                       "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [5, 0], [5, 8], [0, 8], [0, 0]]]},},
                      {"type": "Feature", "properties": {"classification": {"name": "Stroma"}},
                       "geometry": {"type": "Polygon", "coordinates": [[[5, 0], [10, 0], [10, 8], [5, 8], [5, 0]]]},}
                  ]}
    path = tmp_path / "expert.geojson"
    path.write_text(json.dumps(collection))
    annotation = read_annotation(path, (8, 10), {"Tumor": 1, "Stroma": 2},
        field="classification.name", coordinate_space="native_image_pixel_edges")
    assert np.all(annotation[:, :5] == 1) and np.all(annotation[:, 5:] == 2)
    groups = np.repeat(np.arange(1, 11, dtype=np.uint8)[None, :], 8, axis=0)
    kept, labels, excluded = group_targets(groups, np.arange(1, 11), annotation)
    assert kept.tolist() == list(range(1, 11)) and labels.tolist() == [1] * 5 + [2] * 5
    assert excluded == []
    with pytest.raises(ValueError, match="annotation_coordinate_space"):
        read_annotation(path, (8, 10), {"Tumor": 1, "Stroma": 2}, field="classification.name")
    collection["features"][1]["geometry"]["coordinates"][0][0][0] = 4
    collection["features"][1]["geometry"]["coordinates"][0][3][0] = 4
    collection["features"][1]["geometry"]["coordinates"][0][4][0] = 4
    path.write_text(json.dumps(collection))
    with pytest.raises(ValueError, match="overlap"):
        read_annotation(path, (8, 10), {"Tumor": 1, "Stroma": 2},
            field="classification.name", coordinate_space="native_image_pixel_edges")


def test_zero_can_be_a_real_tissue_class_in_mask(tmp_path):
    path = tmp_path / "expert.tif"
    tifffile.imwrite(path, np.array([[0, 1], [0, 1]], np.uint8))
    annotated = read_annotation(path, (2, 2), {"0": 1, "1": 2})
    np.testing.assert_array_equal(annotated, [[1, 2], [1, 2]])


@pytest.mark.parametrize("target", [0, 256, 1.5, True])
def test_annotation_class_map_rejects_targets_that_cannot_be_encoded(target):
    with pytest.raises(ValueError):
        _class_map({"Tumor": target}, {1: "tumor"})
