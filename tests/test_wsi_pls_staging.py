"""A published WSI can gain PLS names only from a bound Benchmark 1 model."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from bora.region_identity import geometry_digest


SCRIPT = (Path(__file__).parents[1] / "migration" /
          "benchmark1-pls-bridge-2026-11-01-v1" / "stage_wsi_prediction.py")
spec = importlib.util.spec_from_file_location("stage_wsi_prediction", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def test_stages_prediction_without_changing_polygon_or_group_id(tmp_path):
    frozen = json.loads(module.FROZEN.read_text())
    site = tmp_path / "site"
    polygon = {"type": "Polygon", "coordinates": [[[0, 0], [2, 0], [2, 2],
                                                      [0, 2], [0, 0]]]}
    original = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"value": 1, "name": "group_01"},
         "geometry": polygon}]}
    geo = write(site / "visium-hd" / "case" / "original.geojson", original)
    page = site / "visium-hd" / "case.html"
    page.write_text("A fastPLS tissue-name assignment is not available for this saved result."
                    "<h2>Full-slide overview</h2>")
    record = {"id": "visium-hd-case", "section": "visium-hd",
              "page": "visium-hd/case.html#inspect-geojson",
              "geometry": "visium-hd/case/original.geojson",
              "stored_geojson_sha256": module.sha(geo),
              "decoded_geojson_sha256": module.sha(geo), "source_feature_count": 1,
              "fastpls_tissue_names_available": False,
              "groups": [{"value": 1, "name": "Group 01", "color": "#ffffff"}]}
    write(site / "geojson-inspectors.json", {"records": [record]})
    model = tmp_path / "model.npz"
    model.write_bytes(b"model for bound test")
    accepted = {"status": "ACCEPTED", "training_set_sha256": module.sha(module.FROZEN),
                "training_images": frozen["admitted_training_images"],
                "training_groups": frozen["class_eligible_groups"],
                "fit_scope": "Benchmark 1 admitted training images only",
                "model_path": str(model), "model_sha256": module.sha(model)}
    receipt = write(tmp_path / "accepted.json", accepted)
    classified = tmp_path / "classified"
    named = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"source_group_id": 1,
         "predicted_tissue_id": 3, "predicted_tissue": "stroma"},
         "geometry": polygon}],
        "tissue_identity_model_sha256": accepted["model_sha256"],
        "tissue_identity_status": "unvalidated_model_transfer"}
    named_file = write(classified / "tissue-annotations.geojson", named)
    report = {"format": "bora-region-identity-v1", "complete": True,
              "geometry_unchanged": True, "group_count": 1,
              "geometry_sha256": geometry_digest(original["features"]),
              "input_sha256": {"model": accepted["model_sha256"]},
              "output_sha256": {"tissue-annotations.geojson": module.sha(named_file)},
              "biological_accuracy_qualified": False,
              "groups": [{"source_group_id": 1, "prediction_id": 3,
                          "prediction_name": "stroma", "physical_area_um2": 2000000}]}
    write(classified / "report.json", report)
    staged = tmp_path / "staged"
    result = module.stage(site, "visium-hd-case", classified, receipt, staged)
    assert result["status"] == "STAGED_PLS_WSI"
    new_record = json.loads((staged / "geojson-inspectors.json").read_text())["records"][0]
    assert new_record["fastpls_tissue_names_available"] is True
    with gzip.open(staged / new_record["geometry"], "rt") as stream:
        updated = json.load(stream)
    assert updated["features"][0]["geometry"] == polygon
    assert updated["features"][0]["properties"]["value"] == 1
    assert updated["features"][0]["properties"]["predicted_tissue"] == "stroma"
    assert "PLS tissue-type predictions" in (staged / "visium-hd/case.html").read_text()
    # The original web result stays untouched until staged artifacts are reviewed.
    assert module.sha(geo) == record["stored_geojson_sha256"]
    report["input_sha256"]["model"] = hashlib.sha256(b"another model").hexdigest()
    write(classified / "report.json", report)
    with pytest.raises(ValueError, match="another model"):
        module.stage(site, "visium-hd-case", classified, receipt, tmp_path / "bad")
