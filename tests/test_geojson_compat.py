import json

import pytest

from bora.geojson_compat import enrich_wsitools_geojson


def test_annotation_metadata_retains_every_class(tmp_path):
    metrics = tmp_path / "metrics_cpp.json"
    metrics.write_text(json.dumps({"classes": [
        {"value": 0, "name": "background", "color": [0, 0, 0]},
        {"value": 1, "name": "stroma", "color": [69, 117, 180]},
        {"value": 2, "name": "normal", "color": [26, 152, 80]},
    ]}))
    vector = tmp_path / "refined.geojson"
    feature = lambda value: {"type": "Feature", "geometry": {"type": "Polygon",
        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
        "properties": {"value": value, "classification": f"group_{value}"}}
    vector.write_text(json.dumps({"type": "FeatureCollection",
                                  "features": [feature(1), feature(2)]}))
    report = enrich_wsitools_geojson(vector, metrics, 200, 100)
    data = json.loads(vector.read_text())
    assert report == {"classes": [1, 2], "features": 2}
    assert data["features"][0]["properties"]["classification"] == {
        "name": "stroma", "color": "#4575B4"}
    assert data["features"][1]["properties"]["wsiTools"]["slide_width"] == 200
    assert all(f["properties"]["objectType"] == "annotation"
               for f in data["features"])


def test_missing_annotation_class_fails_before_replacing_file(tmp_path):
    metrics = tmp_path / "metrics_cpp.json"
    metrics.write_text(json.dumps({"classes": [
        {"value": 1, "name": "a", "color": [1, 2, 3]},
        {"value": 2, "name": "b", "color": [4, 5, 6]},
    ]}))
    vector = tmp_path / "refined.geojson"
    original = json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": None, "properties": {"value": 1}}]})
    vector.write_text(original)
    with pytest.raises(ValueError, match="missing classes"):
        enrich_wsitools_geojson(vector, metrics, 10, 10)
    assert vector.read_text() == original
