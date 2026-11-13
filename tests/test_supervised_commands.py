"""Small CPU contracts for the new supervised package interface."""
import hashlib
import json

import numpy as np
from PIL import Image

from bora import supervised
from bora.cli import main


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fit_predict_evaluate_verify(tmp_path, capsys):
    rng = np.random.default_rng(13)
    labels = np.tile(np.arange(1, 9, dtype=np.uint8), 8)
    features = rng.normal(size=(len(labels), 1280)).astype(np.float16)
    for cls in range(1, 9):
        features[labels == cls, cls-1] += 7
    x_path, y_path = tmp_path / "training.npy", tmp_path / "labels.npy"
    np.save(x_path, features)
    np.save(y_path, labels)
    fit_manifest = tmp_path / "fit.json"
    fit_manifest.write_text(json.dumps({"format": "bora-supervised-fit-v1", "split": "train",
        "source_full_channels": 1280,
        "features": {"path": x_path.name, "sha256": sha(x_path)},
        "labels": {"path": y_path.name, "sha256": sha(y_path)}}))
    model = tmp_path / "model.npz"
    main(["fit", str(fit_manifest), "--output", str(model)])
    assert supervised.verify(model)["selected_channels"] >= 8
    assert json.loads(capsys.readouterr().out)["status"] == "COMPLETE"

    grid = features[:16].reshape(4, 4, 1280).copy()
    grid[0, 0, :] = np.nan
    grid_path = tmp_path / "grid.npy"
    np.save(grid_path, grid)
    pred_manifest = tmp_path / "predict.json"
    pred_manifest.write_text(json.dumps({"format": "bora-supervised-predict-v1",
        "model": {"path": model.name, "sha256": sha(model)},
        "features": {"path": grid_path.name, "sha256": sha(grid_path)},
        "roi": {"working_size_wh": [56, 56], "display_size_wh": [56, 56],
                "allowed_classes": list(range(1, 9)), "polish": False}}))
    folder = tmp_path / "prediction"
    main(["predict", str(pred_manifest), "--output", str(folder)])
    assert supervised.verify(folder)["files"] == 3
    assert json.loads((folder / "PREDICTION_COMPLETE.json").read_text())["nearest_imputed_token_count"] == 1
    mask = np.asarray(Image.open(folder / "class-mask.png"))
    assert mask.shape == (56, 56) and np.isin(mask, range(1, 9)).all()

    target = tmp_path / "target.png"
    Image.fromarray(mask, "L").save(target)
    eval_manifest = tmp_path / "evaluate.json"
    eval_manifest.write_text(json.dumps({"format": "bora-supervised-evaluate-v1",
        "is_independent_test": False, "cases": [{"id": "fixture", "spacing_yx_um": [0.5, 0.5],
        "prediction": {"path": "prediction/class-mask.png", "sha256": sha(folder / "class-mask.png")},
        "target": {"path": target.name, "sha256": sha(target)}}]}))
    scores = tmp_path / "scores.json"
    main(["evaluate", str(eval_manifest), "--output", str(scores)])
    assert supervised.verify(scores)["status"] == "VERIFIED"
    reported = json.loads(scores.read_text())
    assert reported["mean_case_macro_iou"] == 1
    assert reported["mean_symmetric_region_covering"] == 1
    assert reported["mean_boundary_f1_5um"] == 1
    capsys.readouterr()
    main(["verify", str(scores)])
    assert json.loads(capsys.readouterr().out)["status"] == "VERIFIED"


def test_fit_rejects_incomplete_encoder_cache(tmp_path):
    x, y = tmp_path / "features.npy", tmp_path / "labels.npy"
    np.save(x, np.zeros((16, 203), np.float16))
    np.save(y, np.tile(np.arange(1, 9), 2))
    manifest = tmp_path / "fit.json"
    manifest.write_text(json.dumps({"format": "bora-supervised-fit-v1", "split": "train",
        "source_full_channels": 1280,
        "features": {"path": x.name, "sha256": sha(x)},
        "labels": {"path": y.name, "sha256": sha(y)}}))
    import pytest
    with pytest.raises(ValueError, match="N×1280"):
        supervised.fit(manifest, tmp_path / "model.npz")


def test_evaluate_rejects_unregistered_frames(tmp_path):
    import pytest
    prediction, target = tmp_path / "prediction.png", tmp_path / "target.png"
    Image.fromarray(np.ones((20, 20), np.uint8), "L").save(prediction)
    Image.fromarray(np.ones((21, 20), np.uint8), "L").save(target)
    manifest = tmp_path / "eval.json"
    manifest.write_text(json.dumps({"format": "bora-supervised-evaluate-v1", "cases": [{
        "id": "unregistered", "spacing_yx_um": [0.5, 0.5],
        "prediction": {"path": prediction.name, "sha256": sha(prediction)},
        "target": {"path": target.name, "sha256": sha(target)}}]}))
    with pytest.raises(ValueError, match="frames must match"):
        supervised.evaluate(manifest, tmp_path / "scores.json")


def test_peso_binary_head_is_distinct_and_uses_fixed_threshold(tmp_path):
    rng = np.random.default_rng(4)
    labels = np.tile(np.array([2, 4], np.uint8), 24)
    x = rng.normal(size=(len(labels), 1280)).astype(np.float32)
    x[labels == 4, 0] += 5
    feature, target = tmp_path / "x.npy", tmp_path / "y.npy"
    np.save(feature, x)
    np.save(target, labels)
    manifest = tmp_path / "train.json"
    manifest.write_text(json.dumps({"format": "bora-supervised-fit-v1", "split": "train",
        "method": "peso-binary-ridge64", "source_full_channels": 1280,
        "features": {"path": feature.name, "sha256": sha(feature)},
        "labels": {"path": target.name, "sha256": sha(target)}}))
    model = tmp_path / "peso.npz"
    supervised.fit(manifest, model)
    with np.load(model, allow_pickle=False) as arrays:
        assert np.array_equal(arrays["classes"], [2, 4])
        assert len(arrays["selected"]) == 64
    grid = tmp_path / "grid.npy"
    np.save(grid, x[:16].reshape(4, 4, 1280))
    support = tmp_path / "support.png"
    support_mask = np.ones((56, 56), np.uint8)
    support_mask[:4] = 0
    Image.fromarray(support_mask, "L").save(support)
    pred_manifest = tmp_path / "prediction.json"
    pred_manifest.write_text(json.dumps({"format": "bora-supervised-predict-v1",
        "model": {"path": model.name, "sha256": sha(model)},
        "features": {"path": grid.name, "sha256": sha(grid)},
        "roi": {"working_size_wh": [56, 56], "display_size_wh": [56, 56],
                "allowed_classes": [2, 4], "polish": False,
                "support_mask": {"path": support.name, "sha256": sha(support)}}}))
    result = tmp_path / "result"
    supervised.predict(pred_manifest, result)
    mask = np.asarray(Image.open(result / "class-mask.png"))
    assert set(np.unique(mask)) == {0, 2, 4}
    assert supervised.verify(result)["status"] == "VERIFIED"


def test_wsi_projection_keeps_reviewed_background(tmp_path):
    from shapely.geometry import box
    from bora import supervised_wsi
    scores = np.zeros((8, 20, 20), np.float32)
    scores[0, :, :10] = 4
    scores[1, :, 10:] = 4
    tissue = [box(28, 28, 252, 252)]
    classes, selected = supervised_wsi.select_classes(scores, tissue, .02)
    assert classes == [1, 2]
    assert selected["excluded_class_ids"] == [3, 4, 5, 6, 7, 8]
    support = supervised_wsi.reviewed_support(tissue, (280, 280), tmp_path / "support.dat")
    projected = supervised_wsi.project_mask(scores, support, classes, tmp_path / "raw.dat")
    assert np.array_equal(projected > 0, support > 0)
    assert set(np.unique(projected)) == {0, 1, 2}
    repaired, _ = supervised_wsi.area_repair(projected, support, tmp_path / "clean.dat", (.5, .5), 400)
    assert np.array_equal(repaired > 0, support > 0)


def test_public_full_slide_function_exports_clean_native_vectors(tmp_path):
    import rasterio
    from rasterio.transform import Affine
    from shapely.geometry import box, mapping
    from bora import segment_full_slide
    from bora.supervised_wsi import verify_wsi

    model = tmp_path / "ridge.npz"
    coefficients = np.zeros((8, 2), np.float64)
    coefficients[0, 0] = coefficients[1, 1] = 1
    intercept = np.array([0, 0] + [-100] * 6, np.float64)
    np.savez(model, format="bora-supervised-ridge-v1",
             selected=np.array([0, 1], np.int32), mean=np.zeros(2),
             std=np.ones(2), coef=coefficients, intercept=intercept,
             classes=np.arange(1, 9), training_manifest_sha256="a" * 64,
             method="eight-class-ridge")
    grid = np.zeros((20, 20, 1280), np.float16)
    grid[:, :10, 0] = 10
    grid[:, 10:, 1] = 10
    features = tmp_path / "features.npy"
    np.save(features, grid)
    source = tmp_path / "calibrated.tif"
    with rasterio.open(source, "w", driver="GTiff", height=280, width=280,
                       count=3, dtype="uint8", transform=Affine.identity()) as dst:
        for band in range(1, 4):
            dst.write(np.full((280, 280), 180, np.uint8), band)
    support = tmp_path / "support.geojson"
    support.write_text(json.dumps({
        "coordinate_space": "original_WSI_native_pixel_edges",
        "native_origin_xy": [0, 0], "native_shape_yx": [280, 280],
        "native_mpp_xy": [.5, .5], "working_mpp_xy": [.5, .5],
        "working_pixel_to_native_scale_xy": [1, 1],
        "full_slide_input_identity": {"shape_yx": [280, 280]},
        "features": [{"geometry": mapping(box(10, 10, 270, 270))}]}))
    presence = tmp_path / "presence.json"
    presence.write_text(json.dumps({
        "status": "TRAINING_ONLY_RIDGE_CLASS_PRESENCE_RULE_FROZEN",
        "minimum_predicted_tissue_fraction": .02,
        "development_masks_opened": 0, "whole_slide_expert_masks_opened": 0}))
    ontology = tmp_path / "ontology.json"
    ontology.write_text(json.dumps({"classes": {
        str(i): {"name": f"class_{i}"} for i in range(1, 9)}}))
    manifest = tmp_path / "predict.json"
    manifest.write_text(json.dumps({
        "format": "bora-supervised-predict-v1",
        "model": {"path": model.name, "sha256": sha(model)},
        "features": {"path": features.name, "sha256": sha(features)},
        "wsi": {"calibrated_shape_yx": [280, 280],
                "source_frame_id": "synthetic:calibrated_0p5um",
                "source_image": {"path": source.name, "sha256": sha(source)},
                "support_geojson": {"path": support.name, "sha256": sha(support)},
                "ontology": {"path": ontology.name, "sha256": sha(ontology)},
                "class_presence_receipt": {"path": presence.name, "sha256": sha(presence)},
                "minimum_predicted_tissue_fraction": .02}}))
    output = tmp_path / "segmented"
    result = segment_full_slide(manifest, output)
    assert result["status"] == "VERIFIED"
    assert verify_wsi(output / "full-slide")["status"] == "VERIFIED"
    with rasterio.open(result["mask_tiff"]) as src:
        mask = src.read(1)
    assert set(np.unique(mask)) == {0, 1, 2}
    assert np.all(mask[:10] == 0) and np.all(mask[:, :10] == 0)
    native = json.loads((output / "full-slide/supervised-bora.native.geojson").read_text())
    assert native["coordinate_space"] == "original_WSI_native_pixel_edges"
    receipt = json.loads((output / "full-slide/FULL_SLIDE_VERIFIED.json").read_text())
    assert receipt["contour_smoothing"]["sigma_um"] == 6
    assert receipt["independent_geometry_audit"]["physical_minimum_um2"] == 2000
    assert receipt["independent_geometry_audit"]["native_geojson_roundtrip_different_pixels"] == 0
