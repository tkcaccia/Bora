import json
import numpy as np
import pytest
import tifffile
from PIL import Image
from rasterio.features import rasterize
from bora.pipeline import Case, prepare, run, load_case, expand_tile_labels, ENCODER_REVISION, ENCODER_SHA256
from bora.classifier import NeighborContextClassifier
from bora.cli import parser, train_file


def case():
    rng = np.random.default_rng(21)
    h = w = 360
    coarse = np.ones((h, w), np.uint8); coarse[:, 180:] = 2
    he = rng.integers(40, 230, (h, w, 3), dtype=np.uint8)
    rgb = np.zeros_like(he); rgb[:, :180] = [220, 40, 70]; rgb[:, 180:] = [40, 180, 70]
    tokens = rng.normal(size=(26, 26, 1280)).astype(np.float32)
    tokens[:, 13:, :32] += 3
    tissue = np.ones((h, w), bool); tissue[:3] = False
    return Case(he, coarse, tissue, tokens, np.array([.5, .5]), np.array([.5, .5]), tissue, rgb=rgb)


def test_full_native_pipeline_and_vector_roundtrip(tmp_path):
    c = case(); data = prepare(c)
    rng = np.random.default_rng(7)
    x = np.tile(data["descriptors"], (20, 1)) + rng.normal(0, .01, (40, 2560))
    y = np.tile([1, 2], 20)
    model = NeighborContextClassifier().fit(x, y, {1: "tumor", 2: "stroma"})
    output = tmp_path / "result"
    report = run(c, model, output)
    vector = json.loads((output / "mask.geojson").read_text())
    pairs = [(f["geometry"], f["properties"]["value"]) for f in vector["features"]]
    raster = rasterize(pairs, out_shape=(360, 360), fill=0, dtype=np.uint8)
    np.testing.assert_array_equal(raster, tifffile.imread(output / "mask.ome.tif"))
    assert report["classifier"] == "Neighbor context"
    assert report["fine_annotations_read"] is False and report["native_head_count"] == 1
    assert vector["final_native_pixel_region_cleanup"]["positive_polygon_parts_below_threshold"] == 0
    with pytest.raises(FileExistsError): run(c, model, output)


@pytest.mark.parametrize("bad", ["mpp", "fov", "small", "single", "tokens"])
def test_bad_calibration_or_insufficient_tissue_fields_are_rejected(bad):
    c = case()
    if bad == "mpp": c.mpp_xy = [1., 1.]
    if bad == "fov": c.native_mpp_xy = [.25, .5]
    if bad == "small":
        c.image = c.image[:200]; c.coarse = c.coarse[:200]; c.tissue = c.tissue[:200]
    if bad == "single": c.coarse[:] = 1
    if bad == "tokens": c.tokens = c.tokens[:-1]
    with pytest.raises(ValueError): c.validate()


def test_reference_paths_are_not_accepted(tmp_path):
    p = tmp_path / "case.json"
    p.write_text(json.dumps({"reference_mask": "hidden.tif"}))
    with pytest.raises(ValueError, match="Unknown case fields"): load_case(p)


def test_training_refuses_validation_descriptors_before_opening_them(tmp_path):
    p = tmp_path / "training.json"
    p.write_text(json.dumps({"format": "bora-training-v1", "classes": {"1": "tumor"},
        "cases": [{"split": "test", "source_group": "patient-1", "descriptors": "never-open.npz"}]}))
    with pytest.raises(ValueError, match="split=train"): train_file(p, tmp_path / "model.npz")


def test_cli_has_one_pipeline_and_no_classifier_selector():
    p = parser()
    assert "Neighbor context" in p.description
    with pytest.raises(SystemExit): p.parse_args(["refine", "case.json", "--model", "m.npz", "--output", "result", "--classifier", "opls"])


def test_tile_grid_expands_without_changing_anonymous_group_ids():
    grid = np.array([[1, 2], [3, 0], [3, 0]], np.uint8)
    coarse = expand_tile_labels(grid, (181, 179), [90, 90])
    assert coarse.shape == (181, 179)
    assert coarse[0, 0] == 1 and coarse[0, 100] == 2
    assert coarse[100, 0] == 3 and coarse[180, 178] == 0
    with pytest.raises(ValueError, match="grid shape"):
        expand_tile_labels(grid, (271, 179), [90, 90])


def test_case_manifest_accepts_tile_grid_and_preserves_calibration(tmp_path):
    original = case()
    Image.fromarray(original.image).save(tmp_path / "he.png")
    Image.fromarray(original.rgb).save(tmp_path / "evidence.png")
    tifffile.imwrite(tmp_path / "tissue.tif", original.tissue.astype(np.uint8))
    np.save(tmp_path / "tokens.npy", original.tokens)
    np.save(tmp_path / "clusters.npy", np.tile([1, 1, 2, 2], (4, 1)).astype(np.uint8))
    spec = dict(image="he.png", tile_labels="clusters.npy", tile_size_yx=[90, 90],
        tissue="tissue.tif", tokens="tokens.npy", evidence_rgb="evidence.png",
        mpp_xy=[.5, .5], encoder_revision=ENCODER_REVISION, encoder_sha256=ENCODER_SHA256)
    path = tmp_path / "case.json"
    path.write_text(json.dumps(spec))
    loaded = load_case(path)
    np.testing.assert_array_equal(loaded.coarse, original.coarse)
    assert len(prepare(loaded)["group_ids"]) == 2
