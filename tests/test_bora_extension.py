import json
from pathlib import Path
import stat
import zipfile

import numpy as np
from PIL import Image
import pytest
import tifffile

from experiments.bora_extension import plan
from experiments.bora_extension.acquire_ocelot import check_members
from experiments.virchow2_inverse.common import digest


@pytest.mark.parametrize("name", ["../x", "/x", "a\\b", "C:/x"])
def test_rejects_unsafe_archive_names(name):
    with pytest.raises(ValueError):
        check_members([zipfile.ZipInfo(name)])


def test_rejects_archive_links_and_duplicates():
    info = zipfile.ZipInfo("link")
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(ValueError):
        check_members([info])
    with pytest.raises(ValueError):
        check_members([zipfile.ZipInfo("x"), zipfile.ZipInfo("x")])
    check_members([zipfile.ZipInfo("data/"), zipfile.ZipInfo("data/mask.png")])


def test_global_overlap_is_slide_specific_and_checks_bbox():
    row = dict(dataset="tiger", id="source_[100, 200, 600, 800]",
               slide_id="123A", width=500, height=600)
    assert plan.global_rect(row, [20, 40, 100, 200]) == ("123A", (120, 240, 220, 440))
    assert plan.intersects((0, 0, 10, 10), (9, 9, 20, 20))
    assert not plan.intersects((0, 0, 10, 10), (10, 0, 20, 10))
    with pytest.raises(ValueError):
        plan.global_rect({**row, "width": 499}, [0, 0, 100, 100])


def test_ocelot_tissue_spacing_uses_resized_axes_not_slide_mpp():
    from experiments.bora_extension.prepare_ocelot import tissue_spacing
    sample = dict(mpp_x=.25, mpp_y=.26,
        tissue=dict(x_start=10, x_end=3210, y_start=20, y_end=3220,
                    resized_mpp_x=.78125, resized_mpp_y=.8125))
    np.testing.assert_allclose(tissue_spacing(sample), [.78125, .8125])
    sample["tissue"]["resized_mpp_x"] = .25
    with pytest.raises(ValueError):
        tissue_spacing(sample)


def source_pair(root, dataset, raw, ignore):
    folder = root / "source"
    folder.mkdir(exist_ok=True)
    rgb = np.full((*raw.shape, 3), [210, 130, 180], np.uint8)
    image, mask = folder / (dataset + ".png"), folder / (dataset + "_mask.png")
    Image.fromarray(rgb).save(image)
    Image.fromarray(raw).save(mask)
    return dict(dataset=dataset, id="sample", group_id="source-group",
        image=str(image.relative_to(root)), mask=str(mask.relative_to(root)),
        image_sha256=digest(image), mask_sha256=digest(mask), ignore_labels=ignore,
        height=raw.shape[0], width=raw.shape[1])


def test_unknown_spacing_and_dataset_specific_zero_are_not_fabricated(tmp_path, monkeypatch):
    # WSSS zero is tumor, whereas RINGS zero is a valid negative target.
    raw = np.tile(np.repeat(np.arange(4, dtype=np.uint8), 90), (360, 1))
    wsss = source_pair(tmp_path, "wsss4luad", raw, [3])
    # Different image bytes avoid treating these synthetic datasets as duplicates.
    rings = source_pair(tmp_path, "rings-tumor", np.zeros((360, 360), np.uint8), [])
    image = tmp_path / rings["image"]
    Image.new("RGB", (360, 360), (190, 120, 140)).save(image)
    rings["image_sha256"] = digest(image)
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "PLAN.json").write_text(json.dumps(dict(samples=[])))
    monkeypatch.setattr(plan, "ADDITIONS", {"wsss4luad": 1, "rings-tumor": 1})
    monkeypatch.setattr(plan, "read_rows", lambda data, dataset: [wsss if dataset == "wsss4luad" else rings])
    root = tmp_path / "extension"
    record = plan.prepare(tmp_path, previous, root)
    assert record["all_requested_prepared"]
    for sample in record["samples"]:
        assert sample["native_mpp_xy"] is None
        assert sample["working_mpp_xy"] is None
        assert sample["fov_um_xy"] is None
        assert not sample["physical_metrics_allowed"]
    truth = tifffile.imread(root / "cases/wsss4luad_01/reference.native.tif")
    np.testing.assert_array_equal(truth[0], np.repeat([1, 2, 3, 0], 90))
    assert record["samples"][0]["class_names"][1] == "tumor epithelium"
    rings_truth = tifffile.imread(root / "cases/rings-tumor_01/reference.native.tif")
    assert np.all(rings_truth == 1)
    assert record["samples"][1]["single_coarse_class"]
    assert digest(previous / "PLAN.json") == record["previous_plan_sha256"]
    with pytest.raises(FileExistsError):
        plan.prepare(tmp_path, previous, root)
