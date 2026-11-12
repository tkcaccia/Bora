"""Completed whole-slide outputs keep final assets and hash evidence."""

import hashlib
import json

import pytest

from bora.intermediate_retention import retire_polished_work_arrays


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_output(tmp_path):
    final = ("BORA_FULL_SLIDE.mask-0p5um.tif", "BORA_FULL_SLIDE.working.geojson")
    keep = ("support.npy", "global-prepolish-cleaned-mask.npy",
            "contour-smoothed-mask.npy", "global-cleaned-mask.npy")
    work = ("diffusion-1um.npy", "smoothed-mask.npy", "global-rasterized-mask.npy",
            "polished-area-clean-0.npy", "global-raster-clean-0.npy",
            "global-raster-repaired-0.npy")
    for name in final + keep + work:
        (tmp_path / name).write_bytes(name.encode())
    contour = dict(prepolish_mask_sha256=digest(tmp_path / keep[1]),
                   smoothed_mask_sha256=digest(tmp_path / keep[2]),
                   final_cleaned_mask_sha256=digest(tmp_path / keep[3]))
    (tmp_path / "CONTOUR_POLISH_COMPLETE.json").write_text(json.dumps(contour))
    complete = dict(status="COMPLETE", files_sha256={name: digest(tmp_path / name) for name in final})
    (tmp_path / "FULL_SLIDE_COMPLETE.json").write_text(json.dumps(complete))
    return complete, final, keep, work


def test_retirement_preserves_final_and_removes_all_work_masks(tmp_path):
    complete, final, keep, work = fixture_output(tmp_path)
    receipt = retire_polished_work_arrays(tmp_path, complete)
    assert receipt["status"] == "RETIRED"
    assert set(receipt["removed"]) == set(work + keep)
    assert all((tmp_path / name).exists() for name in final)
    assert all(not (tmp_path / name).exists() for name in work + keep)
    assert (tmp_path / "INTERMEDIATE_MASKS_RETIRED.json").exists()


def test_bad_final_hash_prevents_any_retirement(tmp_path):
    complete, _, _, work = fixture_output(tmp_path)
    (tmp_path / "BORA_FULL_SLIDE.mask-0p5um.tif").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Final full-slide asset differs"):
        retire_polished_work_arrays(tmp_path, complete)
    assert all((tmp_path / name).exists() for name in work)


def test_in_memory_tuple_matches_persisted_json_array(tmp_path):
    complete, _, _, _ = fixture_output(tmp_path)
    complete["measured_mpp_xy"] = (0.5, 0.5)
    (tmp_path / "FULL_SLIDE_COMPLETE.json").write_text(json.dumps(complete))
    assert retire_polished_work_arrays(tmp_path, complete)["status"] == "RETIRED"
