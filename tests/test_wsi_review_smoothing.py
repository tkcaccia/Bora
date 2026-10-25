"""Focused tests for exact-preserving Bora review smoothing."""
import json
from pathlib import Path

import pytest
import numpy as np
from shapely.geometry import Polygon, box, shape
from shapely.ops import unary_union

from bora.wsi_global_vector import (
    _agreement, _bounded_support_exchange, _parts, _require_large_parts,
)
from bora.wsi_review_smoothing import (
    _restore_small_complementary_pairs,
    choose_review_geometry,
)


def test_small_island_and_opposite_hole_are_restored_together():
    support = box(0, 0, 50, 50)
    exact_island = box(20, 20, 31, 31)  # 121 px², above 100
    shrunken_island = box(21, 20, 30, 31)  # 99 px², below 100
    exact = {1: support.difference(exact_island), 2: exact_island}
    candidate = {1: support.difference(shrunken_island), 2: shrunken_island}
    review, repairs = _restore_small_complementary_pairs(exact, candidate, 100)
    assert len(repairs) == 1
    assert review[2].equals(exact_island)
    assert review[1].equals(exact[1])
    assert review[1].intersection(review[2]).area == 0
    assert unary_union(list(review.values())).equals(support)
    assert all(_require_large_parts(value, 100) for value in review.values())


def test_background_touching_island_cannot_be_repaired_as_internal_hole():
    support = box(0, 0, 50, 50)
    exact_part = box(0, 0, 11, 11)
    candidate_part = box(0, 0, 9, 11)
    exact = {1: support.difference(exact_part), 2: exact_part}
    candidate = {1: support.difference(candidate_part), 2: candidate_part}
    with pytest.raises(ValueError, match="Cannot safely restore"):
        _restore_small_complementary_pairs(exact, candidate, 100)


def test_three_class_island_restores_only_its_unique_opposing_class():
    support = box(0, 0, 60, 50)
    right = box(40, 0, 60, 50)
    original = box(20, 20, 31, 31)
    shrunken = box(21, 20, 30, 31)
    exact = {1: support.difference(right).difference(original),
             2: original, 3: right}
    candidate = {1: support.difference(right).difference(shrunken),
                 2: shrunken, 3: right}
    review, repairs = _restore_small_complementary_pairs(exact, candidate, 100)
    assert repairs[0]["opposing_class_id"] == 1
    assert all(review[value].equals(exact[value]) for value in exact)
    assert unary_union(list(review.values())).equals(support)


def test_equal_tissue_loss_and_gain_are_reported_separately():
    confusion = np.zeros((256, 256), np.int64)
    confusion[0, 0] = 100
    confusion[1, 1] = 100
    confusion[2, 2] = 100
    confusion[1, 0] = 2
    confusion[0, 2] = 2
    agreement = _agreement(confusion, (1, 2))
    assert agreement["changed_tissue_area_fraction"] == 0
    assert agreement["new_background_in_exact_tissue"] == 2
    assert agreement["new_tissue_in_exact_background"] == 2
    assert not _bounded_support_exchange(agreement, 4., 202.,
        maximum_one_way_fraction=.005, maximum_total_fraction=.01)


def test_whole_boundary_mode_turns_stairs_diagonal_with_separate_support_guards():
    steps = [(100, 20)]
    for i in range(80):
        steps.extend([(101+i, 20+i), (101+i, 21+i)])
    support = Polygon([(20,20), *steps, (490,100), (490,490), (20,490)])
    assert support.is_valid
    exact = {1:support.intersection(box(0,0,256,512)),
             2:support.intersection(box(256,0,512,512))}
    review, selected, _ = choose_review_geometry(exact, [512,512], [1.,1.],
        "0"*64, minimum_area_um2=100., tolerances_um=(3., 1.5, 1.),
        simplify_outer_boundary=True)
    stats = selected["segment_stats_after"]
    assert stats["diagonal_segments"] > 0
    assert stats["axis_aligned_segments"] < .5 * \
        selected["segment_stats_before"]["axis_aligned_segments"]
    assert selected["review_coordinate_count"] < .5 * sum(
        len(p.exterior.coords) for g in exact.values() for p in _parts(g))
    agreement = selected["full_native_raster_agreement"]
    assert agreement["new_background_in_exact_tissue"] <= \
        agreement["exact_tissue_pixels"] * .0005
    assert agreement["new_tissue_in_exact_background"] <= \
        agreement["exact_tissue_pixels"] * .0005
    assert agreement["minimum_class_iou"] >= .995
    assert selected["maximum_changed_pixel_distance_um"] <= 5
    assert [len(_parts(review[v])) for v in (1,2)] == [1,1]


def test_pinned_pm016_r4_vectors_select_a_guarded_nonzero_review():
    root = Path(__file__).resolve().parents[2] / "Development Coordinator" / \
        "validation" / "pm016-recovered-bora-full-diagnostic-20261025" / "result"
    source = root / "mask.geojson"
    receipt = root / "GLOBAL_VECTOR_COMPLETE.json"
    if not source.is_file() or not receipt.is_file():
        pytest.skip("Read-only PM016 r4 vector fixture is not installed")
    document = json.loads(source.read_text())
    vector_receipt = json.loads(receipt.read_text())
    exact = {value: unary_union([shape(feature["geometry"])
        for feature in document["features"] if feature["properties"]["value"] == value])
        for value in (1, 2)}
    review, selected, attempts = choose_review_geometry(exact,
        document["native_shape_yx"], document["native_mpp_xy"],
        vector_receipt["source_semantic_mask_sha256"],
        simplify_outer_boundary=False)
    assert attempts[0]["status"] == "REJECTED"  # 3 µm exceeds IoU gate
    assert selected["tolerance_um"] == 1.5
    assert len(selected["restored_pairs"]) == 16
    assert [len(_parts(review[value])) for value in (1, 2)] == [130, 210]
    assert selected["full_native_raster_agreement"]["minimum_class_iou"] >= .995
    assert selected["maximum_changed_pixel_distance_um"] <= 5
    assert selected["vertex_reduction_fraction"] > .8
    assert selected["outer_support_symmetric_difference_px2"] < 1e-6
    assert selected["segment_stats_after"]["diagonal_segment_fraction"] > .5
