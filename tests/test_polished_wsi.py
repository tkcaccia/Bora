"""Regression checks for physical contour polish and strict area cleanup."""

import numpy as np
import pytest
from shapely.geometry import box, mapping

from bora.contour_polish import smooth_global
from bora.selected_wsi_polished import audit, clean_global
from bora.wsi_polished import native_vectors, polished_parameters
from bora.cli import parser


def test_contour_polish_preserves_voids_and_has_exact_block_seams(tmp_path):
    shape = (384, 384)
    support = np.ones(shape, np.uint8)
    labels = np.ones(shape, np.uint8)
    labels[:, 191:] = 2
    labels[170:210, 170:210] = 0
    support[170:210, 170:210] = 0
    labels[280:300, 40:60] = 0  # Existing unassigned tissue is also fixed.
    labels[140:160, 110:120] = 2
    receipt = smooth_global(labels, support, (0.5, 0.5),
                            tmp_path / 'smoothed.npy', sigma_um=6, block_size=128)
    output = np.load(tmp_path / 'smoothed.npy')
    assert np.array_equal(output > 0, labels > 0)
    assert np.all(output[170:210, 170:210] == 0)
    assert receipt['changed_supported_pixels'] > 0
    assert len(receipt['exact_block_seam_checks']) == 3
    assert all(check['different_pixels'] == 0
               for check in receipt['exact_block_seam_checks'])


def test_strict_2000_um2_cleanup_retains_exact_threshold(tmp_path):
    labels = np.ones((120, 120), np.uint8)
    labels[10:30, 10:30] = 2  # 400 px * 5 µm² = exactly 2000 µm².
    labels[50:69, 50:70] = 2  # 380 px = 1900 µm².
    result, counts = clean_global(labels, (1.0, 5.0),
                                  tmp_path / 'cleaned.npy', 2000)
    assert np.all(result[10:30, 10:30] == 2)
    assert np.all(result[50:69, 50:70] == 1)
    assert counts['parts'] == 1
    assert counts['pixels'] == 380


def test_polished_inputs_keep_defaults_and_allow_smaller_minimum():
    args = parser().parse_args(['wsi-polished', 'case.json', '--output', 'result'])
    assert (args.minimum_area_um2, args.contour_sigma_um) == (2000, 6)
    args = parser().parse_args(['wsi-polished', 'case.json', '--output', 'result',
                                '--minimum-area-um2', '100',
                                '--contour-smoothing-um', '2.5'])
    assert (args.minimum_area_um2, args.contour_sigma_um) == (100, 2.5)
    params = polished_parameters(args.minimum_area_um2, args.contour_sigma_um)
    assert params['prepolish_minimum_um2'] == 100
    for bad in (0, -1, float('nan'), float('inf'), None, True):
        with pytest.raises(ValueError):
            polished_parameters(bad, 6)
        with pytest.raises(ValueError):
            polished_parameters(2000, bad)


def test_global_audit_uses_requested_physical_minimum():
    labels = np.ones((40, 40), dtype=np.uint8)
    labels[10:22, 10:22] = 2
    tissue = box(0, 0, 40, 40).difference(box(10, 10, 22, 22))
    data = {'features': [
        {'properties': {'value': 1}, 'geometry': mapping(tissue)},
        {'properties': {'value': 2}, 'geometry': mapping(box(10, 10, 22, 22))},
    ]}
    assert audit(data, labels, (1, 1), minimum_area_um2=100)['minimum_area_um2'] == 100
    with pytest.raises(ValueError, match='Global polygon below physical minimum'):
        audit(data, labels, (1, 1), minimum_area_um2=200)
    identity = {'mpp_xy': [1, 1], 'native_mpp_xy': [1, 1],
                'native_shape_yx': [40, 40],
                'analysis_bounds_native_xyxy': [0, 0, 40, 40]}
    native = native_vectors(data, identity, minimum_area_um2=100)
    assert native['physical_area_rule']['minimum_component_and_enclosed_hole_um2'] == 100
    assert native['native_coordinate_final_cleanup']['minimum_part_and_hole_um2'] == 100
