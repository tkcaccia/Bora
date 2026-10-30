"""Regression checks for physical contour polish and strict area cleanup."""

import numpy as np

from bora.contour_polish import smooth_global
from bora.selected_wsi_polished import clean_global


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
