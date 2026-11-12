import numpy as np
import pytest

from benchmark.frozen_group_recovery import recover_precleanup_groups


def test_recovers_removed_expert_group_with_native_scaling_and_exact_support(tmp_path):
    mask = np.zeros((12, 16), np.uint8)
    mask[1:11, 1:7] = 1
    mask[1:11, 9:15] = 2
    mask[3, 4] = 3
    mask[8, 12] = 3
    path = tmp_path / 'smoothed-mask.npy'
    np.save(path, mask)

    features, receipt = recover_precleanup_groups(
        path, native_shape_yx=(24, 32), working_content_shape_yx=mask.shape,
        required_group_ids={1, 2, 3}, final_group_ids={1, 2})

    assert {f['properties']['value'] for f in features} == {1, 2, 3}
    assert receipt['recovered_expert_group_ids'] == [3]
    assert receipt['exact_raster_roundtrip'] is True
    assert receipt['native_shape_yx'] == [24, 32]


def test_rejects_missing_frozen_group(tmp_path):
    path = tmp_path / 'smoothed-mask.npy'
    np.save(path, np.ones((4, 4), np.uint8))
    with pytest.raises(ValueError, match='expert or final group absent'):
        recover_precleanup_groups(
            path, native_shape_yx=(8, 8), working_content_shape_yx=(4, 4),
            required_group_ids={1, 2}, final_group_ids={1})
