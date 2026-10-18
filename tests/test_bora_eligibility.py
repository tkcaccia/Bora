import copy
import json

import numpy as np
import pytest
import tifffile
from PIL import Image

from experiments.bora_extension.eligibility import (
    POLICY_PATH, inspect, digest, audit, prepare_view)
from experiments.bora_cycles.benchmark_scope import validate_scope, EXPECTED

POLICY = json.loads(POLICY_PATH.read_text())


def two_groups(shape=(896, 896)):
    coarse = np.ones(shape, np.uint8)
    coarse[:, 450:] = 2
    return coarse


def test_single_input_group_and_ignore_never_qualify():
    coarse = np.ones((1280, 1280), np.uint8)
    coarse[:, :90] = 0
    result = inspect(coarse, coarse.shape, POLICY)
    assert result['input_groups'] == [1]
    assert not result['eligible']
    assert 'fewer_than_two_input_groups' in result['exclusion_reasons']


def test_second_group_without_a_usable_tile_cannot_qualify():
    coarse = np.ones((1280, 1280), np.uint8)
    coarse[5, 5] = 2
    result = inspect(coarse, coarse.shape, POLICY)
    assert result['input_groups'] == [1, 2]
    assert result['observed_groups'] == [1]
    assert not result['eligible']


def test_small_real_field_not_qualified_by_overlapping_window_count():
    coarse = two_groups((370, 419))
    coarse[:, 180:] = 2
    result = inspect(coarse, coarse.shape, POLICY)
    assert result['complete_base_field_capacity'] == 1
    assert not result['eligible']
    assert 'fewer_than_four_real_encoder_fields_on_each_axis' in result['exclusion_reasons']


def test_both_axis_extent_and_label_observation_count_required():
    narrow = two_groups((448, 1792))
    r = inspect(narrow, narrow.shape, POLICY)
    assert r['complete_base_field_capacity'] == 16
    assert not r['eligible']
    sparse = np.zeros((1280, 1280), np.uint8)
    sparse[:90, :90] = 1
    sparse[90:180, :90] = 2
    r = inspect(sparse, sparse.shape, POLICY)
    assert r['observed_groups'] == [1, 2] and r['usable_coarse_tiles'] == 2
    assert not r['eligible']


def test_threshold_and_rare_group_preserved_including_rings_negative():
    coarse = two_groups()
    result = inspect(coarse, coarse.shape, POLICY)
    assert result['eligible']
    assert result['complete_base_fields_yx'] == [4, 4]
    coarse[:] = 1
    coarse[:90, :90] = 2
    result = inspect(coarse, coarse.shape, POLICY)
    assert result['eligible'] and result['coarse_tiles_by_group']['2'] == 1


def test_scope_audit_and_view_never_need_fine_reference_or_scores(tmp_path):
    source = tmp_path/'source'
    samples, manifests = [], {}
    for name, coarse in [('two', two_groups()), ('one', np.ones((896,896), np.uint8))]:
        folder = source/'cases'/name
        folder.mkdir(parents=True)
        Image.new('RGB', (896,896), (180,90,140)).save(folder/'original.png')
        tifffile.imwrite(folder/'coarse.tif', coarse)
        (folder/'selection.json').write_text(json.dumps(dict(classes=sorted(np.unique(coarse).tolist()))))
        files = {n: digest(folder/n) for n in ['original.png','coarse.tif','selection.json']}
        (folder/'INPUTS_FROZEN.json').write_text(json.dumps(dict(files_sha256=files)))
        manifests[name] = digest(folder/'INPUTS_FROZEN.json')
        samples.append(dict(case=name,dataset='test',group=name,working_mpp_xy=None,
            working_shape_yx=[896,896], class_names={'1':'a','2':'b'}))
    (source/'PLAN.json').write_text(json.dumps(dict(samples=samples,historical_cases=['two'],additional_cases=['one'])))
    (source/'ALL_INPUTS_FROZEN.json').write_text(json.dumps(dict(plan_sha256=digest(source/'PLAN.json'),manifests=manifests)))
    report = audit(source, tmp_path/'report')
    assert report['eligible_cases'] == ['two']
    assert report['excluded_cases'] == ['one']
    assert not report['protocol']['fine_reference_read']
    view = prepare_view(source, tmp_path/'report', tmp_path/'view')
    assert [s['case'] for s in view['samples']] == ['two']
    assert (tmp_path/'view/diagnostics/one').is_symlink()
    # Immutable audit supports an exact repeat but detects a changed input.
    assert audit(source, tmp_path/'report') == report
    with (source/'cases/two/coarse.tif').open('ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError, match='Changed RGB/coarse'):
        audit(source, tmp_path/'report')


def test_future_cycle_cannot_accidentally_reuse_old_unfiltered_scope():
    with pytest.raises(ValueError, match='eligibility audit'):
        validate_scope(dict(cycle=11, benchmark_scope=dict(version='expanded349-v1')))


def qualified_fixture():
    ids = [f'fixture-{i}' for i in range(349)]
    scope = dict(version='qualified-multigroup-v1',required_cases=ids[:2],diagnostic_cases=ids[2:],
        source_cases=ids,source_dataset_counts=EXPECTED,dataset_counts={'bcss':2},
        plan_sha256='fixture-plan',historical_cases=ids[:2],
        eligible_case_checks={case:dict(eligible=True,exclusion_reasons=[],input_groups=[1,2],
            observed_groups=[1,2],complete_base_fields_yx=[4,4],usable_coarse_tiles=16) for case in ids[:2]})
    return dict(cycle=11,cases=ids[:2],input_plan_sha256='fixture-plan',benchmark_scope=scope)


def test_qualified_scope_requires_all_eligible_cases_and_no_diagnostic_cases():
    protocol = qualified_fixture()
    validate_scope(protocol)
    protocol['cases'] = protocol['cases'][:1]
    with pytest.raises(ValueError, match='every qualified image'):
        validate_scope(protocol)
    validate_scope(protocol,pilot=True)
    protocol['cases'].append('fixture-2')
    with pytest.raises(ValueError, match='Unexpected'):
        validate_scope(protocol,pilot=True)


def test_qualified_scope_cannot_admit_single_group_or_rewrite_old_cycle():
    protocol = qualified_fixture()
    protocol['benchmark_scope']['eligible_case_checks']['fixture-0']['observed_groups'] = [1]
    with pytest.raises(ValueError, match='Single-group'):
        validate_scope(protocol)
    protocol = qualified_fixture()
    protocol['cycle'] = 10
    with pytest.raises(ValueError, match='completed cycles'):
        validate_scope(protocol)
