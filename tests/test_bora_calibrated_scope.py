import copy
import pytest

from experiments.bora_cycles.benchmark_scope import validate_scope,requires_parent_parity
from experiments.bora_extension.calibration import require


def fixture():
    old=[f'old{i}' for i in range(3)];added=[f'new{i}' for i in range(40)];ids=old+added
    checks={}
    for case in ids:
        physical=require(dict(case=case,native_shape_yx=[2560,2560],working_shape_yx=[1280,1280],
            native_mpp_xy=[.25,.25],working_mpp_xy=[.5,.5],fov_um_xy=[640,640]))
        checks[case]=dict(eligible=True,exclusion_reasons=[],input_groups=[1,2],observed_groups=[1,2],
            complete_base_fields_yx=[5,5],usable_coarse_tiles=100,physical_calibration=physical)
    scope=dict(version='calibrated-tissues-v1',introduced_cycle=11,required_cases=ids,
        source_cases=ids+['diagnostic0','diagnostic1'],diagnostic_cases=['diagnostic0','diagnostic1'],
        historical_cases=old,additional_cases=added,dataset_counts={'old':3,'new':40},
        source_dataset_counts={'old':5,'new':40},new_dataset_counts={'new':40},
        eligible_case_checks=checks,plan_sha256='fixture')
    return dict(cycle=11,cases=ids,benchmark_scope=scope,input_plan_sha256='fixture')


def test_calibrated_full_scope_and_explicit_pilot_subset():
    protocol=fixture();validate_scope(protocol)
    protocol['cases']=protocol['cases'][:2]
    with pytest.raises(ValueError,match='every calibrated'):
        validate_scope(protocol)
    validate_scope(protocol,pilot=True)


def test_known_new_cases_have_matched_parents_now_and_exact_parity_next_cycle():
    p=fixture()
    assert requires_parent_parity(p,'old0')
    assert not requires_parent_parity(p,'new0')
    p['cycle']=12
    assert requires_parent_parity(p,'new0')


def test_unknown_calibration_single_group_or_partial_dataset_cannot_enter():
    for mutation in ['calibration','group','quota']:
        p=fixture()
        if mutation=='calibration':p['benchmark_scope']['eligible_case_checks']['new0']['physical_calibration']['eligible']=False
        elif mutation=='group':p['benchmark_scope']['eligible_case_checks']['new0']['observed_groups']=[1]
        else:p['benchmark_scope']['new_dataset_counts']['new']=39
        with pytest.raises(ValueError):validate_scope(p)


def test_case_removal_cannot_silently_change_the_full_inventory_partition():
    p=fixture();p['benchmark_scope']['diagnostic_cases']=[]
    with pytest.raises(ValueError,match='partition'):validate_scope(p)
