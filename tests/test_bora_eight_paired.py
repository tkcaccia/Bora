from copy import deepcopy
import pytest

from experiments.bora_cycles.paired_eight import SCORE_CORE, delta_summary, parent_score_parity
from experiments.bora_cycles.runtime_policy import DEFAULT_POLICY


def test_paired_utility_penalizes_slow_child_per_case_without_hidden_exclusions():
    rows = []
    for case,ci,pi,seconds in [('a',.7,.6,.1),('b',.8,.8,3.)]:
        rows.append(dict(case=case,methods={
            'C':dict(status='ok',score=dict(macro_iou=ci),median_seconds=seconds),
            'P':dict(status='ok',score=dict(macro_iou=pi),median_seconds=.2)}))
    report = delta_summary(rows,'C','P',DEFAULT_POLICY)
    assert report['mean_iou_delta'] == pytest.approx(.05)
    assert report['mean_utility_delta'] == pytest.approx(0)
    assert (report['wins'],report['losses'],report['ties']) == (1,0,1)
    rows[0]['methods']['C']['status'] = 'failed'
    assert not delta_summary(rows,'C','P',DEFAULT_POLICY)['complete']
    assert delta_summary([],'C','P',DEFAULT_POLICY)['mean_iou_delta'] is None


def test_historical_parity_covers_component_scores_and_deleted_pixels():
    score = {key:{'nested':.5} if key in ['per_class','regions'] else [] if key=='missed_regions'
             else .5 for key in SCORE_CORE}
    old = [dict(case='old',methods={'P':dict(status='ok',score=score)})]
    new = deepcopy(old)
    new[0]['methods']['P']['score']['coordinate_units'] = 'micrometres'
    assert parent_score_parity(new,old,['old'],['P']) == 1
    for key in ['annotated_unassigned_pixels','regions','missed_regions']:
        altered = deepcopy(new)
        altered[0]['methods']['P']['score'][key] = 'changed'
        with pytest.raises(ValueError,match='score changed'):
            parent_score_parity(altered,old,['old'],['P'])
    with pytest.raises(ValueError,match='scope'):
        parent_score_parity(new,old,['other'],['P'])
