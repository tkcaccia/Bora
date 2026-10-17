import pytest
from experiments.bora_cycles.runtime_report import matched_history_ranking


def test_expansion_does_not_turn_easy_new_cases_into_a_historical_gain():
    rows = [dict(case=case, methods={'M':dict(status='ok', median_seconds=.2,
        score=dict(macro_iou=iou))}) for case,iou in [('old-a',.4),('old-b',.6),('new',1.)]]
    result = dict(protocol=dict(benchmark_scope=dict(historical_cases=['old-a','old-b'])),
        cases=rows, ranking=[dict(method='M', complete=True, mean_iou=2/3)])
    matched = matched_history_ranking(result)[0]
    assert matched['mean_iou'] == pytest.approx(.5)
    assert matched['runtime_adjusted_score'] == pytest.approx(.5)
    assert matched['cases'] == 2
    assert matched['mean_seconds'] == pytest.approx(.2)
    assert result['ranking'][0]['mean_iou'] == 2/3  # Primary349-style score untouched.
    result['cases'] = rows[1:]
    with pytest.raises(ValueError, match='incomplete'):
        matched_history_ranking(result)
