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


def test_rendered_history_labels_scope_change_and_keeps_primary_scores(tmp_path):
    import json
    from experiments.bora_cycles.runtime_report import build_history
    from experiments.bora_cycles.run import digest
    # Tiny report fixtures, not study selections or visual-review evidence.
    for cycle in [7, 8]:
        root = tmp_path/f'cycle-{cycle:02d}'
        root.mkdir()
        cases = [('old-a', .4), ('old-b', .6)]+([('new', 1.)] if cycle == 8 else [])
        protocol = dict(host='unit-test', roster={'M':dict(title='Fixture model')},
            _fixture_no_study_review=True)
        if cycle == 8:
            protocol['benchmark_scope'] = dict(version='expanded349-v1', historical_cases=['old-a','old-b'])
        rows = [dict(case=c, methods={'M':dict(status='ok', median_seconds=.2,
            score=dict(macro_iou=v))}) for c,v in cases]
        result = dict(protocol=protocol, cases=rows, ranking=[dict(method='M', complete=True,
            mean_iou=sum(v for _,v in cases)/len(cases), mean_seconds=.2)])
        for name, value in [('RESULTS.json', result), ('PROTOCOL.json', protocol),
                            ('VISUAL_REVIEW.json', dict(unit_test_fixture=True))]:
            (root/name).write_text(json.dumps(value))
        selection = dict(cycle=cycle, survivors=['M'],
            results_sha256=digest(root/'RESULTS.json'), protocol_sha256=digest(root/'PROTOCOL.json'),
            review_sha256=digest(root/'VISUAL_REVIEW.json'))
        (root/'SELECTION.json').write_text(json.dumps(selection))
    history = build_history(tmp_path)
    assert history['cycles'][0]['raw_accuracy_leader_row']['mean_iou'] == .5
    assert history['cycles'][1]['raw_accuracy_leader_row']['mean_iou'] == pytest.approx(2/3)
    assert history['cycles'][1]['matched_historical_ranking'][0]['mean_iou'] == .5
    html = (tmp_path/'runtime_history.html').read_text()
    assert 'not comparable gains' in html and 'Original-cohort IoU' in html
