import copy
import pytest
from experiments.bora_cycles import posthoc_calibrated as module
from experiments.bora_cycles.paired_eight import SCORE_CORE
from experiments.bora_cycles.runtime_policy import DEFAULT_POLICY


def fixture(cycle=12):
    parents=['P11','N08','M04','J01','C17']
    children=[dict(id=f'Q{i:02d}',parent=parents[(i-1)//3],title=f'Child{i}') for i in range(1,16)]
    methods=parents+[r['id'] for r in children]
    score={key:.7 for key in SCORE_CORE}
    entries={key:dict(status='ok',score=copy.deepcopy(score),median_seconds=.2) for key in methods}
    rows=[dict(case='ocelot_01',dataset='ocelot',historical_subset=True,methods=copy.deepcopy(entries)),
          dict(case='puma_01',dataset='puma',historical_subset=False,methods=copy.deepcopy(entries))]
    protocol=dict(cycle=cycle,pilot=True,cases=[r['case'] for r in rows],methods=methods,roster={key:{} for key in methods},
        benchmark_scope=dict(version='calibrated-tissues-v1'),
        lineage=dict(unchanged_survivors=parents,children=children))
    return dict(protocol=protocol,cases=rows),dict(cases=copy.deepcopy(rows)),dict(cases=2,methods=20,unchanged_survivor_comparisons=10)


def configure(monkeypatch):
    # Scope has its own positive/negative tests; exercise paired arithmetic and
    # the actual versioned ontology rule here with a small frozen-like fixture.
    monkeypatch.setattr(module,'validate_scope',lambda *a,**k:None)
    monkeypatch.setattr(module,'requires_parent_parity',lambda *a,**k:True)
    monkeypatch.setattr(module,'active_policy',lambda *a,**k:DEFAULT_POLICY)


def test_cycle12_uses_frozen_children_and_requires_all_parent_score_parity(monkeypatch):
    configure(monkeypatch);result,old,audit=fixture()
    report=module.paired(result,old,audit)
    assert set(report['children'])=={f'Q{i:02d}' for i in range(1,16)}
    assert report['unchanged_ontology_parent_score_comparisons']==10
    assert report['prospective_ocelot_rescored_comparisons']==0
    result['cases'][0]['methods']['P11']['score']['macro_iou']=.1
    with pytest.raises(ValueError,match='parent score changed'):module.paired(result,old,audit)


def test_cycle11_keeps_its_historical_ocelot_rescore_exception(monkeypatch):
    configure(monkeypatch);result,old,audit=fixture(11)
    result['cases'][0]['methods']['P11']['score']['macro_iou']=.1
    report=module.paired(result,old,audit)
    assert report['prospective_ocelot_rescored_comparisons']==5
    assert report['unchanged_ontology_parent_score_comparisons']==5


def test_changed_frozen_lineage_cannot_silently_compare_wrong_children(monkeypatch):
    configure(monkeypatch);result,old,audit=fixture()
    result['protocol']['lineage']['children'][0]['id']='P01'
    with pytest.raises(ValueError,match='lineage'):module.paired(result,old,audit)
