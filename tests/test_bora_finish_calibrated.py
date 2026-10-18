import json

import pytest

from experiments.bora_cycles.finish_calibrated import process_identity,stage_ready
from experiments.bora_cycles.run import save,digest


def fixture(root):
    protocol=dict(pilot=False,cases=['a','b'])
    save(root/'PROTOCOL.json',protocol);sha=digest(root/'PROTOCOL.json')
    save(root/'ALL_PREDICTIONS_FROZEN.json',dict(protocol_sha256=sha,fine_reference_read=False,manifests={'a':'a','b':'b'}))
    save(root/'VISUAL_MANIFEST.json',dict(contact_sha256={'a':'a','b':'b'}))
    save(root/'RESULTS.json',dict(protocol=protocol,cases=[dict(case='a'),dict(case='b')]))
    return sha


def test_absent_terminal_stage_waits_even_with_some_outputs(tmp_path):
    sha=fixture(tmp_path)
    assert not stage_ready(tmp_path,sha)
    save(tmp_path/'STAGE.json',dict(state='awaiting_visual_review',cases=2))
    assert stage_ready(tmp_path,sha)


@pytest.mark.parametrize('problem',['render','score','freeze','protocol','stage'])
def test_terminal_marker_alone_cannot_release_incomplete_or_changed_evidence(tmp_path,problem):
    sha=fixture(tmp_path);save(tmp_path/'STAGE.json',dict(state='awaiting_visual_review',cases=2))
    if problem=='render':save(tmp_path/'VISUAL_MANIFEST.json',dict(contact_sha256={'a':'a'}))
    elif problem=='score':
        result=json.loads((tmp_path/'RESULTS.json').read_text());result['cases']=[dict(case='a'),dict(case='a')];save(tmp_path/'RESULTS.json',result)
    elif problem=='freeze':save(tmp_path/'ALL_PREDICTIONS_FROZEN.json',dict(protocol_sha256=sha,fine_reference_read=True,manifests={'a':'a','b':'b'}))
    elif problem=='protocol':save(tmp_path/'PROTOCOL.json',dict(pilot=False,cases=['a']))
    else:save(tmp_path/'STAGE.json',dict(state='awaiting_visual_review',cases=1))
    with pytest.raises(ValueError):stage_ready(tmp_path,sha)


def test_process_identity_distinguishes_missing_zombie_and_pid_reuse(tmp_path):
    process=tmp_path/'123';process.mkdir()
    # Linux fields3..22, including a process name containing spaces/parentheses.
    tail=['S']+['0']*18+['123456']
    (process/'stat').write_text('123 (name with (parenthesis)) '+' '.join(tail))
    assert process_identity(123,tmp_path)=='123456'
    tail[19]='654321';(process/'stat').write_text('123 (same name) '+' '.join(tail))
    assert process_identity(123,tmp_path)=='654321'
    tail[0]='Z';(process/'stat').write_text('123 (same name) '+' '.join(tail))
    assert process_identity(123,tmp_path) is None
    assert process_identity(456,tmp_path) is None
