import json

import numpy as np
import pytest
import tifffile

from experiments.bora_cycles.audit_repeatability import compare
from experiments.bora_cycles.run import digest,save


def write_run(root,*,pilot):
    cases=['x'] if pilot else ['x','y'];root.mkdir()
    source=root/'runtime_sources/experiments/bora_cycles/run.py';source.parent.mkdir(parents=True)
    source.write_text('frozen prediction implementation')
    protocol=dict(pilot=pilot,cycle=11,cases=cases,methods=['J01'],input_plan_sha256='same',
        primary_conversion='ensemble',benchmark_scope={'version':'same-frozen-scope'},study_protocol={'metric':'same'},
        source_hashes={'experiments/bora_cycles/run.py':digest(source)},repeats=2 if pilot else 1)
    save(root/'PROTOCOL.json',protocol)
    rows=[];manifests={}
    for case in cases:
        out=root/'cases'/case;target=out/'J01';target.mkdir(parents=True)
        tifffile.imwrite(target/'native.tif',np.array([[1,1],[0,2]],np.uint8))
        tifffile.imwrite(target/'support.native.tif',np.array([[1,1],[0,1]],np.uint8))
        np.savez_compressed(out/'RARITY.npz',score=np.zeros((2,2)),regions=np.zeros((2,2),np.int32))
        files={n:digest(out/n) for n in ('J01/native.tif','J01/support.native.tif','RARITY.npz')}
        save(out/'PREDICTIONS.json',dict(files_sha256=files,input_manifest_sha256=case+'input',rgb_sha256=case+'rgb'))
        manifests[case]=digest(out/'PREDICTIONS.json')
        score=dict(macro_iou=.7,macro_dice=.8,pixel_accuracy=.8,annotated_unassigned_pixels=0,
            valid_pixels=3,per_class={'1':{'iou':.5}},regions={'components':2},missed_regions=[])
        rows.append(dict(case=case,sample={'native_mpp_xy':[.5,.5]},methods={'J01':dict(status='ok',
            median_seconds=3 if pilot else .1,score=score)}))
    save(root/'ALL_PREDICTIONS_FROZEN.json',dict(protocol_sha256=digest(root/'PROTOCOL.json'),fine_reference_read=False,manifests=manifests))
    save(root/'RESULTS.json',dict(protocol=protocol,cases=rows))
    return root


def refresh_prediction(root,case='x'):
    p=root/'cases'/case/'PREDICTIONS.json';record=json.loads(p.read_text())
    record['files_sha256']={n:digest(p.parent/n) for n in record['files_sha256']};save(p,record)
    f=root/'ALL_PREDICTIONS_FROZEN.json';record=json.loads(f.read_text());record['manifests'][case]=digest(p);save(f,record)


def test_exact_overlap_compares_masks_supports_scores_and_proposals_without_comparing_times(tmp_path):
    full=write_run(tmp_path/'full',pilot=False);pilot=write_run(tmp_path/'pilot',pilot=True)
    report=compare(full,pilot)
    assert report['full_cases']==2 and report['pilot_cases']==1
    assert report['native_masks']==report['native_supports']==report['accuracy_core_comparisons']==1
    assert report['rarity_arrays']==2 and not report['timings_compared']


@pytest.mark.parametrize('changed', ['mask','support','rarity','score','input','rgb'])
def test_equal_aggregate_cannot_hide_a_changed_mask_support_score_or_input(tmp_path,changed):
    full=write_run(tmp_path/'full',pilot=False);pilot=write_run(tmp_path/'pilot',pilot=True)
    case=full/'cases/x'
    if changed in ('mask','support'):
        name='native.tif' if changed=='mask' else 'support.native.tif'
        labels=tifffile.imread(case/'J01'/name);labels[0,0]=0;tifffile.imwrite(case/'J01'/name,labels)
    elif changed=='rarity':np.savez_compressed(case/'RARITY.npz',score=np.ones((2,2)),regions=np.zeros((2,2),np.int32))
    elif changed=='score':
        p=full/'RESULTS.json';result=json.loads(p.read_text());result['cases'][0]['methods']['J01']['score']['macro_iou']=.6;save(p,result)
    else:
        p=case/'PREDICTIONS.json';record=json.loads(p.read_text());record['input_manifest_sha256' if changed=='input' else 'rgb_sha256']='changed';save(p,record)
    refresh_prediction(full)
    with pytest.raises(ValueError,match='differs|differ'):compare(full,pilot)


def test_an_changed_prediction_implementation_cannot_pass_as_a_report_edit(tmp_path):
    full=write_run(tmp_path/'full',pilot=False);pilot=write_run(tmp_path/'pilot',pilot=True)
    source=full/'runtime_sources/experiments/bora_cycles/run.py';source.write_text('changed prediction')
    p=full/'PROTOCOL.json';protocol=json.loads(p.read_text());protocol['source_hashes']['experiments/bora_cycles/run.py']=digest(source);save(p,protocol)
    f=full/'ALL_PREDICTIONS_FROZEN.json';frozen=json.loads(f.read_text());frozen['protocol_sha256']=digest(p);save(f,frozen)
    r=full/'RESULTS.json';result=json.loads(r.read_text());result['protocol']=protocol;save(r,result)
    with pytest.raises(ValueError,match='Prediction source'):compare(full,pilot)
