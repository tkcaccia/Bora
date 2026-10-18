import json
import pytest

from experiments.bora_extension.calibrated_cohort import (cache_index,partition,source_group,
    MODEL_REVISION,MODEL_SHA256)
from experiments.bora_extension.tissue_fields import sha


def test_full_inventory_partition_cannot_hide_diagnostics_or_repeat_cases():
    partition(['a','b','c'],['a','b'],['c'])
    for source,required,diagnostic in [(['a','b','c'],['a'],['c']),
        (['a','b'],['a','b'],['b']),(['a','a'],['a'],[])]:
        with pytest.raises(ValueError,match='partition'):
            partition(source,required,diagnostic)


def test_tcga_cases_cluster_across_sources_but_anonymous_ids_do_not():
    a=dict(dataset='first',group='TCGA-AB-1234')
    b=dict(dataset='second',group='unknown',source=dict(slide_id='tcga-ab-1234-01Z-DX1'))
    assert source_group(a)==source_group(b)
    assert source_group(dict(dataset='first',group='patient1'))!=source_group(dict(dataset='second',group='patient1'))


@pytest.fixture
def cache(tmp_path):
    root=tmp_path/'features';case=root/'cases'/'case';case.mkdir(parents=True)
    source=tmp_path/'source';source.mkdir();(source/'PLAN.json').write_text('{}')
    (case/'FEATURES_FROZEN.json').write_text('{}')
    protocol=dict(cases=['case'],source_plans={str(source):sha(source/'PLAN.json')},
        encoder_checkpoint_sha256=MODEL_SHA256,encoder_revision=MODEL_REVISION)
    (root/'PROTOCOL.json').write_text(json.dumps(protocol));psha=sha(root/'PROTOCOL.json');msha=sha(case/'FEATURES_FROZEN.json')
    (root/'ALL_FEATURES_FROZEN.json').write_text(json.dumps(dict(cases=1,protocol_sha256=psha,case_manifest_sha256={'case':msha})))
    audit=dict(passed=True,full_corpus=True,cases=1,arrays=3,calibrated=1,pixel_only=0,single_coarse_class=0,
        protocol_sha256=psha,rows=[dict(case='case',feature_manifest_sha256=msha)])
    (root/'FEATURE_AUDIT.json').write_text(json.dumps(audit))
    return root,source


def test_feature_index_requires_full_calibration_and_fixed_encoder_identity(cache):
    root,source=cache
    index,_=cache_index([root]);assert index=={'case':str(root)}
    path=root/'FEATURE_AUDIT.json';audit=json.loads(path.read_text());audit['calibrated']=0;audit['pixel_only']=1;path.write_text(json.dumps(audit))
    with pytest.raises(ValueError,match='Complete calibrated'):
        cache_index([root])


def test_feature_manifest_or_source_mutation_blocks_assembly(cache):
    root,source=cache
    (source/'PLAN.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='source plan changed'):
        cache_index([root])
