"""Assemble an audited calibrated cohort without decoding fine references.

Old RGB/input folders are reused unchanged. New fields use the same fixed
conversion functions and already audited Virchow2 caches. Scores and predicted
masks are neither accepted nor read by this command.
"""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
import re
import shutil

import numpy as np
from PIL import Image
import tifffile

from .calibration import require,POLICY_PATH as PHYSICAL_POLICY
from .eligibility import inspect,POLICY_PATH,save_new
from .features import verify_case
from .tissue_fields import sha
from experiments.virchow2_inverse.common import MODEL_REVISION,MODEL_SHA256

VERSION='calibrated-tissues-v1'


def load(path):return json.loads(Path(path).read_text())


def partition(source,required,diagnostic):
    if (len(set(source))!=len(source) or len(set(required))!=len(required)
            or len(set(diagnostic))!=len(diagnostic) or set(required)&set(diagnostic)
            or set(required)|set(diagnostic)!=set(source)):
        raise ValueError('Required and diagnostic cases must partition the full frozen source inventories')


def cache_index(roots):
    result={};proof={}
    for root in map(Path,roots):
        frozen=load(root/'ALL_FEATURES_FROZEN.json');audit=load(root/'FEATURE_AUDIT.json');protocol=load(root/'PROTOCOL.json')
        if (not audit['passed'] or not audit['full_corpus'] or audit['cases']!=frozen['cases']
                or audit['arrays']!=3*frozen['cases'] or audit['calibrated']!=frozen['cases']
                or audit['pixel_only'] or audit['single_coarse_class']
                or protocol['encoder_checkpoint_sha256']!=MODEL_SHA256 or protocol['encoder_revision']!=MODEL_REVISION
                or audit['protocol_sha256']!=sha(root/'PROTOCOL.json')
                or frozen['protocol_sha256']!=sha(root/'PROTOCOL.json')
                or set(protocol['cases'])!=set(frozen['case_manifest_sha256'])):
            raise ValueError('Complete calibrated feature audit required')
        for source,expected in protocol['source_plans'].items():
            if sha(Path(source)/'PLAN.json')!=expected:raise ValueError('Feature source plan changed')
        audit_rows={r['case']:r for r in audit['rows']}
        for case,expected in frozen['case_manifest_sha256'].items():
            if case in result or sha(root/'cases'/case/'FEATURES_FROZEN.json')!=expected or audit_rows[case]['feature_manifest_sha256']!=expected:
                raise ValueError('Duplicate or changed feature cache')
            result[case]=str(root)
        proof[str(root)]=dict(freeze_sha256=sha(root/'ALL_FEATURES_FROZEN.json'),
            audit_sha256=sha(root/'FEATURE_AUDIT.json'),protocol_sha256=sha(root/'PROTOCOL.json'))
    return result,proof


def source_group(sample):
    text=' '.join(str(sample.get(k,'')) for k in ('group','source'))
    found=re.search(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}',text.upper())
    return 'TCGA:'+found.group(0) if found else sample['dataset']+':'+str(sample['group'])


def prepare(base,calibration,acquisition_audit,acquisitions,features,root):
    base,root=Path(base),Path(root);old=load(base/'PLAN.json');cal=load(calibration);acq=load(acquisition_audit)
    old_freeze=load(base/'ALL_INPUTS_FROZEN.json')
    base_eligibility=load(base/'ELIGIBILITY.json')
    if (cal['source_plan_sha256']!=sha(base/'PLAN.json') or cal['policy_sha256']!=sha(PHYSICAL_POLICY)
            or not cal['no_encoder_or_bora_run'] or cal['fine_reference_read'] or cal['predictions_read'] or cal['scores_read']
            or old_freeze['plan_sha256']!=sha(base/'PLAN.json')
            or old['eligibility_sha256']!=sha(base/'ELIGIBILITY.json')
            or not acq['all_passed'] or acq['calibration_policy_sha256']!=sha(PHYSICAL_POLICY)):
        raise ValueError('Calibration/acquisition proof belongs to other inputs or policy')
    old_ready=set(cal['ready_cases']);old_rows=[r for r in old['samples'] if r['case'] in old_ready]
    if {r['case'] for r in old_rows}!=old_ready:raise ValueError('Incomplete calibrated base subset')
    for row in old_rows:require(row)
    new=[];source_roots={};source_plans={str(base):sha(base/'PLAN.json')}
    audited_datasets={r['dataset']:r for r in acq['datasets']}
    for source in map(Path,acquisitions):
        plan=load(source/'PLAN.json');rows=plan['samples'];dataset=plan['dataset']
        if (not plan['quota_met'] or plan['accepted']!=40 or len(rows)!=40
                or dataset not in audited_datasets or audited_datasets[dataset]['physically_calibrated']!=40
                or audited_datasets[dataset]['plan_sha256']!=sha(source/'PLAN.json')):
            raise ValueError('Every admitted new tissue dataset needs40 fully calibrated audited fields')
        for row in rows:
            require(row)
            if row['case'] in source_roots:raise ValueError('Duplicate added case')
            source_roots[row['case']]=str(source);new.append(row)
        source_plans[str(source)]=sha(source/'PLAN.json')
    if {r['case'] for r in new}!=set(acq['physical_calibration_ready_cases']):
        raise ValueError('All new physically eligible datasets in the audited inventory are required')
    caches,feature_proof=cache_index(features)
    if set(caches)!={r['case'] for r in new}:raise ValueError('Feature caches and admitted new cases differ')
    rows=old_rows+new;ids=[r['case'] for r in rows]
    old_source=[r['case'] for r in old['samples']]+old['diagnostic_cases']
    if set(old_source)!={r['case'] for r in base_eligibility['cases']}:
        raise ValueError('Historical eligibility inventory is incomplete')
    inventory=old_source+[r['case'] for r in acq['samples']]
    diagnostics=old['diagnostic_cases']+cal['pending_cases']+acq['calibration_pending_cases']
    partition(inventory,ids,diagnostics)
    groups=defaultdict(list)
    for row in rows:groups[source_group(row)].append(row['case'])
    plan=dict(benchmark_scope_version=VERSION,introduced_cycle=11,samples=rows,
        historical_cases=[r['case'] for r in old_rows],additional_cases=[r['case'] for r in new],
        source_cases=inventory,diagnostic_cases=diagnostics,
        dataset_counts=dict(Counter(r['dataset'] for r in rows)),
        source_dataset_counts=dict(Counter(r['dataset'] for r in base_eligibility['cases'])+Counter(r['dataset'] for r in acq['samples'])),
        acquisition_audit=str(acquisition_audit),acquisition_audit_sha256=sha(acquisition_audit),
        calibration_audit=str(calibration),calibration_audit_sha256=sha(calibration),
        source_plan_sha256=source_plans,added_source_roots=source_roots,added_feature_roots=caches,
        feature_proof=feature_proof,base_input_freeze_sha256=sha(base/'ALL_INPUTS_FROZEN.json'),
        physical_policy_sha256=sha(PHYSICAL_POLICY),eligibility_policy_sha256=sha(POLICY_PATH),
        conversions=['09','10','15','19','ensemble'],primary_conversion='ensemble',
        conversion_mechanisms_changed=False,all_images_are_development=True,all_analysis_images_are_ROIs=True,
        source_group_clusters=dict(groups),source_group_caveat='TCGA cases are grouped across sources. Other provider IDs remain namespaced; unknown linkage is not asserted independent.',
        incomplete_acquisition_datasets=acq['pending'],fine_reference_decoded_by_assembler=False,
        source_code_sha256=sha(__file__),bora_run=False)
    root.mkdir(parents=True,exist_ok=True);(root/'cases').mkdir(exist_ok=True)
    save_new(root/'PLAN.json',plan)
    repo=Path(__file__).resolve().parents[2]
    files=[p for package in ('bora_extension','virchow2_multi','virchow2_inverse','virchow2_tissue','virchow2_ensemble')
        for p in sorted((repo/'experiments'/package).glob('*.py'))]+[PHYSICAL_POLICY,POLICY_PATH]
    protocol=dict(plan_sha256=sha(root/'PLAN.json'),source_sha256={str(p.relative_to(repo)):sha(p) for p in files},
        fixed_conversion_mechanisms_unchanged=True,fine_reference_decoded=False,bora_run=False)
    save_new(root/'ASSEMBLY_PROTOCOL.json',protocol)
    for path in files:
        dest=root/'runtime_sources'/path.relative_to(repo);dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            if sha(dest)!=sha(path):raise ValueError('Archived assembly source changed')
        else:shutil.copy2(path,dest)
    for row in old_rows:
        folder=base/'cases'/row['case'];manifest=folder/'INPUTS_FROZEN.json'
        if sha(manifest)!=old_freeze['manifests'][row['case']]:raise ValueError('Historical input manifest changed')
        target=root/'cases'/row['case']
        if target.exists() or target.is_symlink():
            if not target.is_symlink() or target.resolve()!=folder.resolve():raise ValueError('Historical case target changed')
        else:target.symlink_to(folder.resolve(),target_is_directory=True)
    # Reference files are copied and byte-hashed without reading mask values.
    from .conversions import transfer_native_files
    for row in new:transfer_native_files(Path(source_roots[row['case']]),root,row)
    return plan


def convert(task):
    root,cache,sample=task
    verify_case(Path(cache)/'cases'/sample['case'],sample,calibration=require(sample))
    from .conversions import prepare_case
    return prepare_case((root,cache,sample))


def freeze(root):
    root=Path(root);plan=load(root/'PLAN.json');policy=load(POLICY_PATH);checks={};manifests={};native_hashes={}
    for sample in plan['samples']:
        case=sample['case'];folder=root/'cases'/case;m=load(folder/'INPUTS_FROZEN.json')
        for name,expected in m['files_sha256'].items():
            if sha(folder/name)!=expected:raise ValueError('Frozen input changed: '+case+'/'+name)
        if sha(folder/'reference.native.tif')!=sample['reference_native_sha256']:
            raise ValueError('Reference bytes changed during transfer')
        with Image.open(folder/'original.native.png') as im:
            if [im.height,im.width]!=sample['native_shape_yx']:raise ValueError('Native raster dimensions differ from calibrated plan')
        native_sha=sha(folder/'original.native.png')
        if native_sha in native_hashes:raise ValueError('Duplicate native image fields: '+native_hashes[native_sha]+' and '+case)
        native_hashes[native_sha]=case
        image=np.array(Image.open(folder/'original.png').convert('RGB'))
        if list(image.shape[:2])!=sample['working_shape_yx'] or np.all((image.min(-1)>=230)&(np.ptp(image.astype(np.int16),axis=-1)<=20)):
            raise ValueError('Invalid or empty tissue image')
        decision=inspect(tifffile.imread(folder/'coarse.tif'),image.shape[:2],policy)
        selection=load(folder/'selection.json');observed=selection.get('observed_classes',selection.get('classes'))
        if not decision['eligible'] or sorted(map(int,observed))!=decision['observed_groups']:
            raise ValueError('Input group/tile gate or feature selection changed')
        coverage=load(folder/'encoder_coverage.json')
        if not coverage['all_original_tokens_covered'] or coverage['original_shape_yx']!=sample['working_shape_yx']:
            raise ValueError('Full original-field encoder coverage missing')
        if 'checkpoint_sha256' in coverage:
            if coverage['checkpoint_sha256']!=MODEL_SHA256 or coverage['revision']!=MODEL_REVISION:
                raise ValueError('Wrong legacy encoder checkpoint')
        else:
            cache_path=Path(m['feature_cache']['base']['path']).parent.parent/'FEATURES_FROZEN.json'
            feature_record=load(cache_path)
            if sha(cache_path)!=m['feature_manifest_sha256'] or feature_record['encoder_checkpoint_sha256']!=MODEL_SHA256 or feature_record['encoder_revision']!=MODEL_REVISION:
                raise ValueError('Wrong or unbound encoder feature manifest')
        physical=require(sample);manifests[case]=sha(folder/'INPUTS_FROZEN.json')
        checks[case]=dict(**decision,physical_calibration=physical,input_manifest_sha256=manifests[case],
            original_sha256=sha(folder/'original.png'),coarse_sha256=sha(folder/'coarse.tif'))
    from .sampling_review import review
    sampling=review(root,root/'SAMPLING_REVIEW.json')
    if any(any(s['minimum_window_reflection_padding_yx']) or s['calibrated_axes_beyond_documented_approx_range']
           for r in sampling['cases'] for s in r['scales'].values()):
        raise ValueError('Incomplete physical context or unsupported cached model scale')
    report=dict(version=VERSION,cases=checks,source_code_sha256=sha(__file__),plan_sha256=sha(root/'PLAN.json'),
        assembly_protocol_sha256=sha(root/'ASSEMBLY_PROTOCOL.json'),
        eligibility_policy_sha256=sha(POLICY_PATH),physical_policy_sha256=sha(PHYSICAL_POLICY),
        sampling_review_sha256=sha(root/'SAMPLING_REVIEW.json'),
        fine_reference_read=False,predictions_read=False,scores_read=False,all_passed=True,bora_run=False)
    save_new(root/'COHORT_AUDIT.json',report)
    save_new(root/'ALL_INPUTS_FROZEN.json',dict(plan_sha256=sha(root/'PLAN.json'),cases=len(manifests),
        manifests=manifests,cohort_audit_sha256=sha(root/'COHORT_AUDIT.json'),
        assembly_protocol_sha256=sha(root/'ASSEMBLY_PROTOCOL.json'),
        conversions=plan['conversions'],physical_policy_sha256=sha(PHYSICAL_POLICY),
        eligibility_policy_sha256=sha(POLICY_PATH),all_original_fields_encoded=True,
        fine_references_decoded_by_converter=False,bora_run=False))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--acquisition-audit',type=Path,required=True);p.add_argument('--acquisitions',type=Path,nargs='+',required=True)
    p.add_argument('--features',type=Path,nargs='+',required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--workers',type=int,choices=[1,2,3],default=3);p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args()
    if shutil.disk_usage(a.root if a.root.exists() else a.root.parent).free<20*2**30:raise ValueError('Keep20GiB free on the compute volume')
    plan=prepare(a.base,a.calibration,a.acquisition_audit,a.acquisitions,a.features,a.root)
    if a.prepare_only:
        print(json.dumps(dict(prepared=len(plan['samples']),converted=False)));return
    tasks=[(a.root,plan['added_feature_roots'][r['case']],r) for r in plan['samples'] if r['case'] in plan['added_feature_roots']]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=get_context('spawn')) as pool:list(pool.map(convert,tasks))
    report=freeze(a.root);print(json.dumps(dict(frozen_cases=len(report['cases']),datasets=plan['dataset_counts'])))


if __name__=='__main__':main()
