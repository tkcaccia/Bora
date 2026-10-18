"""Versioned scope gates; completed cycles retain their original membership."""
from collections import Counter
import json
from pathlib import Path

EXPECTED = {"bcss": 132, "tiger": 80, "wsss4luad": 40, "rings-tumor": 40,
            "ocelot": 40, "beetle": 10, "camelyon16-subset": 7}


def validate_scope(protocol, *, pilot=False):
    scope = protocol.get("benchmark_scope")
    if scope is None:
        if protocol.get("cycle", 1) >= 8:
            raise ValueError("Cycle8 and later require the expanded benchmark contract")
        if not pilot and len(protocol["cases"]) != 149:
            raise ValueError("Historical full cycles require149 images")
        return
    if scope["version"] == "qualified-multigroup-v1":
        return validate_qualified(protocol, pilot=pilot)
    if scope['version']=='calibrated-tissues-v1':
        return validate_calibrated(protocol,pilot=pilot)
    if protocol.get("cycle", 1) >= 11:
        raise ValueError("Cycle11 onward requires the multigroup/real-tile eligibility audit")
    if protocol["cycle"] < 8 or scope["version"] != "expanded349-v1":
        raise ValueError("Expanded scope is prospective from cycle8")
    required, historical = scope["required_cases"], scope["historical_cases"]
    if (len(required) != 349 or len(set(required)) != 349 or len(historical) != 149
            or len(set(historical)) != 149 or not set(historical) <= set(required)):
        raise ValueError("Expanded scope must retain149 historical and200 added cases")
    if scope["dataset_counts"] != EXPECTED or scope["plan_sha256"] != protocol["input_plan_sha256"]:
        raise ValueError("Expanded dataset counts/plan identity changed")
    if not set(protocol["cases"]) <= set(required):
        raise ValueError("Unexpected benchmark case")
    if len(protocol["cases"]) != len(set(protocol["cases"])):
        raise ValueError("Duplicate benchmark case")
    if not pilot and set(protocol["cases"]) != set(required):
        raise ValueError("Full expanded cycle requires every349 case")


def validate_qualified(protocol, *, pilot=False):
    scope = protocol["benchmark_scope"]
    if protocol["cycle"] < 11:
        raise ValueError("Qualified scope must not rewrite completed cycles1-10")
    required, excluded = scope["required_cases"], scope["diagnostic_cases"]
    source = scope["source_cases"]
    if (not required or len(set(source)) != 349 or len(source) != 349
            or len(set(required)) != len(required) or len(set(excluded)) != len(excluded)
            or set(required) & set(excluded) or set(required) | set(excluded) != set(source)):
        raise ValueError("Qualified and diagnostic inventories must partition the original349")
    if (scope["source_dataset_counts"] != EXPECTED
            or sum(scope["dataset_counts"].values()) != len(required)
            or scope["plan_sha256"] != protocol["input_plan_sha256"]
            or not set(scope["historical_cases"]) <= set(required)):
        raise ValueError("Qualified dataset counts/plan identity changed")
    checks = scope["eligible_case_checks"]
    if set(checks) != set(required):
        raise ValueError("Every benchmark image needs its own positive eligibility decision")
    for check in checks.values():
        if (not check["eligible"] or check["exclusion_reasons"]
                or len(check["input_groups"]) < 2 or len(check["observed_groups"]) < 2
                or min(check["complete_base_fields_yx"]) < 4
                or check["usable_coarse_tiles"] < 16):
            raise ValueError("Single-group or undersized image cannot enter the benchmark")
    if len(protocol["cases"]) != len(set(protocol["cases"])) or not set(protocol["cases"]) <= set(required):
        raise ValueError("Unexpected or duplicate qualified benchmark case")
    if not pilot and set(protocol["cases"]) != set(required):
        raise ValueError("A full cycle must include every qualified image")


def requires_parent_parity(protocol, case):
    """New images have no previous mask; old images retain exact parity checks."""
    scope = protocol.get("benchmark_scope")
    if scope and scope['version']=='calibrated-tissues-v1':
        return protocol['cycle']>scope['introduced_cycle'] or case in scope['historical_cases']
    return scope is None or case in scope["historical_cases"] or protocol["cycle"] > 8


def validate_calibrated(protocol,*,pilot=False):
    from experiments.bora_extension.calibrated_cohort import partition
    scope=protocol['benchmark_scope'];required=scope['required_cases'];old=scope['historical_cases'];added=scope['additional_cases']
    if not required or scope['introduced_cycle']<11 or protocol['cycle']<scope['introduced_cycle']:
        raise ValueError('Calibrated scope is prospective from its declared cycle')
    partition(scope['source_cases'],required,scope['diagnostic_cases'])
    if (len(set(old))!=len(old) or len(set(added))!=len(added) or set(old)&set(added)
            or set(old)|set(added)!=set(required) or sum(scope['dataset_counts'].values())!=len(required)
            or sum(scope['source_dataset_counts'].values())!=len(scope['source_cases'])
            or sum(scope['new_dataset_counts'].values())!=len(added)
            or any(v!=40 for v in scope['new_dataset_counts'].values())
            or scope['plan_sha256']!=protocol['input_plan_sha256']):
        raise ValueError('Calibrated membership, dataset quotas or plan identity changed')
    checks=scope['eligible_case_checks']
    if set(checks)!=set(required):raise ValueError('Every calibrated case needs an eligibility decision')
    for c in checks.values():
        physical=c['physical_calibration']
        if (not c['eligible'] or c['exclusion_reasons'] or len(c['input_groups'])<2 or len(c['observed_groups'])<2
                or min(c['complete_base_fields_yx'])<4 or c['usable_coarse_tiles']<16
                or not physical['eligible'] or physical['reasons'] or physical['native_mpp_xy'] is None
                or physical['working_mpp_xy'] is None or set(physical['scales'])!={'1','2','4'}):
            raise ValueError('An uncalibrated, single-group or undersized case entered the scope')
    cases=protocol['cases']
    if len(cases)!=len(set(cases)) or not set(cases)<=set(required):raise ValueError('Unexpected calibrated benchmark case')
    if not pilot and set(cases)!=set(required):raise ValueError('Full cycle requires every calibrated benchmark case')


def load_scope(source, cycle):
    from .run import digest
    source = Path(source)
    if cycle < 8:
        return None
    plan = json.loads((source/"PLAN.json").read_text())
    frozen = json.loads((source/"ALL_INPUTS_FROZEN.json").read_text())
    if cycle >= 11:
        from experiments.bora_extension.calibration import require
        for sample in plan["samples"]:
            require(sample)
    ids = [s["case"] for s in plan["samples"]]
    counts = dict(Counter(s["dataset"] for s in plan["samples"]))
    sha = digest(source/"PLAN.json")
    if frozen["plan_sha256"] != sha or set(frozen["manifests"]) != set(ids) or frozen["cases"] != len(ids):
        raise ValueError("Expanded all-input freeze missing or changed")
    for case, expected in frozen["manifests"].items():
        if digest(source/"cases"/case/"INPUTS_FROZEN.json") != expected:
            raise ValueError("Expanded case input manifest changed")
    if plan.get('benchmark_scope_version')=='calibrated-tissues-v1':
        from experiments.bora_extension.calibration import POLICY_PATH as PHYSICAL_POLICY,require
        from experiments.bora_extension.eligibility import POLICY_PATH
        audit= json.loads((source/'COHORT_AUDIT.json').read_text())
        assembly=json.loads((source/'ASSEMBLY_PROTOCOL.json').read_text())
        if (not audit['all_passed'] or audit['fine_reference_read'] or audit['scores_read'] or audit['predictions_read']
                or audit['plan_sha256']!=sha or frozen['cohort_audit_sha256']!=digest(source/'COHORT_AUDIT.json')
                or assembly['plan_sha256']!=sha or frozen['assembly_protocol_sha256']!=digest(source/'ASSEMBLY_PROTOCOL.json')
                or audit['assembly_protocol_sha256']!=digest(source/'ASSEMBLY_PROTOCOL.json')
                or audit['sampling_review_sha256']!=digest(source/'SAMPLING_REVIEW.json')
                or any(r.get('physical_policy_sha256')!=digest(PHYSICAL_POLICY) for r in (plan,audit,frozen))
                or any(r.get('eligibility_policy_sha256')!=digest(POLICY_PATH) for r in (plan,audit,frozen))
                or set(audit['cases'])!=set(ids)):
            raise ValueError('Calibrated input, policy, assembly or sampling evidence changed')
        for name,expected in assembly['source_sha256'].items():
            if digest(source/'runtime_sources'/name)!=expected:raise ValueError('Archived RGB assembly source changed')
        for path,expected in plan['source_plan_sha256'].items():
            if digest(Path(path)/'PLAN.json')!=expected:raise ValueError('Source inventory plan changed')
        for name,field in [('acquisition_audit','acquisition_audit_sha256'),('calibration_audit','calibration_audit_sha256')]:
            if digest(Path(plan[name]))!=plan[field]:raise ValueError('Acquisition/calibration audit changed')
        checks={}
        for sample in plan['samples']:
            case=sample['case'];c=audit['cases'][case]
            if c['input_manifest_sha256']!=frozen['manifests'][case] or c['physical_calibration']!=require(sample):
                raise ValueError('Eligibility/calibration belongs to different case inputs')
            checks[case]={k:c[k] for k in ('eligible','exclusion_reasons','input_groups','observed_groups',
                'complete_base_fields_yx','usable_coarse_tiles','physical_calibration','input_manifest_sha256')}
        added=set(plan['additional_cases'])
        scope=dict(version='calibrated-tissues-v1',introduced_cycle=plan['introduced_cycle'],required_cases=ids,
            source_cases=plan['source_cases'],diagnostic_cases=plan['diagnostic_cases'],historical_cases=plan['historical_cases'],
            additional_cases=plan['additional_cases'],dataset_counts=counts,source_dataset_counts=plan['source_dataset_counts'],
            new_dataset_counts=dict(Counter(r['dataset'] for r in plan['samples'] if r['case'] in added)),
            eligible_case_checks=checks,plan_sha256=sha,all_inputs_sha256=digest(source/'ALL_INPUTS_FROZEN.json'),
            cohort_audit_sha256=digest(source/'COHORT_AUDIT.json'),physical_policy_sha256=digest(PHYSICAL_POLICY),
            eligibility_policy_sha256=digest(POLICY_PATH))
        validate_scope(dict(cycle=cycle,cases=ids,benchmark_scope=scope,input_plan_sha256=sha))
        return scope
    if plan.get("benchmark_scope_version") == "qualified-multigroup-v1":
        from experiments.bora_extension.eligibility import POLICY_PATH
        audit_path, policy_path = source/"ELIGIBILITY.json", source/"POLICY.json"
        audit = json.loads(audit_path.read_text())
        if (digest(audit_path) != plan["eligibility_sha256"]
                or digest(audit_path) != frozen["eligibility_sha256"]
                or digest(policy_path) != plan["eligibility_policy_sha256"]
                or json.loads(policy_path.read_text()) != json.loads(POLICY_PATH.read_text())
                or audit["protocol"]["policy_sha256"] != digest(policy_path)
                or audit["protocol"]["input_plan_sha256"] != plan["eligibility_source_plan_sha256"]
                or audit["protocol"]["fine_reference_read"] or audit["protocol"]["scores_read"]
                or audit["protocol"]["predictions_read"]):
            raise ValueError("Frozen score-blind eligibility evidence changed")
        if set(ids) != set(audit["eligible_cases"]) or set(plan["diagnostic_cases"]) != set(audit["excluded_cases"]):
            raise ValueError("Benchmark membership differs from the eligibility audit")
        checks = {r["case"]: r for r in audit["cases"] if r["eligible"]}
        if any(checks[case]["input_manifest_sha256"] != frozen["manifests"][case] for case in ids):
            raise ValueError("Eligibility decision belongs to different frozen inputs")
        compact_checks = {case: {key: check[key] for key in (
            "eligible", "exclusion_reasons", "input_groups", "observed_groups",
            "complete_base_fields_yx", "usable_coarse_tiles", "input_manifest_sha256")}
            for case, check in checks.items()}
        scope = dict(version="qualified-multigroup-v1", required_cases=ids,
            diagnostic_cases=plan["diagnostic_cases"],
            historical_cases=plan["historical_cases"], dataset_counts=counts,
            source_cases=[r["case"] for r in audit["cases"]],
            source_dataset_counts={d: r["total"] for d,r in audit["datasets"].items()},
            eligible_case_checks=compact_checks, eligibility_sha256=digest(audit_path),
            eligibility_policy_sha256=digest(policy_path),
            plan_sha256=sha, all_inputs_sha256=digest(source/"ALL_INPUTS_FROZEN.json"))
        validate_scope(dict(cycle=cycle, cases=ids, benchmark_scope=scope, input_plan_sha256=sha))
        return scope
    scope = dict(version="expanded349-v1", required_cases=ids,
        historical_cases=plan["historical_cases"], dataset_counts=counts,
        plan_sha256=sha, all_inputs_sha256=digest(source/"ALL_INPUTS_FROZEN.json"))
    validate_scope(dict(cycle=cycle, cases=ids, benchmark_scope=scope, input_plan_sha256=sha))
    return scope
