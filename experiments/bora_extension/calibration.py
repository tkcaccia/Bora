"""Physical Virchow2 sampling gate; no model, fine mask or score access."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

POLICY_PATH=Path(__file__).with_name('CALIBRATION_POLICY.json')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(sample):
    policy=json.loads(POLICY_PATH.read_text());reasons=[]
    def axes(key):
        value=sample.get(key)
        if not isinstance(value,(list,tuple)) or len(value)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in value):
            reasons.append('missing_or_invalid_'+key);return None
        return list(map(float,value))
    native=axes('native_mpp_xy');work=axes('working_mpp_xy')
    shape=sample.get('working_shape_yx');native_shape=sample.get('native_shape_yx')
    def valid_shape(value):
        return isinstance(value,(list,tuple)) and len(value)==2 and all(type(v) is int and v>0 for v in value)
    geometry_ok=valid_shape(shape) and valid_shape(native_shape)
    if not geometry_ok:
        reasons.append('invalid_physical_image_geometry')
    if native and work and (sample.get('pixel_only') is True or sample.get('physical_metrics_allowed') is False):
        reasons.append('calibration_contradicts_pixel_only_status')
    scales={};physical=None
    if native and work and geometry_ok:
        physical=[native_shape[1]*native[0],native_shape[0]*native[1]]
        expected=[shape[1]*work[0],shape[0]*work[1]]
        if any(abs(a-b)>max(1e-6,abs(a)*1e-8) for a,b in zip(physical,expected)):
            reasons.append('native_and_working_fields_disagree_in_micrometres')
        stated=sample.get('fov_um_xy')
        stated_ok=isinstance(stated,(list,tuple)) and len(stated)==2 and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in stated)
        if stated is not None and (not stated_ok or any(abs(a-b)>max(1e-6,abs(a)*1e-8) for a,b in zip(physical,stated))):
            reasons.append('declared_physical_field_disagrees')
        if min(physical)<policy['minimum_base_physical_field_um_per_axis']*(1-policy['relative_sampling_rounding_tolerance']):
            reasons.append('insufficient_physical_tissue_field')
        for factor,target in zip(policy['scale_factors'],policy['target_scale_mpp']):
            h,w=round(shape[0]/factor),round(shape[1]/factor)
            if min(h,w)<224:
                reasons.append('no_real_encoder_window_at_scale_'+str(factor));continue
            spacing=[physical[0]/w,physical[1]/h]
            if any(abs(v/target-1)>policy['relative_sampling_rounding_tolerance'] for v in spacing):
                reasons.append('unsupported_or_inconsistent_physical_sampling_at_scale_'+str(factor))
            scales[str(factor)]=dict(shape_yx=[h,w],actual_mpp_xy=spacing,target_mpp=target,
                window_um_xy=[224*v for v in spacing],token_um_xy=[14*v for v in spacing],
                stride_um_xy=[112*v for v in spacing],source_pixels_per_window_xy=[224*v/n for v,n in zip(spacing,native)],
                interpolation_upscale_xy=[max(1.,n/v) for n,v in zip(native,spacing)])
    source=sample.get('source') or {}
    evidence={k:source[k] for k in ('microscopy','calibration_source','resolution_status','source_record','record',
        'mpp_um','mpp_x_um','mpp_y_um','source_mpp_xy','source_coordinates','source_tiff_mpp_xy',
        'source_tiff_resolution_units','metadata_sha256') if k in source}
    return dict(case=sample.get('case'),eligible=not reasons,reasons=reasons,
        native_mpp_xy=native,working_mpp_xy=work,physical_field_um_xy=physical,
        coarse_tile_um_xy=None if work is None else [90*v for v in work],scales=scales,
        optical_resolving_power_um=None,
        calibration_evidence=dict(resolution_record=sample.get('resolution'),source_metadata=evidence),
        objective_or_scanner_report=source.get('microscopy'),
        unknown_mpp_never_inferred=True)


def require(sample):
    result=evaluate(sample)
    if not result['eligible']:
        raise ValueError('Virchow2 physical calibration required for '+str(sample.get('case'))+': '+', '.join(result['reasons']))
    return result


def audit_plan(source,out):
    source,out=Path(source),Path(out);out.mkdir(parents=True,exist_ok=True)
    plan=json.loads((source/'PLAN.json').read_text());rows=[evaluate(s) for s in plan['samples']]
    result=dict(policy=json.loads(POLICY_PATH.read_text()),policy_sha256=digest(POLICY_PATH),
        source_plan_sha256=digest(source/'PLAN.json'),source_code_sha256=digest(__file__),
        cases=rows,ready_cases=[r['case'] for r in rows if r['eligible']],
        pending_cases=[r['case'] for r in rows if not r['eligible']],
        datasets={d:dict(total=sum(s['dataset']==d for s in plan['samples']),ready=sum(s['dataset']==d and r['eligible'] for s,r in zip(plan['samples'],rows))) for d in sorted({s['dataset'] for s in plan['samples']})},
        no_encoder_or_bora_run=True,fine_reference_read=False,predictions_read=False,scores_read=False,
        notice='Calibration audit only. No new benchmark cohort or score is signed. Acquisition/group/tile gates still apply independently.')
    p=out/'CALIBRATION.json'
    if p.exists() and json.loads(p.read_text())!=result:raise FileExistsError('Calibration evidence changed; choose a new output version')
    p.write_text(json.dumps(result,indent=2)+'\n');(out/'POLICY.json').write_text(POLICY_PATH.read_text())
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=audit_plan(a.source,a.out);print(json.dumps(dict(ready=len(r['ready_cases']),pending=len(r['pending_cases']),datasets=r['datasets'])))
