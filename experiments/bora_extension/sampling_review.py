"""Review frozen encoder scale metadata; no inference, labels or score access."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .eligibility import digest, save_new


def review(source, output):
    source = Path(source)
    plan = json.loads((source/'PLAN.json').read_text())
    rows = []
    for sample in plan['samples']:
        folder = source/'cases'/sample['case']
        inputs = json.loads((folder/'INPUTS_FROZEN.json').read_text())
        scales = {}
        h, w = sample['working_shape_yx']
        spacing = sample['working_mpp_xy']
        for name, factor in [('base',1),('scale_1',2),('scale_2',4)]:
            cache = inputs['feature_cache'][name]
            path = Path(cache['path']).with_name('READY.json' if name == 'base' else name+'.json')
            metadata = json.loads(path.read_text())
            shape = [round(h/factor),round(w/factor)]
            if metadata.get('original_field_shape_yx',metadata.get('shape_yx',shape)) != shape:
                raise ValueError('Real sampling shape differs from metadata: '+sample['case']+'/'+name)
            if metadata['sha256'] != cache['sha256'] or metadata['window_px'] != 224 or metadata['patch_px'] != 14:
                raise ValueError('Encoder identity differs from the frozen cache')
            if name == 'base' and metadata['image_sha256'] != sample['original_sha256']:
                raise ValueError('Encoder input RGB differs from the input plan')
            axes = None if spacing is None else [spacing[0]*w/shape[1],spacing[1]*h/shape[0]]
            declared = metadata.get('sampling',{}).get('mpp_xy',metadata.get('mpp_xy'))
            if declared is not None and axes is not None and any(abs(a-b) > 1e-9 for a,b in zip(axes,declared)):
                raise ValueError('Cached axis spacing disagrees with real resized geometry')
            pads = [max(0,224-v) for v in shape]
            capacity = [v//224 for v in shape]
            nominal = 10/factor if sample['dataset'] == 'wsss4luad' else None
            scales[name] = dict(factor=factor, real_shape_yx=shape,
                minimum_window_reflection_padding_yx=pads,
                first_window_real_area_fraction=min(shape[0],224)*min(shape[1],224)/(224*224),
                complete_disjoint_field_capacity=capacity[0]*capacity[1],
                mpp_xy=axes, encoder_window_um_xy=None if axes is None else [224*v for v in axes],
                encoder_token_um_xy=None if axes is None else [14*v for v in axes],
                nominal_objective_x=nominal,
                nominal_below_documented_5x=nominal is not None and nominal < 5,
                calibrated_axes_beyond_documented_approx_range=(None if axes is None else
                    any(v < .25*.99 or v > 2*1.01 for v in axes)),
                metadata_path=str(path), metadata_sha256=digest(path),
                array_declared_sha256=cache['sha256'], array_bytes_rehashed_in_this_review=False)
        rows.append(dict(case=sample['case'],dataset=sample['dataset'],
            native_shape_yx=sample['native_shape_yx'], native_mpp_xy=sample['native_mpp_xy'],
            working_shape_yx=[h,w], working_mpp_xy=spacing,
            coarse_tile_px=90, coarse_tile_um_xy=None if spacing is None else [90*v for v in spacing],
            scales=scales))
    datasets = {}
    for dataset in sorted({r['dataset'] for r in rows}):
        subset = [r for r in rows if r['dataset'] == dataset]
        datasets[dataset] = dict(cases=len(subset),
            calibrated=sum(r['working_mpp_xy'] is not None for r in subset),
            scales={name:dict(minimum_window_padded_cases=sum(any(r['scales'][name]['minimum_window_reflection_padding_yx']) for r in subset),
                no_complete_real_window_cases=sum(r['scales'][name]['complete_disjoint_field_capacity'] == 0 for r in subset),
                nominal_below_documented_5x=sum(r['scales'][name]['nominal_below_documented_5x'] for r in subset))
                for name in ('base','scale_1','scale_2')})
    result = dict(cases=rows,datasets=datasets,input_plan_sha256=digest(source/'PLAN.json'),
        source_code_sha256=digest(__file__),
        model_source='https://huggingface.co/paige-ai/Virchow2/blob/main/README.md',
        wsss_source='https://arxiv.org/html/2204.06455v2',
        documented_model_sampling_mpp=[.25,.5,1,2],
        note='Mixed-magnification training does not imply invariance to arbitrary scales. Reported magnification does not establish per-file MPP. Array identities are compared to frozen declarations; array bytes were audited previously, not rehashed here.',
        fine_reference_read=False,predictions_read=False,scores_read=False,bora_run=False,encoder_run=False)
    save_new(output,result)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=review(args.source,args.output)
    print(json.dumps(dict(datasets=result['datasets'])))
