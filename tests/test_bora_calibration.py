import pytest
import json

from experiments.bora_extension.calibration import evaluate,require


def calibrated():
    return dict(case='physical_fixture',native_shape_yx=[2560,2560],working_shape_yx=[1280,1280],
        native_mpp_xy=[.25,.25],working_mpp_xy=[.5,.5],fov_um_xy=[640,640])


def test_window_field_and_token_units_follow_physical_scale():
    r=require(calibrated())
    assert [r['scales'][str(f)]['window_um_xy'][0] for f in (1,2,4)]==[112,224,448]
    assert r['scales']['1']['token_um_xy']==[7,7]
    assert r['scales']['1']['source_pixels_per_window_xy']==[448,448]
    assert r['coarse_tile_um_xy']==[45,45]


def test_magnification_or_large_pixel_count_cannot_replace_mpp():
    sample=calibrated();sample.update(native_mpp_xy=None,working_mpp_xy=None,source=dict(magnification=40))
    with pytest.raises(ValueError,match='physical calibration required'):require(sample)


def test_wrong_axis_projection_and_physical_context_are_rejected():
    sample=calibrated();sample['working_mpp_xy']=[.5,1.]
    assert 'native_and_working_fields_disagree_in_micrometres' in evaluate(sample)['reasons']
    sample=calibrated();sample.update(working_mpp_xy=[1,1],native_mpp_xy=[.5,.5],fov_um_xy=[1280,1280])
    assert 'unsupported_or_inconsistent_physical_sampling_at_scale_4' in evaluate(sample)['reasons']


def test_interpolation_is_reported_without_claiming_new_optical_detail():
    sample=calibrated();sample.update(native_mpp_xy=[1,1],native_shape_yx=[896,896],working_shape_yx=[1792,1792],fov_um_xy=[896,896])
    r=require(sample)
    assert r['scales']['1']['interpolation_upscale_xy']==[2,2]
    assert r['scales']['1']['source_pixels_per_window_xy']==[112,112]
    assert r['optical_resolving_power_um'] is None


def test_unknown_calibration_is_rejected_before_loading_or_encoding(tmp_path):
    from experiments.bora_extension.features import freeze_case
    called=[]
    sample=calibrated();sample['native_mpp_xy']=None
    with pytest.raises(ValueError,match='physical calibration required'):
        freeze_case(tmp_path,tmp_path,sample,None,lambda *args:called.append(args))
    assert not called and not list(tmp_path.iterdir())


@pytest.mark.parametrize('change',[
    {'pixel_only':True},{'physical_metrics_allowed':False},
    {'native_shape_yx':[0,2560]},{'working_shape_yx':[1280,float('nan')]},
    {'native_mpp_xy':[True,.25]},{'fov_um_xy':[float('nan'),640]},
])
def test_invalid_or_contradictory_physical_metadata_fails_closed(change):
    sample=calibrated();sample.update(change)
    assert not evaluate(sample)['eligible']


def test_partial_request_cannot_sign_a_plan_containing_uncalibrated_caches(tmp_path,monkeypatch):
    import builtins
    import sys
    from experiments.bora_extension import features
    source=tmp_path/'source';source.mkdir();model=tmp_path/'model';model.mkdir()
    (model/'config.json').write_text('{}')
    known=calibrated();unknown=dict(known,case='unknown',native_mpp_xy=None,working_mpp_xy=None)
    (source/'PLAN.json').write_text(json.dumps(dict(samples=[known,unknown])))
    monkeypatch.setattr(sys,'argv',['features','--sources',str(source),'--root',str(tmp_path/'out'),
        '--model-dir',str(model),'--cases',known['case']])
    original_import=builtins.__import__
    def no_gpu(name,*args,**kwargs):
        if name=='torch':raise AssertionError('Uncalibrated plan reached GPU import')
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',no_gpu)
    with pytest.raises(ValueError,match='physical calibration required for unknown'):
        features.main()
    assert not (tmp_path/'out/ALL_FEATURES_FROZEN.json').exists()
