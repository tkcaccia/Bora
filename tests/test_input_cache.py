import importlib.util,json
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('input_cache',Path(__file__).parents[1]/'benchmark/input_cache.py');cache=importlib.util.module_from_spec(spec);spec.loader.exec_module(cache)
def test_cache_reuse_and_corruption(tmp_path):
 asset=tmp_path/'features.npy';asset.write_bytes(b'encoded-input');identity=dict(image_sha256='a'*64,encoder=dict(checkpoint_sha256='b'*64,channels=1280))
 manifest=cache.register_stage(tmp_path,'virchow2_spatial',identity,{'features':asset})
 assert cache.register_stage(tmp_path,'virchow2_spatial',identity,{'features':asset})==manifest
 assert cache.verify_stage(tmp_path,manifest)['identity']==identity
 asset.write_bytes(b'different-input')
 with pytest.raises(ValueError):cache.verify_stage(tmp_path,manifest)
 with pytest.raises(ValueError):cache.register_stage(tmp_path,'virchow2_spatial',identity,{'features':asset})
def test_model_identity_and_no_bora_dependency():
 a=dict(image_sha256='a',encoder=dict(checkpoint_sha256='b'));b=dict(image_sha256='a',encoder=dict(checkpoint_sha256='c'))
 assert cache.cache_key('virchow2_pooled',a)!=cache.cache_key('virchow2_pooled',b)
 with pytest.raises(ValueError):cache.cache_key('virchow2_spatial',dict(a,selected_dimensions=[1,2]))
def test_no_partial_or_external_registration(tmp_path):
 with pytest.raises(ValueError):cache.register_stage(tmp_path,'grandqc',{}, {})
 with pytest.raises(ValueError):cache.checked_path(tmp_path,'../outside')
 with pytest.raises(ValueError):cache.checked_path(tmp_path,'/etc/passwd')

def test_benchmark_gate_requires_every_stage_and_same_image(tmp_path):
 asset=tmp_path/'asset';asset.write_bytes(b'cached-bytes');stages={}
 for stage in ('preprocessing','virchow2_pooled','virchow2_spatial','grandqc'):
  identity=dict(calibrated_sha256='same-image') if stage=='preprocessing' else dict(image_sha256='same-image')
  if stage.startswith('virchow2'):identity['encoder']=dict(channels=1280)
  outputs={'asset':asset}
  if stage=='grandqc':outputs={'x'+suffix:asset for suffix in ('_grandqc_class_mask.tif','_grandqc_tissue_detection_mask.tif','_grandqc_background_mask.tif','_grandqc_artifact_mask.tif','_grandqc_clean_tissue_mask.tif')}
  stages[stage]=cache.register_stage(tmp_path,stage,identity,outputs)
 assert set(cache.require_image_inputs(tmp_path,stages))==set(stages)
 with pytest.raises(ValueError,match='all required'):cache.require_image_inputs(tmp_path,{k:v for k,v in stages.items() if k!='grandqc'})
 wrong=cache.register_stage(tmp_path,'virchow2_pooled',dict(image_sha256='other-image',encoder=dict(channels=1280)),{'asset':asset})
 with pytest.raises(ValueError,match='different calibrated'):cache.require_image_inputs(tmp_path,dict(stages,virchow2_pooled=wrong))
