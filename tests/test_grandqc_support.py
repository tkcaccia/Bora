import json
import numpy as np
import pytest
from PIL import Image
import tifffile
from bora.selected_wsi import sha
from bora.grandqc_support import bind_support,stream_support
from bora.wsi_frozen import read_case

def inputs(folder):
 image=folder/'he.png';Image.new('RGB',(600,400),(180,100,140)).save(image)
 mask=np.zeros((4,6),np.uint8);mask[:,:3]=255;path=folder/'x_grandqc_clean_tissue_mask.tif';tifffile.imwrite(path,mask)
 summary=dict(tissue_mask_source='grandqc_model',level0_shape_yx=[400,600],source_mpp=.5,qc_mask_shape_yx=[4,6],tissue_mask_shape_yx=[4,6],model_provenance={k:dict(checkpoints=[dict(status='verified')]) for k in ('tissue','artifact')})
 meta=folder/'x_grandqc_summary.json';meta.write_text(json.dumps(summary))
 cache=folder/'qc-cache.json';cache.write_text(json.dumps(dict(status='COMPLETE',stage='grandqc',identity=dict(image_sha256=sha(image)),outputs={path.name:dict(sha256=sha(path)),meta.name:dict(sha256=sha(meta))})))
 spec=dict(format='bora-grandqc-support-v1',mode='artifact-free',mask=dict(path=path.name,sha256=sha(path)),summary=dict(path=meta.name,sha256=sha(meta)),cache_manifest=dict(path=cache.name,sha256=sha(cache)))
 return spec,image

def test_cached_mask_is_aligned_without_rerunning_qc(tmp_path):
 spec,image=inputs(tmp_path);identity=bind_support(spec,tmp_path,sha(image),[400,600],[.5,.5]);out=tmp_path/'out';out.mkdir();support=stream_support(identity,[240,450],out)
 assert support.shape==(240,450) and support.dtype==np.uint8
 assert np.all(support[:,:300]==1) and np.all(support[:,300:]==0)
 rec=json.loads((out/'GRANDQC_SUPPORT_USED.json').read_text());assert rec['no_inference_rerun'] and not rec['expert_annotation_used']

def test_foreign_image_or_wrong_spacing_is_rejected(tmp_path):
 spec,image=inputs(tmp_path)
 with pytest.raises(ValueError,match='different calibrated'):bind_support(spec,tmp_path,'f'*64,[400,600],[.5,.5])
 with pytest.raises(ValueError,match='spacing'):bind_support(spec,tmp_path,sha(image),[400,600],[.25,.25])
 spec['annotation']='expert.png'
 with pytest.raises(ValueError,match='predictor mask'):bind_support(spec,tmp_path,sha(image),[400,600],[.5,.5])

def test_wsi_input_binds_grandqc_predictor(tmp_path):
 spec,image=inputs(tmp_path);groups=np.ones((4,4),np.uint8);groups[:,2:]=2;np.save(tmp_path/'groups.npy',groups);np.save(tmp_path/'prob.npy',np.full((64,64,2),.5,np.float16))
 response=dict(format='bora-slide-global-responses-v1',status='COMPLETE',source_frame_id='qc-frame',source_image_sha256=sha(image),group_grid_sha256=sha(tmp_path/'groups.npy'),probability_sha256=sha(tmp_path/'prob.npy'),shape_yxc=[64,64,2],class_ids=[1,2],one_fit_used_across_slide=True,independently_fitted_windows=False,model_sha256='a'*64)
 (tmp_path/'response.json').write_text(json.dumps(response))
 case=dict(format='bora-wsi-colorectal-v1',source_frame_id='qc-frame',image='he.png',group_grid='groups.npy',probabilities='prob.npy',response_receipt='response.json',shape_yx=[400,600],working_mpp_xy=[.5,.5],native_mpp_xy=[.25,.25],native_shape_yx=[800,1200],quality_control=spec)
 path=tmp_path/'case.json';path.write_text(json.dumps(case));_,_,_,identity=read_case(path)
 assert identity['quality_control']['mode']=='artifact-free'
 assert identity['quality_control']['mask_sha256']==spec['mask']['sha256']

def test_full_geometry_uses_saved_qc_on_cpu(tmp_path):
 from bora.wsi_frozen import run_wsi
 spec,image=inputs(tmp_path);groups=np.ones((4,4),np.uint8);groups[:,2:]=2;np.save(tmp_path/'groups.npy',groups);np.save(tmp_path/'prob.npy',np.full((64,64,2),.5,np.float16))
 response=dict(format='bora-slide-global-responses-v1',status='COMPLETE',source_frame_id='qc-geometry',source_image_sha256=sha(image),group_grid_sha256=sha(tmp_path/'groups.npy'),probability_sha256=sha(tmp_path/'prob.npy'),shape_yxc=[64,64,2],class_ids=[1,2],one_fit_used_across_slide=True,independently_fitted_windows=False,model_sha256='a'*64)
 (tmp_path/'response.json').write_text(json.dumps(response))
 case=dict(format='bora-wsi-colorectal-v1',source_frame_id='qc-geometry',image='he.png',group_grid='groups.npy',probabilities='prob.npy',response_receipt='response.json',shape_yx=[400,600],working_mpp_xy=[.5,.5],native_mpp_xy=[.25,.25],native_shape_yx=[800,1200],quality_control=spec)
 path=tmp_path/'case.json';path.write_text(json.dumps(case));out=tmp_path/'result';run_wsi(path,out,minimum_free_bytes=0,memory_budget_bytes=1<<30)
 support=np.load(out/'support.npy',mmap_mode='r');assert np.all(support[:,:300]==1) and np.all(support[:,300:]==0)
 result=json.loads((out/'FULL_SLIDE_COMPLETE.json').read_text());assert result['status']=='COMPLETE'
 assert 'GRANDQC_SUPPORT_USED.json' in result['files_sha256']
 assert (out/'BORA_FULL_SLIDE.wsi.geojson').is_file()
