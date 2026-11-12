"""Final cache assembly retains every original encoder channel and row."""
from pathlib import Path
import importlib.util,json
import numpy as np

def test_full_cache_assembly_is_byte_exact_and_registered(tmp_path):
 project=Path(__file__).parents[1];source=project/'migration/full1280-all-benchmark-cache-2026-10-29-v1/finalize_cache.py'
 spec=importlib.util.spec_from_file_location('cache_assembly',source);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
 r.DATA=tmp_path;r.ROOT=tmp_path/'analysis/cache-test';root=r.ROOT/'image';cal=root/'calibrated-01';cal.mkdir(parents=True)
 def save(path,data):path.write_text(json.dumps(data))
 save(cal/'PREPROCESSED_COMPLETE.local.json',dict(grid_shape_yx=[4,4],target_sha256='a'*64,source_sha256='b'*64,effective_mpp_xy=[.5,.5]));np.save(cal/'eligible-grid.npy',np.ones((4,4),bool))
 control=root/'GPU_CONTROL';control.mkdir();save(control/'GPU_FEATURES_COMPLETE.json',dict(channels=1280,all_gpu_cleanup_verified=True))
 originals={}
 for factor in (1,2,4):
  folder=root/'spatial-01'/f'factor-{factor}';folder.mkdir(parents=True);rows=[];width=64//factor
  for row in range(4//factor):
   values=np.broadcast_to(np.arange(1280,dtype=np.float16),(16,width,1280)).copy();values+=row;name=f'row-{row:03d}.npy';np.save(folder/name,values);rows.append(dict(row=row,file=name,sha256=r.sha256(folder/name)));originals[factor,row]=values
  save(folder/'FEATURES_COMPLETE.json',dict(status='COMPLETE',channel_count=1280,calibrated_image_sha256='a'*64,row_receipts=rows,checkpoint_sha256='c'*64))
 pooled=root/'pooled-full-01';pooled.mkdir();rows=[]
 for row in range(4):
  p=pooled/f'row-{row:03d}.npy';np.save(p,np.zeros((4,1280),np.float32));rows.append(dict(row=row,file=p.name,sha256=r.sha256(p)))
 save(pooled/'GRID_COMPLETE.json',dict(status='COMPLETE',rows=rows));r.main('image')
 complete=json.loads((root/'CACHE_COMPLETE.json').read_text());assert complete['channels']==1280
 for factor in (1,2,4):
  a=np.load(complete['features'][str(factor)]['path'],mmap_mode='r');assert a.shape[-1]==1280
  for row in range(4//factor):assert np.array_equal(a[row*16:(row+1)*16],originals[factor,row])
 record=r.verify_stage(tmp_path,complete['spatial_cache_manifest']);assert record['identity']['encoder']['channels']==1280
