"""Bora-owned selection from immutable, full 1280-channel Virchow2 inputs."""
from pathlib import Path
import json
import numpy as np
from .conversion import select_dimensions
from .selected_wsi import sha,atomic
from .wsi_frozen import bound_file
FORMAT='bora-wsi-response-input-v2-full1280'

def is_full_input(manifest):
 return json.loads(Path(manifest).read_text()).get('format')==FORMAT

def read_full_input(manifest):
 manifest=Path(manifest).resolve();spec=json.loads(manifest.read_text())
 allowed={'format','source_frame_id','image','shape_yx','group_grid','features'}
 if spec.get('format')!=FORMAT or set(spec)!=allowed:raise ValueError('Full-channel Bora inputs must contain all channels, with no caller-selected dimensions')
 if not isinstance(spec['source_frame_id'],str) or not spec['source_frame_id'].strip():raise ValueError('Source frame required')
 image,image_sha=bound_file(manifest.parent,spec['image'],'image');groups,group_sha=bound_file(manifest.parent,spec['group_grid'],'group_grid')
 grid=np.load(groups,allow_pickle=False)
 if grid.ndim!=2 or grid.dtype.kind not in 'iu' or grid.min()<0 or grid.max()>255 or any(n%4 for n in grid.shape):raise ValueError('Categorical padded tile clusters required')
 eligible=grid>0;labels=grid[eligible];classes=np.unique(labels)
 if len(classes)<2 or not np.array_equal(classes,np.arange(1,len(classes)+1)):raise ValueError('At least two contiguous anonymous tile clusters required')
 shape=spec['shape_yx']
 if not isinstance(shape,list) or len(shape)!=2 or any(type(n) is not int or n<1 for n in shape) or np.any(np.array(shape)>np.array(grid.shape)*224):raise ValueError('Complete image frame required')
 if set(spec['features'])!={'1','2','4'}:raise ValueError('Three full-channel scales required')
 arrays={};hashes={}
 for factor in (1,2,4):
  path,digest=bound_file(manifest.parent,spec['features'][str(factor)],f'factor_{factor}')
  a=np.load(path,mmap_mode='r',allow_pickle=False)
  if a.shape!=tuple(np.array(grid.shape)*16//factor)+(1280,) or a.dtype.kind!='f':raise ValueError('All 1280 channels are required at every scale')
  arrays[factor]=a;hashes[str(factor)]=digest
 return spec,arrays,eligible,labels,dict(full_input_manifest_sha256=sha(manifest),full_features_sha256=hashes,group_grid_sha256=group_sha,source_image_sha256=image_sha)

def prepare_bora_selection(manifest,output):
 spec,arrays,eligible,labels,identity=read_full_input(manifest);output=Path(output)
 if output.exists():raise FileExistsError('Use a fresh Bora selection output')
 # Selection is fitted here, from all channels and the current tile clusters.
 values=np.empty((int(eligible.sum()),1280),np.float32)
 for i,(y,x) in enumerate(np.argwhere(eligible)):
  block=arrays[1][y*16:(y+1)*16,x*16:(x+1)*16]
  if not np.isfinite(block).all():raise ValueError('Fitting tile lacks full-channel tokens')
  values[i]=block.mean((0,1),dtype=np.float64)
 dimensions=select_dimensions(values,labels).astype(int);output.mkdir(parents=True)
 files={}
 for factor,a in arrays.items():
  path=output/f'bora-selected-factor-{factor}.npy'
  selected=np.lib.format.open_memmap(path,mode='w+',dtype=a.dtype,shape=a.shape[:2]+(64,))
  for y in range(0,a.shape[0],16):selected[y:y+16]=np.take(a[y:y+16],dimensions,axis=2)
  selected.flush();del selected;files[str(factor)]=dict(path=str(path.resolve()),sha256=sha(path))
 selected_spec=dict(spec,format='bora-wsi-response-input-v1',image=dict(path=str(bound_file(Path(manifest).resolve().parent,spec['image'],'image')[0]),sha256=identity['source_image_sha256']),group_grid=dict(path=str(bound_file(Path(manifest).resolve().parent,spec['group_grid'],'group_grid')[0]),sha256=identity['group_grid_sha256']),features=files,selected_dimensions=dimensions.tolist())
 target=output/'bora-selected-input.json';atomic(target,selected_spec)
 atomic(output/'BORA_SELECTION_COMPLETE.json',dict(identity,status='COMPLETE',input_channels=1280,retained_channels=64,selected_dimensions=dimensions.tolist(),selection_owner='Bora',selection_basis='full 1280-channel base-scale token means over fitting tiles and their anonymous cluster IDs',selected_input_manifest_sha256=sha(target)))
 return target,identity
