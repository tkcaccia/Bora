"""Bind learned GrandQC support to its calibrated image and stream its mask."""
from pathlib import Path
import json
import numpy as np
from .wsi_frozen import bound_file
from .selected_wsi import atomic

def bind_support(spec,base,image_sha,frame_shape,mpp_xy):
 import pyvips
 required={'format','mask','summary','cache_manifest','mode'}
 if not isinstance(spec,dict) or set(spec)!=required or spec['format']!='bora-grandqc-support-v1' or spec['mode'] not in ('artifact-free','tissue'):raise ValueError('Use a bound GrandQC tissue or artifact-free predictor mask')
 paths={};hashes={}
 for name in ('mask','summary','cache_manifest'):paths[name],hashes[name]=bound_file(base,spec[name],'GrandQC '+name)
 record=json.loads(paths['cache_manifest'].read_text());summary=json.loads(paths['summary'].read_text())
 if record.get('status')!='COMPLETE' or record.get('stage')!='grandqc' or record['identity']['image_sha256']!=image_sha:raise ValueError('GrandQC belongs to a different calibrated image')
 suffix='_grandqc_clean_tissue_mask.tif' if spec['mode']=='artifact-free' else '_grandqc_tissue_detection_mask.tif'
 if not any(name.endswith(suffix) and value['sha256']==hashes['mask'] for name,value in record['outputs'].items()):raise ValueError('Mask is not the cached learned GrandQC output')
 if not any(name.endswith('_grandqc_summary.json') and value['sha256']==hashes['summary'] for name,value in record['outputs'].items()):raise ValueError('GrandQC summary differs from the cached inference')
 if summary['tissue_mask_source']!='grandqc_model' or summary['level0_shape_yx']!=list(frame_shape) or abs(summary['source_mpp']-float(np.mean(mpp_xy)))>.001:raise ValueError('GrandQC image frame, spacing or learned tissue source differs')
 for name in ('tissue','artifact'):
  if not summary['model_provenance'][name]['checkpoints'] or any(p['status']!='verified' for p in summary['model_provenance'][name]['checkpoints']):raise ValueError('GrandQC model identity unverified')
 image=pyvips.Image.new_from_file(str(paths['mask']),access='random');expected=summary['qc_mask_shape_yx'] if spec['mode']=='artifact-free' else summary['tissue_mask_shape_yx']
 if image.bands!=1 or image.format!='uchar' or [image.height,image.width]!=expected:raise ValueError('GrandQC mask geometry or dtype differs')
 histogram=np.frombuffer(image.hist_find().write_to_memory(),dtype=np.uint32)
 if np.any(histogram[1:255]):raise ValueError('GrandQC support must be a binary model mask')
 return dict(mode=spec['mode'],mask_path=str(paths['mask']),mask_sha256=hashes['mask'],summary_sha256=hashes['summary'],cache_manifest_sha256=hashes['cache_manifest'],mask_shape_yx=expected,calibrated_frame_shape_yx=list(frame_shape),source='learned GrandQC predictor, not expert annotations',mapping='nearest neighbor over the complete padded calibrated image; crop to analysis frame')

def stream_support(identity,shape_yx,out):
 import pyvips
 h,w=shape_yx;fh,fw=identity['calibrated_frame_shape_yx'];image=pyvips.Image.new_from_file(identity['mask_path'],access='random')
 image=image.resize(fw/image.width,vscale=fh/image.height,kernel='nearest')
 if (image.height,image.width)!=(fh,fw):raise ValueError('GrandQC resampling frame differs')
 support=np.lib.format.open_memmap(Path(out)/'support.npy',mode='w+',dtype=np.uint8,shape=(h,w))
 for y in range(0,h,512):
  rows=min(512,h-y);pixels=np.frombuffer(image.crop(0,y,w,rows).write_to_memory(),np.uint8).reshape(rows,w);support[y:y+rows]=pixels>0
 support.flush();atomic(Path(out)/'GRANDQC_SUPPORT_USED.json',dict(identity,status='COMPLETE',working_shape_yx=[h,w],no_inference_rerun=True,expert_annotation_used=False))
 return support
