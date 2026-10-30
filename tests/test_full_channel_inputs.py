import json
import numpy as np
import pytest
from bora.full_channel_inputs import FORMAT,prepare_bora_selection,read_full_input
from bora.slide_responses import _inputs,fit_slide_responses,project_slide_responses
from bora.selected_wsi import sha

def fixture(folder,channels=1280):
 rng=np.random.default_rng(902);(folder/'he.bin').write_bytes(b'image-identity')
 groups=np.ones((4,4),np.uint8);groups[:,2:]=2;np.save(folder/'groups.npy',groups)
 features={}
 for factor in (1,2,4):
  n=64//factor;a=rng.normal(0,.15,(n,n,channels)).astype(np.float16)
  # Only high-numbered channels carry tile-cluster information.
  if channels==1280:a[:,:n//2,1000:1080]+=2
  p=folder/f'full-{factor}.npy';np.save(p,a);features[str(factor)]=dict(path=p.name,sha256=sha(p))
 spec=dict(format=FORMAT,source_frame_id='full-channel-test',image='he.bin',shape_yx=[896,896],group_grid='groups.npy',features=features)
 manifest=folder/'full-input.json';manifest.write_text(json.dumps(spec));return manifest,spec

def test_bora_selects_from_all_channels_using_clusters(tmp_path):
 manifest,spec=fixture(tmp_path);before={p:sha(p) for p in tmp_path.glob('full-*.npy')}
 target,identity=prepare_bora_selection(manifest,tmp_path/'derived')
 selected=json.loads(target.read_text())['selected_dimensions']
 assert len(selected)==64 and len(set(selected))==64 and any(v>=1000 for v in selected)
 arrays,_,_,info=_inputs(target)
 for factor in (1,2,4):assert np.array_equal(arrays[factor],np.load(tmp_path/f'full-{factor}.npy')[...,selected])
 assert all(sha(p)==digest for p,digest in before.items())
 assert info['selected_dimensions']==selected
 # Different tile clusters can refit selection against the same immutable features.
 groups=np.load(tmp_path/'groups.npy');groups[:2,:]=1;groups[2:,:]=2;np.save(tmp_path/'groups.npy',groups)
 other,_=prepare_bora_selection(manifest,tmp_path/'other');assert json.loads(other.read_text())['selected_dimensions']!=selected
 assert all(sha(p)==digest for p,digest in before.items())

def test_rejects_reduced_or_caller_selected_inputs(tmp_path):
 manifest,spec=fixture(tmp_path,channels=64)
 with pytest.raises(ValueError,match='1280'):read_full_input(manifest)
 spec['selected_dimensions']=list(range(64));manifest.write_text(json.dumps(spec))
 with pytest.raises(ValueError,match='caller-selected'):read_full_input(manifest)

def test_fit_accepts_full_channels_and_records_bora_selection(tmp_path):
 manifest,spec=fixture(tmp_path)
 result=fit_slide_responses(manifest,tmp_path/'fit')
 assert result['full_input_channels']==1280 and result['selection_owner']=='Bora'
 assert result['full_input_manifest_sha256']==sha(manifest)
 projected=project_slide_responses(manifest,tmp_path/'fit',tmp_path/'projection-valid')
 assert projected['full_input_channels']==1280 and projected['selection_owner']=='Bora'
 # A changed full feature identity is rejected even if the retained subset exists.
 spec['source_frame_id']='changed';manifest.write_text(json.dumps(spec))
 with pytest.raises(ValueError,match='Full-channel input differs'):project_slide_responses(manifest,tmp_path/'fit',tmp_path/'projection')
