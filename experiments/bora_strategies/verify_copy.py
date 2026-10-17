"""Standard-library verification of a local delivery; feature caches remain remote."""
import argparse,hashlib,json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify(base,entries):
    for name,sha in entries.items():
        if digest(base/name)!=sha:raise ValueError('Copy mismatch: '+str(base/name))
    return len(entries)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    frozen=json.loads((a.root/'ALL_PREDICTIONS_FROZEN.json').read_text());protocol=json.loads((a.root/'PROTOCOL.json').read_text());inputs=json.loads((a.source/'ALL_INPUTS_FROZEN.json').read_text())
    assert frozen['protocol_sha256']==digest(a.root/'PROTOCOL.json') and protocol['inputs_sha256']==digest(a.source/'ALL_INPUTS_FROZEN.json')
    assert inputs['plan_sha256']==digest(a.source/'PLAN.json');plan=json.loads((a.source/'PLAN.json').read_text());counts={}
    counts['prediction_manifests']=len(frozen['conversion_manifest_sha256']);counts['frozen_prediction_files']=0;counts['input_files']=0
    for key,sha in frozen['conversion_manifest_sha256'].items():
        folder=a.root/'cases'/key;assert digest(folder/'FROZEN.json')==sha;record=json.loads((folder/'FROZEN.json').read_text())
        counts['frozen_prediction_files']+=verify(folder,record['files_sha256']);assert digest(folder.parent/'tiles.native.tif')==record['tiles_native_sha256']
    for sample in plan['samples']:
        folder=a.source/'cases'/sample['case'];assert digest(folder/'INPUTS_FROZEN.json')==inputs['manifests'][sample['case']]
        counts['input_files']+=verify(folder,json.loads((folder/'INPUTS_FROZEN.json').read_text())['files_sha256'])
        assert digest(folder/'reference.native.tif')==sample['reference_native_sha256']
    counts['visuals']=verify(a.root,json.loads((a.root/'VISUAL_MANIFEST.json').read_text())['files_sha256'])
    counts['runtime_sources']=verify(a.root/'runtime_sources',json.loads((a.root/'RUNTIME_CODE.json').read_text())['files_sha256'])
    counts['delivery_metadata']=verify(a.root,json.loads((a.root/'DELIVERY_MANIFEST.json').read_text())['files_sha256'])
    counts['all_checks_passed']=True;(a.root/'LOCAL_COPY_AUDIT.json').write_text(json.dumps(counts,indent=2)+'\n');print(json.dumps(counts,indent=2))

if __name__=='__main__':main()
