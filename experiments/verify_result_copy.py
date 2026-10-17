"""Byte-level transport audit for a previously scientifically audited result."""
import argparse,hashlib,json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    audit=json.loads((root/'AUDIT.json').read_text());assert audit['all_checks_passed']
    frozen=json.loads((root/'ALL_PREDICTIONS_FROZEN.json').read_text());assert sha(root/'PROTOCOL.json')==frozen['protocol_sha256']
    pairs=frozen.get('conversion_manifest_sha256',frozen.get('case_manifest_sha256'));files=0;masks=0
    for key,digest in pairs.items():
        folder=root/'cases'/key;assert sha(folder/'FROZEN.json')==digest
        record=json.loads((folder/'FROZEN.json').read_text())
        for name,expected in record['files_sha256'].items():
            assert sha(folder/name)==expected,str(folder/name);files+=1;masks+=name.endswith('/bora.native.tif') or name=='bora.native.tif'
    visuals=json.loads((root/'VISUAL_MANIFEST.json').read_text())['files_sha256']
    for name,digest in visuals.items():assert sha(root/name)==digest,name
    runtime=json.loads((root/'RUNTIME_CODE.json').read_text())['files_sha256']
    for name,digest in runtime.items():assert sha(root/'runtime_sources'/name)==digest,name
    metadata=parents=0
    for manifest,counter in [('METADATA_MANIFEST.json','metadata'),('PARENT_INPUT_MANIFEST.json','parents')]:
        if (root/manifest).exists():
            entries=json.loads((root/manifest).read_text())['files_sha256']
            for name,digest in entries.items():assert sha(root/name)==digest,name
            if counter=='metadata':metadata=len(entries)
            else:parents=len(entries)
    assert masks==frozen['native_masks']
    result=dict(scientific_remote_audit_passed=True,prediction_manifests=len(pairs),frozen_files_verified=files,native_masks_verified=masks,
        visual_files_verified=len(visuals),runtime_sources_verified=len(runtime),metadata_files_verified=metadata,parent_inputs_verified=parents,
        all_checks_passed=True,verification_source_sha256=sha(Path(__file__)))
    (root/'LOCAL_COPY_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
