"""Acquire pinned public tissue releases, without models or upstream scripts.

The input catalog contains explicit provider file URLs, sizes and MD5s.
Download/extraction is separate from the forty-eligible-image planning stage.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

from .acquire_ocelot import digest, check_members

RESERVE = 20 * 2**30


def acquire_file(root, dataset, entry):
    folder = root/'raw'/dataset
    folder.mkdir(parents=True,exist_ok=True)
    name = entry['name']
    if Path(name).name != name or name in ('.','..'):
        raise ValueError('Provider filename must be a plain basename')
    target = folder/name
    partial = target.with_suffix(target.suffix+'.part')
    if not target.exists():
        start = partial.stat().st_size if partial.exists() else 0
        if start >= entry['bytes']:
            raise ValueError('Incomplete download already reaches expected size; inspect retained bytes')
        headers = {'User-Agent':'Bora-tissue-benchmark/0.3'}
        if start:
            headers['Range'] = f'bytes={start}-'
        with urllib.request.urlopen(urllib.request.Request(entry['url'],headers=headers),timeout=60) as response:
            if start and (response.status != 206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-')):
                raise ValueError('Provider did not honor resume; partial file retained')
            if 'text/html' in response.headers.get('Content-Type',''):
                raise ValueError('Provider returned HTML rather than dataset bytes')
            with partial.open('ab' if start else 'wb') as handle:
                for block in iter(lambda:response.read(2**20),b''):
                    if shutil.disk_usage(folder).free < RESERVE+len(block):
                        raise ValueError('Preserve20GiB free while acquiring data')
                    handle.write(block)
        algorithm='md5' if 'md5' in entry else 'sha256'
        if partial.stat().st_size != entry['bytes'] or digest(partial,algorithm) != entry[algorithm]:
            raise ValueError('Size/publisher MD5 mismatch: '+name)
        partial.rename(target)
    algorithm='md5' if 'md5' in entry else 'sha256'
    if target.stat().st_size != entry['bytes'] or digest(target,algorithm) != entry[algorithm]:
        raise ValueError('Existing public download differs from pinned release')
    record = dict(entry,sha256=digest(target),path=str(target.relative_to(root)))
    if entry.get('extract'):
        out = folder/'extracted'/target.stem
        marker = out/'.bora-extraction.json'
        if not marker.exists():
            with zipfile.ZipFile(target) as z:
                infos=z.infolist()
                check_members(infos)
                if any(i.file_size > 2**31 for i in infos):
                    raise ValueError('Large WSI members must be extracted selectively')
                if shutil.disk_usage(folder).free < RESERVE+sum(i.file_size for i in infos):
                    raise ValueError('Insufficient extraction space and reserve')
                out.mkdir(parents=True,exist_ok=True)
                z.extractall(out)
            marker.write_text(json.dumps(dict(archive_sha256=record['sha256'],members=len(infos)))+'\n')
        elif json.loads(marker.read_text())['archive_sha256'] != record['sha256']:
            raise ValueError('Extraction belongs to different archive')
        record['extracted']=str(out.relative_to(root))
    print(json.dumps(dict(acquired=dataset,file=name,bytes=entry['bytes'])),flush=True)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--catalog',type=Path,required=True)
    parser.add_argument('--receipt-name',default='tissue-catalog-acquisition-2026-10-18-v1.json')
    args=parser.parse_args()
    catalog=json.loads(args.catalog.read_text())
    tasks=[(name,e) for name,v in catalog['datasets'].items() for e in v.get('files',[])]
    args.root.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda task:acquire_file(args.root,*task),tasks))
    record=dict(catalog_sha256=digest(args.catalog),source_code_sha256=digest(__file__),
        files=results,downloaded_utc=datetime.now(timezone.utc).isoformat(),
        segmentation_run=False,encoder_run=False,external_scripts_run=False,
        benchmark_extended=False,note='Acquired files only; actual forty-image qualification is separate')
    if Path(args.receipt_name).name!=args.receipt_name:raise ValueError('Receipt name must be a basename')
    output=args.root/'metadata'/args.receipt_name
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':
    main()
