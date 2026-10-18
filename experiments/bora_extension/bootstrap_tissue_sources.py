"""Reproducible public source inventories needed by the bounded downloaders.

Run after primary archive acquisition. List ZIP members through checked HTTPS
ranges; never fetch whole large image archives. Query only open GDC source-file
metadata, and list the author's public ORCA folder without authenticating.
"""
import argparse
import csv
import json
from pathlib import Path
import urllib.request
import zipfile
from .http_ranges import RangeFile


def get_json(url,payload=None):
    data=None if payload is None else json.dumps(payload).encode()
    request=urllib.request.Request(url,data=data,headers={'User-Agent':'Bora-tissue-benchmark/0.3','Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=60) as response:return json.load(response)


def zip_index(data,record_id,key,filename,cache_name):
    output=data/'metadata'/filename
    record=get_json('https://zenodo.org/api/records/'+str(record_id))
    file=next(f for f in record['files'] if f['key']==key)
    url=f'https://zenodo.org/api/records/{record_id}/files/{key}/content'
    if output.exists():
        existing=json.loads(output.read_text())
        if existing['source']!=url or existing['source_bytes']!=file['size']:raise ValueError('Original ZIP identity changed')
        return
    with RangeFile(url,file['size'],data/'raw'/cache_name/'source-blocks-v1') as reader:
        with zipfile.ZipFile(reader) as z:
            members=[dict(name=i.filename,bytes=i.file_size,compressed=i.compress_size,offset=i.header_offset,crc32=i.CRC) for i in z.infolist()]
        value=dict(source=url,source_bytes=file['size'],publisher_checksum=file['checksum'],members=members,receipt=reader.receipt())
    output.write_text(json.dumps(value,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);a=p.parse_args();data=a.data
    (data/'metadata').mkdir(exist_ok=True)
    for args in [(17735903,'images.zip','ignite-images-zip-index-v1.json','ignite-images'),(15050523,'01_training_dataset_tif_context_ROIs.zip','puma-context-zip-index-v1.json','puma-context')]:zip_index(data,*args)
    output=data/'metadata/gcss-source-gdc-files-v1.json'
    if not output.exists():
        rows=list(csv.DictReader((data/'raw/gcss/extracted/GCSS/GCTS/ROIs_in_TCGA.csv').open()))
        names=sorted({r['TCGA_FILE_NAME'] for r in rows})
        value=get_json('https://api.gdc.cancer.gov/files',dict(filters=dict(op='in',content=dict(field='file_name',value=names)),fields='file_id,file_name,file_size,access,cases.submitter_id',size=200))
        found=value['data']['hits']
        if {r['file_name'] for r in found}!=set(names) or any(r['access']!='open' for r in found):raise ValueError('Not every GCSS source slide has open GDC metadata')
        output.write_text(json.dumps(value,indent=2)+'\n')
    from .acquire_orca import FOLDERS
    for split,url in FOLDERS.items():
        output=data/'metadata'/('orca-provider-validation-list-v1.json' if split=='validation' else 'orca-test-folder-original-files-v1.json')
        if not output.exists():
            import gdown
            files=gdown.download_folder(url=url,quiet=True,use_cookies=False,skip_download=True)
            value=[dict(id=f.id,path=f.path) for f in files]
            if len(value)!=200:raise ValueError('Original ORCA '+split+' folder inventory changed; inspect before acquisition')
            output.write_text(json.dumps(value,indent=2)+'\n')
    print('Public source inventories ready. No encoder or Bora inference.')


if __name__=='__main__':main()
