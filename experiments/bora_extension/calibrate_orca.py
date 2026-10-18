"""Trace original ORCA core spacing to its exact public TCGA source slide.

The author's extraction protocol uses native4500px cores from40x slides and
2250px cores from20x slides. This verifies that protocol against the matched
source header; MPP itself is read from the source, never inferred from40x.
No image is resized, no field is selected and no inference is performed.
"""
import argparse
import json
from pathlib import Path
import re

import tifffile
from PIL import Image

from .bootstrap_tissue_sources import get_json
from .http_ranges import RangeFile
from .tissue_fields import sha

PROTOCOL='https://www.mdpi.com/2076-3417/10/22/8285'


def source_name(core_id):
    stem,suffix=core_id.rsplit('_',1)
    if not suffix.isdigit() or not stem.startswith('TCGA-'):
        raise ValueError('Unrecognized original core/source filename mapping')
    return stem+'.svs'


def read_source(data,hit):
    if hit['access']!='open':raise ValueError('Source slide is not public')
    url='https://api.gdc.cancer.gov/data/'+hit['file_id']
    cache=data/'raw/orca/source-wsi-metadata-blocks'/hit['file_id']
    with RangeFile(url,hit['file_size'],cache,block_size=64*1024) as reader:
        with tifffile.TiffFile(reader) as t:
            p=t.pages[0];description=p.description or ''
            mpp=re.search(r'\bMPP\s*=\s*([\d.]+)',description)
            objective=re.search(r'\bAppMag\s*=\s*([\d.]+)',description)
            if mpp:
                axes=[float(mpp.group(1))]*2;method='Aperio MPP in native source TIFF description'
            else:
                factor={2:25400.,3:10000.}[int(p.tags['ResolutionUnit'].value)]
                axes=[factor*p.tags[k].value[1]/p.tags[k].value[0] for k in ('XResolution','YResolution')]
                method='Source TIFF per-axis calibrated resolution tags'
            if not all(0<v<10 for v in axes):raise ValueError('Invalid source MPP')
            result=dict(source_mpp_xy=axes,source_shape_yx=[p.imagelength,p.imagewidth],
                source_reported_appmag=None if objective is None else float(objective.group(1)),
                calibration_method=method,source_description=description)
        result.update(source_range_receipt=reader.receipt(),source_gdc_file=hit)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--receipts',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    rows=[]
    for path in a.receipts:
        record=json.loads(path.read_text())
        rows.extend(dict(row,provider_release=record['release'],acquisition_receipt=str(path),
            acquisition_receipt_sha256=sha(path)) for row in record['selected'])
    names=sorted({source_name(r['id']) for r in rows})
    value=get_json('https://api.gdc.cancer.gov/files',dict(filters=dict(op='in',
        content=dict(field='file_name',value=names)),fields='file_id,file_name,file_size,access,cases.submitter_id',size=200))
    hits={h['file_name']:h for h in value['data']['hits']};sources={};ready=[];pending=[]
    for row in rows:
        try:
            name=source_name(row['id']);hit=hits[name]
            if name not in sources:sources[name]=read_source(a.data,hit)
            source=sources[name]
            expected={40.:4500,20.:2250}.get(source['source_reported_appmag'])
            with Image.open(a.data/row['image']) as im:shape=[im.height,im.width]
            if expected is None or shape!=[expected,expected] or shape!=row['shape_yx']:
                raise ValueError('Distributed grid does not match author native extraction protocol')
            if sha(a.data/row['image'])!=row['image_sha256']:
                raise ValueError('Original core bytes changed')
            ready.append(dict(row,**source,native_mpp_xy=source['source_mpp_xy'],
                extraction_protocol=PROTOCOL,
                extraction_basis='Original author native-grid extraction; source core coordinates are not supplied, so no exact source-pixel correspondence is asserted',
                magnification_used_to_infer_mpp=False))
        except Exception as exc:
            pending.append(dict(id=row['id'],reason=type(exc).__name__+': '+str(exc)))
    result=dict(sources=value,ready=ready,pending=pending,source_code_sha256=sha(__file__),
        notice='Source calibration inventory only. New physical fields and all eligibility gates must be prepared before inference.',encoder_run=False,bora_run=False)
    if a.out.exists() and json.loads(a.out.read_text())!=result:
        raise FileExistsError('Calibration inventory changed; choose a new output version')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(calibrated_sources=len(ready),pending=len(pending))))


if __name__=='__main__':main()
