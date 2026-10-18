"""Prepare physically calibrated, non-overlapping ORCA tissue fields.

Use original author core images and source-header calibration inventories.
Try one eligible quadrant per TCGA case before reusing cases. Forty fields
are not described as forty patients when several share a source core/case.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .calibration import require
from .orca_labels import decode_orca
from .prepare_tissue_catalog import CATALOG
from .tissue_fields import FieldWriter,sha


def quadrants(shape):
    h,w=shape;ys=[0,h//2,h];xs=[0,w//2,w]
    return [(xs[x],ys[y],xs[x+1]-xs[x],ys[y+1]-ys[y]) for y in range(2) for x in range(2)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--calibrations',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    catalog=dict(CATALOG['orca'],labels={127:'non-carcinoma tissue',255:'carcinoma'},
        valid_negative_raw_label=127,
        annotation_provenance='Original author expert carcinoma contours plus image-derived tissue extent; original validation/test masks, explicit antialias intensity decoding, no model-generated tumour masks')
    writer=FieldWriter(a.data,a.out,'orca',catalog);sources={}
    for path in a.calibrations:
        inventory=json.loads(path.read_text())
        if inventory['encoder_run'] or inventory['bora_run']:raise ValueError('Not a calibration-only inventory')
        for row in inventory['ready']:
            if row['id'] in sources:raise ValueError('Duplicate calibrated ORCA core')
            sources[row['id']]=dict(row,calibration_inventory=str(path),calibration_inventory_sha256=sha(path))
    # Existing fields also retain their source quadrant, so resumed selections
    # never overlap or silently replace a previously saved field.
    used={r['source']['core_id']:set() for r in writer.rows}
    for row in writer.rows:used[row['source']['core_id']].add(row['source']['quadrant_index'])
    for round_index in range(4):
        for core_id,row in sorted(sources.items()):
            group=row['source_group']
            count=sum(r['group']==group for r in writer.rows)
            if count>round_index:continue
            image=a.data/row['image'];mask=a.data/row['mask']
            if sha(image)!=row['image_sha256'] or sha(mask)!=row['mask_sha256']:
                raise ValueError('Original ORCA pair bytes changed')
            with Image.open(image) as im:rgb=np.array(im.convert('RGB'))
            raw,normalization=decode_orca(np.array(Image.open(mask)))
            if rgb.shape!=(*raw.shape,3) or list(raw.shape)!=row['shape_yx']:
                raise ValueError('Original core geometry changed')
            for i,(x,y,w,h) in enumerate(quadrants(raw.shape)):
                if i in used.get(core_id,set()):continue
                source=dict(row,id=core_id+f'__quadrant-{i}',core_id=core_id,quadrant_index=i,
                    group_id=group,group_level='TCGA case; potentially multiple disjoint fields per original core',
                    source_native_shape_yx=list(raw.shape),source_native_xywh=[x,y,w,h],
                    decoded_mask=normalization,record='https://sites.google.com/unibas.it/orca/download',
                    microscopy='Matched original TCGA native TIFF MPP, with the author native core-extraction grid checked against source AppMag; MPP never inferred from AppMag',
                    preparation_code_sha256=sha(__file__),provider_split=row['provider_release']+' (development)',
                    license='Provider educational-use notice; no broader license inferred')
                # Paths in source provenance continue to refer to the original
                # whole core; source_native_xywh identifies this true quadrant.
                if writer.consider(rgb[y:y+h,x:x+w],raw[y:y+h,x:x+w],source,row['native_mpp_xy']):
                    require(writer.rows[-1]);used.setdefault(core_id,set()).add(i);break
            if writer.complete:break
        if writer.complete:break
    result=writer.finish()
    print(json.dumps(dict(dataset='orca',accepted=result['accepted'],quota_met=result['quota_met'],
        source_cases=len(writer.groups),source_cores=len({r['source']['core_id'] for r in writer.rows}))))


if __name__=='__main__':main()
