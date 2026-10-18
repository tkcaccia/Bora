"""Original PUMA tissue polygons with verified, genuinely acquired H&E context.

No model is run. The1024px annotation is embedded in its true2048px context
crop, never enlarged to manufacture eligibility. Central source pixels must
match the independently released original ROI exactly in all RGBA channels.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image
import tifffile

from .http_ranges import RangeFile
from .tissue_fields import FieldWriter, choose_field, sha

CATALOG=dict(labels={1:'tumor tissue',2:'stroma',3:'blood vessel',4:'epidermis',5:'necrosis'},
    ignore=[0],task='melanoma tissue-region segmentation; nuclei annotations unused',
    annotation_provenance='Original expert PUMA training GeoJSON tissue polygons, rasterized at native pixel centres; only the1024px annotated centre is scored',
    palette={1:[220,45,65],2:[55,165,75],3:[55,110,230],4:[220,180,55],5:[115,120,135]})
NAMES={'tissue_tumor':1,'tissue_stroma':2,'tissue_blood_vessel':3,'tissue_epidermis':4,
       'tissue_necrosis':5,'tissue_white_background':0}


def rasterize(document):
    mask=np.zeros((1024,1024),np.uint8)
    conflicts=0
    for feature in document['features']:
        label=NAMES[feature['properties']['classification']['name']]
        geometry=feature['geometry'];kind=geometry['type']
        if kind not in ('Polygon','MultiPolygon'):raise ValueError('Unsupported tissue geometry')
        polygons=[geometry['coordinates']] if kind=='Polygon' else geometry['coordinates']
        for polygon in polygons:
            layer=np.zeros(mask.shape,np.uint8)
            for i,ring in enumerate(polygon):
                points=np.asarray(ring,dtype=float)
                if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all():
                    raise ValueError('Invalid tissue polygon')
                # One released polygon (metastatic ROI018) extends to -1px.
                # Permit this bounded edge excursion and let the fixed1024px
                # canvas clip it. Do not rescale coordinates or label context.
                if np.min(points)<-1.01 or np.max(points)>1025.01:
                    raise ValueError('Polygon is outside original annotated ROI')
                # Source vector coordinates describe pixel edges. Shift by half
                # a pixel before fixed-point rasterization (1/16 pixel precision).
                points=np.round((points-.5)*16).astype(np.int32)
                cv2.fillPoly(layer,[points],1 if i==0 else 0,shift=4)
            foreground=layer.astype(bool)
            conflicts+=int(np.count_nonzero(foreground&(mask>0)&(mask!=label)))
            mask[foreground]=label
    return mask,conflicts


def read_tiff(payload):
    with tifffile.TiffFile(io.BytesIO(payload)) as t:
        page=t.pages[0];unit=int(page.tags['ResolutionUnit'].value)
        factor={2:25400.,3:10000.}[unit]
        spacing=[]
        for key in ('XResolution','YResolution'):
            a,b=page.tags[key].value;spacing.append(factor*b/a)
        image=page.asarray()
    if image.ndim!=3 or image.shape[2]!=4:raise ValueError('Expected original RGBA TIFF')
    return image,spacing


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data=a.data;raw=data/'raw/puma';writer=FieldWriter(data,a.output,'puma',CATALOG)
    paths=sorted((raw/'extracted').rglob('*_tissue.geojson'))
    buckets=[[q for q in paths if f'_{kind}_' in q.name] for kind in ('primary','metastatic')]
    paths=[q for pair in zip(*buckets) for q in pair]
    ci=json.loads((data/'metadata/puma-context-zip-index-v1.json').read_text())
    roi_url='https://zenodo.org/api/records/15050523/files/01_training_dataset_tif_ROIs.zip/content'
    cache=data/'raw/puma-context/source-blocks-v1';roi_cache=data/'raw/puma-rois/source-blocks-v1'
    with RangeFile(ci['source'],ci['source_bytes'],cache,concurrency=2) as cr,RangeFile(roi_url,646953540,roi_cache,block_size=1024*1024) as rr:
        with zipfile.ZipFile(cr) as contexts,zipfile.ZipFile(rr) as rois:
            for annotation in paths:
                name=annotation.name.removesuffix('_tissue.geojson')
                if any(row['source']['id']==name for row in writer.rows):continue
                mask,conflicts=rasterize(json.loads(annotation.read_text()))
                field=np.zeros((2048,2048),np.uint8);field[512:1536,512:1536]=mask
                approx=[10000/44186,10000/44185]
                eligible,why=choose_field(np.full((2048,2048,3),180,np.uint8),field,CATALOG,approx)
                if eligible is None:
                    writer.excluded.append(dict(source=dict(id=name,annotation=str(annotation.relative_to(data))),**why));writer.checkpoint();continue
                rn='01_training_dataset_tif_ROIs/'+name+'.tif'
                cn='01_training_dataset_tif_context_ROIs/'+name+'_context.tif'
                rp=rois.read(rn);cp=contexts.read(cn)
                roi,rmpp=read_tiff(rp);context,cmpp=read_tiff(cp)
                if roi.shape!=(1024,1024,4) or context.shape!=(5120,5120,4):raise ValueError('Original PUMA geometry changed')
                if not np.array_equal(context[2048:3072,2048:3072],roi):
                    raise ValueError('PUMA context centre does not reproduce the independent original ROI')
                if not np.allclose(rmpp,cmpp,rtol=0,atol=1e-10):raise ValueError('Context and ROI spacing disagree')
                crop=context[1536:3584,1536:3584]
                if np.any(crop[...,3]!=255):
                    writer.excluded.append(dict(source=dict(id=name),reason='real_context_contains_transparent_padding'));writer.checkpoint();continue
                target=raw/'selected-context-crops'/(name+'.png');target.parent.mkdir(exist_ok=True)
                Image.fromarray(crop[...,:3]).save(target)
                source=dict(id=name,group_id='puma-roi-'+name,group_level='provider_ROI_only_patient_or_WSI_linkage_not_released_here',
                    image=str(target.relative_to(data)),image_sha256=sha(target),mask=str(annotation.relative_to(data)),mask_sha256=sha(annotation),
                    context_member=cn,context_member_sha256=hashlib.sha256(cp).hexdigest(),context_crc32=contexts.getinfo(cn).CRC,
                    original_roi_member=rn,original_roi_sha256=hashlib.sha256(rp).hexdigest(),original_roi_crc32=rois.getinfo(rn).CRC,
                    original_context_shape_yx=[5120,5120],source_context_crop_xywh=[1536,1536,2048,2048],
                    annotated_centre_xywh=[512,512,1024,1024],unannotated_context_ignored=True,
                    central_RGBA_exact_match=True,polygon_overlap_conflict_pixels_in_provider_order=conflicts,
                    polygon_canvas_clipping='Original1024px canvas clips at most1px source edge excursions; no coordinate rescaling or annotation beyond ROI',
                    source_tiff_mpp_xy=cmpp,source_tiff_resolution_units='centimetre',
                    record='https://zenodo.org/records/15050523',license='CC0-1.0',provider_split='training (development)',
                    microscopy='Native TIFF axis calibration; published0.22micrometres/px is rounded',
                    full_archive_checksum_verified=False)
                writer.consider(crop[...,:3],field,source,cmpp)
                if writer.complete:break
        (writer.root/'SOURCE_RANGE_RECEIPTS.json').write_text(json.dumps(dict(context=cr.receipt(),roi=rr.receipt()),indent=2)+'\n')
    result=writer.finish();print(json.dumps(dict(dataset='puma',accepted=result['accepted'],quota_met=result['quota_met'])))


if __name__=='__main__':main()
