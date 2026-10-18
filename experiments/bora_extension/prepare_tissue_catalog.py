"""Prepare verified original tissue-region annotations; never run a model."""
import argparse
import csv
import io
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from PIL import Image

from .tissue_fields import FieldWriter,sha
from .http_ranges import RangeFile
from .orca_labels import decode_orca

CATALOG={
 'peso':dict(labels={1:'non-epithelial target (not expert stroma)',2:'epithelial tissue'},ignore=[0],
    valid_negative_raw_label=1,task='epithelial versus non-epithelial tissue; no gland instance task',
    annotation_provenance='Original provider released test ground-truth masks; avoid U-Net-generated training masks',
    palette={1:[55,165,75],2:[220,45,65]}),
 'gcss':dict(labels={1:'tumor',2:'lymphoid stroma',3:'desmoplastic stroma',4:'smooth muscle',5:'necrosis'},
    ignore=[-1,0,6],task='multiclass gastric tissue compartments',
    annotation_provenance='Original GCSS author-provided pixel matrices; annotation-assistance details not assumed',
    palette={1:[220,45,65],2:[200,100,220],3:[55,165,75],4:[220,180,55],5:[115,120,135]}),
 'ignite':dict(labels={2:'tumor epithelium',3:'reactive epithelium',4:'stroma',5:'inflammation',
    6:'alveolar tissue',7:'fatty tissue',8:'necrotic tissue',9:'erythrocytes',10:'bronchial epithelium',
    11:'mucus/plasma/fluids',12:'cartilage/bone',13:'macrophages',14:'muscle',15:'liver',16:'keratinization'},
    ignore=[0,1],task='H&E NSCLC tissue compartment segmentation only; no IHC/nuclei detection',
    annotation_provenance='Original IGNITE H&E manually annotated tissue ROIs; v2 release',
    palette={2:[220,45,65],3:[55,110,230],4:[55,165,75],5:[200,100,220],7:[220,180,55],8:[115,120,135]}),
 'orca':dict(labels={127:'non-carcinoma tissue',128:'non-carcinoma tissue',255:'carcinoma'},ignore=[0],
    task='oral carcinoma versus other tissue; non-tissue black excluded',
    annotation_provenance='Original author validation masks, not resized/augmented third-party derivatives',
    palette={127:[55,165,75],128:[55,165,75],255:[220,45,65]})}


def peso(data,output):
    import openslide
    raw=data/'raw/peso';folder=raw/'extracted'
    rows=list(csv.DictReader((raw/'peso_testset_mapping.csv').open()))
    writer=FieldWriter(data,output,'peso',CATALOG['peso'])
    for row in sorted(rows,key=lambda r:(int(r['slide']),int(r['id']))):
        group='peso-slide-'+row['slide']
        if group in writer.groups:continue
        xmlpath=folder/'peso_testset_regions'/('pds_'+row['slide']+'.xml')
        nodes=ET.parse(xmlpath).findall('./Annotations/Annotation')
        node=next(n for n in nodes if n.attrib['Name']==row['annotation_id'])
        points=[(float(p.attrib['X']),float(p.attrib['Y'])) for p in node.findall('./Coordinates/Coordinate')]
        x0,y0=min(x for x,y in points),min(y for x,y in points)
        x1,y1=max(x for x,y in points),max(y for x,y in points)
        if (x1-x0,y1-y0)!=(2500,2500) or any(p!=int(p) for xy in points for p in xy):
            raise ValueError('Unexpected original PESO region geometry')
        if set(points)!={(x0,y0),(x0,y1),(x1,y0),(x1,y1)}:
            raise ValueError('PESO ROI is not the expected axis-aligned rectangle')
        imagepath=folder/'peso_testset_png'/(row['id']+'.jpg')
        maskpath=folder/'peso_testset_groundtruth_masks'/('pds_'+row['slide']+'.tif')
        with Image.open(imagepath) as image:rgb=np.array(image.convert('RGB'))
        # Read only the specified native2500px annotation window. Whole masks
        # can exceed20 gigapixels; never bypass PIL limits to load them whole.
        with openslide.OpenSlide(str(maskpath)) as mask:
            if x0<0 or y0<0 or 2*x1>mask.dimensions[0] or 2*y1>mask.dimensions[1]:
                raise ValueError('PESO ROI is outside ground-truth slide')
            # Source scans/GT use0.24µm pixels; the provider JPEG/XML export
            # uses0.48µm pixels. At1:1 coordinates all160 masks are unknown.
            # Read the actual5000px GT field and project pixel-centre labels;
            # do not blindly resize a mismatched pair or use a blurred pyramid.
            rgba=np.array(mask.read_region((int(2*x0),int(2*y0)),0,(5000,5000)))
        if not np.all(rgba[...,:3]==rgba[...,0,None]) or np.any(rgba[...,3]!=255):
            raise ValueError('PESO categorical slide mask is not opaque grayscale')
        labels=rgba[...,0]
        if np.mean(labels>0)<.99 or not set(np.unique(labels))<={0,1,2}:
            raise ValueError('PESO source-grid transform does not cover the annotated ROI')
        paired=labels[1::2,1::2].copy()
        source=dict(id=row['id'],group_id=group,group_level='provider_source_slide_patient_mapping_unverified',
            image=str(imagepath.relative_to(data)),mask=str(maskpath.relative_to(data)),
            image_sha256=sha(imagepath),mask_sha256=sha(maskpath),xml_sha256=sha(xmlpath),
            annotation_id=row['annotation_id'],annotation_slide_xywh=[int(x0),int(y0),2500,2500],
            groundtruth_native_xywh=[int(2*x0),int(2*y0),5000,5000],
            groundtruth_grid_mpp_xy=[.24,.24],provider_export_mpp_xy=[.48,.48],
            annotation_projection='Exact factor2 coordinate transform; pixel-centre nearest neighbour [1::2,1::2], no mask averaging',
            slide_class=row['type'],provider_split='test (now used for development)',
            record='https://zenodo.org/records/5137717',license='CC BY-NC-SA4.0',
            microscopy='Provider-exported0.48µm/px ROI grid; not original scanner optical resolution')
        writer.consider(rgb,paired,source,[.48,.48])
        if writer.complete:break
    return writer.finish()


def ignite(data,output):
    raw=data/'raw/ignite'
    rows=list(csv.DictReader((raw/'data_overview.csv').open()))
    writer=FieldWriter(data,output,'ignite',CATALOG['ignite'])
    labelmap=json.loads((raw/'he_label_map.json').read_text())
    if labelmap.get('Tumor epithelium')!=2 or labelmap.get('Stroma')!=4 or labelmap.get('Background')!=1:
        raise ValueError('IGNITE original label ontology changed')
    index=json.loads((data/'metadata/ignite-images-zip-index-v1.json').read_text())
    cache=data/'raw/ignite-images/source-blocks-v1'
    wanted=[r for r in rows if r['task']=='he_tissue_segmentation' and r['split']=='train']
    wanted.sort(key=lambda r:(int(r['patient_id']),int(r['roi_id'])))
    with RangeFile(index['source'],index['source_bytes'],cache) as reader:
        with zipfile.ZipFile(reader) as z:
            names=set(z.namelist())
            for row in wanted:
                group='ignite-patient-'+row['patient_id']
                if group in writer.groups:continue
                maskpath=raw/'extracted/annotations/annotations/he'/(row['name']+'.png')
                if not maskpath.exists():raise FileNotFoundError(maskpath)
                mask=np.array(Image.open(maskpath))
                # Filter absent tissue groups / insufficient real image extent
                # before transferring the paired original RGB member.
                if len(set(np.unique(mask))-{0,1})<2 or min(mask.shape)<896:continue
                name='images/he/'+row['name']+'.png'
                if name not in names:raise ValueError('IGNITE original image member missing')
                imagepath=data/'raw/ignite/selected-originals'/Path(name).name
                imagepath.parent.mkdir(exist_ok=True)
                if not imagepath.exists():imagepath.write_bytes(z.read(name))
                with Image.open(imagepath) as image:rgb=np.array(image.convert('RGB'))
                if rgb.shape[:2]!=mask.shape:raise ValueError('IGNITE original pairing has different geometry')
                source=dict(id=row['name'],group_id=group,group_level='provider_anonymous_patient',
                    image=str(imagepath.relative_to(data)),mask=str(maskpath.relative_to(data)),
                    image_sha256=sha(imagepath),mask_sha256=sha(maskpath),
                    source_zip_member=name,source_zip_crc32=z.getinfo(name).CRC,
                    zip_archive_full_checksum_verified=False,metadata=row,
                    record='https://zenodo.org/records/17735903',license='CC BY-NC-SA4.0',
                    microscopy='Provider0.5µm/px ROI sampling; annotated extent metadata may exclude unannotated context')
                writer.consider(rgb,mask,source,[.5,.5])
                if writer.complete:break
        (writer.root/'SOURCE_RANGE_RECEIPT.json').write_text(json.dumps(reader.receipt(),indent=2)+'\n')
    return writer.finish()


def orca(data,output):
    inputs=json.loads((data/'metadata/orca-native-acquisition-v1.json').read_text())
    writer=FieldWriter(data,output,'orca',CATALOG['orca'])
    for row in inputs['selected']:
        with Image.open(data/row['image']) as im:rgb=np.array(im.convert('RGB'))
        raw,normalization=decode_orca(np.array(Image.open(data/row['mask'])))
        source=dict(row,decoded_mask=normalization,group_id=row['source_group'],group_level='TCGA_case_from_original_filename',
            record='https://sites.google.com/unibas.it/orca/download',
            microscopy='Distributed pixel spacing unverified; preserve original pixels',
            license=inputs['license'],provider_split='validation (now used for development)')
        writer.consider(rgb,raw,source,None)
    return writer.finish()


def gcss(data,output):
    import tifffile
    raw=data/'raw/gcss/extracted/GCSS/GCTS'
    rows={r['IMG']:r for r in csv.DictReader((raw/'ROIs_in_TCGA.csv').open())}
    hits=json.loads((data/'metadata/gcss-source-gdc-files-v1.json').read_text())['data']['hits']
    hits={r['file_name']:r for r in hits}
    writer=FieldWriter(data,output,'gcss',CATALOG['gcss'])
    failures=[]
    with zipfile.ZipFile(raw/'images.zip') as images,zipfile.ZipFile(raw/'labels.zip') as labels:
        names=sorted([n for n in images.namelist() if n.endswith('.png')],key=lambda n:Path(n).name)
        for name in names:
            source_row=rows[Path(name).name]
            group='-'.join(source_row['TCGA_FILE_NAME'].split('-')[:3])
            if group in writer.groups:continue
            label_name='labels/'+Path(name).stem+'_boundary_nuClick.txt'
            with labels.open(label_name) as handle:mask=np.loadtxt(handle,delimiter=',',dtype=np.int16)
            if len(set(np.unique(mask))&set(CATALOG['gcss']['labels']))<2:continue
            payload=images.read(name)
            with Image.open(io.BytesIO(payload)) as image:rgb=np.array(image.convert('RGB'))
            if mask.shape!=rgb.shape[:2]:raise ValueError('Original GCSS pixel matrix/RGB dimensions disagree')
            hit=hits[source_row['TCGA_FILE_NAME']]
            if hit['access']!='open':continue
            url='https://api.gdc.cancer.gov/data/'+hit['file_id']
            cache=data/'raw/gcss/source-wsi-metadata-blocks'/hit['file_id']
            try:
                with RangeFile(url,hit['file_size'],cache,block_size=64*1024) as reader:
                    with tifffile.TiffFile(reader) as slide:
                        page=slide.pages[0]
                        found=re.search(r'\bMPP\s*=\s*([\d.]+)',page.description or '')
                        if found:
                            axes=[float(found.group(1))]*2
                            calibration='Aperio MPP in original source-slide TIFF description'
                        else:
                            unit=int(page.tags['ResolutionUnit'].value)
                            factor={2:25400.,3:10000.}[unit]
                            axes=[]
                            for tag in ('XResolution','YResolution'):
                                a,b=page.tags[tag].value;axes.append(factor*b/a)
                            calibration='Original source TIFF calibrated axis resolution tags'
                        source_shape=[int(page.imagelength),int(page.imagewidth)]
                    receipt=reader.receipt()
                if not all(np.isfinite(v) and 0<v<10 for v in axes):
                    raise ValueError('Invalid source-slide pixel calibration')
            except Exception as exc:
                failures.append(dict(id=source_row['IMG'],file_id=hit['file_id'],reason=type(exc).__name__+': '+str(exc)))
                (writer.root/'CALIBRATION_FAILURES.json').write_text(json.dumps(failures,indent=2)+'\n')
                continue
            x0,y0,x1,y1=map(int,source_row['COORDINATE'].split('_'))
            if min(x0,y0)<0 or x1>source_shape[1] or y1>source_shape[0] or min(x1-x0,y1-y0)<=0:
                raise ValueError('GCSS original ROI coordinates outside source slide')
            # Derive the distributed ROI axes from the exact provider source
            # bounds and actual PNG dimensions. Never assign WSI MPP directly
            # to its downsampled ROI, or infer calibration from magnification.
            spacing=[axes[0]*(x1-x0)/rgb.shape[1],axes[1]*(y1-y0)/rgb.shape[0]]
            source=dict(id=source_row['IMG'],group_id=group,group_level='TCGA_case',
                image_zip_member=name,image_member_sha256=__import__('hashlib').sha256(payload).hexdigest(),
                label_zip_member=label_name,source_csv=source_row,
                source_gdc_file=hit,source_slide_shape_yx=source_shape,source_mpp_xy=axes,
                source_xyxy=[x0,y0,x1,y1],distributed_mpp_xy=spacing,
                calibration_method=calibration+' plus provider crop bounds / distributed image dimensions',
                source_range_receipt=receipt,record='https://zenodo.org/records/8176354',license='CC BY4.0',
                provider_split='unsplit (development)',
                microscopy='Derived distributed-image spacing; pixel bounds/end-point convention can contribute subpixel rounding uncertainty')
            writer.consider(rgb,mask,source,spacing)
            if writer.complete:break
    return writer.finish()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset',choices=['peso','ignite','orca','gcss'],required=True)
    args=parser.parse_args()
    result=globals()[args.dataset](args.data,args.output)
    print(json.dumps(dict(dataset=args.dataset,accepted=result['accepted'],quota_met=result['quota_met'],groups=len({r['group'] for r in result['samples']}))))


if __name__=='__main__':main()
