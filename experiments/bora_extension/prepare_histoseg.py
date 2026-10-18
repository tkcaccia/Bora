"""Bounded native tissue ROIs from original Histo-Seg JPG/PNG slide mosaics.

libvips streams the large sources. Never load a gigapixel slide into a NumPy
array, synthesize pixels, or treat forty ROIs as forty independent patients.
"""
import argparse
from collections import Counter
import json
import os
os.environ.setdefault("VIPS_CONCURRENCY","2")
from pathlib import Path

import numpy as np
import pyvips
pyvips.cache_set_max_mem(64*2**20)
pyvips.cache_set_max_files(10)
from .tissue_fields import FieldWriter,sha

NAMES=['background','keratin','epidermis','papillary dermis','reticular dermis','hypodermis','glands','hair follicles','inflammation','basal cell carcinoma','squamous cell carcinoma','intraepidermal carcinoma']
COLORS=[[0,0,0],[224,224,224],[112,48,160],[0,156,255],[0,0,128],[0,255,0],[127,96,255],[96,96,96],[255,156,0],[127,255,255],[255,0,255],[150,150,0]]
CATALOG=dict(labels={i:n for i,n in enumerate(NAMES) if i},ignore=[0],
    task='multiclass skin layers, tissue compartments and carcinomas; glands are one semantic tissue class',
    annotation_provenance='Original histopathologist-annotated source masks; exact RGB ontology from original dataset paper Table1',
    palette={i:c for i,c in enumerate(COLORS) if i})


def array(image):
    if image.format!='uchar':raise ValueError('Unexpected provider mask sample format')
    return np.frombuffer(image.write_to_memory(),np.uint8).reshape(image.height,image.width,image.bands).copy()


def decode(rgb):
    if rgb.shape[2] not in (3,4):raise ValueError('Expected published RGB tissue ontology')
    if rgb.shape[2]==4 and np.any(rgb[...,3]!=255):raise ValueError('Unreconciled annotation transparency')
    rgb=rgb[...,:3];result=np.full(rgb.shape[:2],255,np.uint8)
    for i,c in enumerate(COLORS):result[np.all(rgb==c,axis=2)]=i
    unknown=result==255
    if unknown.any():
        u,n=np.unique(rgb[unknown],axis=0,return_counts=True)
        raise ValueError('Source colors differ from published ontology: '+str(list(zip(u.tolist()[:15],n.tolist()[:15]))))
    return result


def candidates(path):
    im=pyvips.Image.new_from_file(str(path),access='sequential')
    scale=max(1,int(np.ceil(max(im.width,im.height)/2200)))
    small=array(im.subsample(scale,scale));labels=decode(small)
    ranked=[];size=1280
    for y in range(0,max(1,im.height-size+1),640):
        for x in range(0,max(1,im.width-size+1),640):
            patch=labels[y//scale:min(labels.shape[0],(y+size)//scale),x//scale:min(labels.shape[1],(x+size)//scale)]
            _,n=np.unique(patch[patch>0],return_counts=True)
            if len(n)<2:continue
            p=n/n.sum();entropy=float(-np.sum(p*np.log(p)))
            ranked.append((entropy,float(np.count_nonzero(patch)/patch.size),-y,-x,x,y))
    return [(r[-2],r[-1]) for r in sorted(ranked,reverse=True)]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    folder=a.data/'raw/histoseg';writer=FieldWriter(a.data,a.output,'histoseg',CATALOG)
    images=sorted(folder.glob('*.jpg'));regions={};used={};source_hash={}
    for row in writer.rows:used.setdefault(row['source']['slide_id'],[]).append(row['source']['source_native_xywh'])
    for round_index in range(4):
        for image in images:
            name=image.stem;mask=image.with_suffix('.png')
            if not mask.exists():raise FileNotFoundError(mask)
            if len(used.get(name,[]))>round_index:continue
            rgb_image=pyvips.Image.new_from_file(str(image),access='sequential');mask_image=pyvips.Image.new_from_file(str(mask),access='sequential')
            if (rgb_image.width,rgb_image.height)!=(mask_image.width,mask_image.height):raise ValueError('Native Histo-Seg image/mask dimensions disagree')
            if min(rgb_image.width,rgb_image.height)<896:continue
            if name not in regions:regions[name]=candidates(mask)
            for x,y in regions[name][:150]:
                cw=min(1280,rgb_image.width-x);ch=min(1280,rgb_image.height-y)
                if any(x<u+uw and u<x+cw and y<v+vh and v<y+ch for u,v,uw,vh in used.get(name,[])):continue
                # Separate streaming pipelines are necessary after earlier reads.
                im=pyvips.Image.new_from_file(str(image),access='sequential').crop(x,y,cw,ch)
                ma=pyvips.Image.new_from_file(str(mask),access='sequential').crop(x,y,cw,ch)
                rgb=array(im)[...,:3];raw=decode(array(ma))
                if name not in source_hash:source_hash[name]=[sha(image),sha(mask)]
                source=dict(id=name+f'__x{x}_y{y}',slide_id=name,group_id='histoseg-slide-'+name,
                    group_level='provider_source_mosaic; slides from14 participants total, individual linkage not assumed',
                    source_region_type='WSI mosaic',image=str(image.relative_to(a.data)),mask=str(mask.relative_to(a.data)),
                    image_sha256=source_hash[name][0],mask_sha256=source_hash[name][1],
                    source_native_shape_yx=[rgb_image.height,rgb_image.width],source_native_xywh=[x,y,cw,ch],
                    microscopy='Provider HEIDSTAR HDS-MS-200A20x acquisition; exact MPP not released/verified, never inferred from magnification',
                    record='https://data.mendeley.com/datasets/vccj8mp2cg/2',license='RepositoryCC BY4.0; source article also says non-commercial research only. Preserve both notices.',
                    provider_split='unsplit (development)',ontology_source='https://pmc.ncbi.nlm.nih.gov/articles/PMC11803237/#tbl0001')
                if writer.consider(rgb,raw,source,None):used.setdefault(name,[]).append([x,y,cw,ch]);break
            if writer.complete:break
        if writer.complete:break
    result=writer.finish();print(json.dumps(dict(dataset='histoseg',accepted=result['accepted'],quota_met=result['quota_met'],source_slides=len(writer.groups))))


if __name__=='__main__':main()
