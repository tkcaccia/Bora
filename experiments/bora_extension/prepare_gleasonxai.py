"""Original GleasonXAI tissue-pattern region labels on released TMA images.

Use Gleason patterns3/4/5, not gland instances or classifier image labels.
Normalized expert polygons are rasterized per provider annotator ID in the
provider's grade frame order. The unique plurality of covering annotations is
used; ties and unannotated tissue remain unknown. This is a documented local
consensus, not a claim that the provider distributed this exact hard mask.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from .tissue_fields import FieldWriter,sha

CATALOG=dict(labels={3:'Gleason pattern3 tissue region',4:'Gleason pattern4 tissue region',5:'Gleason pattern5 tissue region'},ignore=[0],
    task='prostate carcinoma architectural tissue-pattern regions; not gland instances',
    annotation_provenance='Original GleasonXAI pathologist region polygons; local unique-plurality consensus across covering provider annotator IDs, ties/unannotated ignored; grade order3,4,5 matches source rasterizer',
    palette={3:[55,165,75],4:[55,110,230],5:[220,45,65]})


def consensus(rows,shape):
    raters=defaultdict(list)
    for row in rows:raters[row['annotator']].append(row)
    votes=np.zeros((*shape,3),np.uint16)
    for annotator,items in sorted(raters.items()):
        mask=np.zeros(shape,np.uint8)
        # Preserve CSV order within a grade, as in the provider implementation.
        for grade in (3,4,5):
            for row in items:
                if int(row['grade'])!=grade:continue
                points=np.asarray(json.loads(row['coords']),float)
                if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all() or points.min()<0 or points.max()>1:
                    raise ValueError('Invalid provider normalized polygon coordinates')
                points=(points*np.array(shape[::-1])).astype(np.int32)
                cv2.fillPoly(mask,[points],grade)
        for i,grade in enumerate((3,4,5)):votes[...,i]+=(mask==grade)
    ordered=np.sort(votes,axis=2)
    valid=(ordered[...,-1]>0)&(ordered[...,-1]>ordered[...,-2])
    result=np.zeros(shape,np.uint8);result[valid]=(np.argmax(votes,axis=2)[valid]+3).astype(np.uint8)
    return result,dict(provider_annotator_ids=sorted(raters),rater_count=len(raters),
        disagreement_tie_pixels=int(np.count_nonzero((ordered[...,-1]>0)&~valid)),
        unannotated_pixels=int(np.count_nonzero(ordered[...,-1]==0)),
        consensus='unique plurality among covering original annotator IDs; ties and no coverage ignored; no pseudo-labels')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    folder=a.data/'raw/gleasonxai';annotation=folder/'final_filtered_explanations_df.csv'
    rows=defaultdict(list)
    for row in csv.DictReader(annotation.open()):rows[row['TMA_identifier']].append(row)
    writer=FieldWriter(a.data,a.output,'gleasonxai',CATALOG)
    paths={p.stem:p for p in (folder/'extracted').rglob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png') and not p.name.startswith('._') and '__MACOSX' not in p.parts}
    failures=[]
    for name,items in sorted(rows.items()):
        if name not in paths or not name.startswith('PR') or len({r['grade'] for r in items})<2:continue
        if 'gleasonxai-core-'+name in writer.groups:continue
        path=paths[name]
        with Image.open(path) as image:rgb=np.array(image.convert('RGB'))
        raw,info=consensus(items,rgb.shape[:2])
        source=dict(id=name,group_id='gleasonxai-core-'+name,group_level='TissueArray provider TMA core; no independent patient linkage assumed',
            source_region_type='TMA core',image=str(path.relative_to(a.data)),image_sha256=sha(path),
            annotation=str(annotation.relative_to(a.data)),annotation_sha256=sha(annotation),
            annotation_rows=len(items),consensus_details=info,source_native_shape_yx=list(raw.shape),
            record='https://doi.org/10.6084/m9.figshare.27301845.v1',license='CC BY4.0',provider_split='source pooled (development)',
            source_mpp_xy=[.5455,.5455],
            microscopy='TissueArray original-image0.5455micrometres/px reported by the author calibration script; no independent scanner measurement',
            calibration_source='https://github.com/DBO-DKFZ/GleasonXAI/blob/main/scripts/create_downscaled_dataset.py')
        writer.consider(rgb,raw,source,[.5455,.5455])
        if writer.complete:break
    result=writer.finish();print(json.dumps(dict(dataset='gleasonxai',accepted=result['accepted'],quota_met=result['quota_met'],matched_native_images=len(paths))))


if __name__=='__main__':main()
