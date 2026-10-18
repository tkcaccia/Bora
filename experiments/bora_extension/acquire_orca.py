"""Bounded original ORCA acquisition from the author's public Drive folder.

First fetch original masks. Select up to40 mixed carcinoma/non-carcinoma source
images after the real-tile gate, preferring different TCGA case IDs. No masks
are enlarged and no third-party processed/augmented derivatives are used.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import gdown
from .tissue_fields import choose_field
from .orca_labels import decode_orca


FOLDER='https://drive.google.com/drive/folders/1XfplgYK5JWzzYWXQhrPUQujXNKUDK-WR'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get_file(root,row):
    target=root/Path(row['path']).name
    if not target.exists():
        if shutil.disk_usage(root).free < 20*2**30+512*2**20:
            raise ValueError('Preserve20GiB reserve during native ORCA acquisition')
        def progress(current,total):
            if current > 128*2**20 or shutil.disk_usage(root).free < 20*2**30:
                raise ValueError('Bounded native-image download/reserve exceeded')
        value=gdown.download(id=row['id'],output=str(target),quiet=True,use_cookies=False,
                             resume=True,progress=progress)
        if not value:
            raise RuntimeError('Author file unavailable: '+row['path'])
    with Image.open(target) as image:
        image.verify()
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--list',type=Path,required=True)
    args=parser.parse_args()
    entries=json.loads(args.list.read_text())
    folder=args.root/'raw/orca/validation-native'
    folder.mkdir(parents=True,exist_ok=True)
    masks=[r for r in entries if r['path'].endswith('_mask.png')]
    image_by_name={Path(r['path']).name:r for r in entries if not r['path'].endswith('_mask.png')}
    unavailable=[]
    def available(row):
        try:return get_file(folder,row)
        except Exception as exc:return dict(id=row['id'],path=row['path'],reason=type(exc).__name__+': '+str(exc))
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths=list(pool.map(available,masks))
    candidates,excluded=[],[]
    for row,path in zip(masks,paths):
        if isinstance(path,dict):unavailable.append(path);continue
        a,normalization=decode_orca(np.array(Image.open(path)))
        codes=sorted(map(int,np.unique(a)))
        groups=int(127 in codes)+int(255 in codes)
        # One grey code plus carcinoma. Zero is actual non-tissue, not a group.
        reason=[]
        if groups<2:reason.append('single_tissue_group')
        if min(a.shape)<896:reason.append('insufficient_real_field')
        if not reason:
            dummy=np.full((*a.shape,3),180,np.uint8)
            test,why=choose_field(dummy,a,dict(labels={127:'non-carcinoma',128:'non-carcinoma',255:'carcinoma'},ignore=[0]),None)
            if test is None:reason.append('no_field_meets_full_coarse_group_tile_gate')
        name=path.name.removesuffix('_mask.png')+'.png'
        item=dict(id=name.removesuffix('.png'),image_name=name,
            mask=str(path.relative_to(args.root)),mask_sha256=sha(path),
            shape_yx=list(a.shape),raw_codes=codes,mask_normalization=normalization,source_group='-'.join(name.split('-')[:3]),
            original_image=image_by_name[name],source_mask=row)
        if reason:excluded.append(dict(**item,reasons=reason))
        else:candidates.append(item)
    # Round robin over source cases before reusing a case. This is a download
    # shortlist only: the full16-coarse-observation/tissue gate follows pairing.
    candidates.sort(key=lambda r:r['id'])
    selected,used=[],set()
    for unique_only in (True,False):
        for item in candidates:
            if item in selected or unique_only and item['source_group'] in used:continue
            selected.append(item);used.add(item['source_group'])
            if len(selected)==40:break
        if len(selected)==40:break
    shortlist=selected+[r for r in candidates if r not in selected]
    selected=[]
    for row in shortlist:
        try:image=get_file(folder,row['original_image'])
        except Exception as exc:
            unavailable.append(dict(id=row['original_image']['id'],path=row['image_name'],reason=type(exc).__name__+': '+str(exc)))
            continue
        with Image.open(image) as im:
            if [im.height,im.width]!=row['shape_yx']:
                raise ValueError('Original author RGB/mask dimensions disagree')
        row.update(image=str(image.relative_to(args.root)),image_sha256=sha(image))
        selected.append(row)
        if len(selected)==40:break
    report=dict(dataset='orca',source='https://sites.google.com/unibas.it/orca/download',
        original_folder=FOLDER,folder_inventory_sha256=sha(args.list),
        annotation='carcinoma255; non-carcinoma tissue127/128; non-tissue0 ignored',
        release='original validation100',license='provider educational-use notice; no broader license inferred',
        microscopy_mpp=None,groups=len({r['source_group'] for r in selected}),selected=selected,excluded=excluded,
        temporarily_unavailable=unavailable,
        quota=40,selected_count=len(selected),full_coarse_gate_pending=True,
        segmentation_run=False,encoder_run=False)
    output=args.root/'metadata/orca-native-acquisition-v1.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(dataset='orca',selected=len(selected),source_groups=report['groups'],excluded=len(excluded))),flush=True)


if __name__=='__main__':main()
