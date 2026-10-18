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
FOLDERS={'validation':FOLDER,'test':'https://drive.google.com/drive/folders/1A_xiKTwBNO9XS_NzdDvWJNAx_z1nRnpS'}


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
    parser.add_argument('--split',choices=sorted(FOLDERS),default='validation')
    parser.add_argument('--existing-masks-only',action='store_true')
    parser.add_argument('--max-images',type=int,default=40,choices=range(1,41))
    parser.add_argument('--max-image-attempts',type=int)
    parser.add_argument('--receipt-name')
    parser.add_argument('--skip-unavailable',type=Path,nargs='*',default=[])
    args=parser.parse_args()
    if (args.existing_masks_only or args.max_images!=40 or args.max_image_attempts is not None or args.skip_unavailable) and not args.receipt_name:
        parser.error('Bounded/custom acquisition requires a new --receipt-name')
    if args.receipt_name and Path(args.receipt_name).name!=args.receipt_name:
        parser.error('Receipt name must be a basename')
    if args.max_image_attempts is not None and args.max_image_attempts<args.max_images:
        parser.error('Attempt cap must reach the requested image count')
    entries=json.loads(args.list.read_text())
    folder=args.root/'raw/orca'/(args.split+'-native')
    folder.mkdir(parents=True,exist_ok=True)
    masks=[r for r in entries if r['path'].endswith('_mask.png')]
    image_by_name={Path(r['path']).name:r for r in entries if not r['path'].endswith('_mask.png')}
    unavailable=[]
    def available(row):
        if args.existing_masks_only and not (folder/Path(row['path']).name).exists():
            return dict(id=row['id'],path=row['path'],reason='Mask not in bounded acquired inventory')
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
    previous_ids=set()
    for path in args.skip_unavailable:
        previous=json.loads(path.read_text())
        previous_ids.update(r['id'] for r in previous['temporarily_unavailable'] if not r['path'].endswith('_mask.png'))
    candidates=[r for r in candidates if r['original_image']['id'] not in previous_ids]
    # Round robin over source cases before reusing a case. This is a download
    # shortlist only: the full16-coarse-observation/tissue gate follows pairing.
    candidates.sort(key=lambda r:r['id'])
    selected,used=[],set()
    for unique_only in (True,False):
        for item in candidates:
            if item in selected or unique_only and item['source_group'] in used:continue
            selected.append(item);used.add(item['source_group'])
            if len(selected)==args.max_images:break
        if len(selected)==args.max_images:break
    shortlist=selected+[r for r in candidates if r not in selected]
    selected=[]
    for attempt,row in enumerate(shortlist,1):
        if args.max_image_attempts is not None and attempt>args.max_image_attempts:break
        try:image=get_file(folder,row['original_image'])
        except Exception as exc:
            unavailable.append(dict(id=row['original_image']['id'],path=row['image_name'],reason=type(exc).__name__+': '+str(exc)))
            continue
        with Image.open(image) as im:
            if [im.height,im.width]!=row['shape_yx']:
                raise ValueError('Original author RGB/mask dimensions disagree')
        row.update(image=str(image.relative_to(args.root)),image_sha256=sha(image))
        selected.append(row)
        if len(selected)==args.max_images:break
    report=dict(dataset='orca',source='https://sites.google.com/unibas.it/orca/download',
        original_folder=FOLDERS[args.split],folder_inventory_sha256=sha(args.list),
        annotation='carcinoma255; normalized non-carcinoma tissue127; non-tissue0 ignored',
        release='original '+args.split+'100',license='provider educational-use notice; no broader license inferred',
        microscopy_mpp=None,groups=len({r['source_group'] for r in selected}),selected=selected,excluded=excluded,
        temporarily_unavailable=unavailable,
        requested_original_images=args.max_images,benchmark_field_quota=40,
        existing_masks_only=args.existing_masks_only,image_attempt_cap=args.max_image_attempts,
        skipped_previous_unavailable_original_ids=sorted(previous_ids),
        previous_unavailable_receipts={str(p):sha(p) for p in args.skip_unavailable},
        quota=args.max_images,selected_count=len(selected),full_coarse_gate_pending=True,
        segmentation_run=False,encoder_run=False)
    output=args.root/'metadata'/(args.receipt_name or ('orca-native-acquisition-v1.json' if args.split=='validation' else 'orca-test-native-acquisition-v1.json'))
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(dataset='orca',selected=len(selected),source_groups=report['groups'],excluded=len(excluded))),flush=True)


if __name__=='__main__':main()
