"""Copy a few aligned native image/mask pairs and their physical metadata.

Only acquisition-manifest datasets with forty audited fields are exported.
Source bytes stay unchanged; this does not run an encoder or segmenter.
"""
import argparse
import json
from pathlib import Path
import shutil

from .tissue_fields import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--count',type=int,default=2,choices=range(1,6))
    a=p.parse_args();manifest=json.loads(a.manifest.read_text());receipt=[]
    for dataset in manifest['datasets']:
        rows=[r for r in manifest['samples'] if r['dataset']==dataset['dataset']][:a.count]
        for i,row in enumerate(rows,1):
            folder=Path(row['acquired_folder']);out=a.out/row['dataset'];out.mkdir(parents=True,exist_ok=True)
            pairs={'original.native.png':'image.png','reference.native.tif':'mask.tif',
                'original.png':'working.png','coarse.tif':'coarse-working.tif'}
            files={}
            for source,suffix in pairs.items():
                target=out/f'{i:02d}-{suffix}'
                if sha(folder/source)!=row['files_sha256'][source]:raise ValueError('Acquired source changed')
                shutil.copyfile(folder/source,target);files[target.name]=sha(target)
            overlay=a.manifest.parent/row['preview']/'overlay.jpg'
            shutil.copyfile(overlay,out/f'{i:02d}-overlay.jpg')
            metadata=dict(row,example_files=files,example_image='native source crop; mask shares native grid',
                working_files='separately named working image/coarse mask use the calibrated working grid',
                acquisition_manifest_sha256=sha(a.manifest),encoder_run=False,bora_run=False)
            (out/f'{i:02d}-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
            receipt.append(dict(dataset=row['dataset'],case=row['case'],files=files))
    (a.out/'EXAMPLES.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(examples=len(receipt),datasets=len(manifest['datasets']),out=str(a.out))))


if __name__=='__main__':main()
