"""Acquire forty eligible tissue fields per source; no model or score access.

Fine masks are used only here, to define controlled coarse inputs and select
boundary-containing fields. They must not be read by later prediction stages.
"""
from collections import Counter
import colorsys
import hashlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import tifffile

from .eligibility import inspect, POLICY_PATH
from experiments.virchow2_multi.plan import coarse_mask


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(2**20),b''):value.update(block)
    return value.hexdigest()


def candidate_fields(mask,spacing):
    h,w=mask.shape
    ch,cw=([min(h,max(896,round(640/spacing[1]))),min(w,max(896,round(640/spacing[0])))]
           if spacing is not None else [min(h,1280),min(w,1280)])
    candidates=[]
    for y in sorted({0,(h-ch)//2,h-ch}):
        for x in sorted({0,(w-cw)//2,w-cw}):
            a=mask[y:y+ch:8,x:x+cw:8]
            _,n=np.unique(a[a>0],return_counts=True)
            p=n/max(n.sum(),1)
            entropy=-float(np.sum(p*np.log(np.maximum(p,1e-30))))
            candidates.append((entropy,float(np.count_nonzero(a)/a.size),-y,-x,y,x,ch,cw))
    return [r[-4:] for r in sorted(candidates,reverse=True)]


def choose_field(rgb,raw,catalog,spacing):
    if rgb.shape != (*raw.shape,3):raise ValueError('Source RGB/mask geometry mismatch')
    if set(map(int,np.unique(raw)))-set(catalog['labels'])-set(catalog['ignore']):
        raise ValueError('Unreconciled source categories')
    codes=sorted(g for g in np.unique(raw) if int(g) in catalog['labels'])
    mapping={int(g):i+1 for i,g in enumerate(codes)}
    mask=np.zeros(raw.shape,np.uint8)
    for g,label in mapping.items():mask[raw==g]=label
    if len(mapping)<2:return None,dict(reason='single_tissue_group')
    policy=json.loads(POLICY_PATH.read_text())
    failures=Counter()
    for y,x,ch,cw in candidate_fields(mask,spacing):
        shape=[round(ch*spacing[1]/.5),round(cw*spacing[0]/.5)] if spacing is not None else [ch,cw]
        # Resampling a coarser provider grid cannot create real pixel fields.
        if min(ch,cw)<896 or min(shape)<4*224:
            failures['insufficient_real_field_at_recorded_sampling']+=1
            continue
        truth=mask[y:y+ch,x:x+cw]
        work=cv2.resize(truth,tuple(shape[::-1]),interpolation=cv2.INTER_NEAREST)
        coarse=coarse_mask(work)
        check=inspect(coarse,shape,policy)
        if not check['eligible']:
            failures.update(check['exclusion_reasons']);continue
        crop=rgb[y:y+ch,x:x+cw]
        he=cv2.resize(crop,tuple(shape[::-1]),interpolation=cv2.INTER_AREA if ch>=shape[0] and cw>=shape[1] else cv2.INTER_LINEAR)
        white=(he.min(axis=2)>=230)&(np.ptp(he.astype(np.int16),axis=2)<=20)
        if white.all():
            failures['no_visible_nonwhite_tissue']+=1
            continue
        observed=Counter()
        negative= mapping.get(catalog.get('valid_negative_raw_label'))
        for tile in check['coarse_observations']:
            ty,tx=tile['y'],tile['x']
            middle=white[ty+22:min(ty+68,shape[0]),tx+22:min(tx+68,shape[1])]
            # Expert tissue categories can include pale fat/lumina. Only a
            # catch-all negative target needs this check to stop glass alone
            # masquerading as the second tissue group.
            if tile['label'] != negative or middle.size and np.mean(~middle)>=.1:
                observed[tile['label']]+=1
        if len(observed)<2:
            failures['fewer_than_two_groups_with_stained_tissue_observations']+=1
            continue
        # Keep every fine class in the selected field, including classes absent
        # from coarse supervision. Compact IDs are just serialization.
        present=sorted(map(int,np.unique(truth[truth>0])))
        compact={g:i+1 for i,g in enumerate(present)}
        fine=np.zeros(truth.shape,np.uint8);cm=np.zeros(coarse.shape,np.uint8)
        for g,label in compact.items():fine[truth==g]=label;cm[coarse==g]=label
        raw_to_compact={g:compact[label] for g,label in mapping.items() if label in compact}
        check=inspect(cm,shape,policy)
        return dict(native=crop,working=he,reference=fine,coarse=cm,
            crop_xywh=[x,y,cw,ch],working_shape_yx=shape,
            raw_to_compact=raw_to_compact,eligibility=check,
            stained_observations_by_group={str(compact[g]):n for g,n in observed.items()},
            native_mpp_xy=spacing,native_capacity_224_yx=[ch//224,cw//224],
            working_mpp_xy=None if spacing is None else [cw*spacing[0]/shape[1],ch*spacing[1]/shape[0]]), None
    return None,dict(reason='no_eligible_boundary_field',window_failure_counts=dict(failures))


class FieldWriter:
    def __init__(self,data,output,dataset,catalog,quota=40):
        self.data,self.output,self.dataset,self.catalog,self.quota=Path(data),Path(output),dataset,catalog,quota
        if self.output.name==dataset:
            raise ValueError('Pass the acquisition output root, not the dataset subfolder; '+dataset+' is appended automatically')
        self.root=self.output/dataset
        self.root.mkdir(parents=True,exist_ok=True)
        if (self.root/'PLAN.json').exists():raise FileExistsError('Dataset fields already frozen')
        self.rows=[];self.excluded=[]
        self.journal=self.root/'ACQUISITION_PROGRESS.json'
        if self.journal.exists():
            previous=json.loads(self.journal.read_text());self.rows=previous['samples'];self.excluded=previous['excluded']
            for row in self.rows:
                for name,expected in row['files_sha256'].items():
                    if sha(self.root/'cases'/row['case']/name)!=expected:raise ValueError('Acquired field changed')

    @property
    def groups(self):return {r['group'] for r in self.rows}
    @property
    def complete(self):return len(self.rows)>=self.quota

    def consider(self,rgb,raw,source,spacing):
        if self.complete:return False
        if any(r['source']['id']==source['id'] for r in self.rows):return False
        selected,reason=choose_field(rgb,raw,self.catalog,spacing)
        if selected is None:
            self.excluded.append(dict(source=source,**reason));self.checkpoint();return False
        encoded=io.BytesIO();Image.fromarray(selected['native']).save(encoded,format='PNG')
        payload=encoded.getvalue();native_digest=hashlib.sha256(payload).hexdigest()
        duplicate=next((r['case'] for r in self.rows if r['files_sha256']['original.native.png']==native_digest),None)
        if duplicate is not None:
            self.excluded.append(dict(source=source,reason='duplicate_native_field',duplicate_of=duplicate))
            self.checkpoint();return False
        case=f'{self.dataset}_{len(self.rows)+1:02d}'
        folder=self.root/'cases'/case
        folder.mkdir(parents=True,exist_ok=True)
        (folder/'original.native.png').write_bytes(payload)
        Image.fromarray(selected['working']).save(folder/'original.png')
        tifffile.imwrite(folder/'reference.native.tif',selected['reference'],compression='deflate')
        tifffile.imwrite(folder/'coarse.tif',selected['coarse'],compression='deflate')
        names={str(label):self.catalog['labels'][rawcode] for rawcode,label in selected['raw_to_compact'].items()}
        ordered=sorted(selected['raw_to_compact'],key=selected['raw_to_compact'].get)
        palette=[self.catalog.get('palette',{}).get(int(g),
            [round(c*255) for c in colorsys.hsv_to_rgb((int(g)*.61803398875)%1,.68,.86)]) for g in ordered]
        files={name:sha(folder/name) for name in ('original.native.png','original.png','reference.native.tif','coarse.tif')}
        row=dict(case=case,dataset=self.dataset,source=source,group=source['group_id'],source_region_type='ROI',
            native_crop_xywh=selected['crop_xywh'],native_shape_yx=list(selected['reference'].shape),
            working_shape_yx=selected['working_shape_yx'],native_capacity_224_yx=selected['native_capacity_224_yx'],native_mpp_xy=selected['native_mpp_xy'],
            working_mpp_xy=selected['working_mpp_xy'],pixel_only=spacing is None,
            physical_metrics_allowed=spacing is not None,raw_to_compact=selected['raw_to_compact'],
            class_names=names,palette=palette,raw_ignore_codes=self.catalog['ignore'],
            coarse_tile_px=90,coarse_tile_um=45 if spacing is not None else None,
            coarse_reference_derived=True,files_sha256=files,eligibility=selected['eligibility'],
            stained_observations_by_group=selected['stained_observations_by_group'],
            tissue_group_gate='At least two observed tissue categories. A declared catch-all negative target counts only if >=10% of its central coarse observation is nonwhite H&E (white min>=230 and range<=20). Explicit expert pale-tissue categories are retained. Dedicated mask background/ignore is excluded.',
            fov_um_xy=None if spacing is None else [selected['reference'].shape[1]*spacing[0],selected['reference'].shape[0]*spacing[1]],
            original_sha256=files['original.png'],coarse_sha256=files['coarse.tif'],
            reference_native_sha256=files['reference.native.tif'],
            annotation_task=self.catalog['task'],valid_negative_raw_label=self.catalog.get('valid_negative_raw_label'),
            annotation_provenance=self.catalog['annotation_provenance'])
        (folder/'INPUTS_ACQUIRED.json').write_text(json.dumps(row,indent=2)+'\n')
        self.rows.append(row);self.checkpoint()
        print(json.dumps(dict(acquired_field=case,groups=len(selected['eligibility']['observed_groups']),usable_tiles=selected['eligibility']['usable_coarse_tiles'])),flush=True)
        return True

    def checkpoint(self):
        value=dict(dataset=self.dataset,quota=self.quota,accepted=len(self.rows),samples=self.rows,
            excluded=self.excluded,source_code_sha256=sha(__file__),
            eligibility_policy_sha256=sha(POLICY_PATH),encoder_run=False,bora_run=False,
            fine_reference_used_only_for_acquisition=True,score_based_selection=False,
            ground_truth_masks_never_supplied_to_prediction=True)
        temporary=self.journal.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(self.journal)

    def finish(self):
        self.checkpoint()
        result=json.loads(self.journal.read_text())
        result['quota_met']=self.complete
        result['stage']='acquired_not_encoded_or_scored'
        if self.complete:(self.root/'PLAN.json').write_text(json.dumps(result,indent=2)+'\n')
        else:(self.root/'QUOTA_SHORTFALL.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
