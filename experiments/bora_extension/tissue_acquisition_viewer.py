"""Audit acquired native/coarse fields and render an image/mask inventory.

This is an acquisition viewer, with no Virchow2 or Bora results or scores.
Only datasets with exactly40 qualifying cases are admitted to its manifest.
"""
import argparse
from collections import Counter
import html
import json
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image
import tifffile
from experiments.virchow2_multi.plan import coarse_mask
from .eligibility import inspect,POLICY_PATH
from .tissue_fields import sha
from .calibration import evaluate as physical_calibration,POLICY_PATH as CALIBRATION_POLICY_PATH

SOURCES={'ignite':'v1','peso':'v2','gcss':'v3','orca':'v6','puma':'v6','histoseg':'v7','gleasonxai':'v4'}


def preview(folder,row,out):
    out.mkdir(parents=True,exist_ok=True)
    native=np.array(Image.open(folder/'original.native.png'))
    ref=tifffile.imread(folder/'reference.native.tif')
    coarse=tifffile.imread(folder/'coarse.tif')
    coarse=cv2.resize(coarse,(native.shape[1],native.shape[0]),interpolation=cv2.INTER_NEAREST)
    def colors(mask):
        rgb=np.zeros((*mask.shape,3),np.uint8)
        for label,color in enumerate(row['palette'],1):rgb[mask==label]=color
        return rgb
    target=colors(ref);tile=colors(coarse)
    overlay=native.copy();overlay[ref>0]=(.55*native[ref>0]+.45*target[ref>0]).astype(np.uint8)
    tiled=native.copy();tiled[coarse>0]=(.55*native[coarse>0]+.45*tile[coarse>0]).astype(np.uint8)
    for name,array in [('he',native),('reference',target),('overlay',overlay),('coarse',tiled)]:
        image=Image.fromarray(array);image.thumbnail((640,640));image.save(out/(name+'.jpg'),quality=87)


def audit(folder,row,policy):
    for name,expected in row['files_sha256'].items():
        if sha(folder/name)!=expected:raise ValueError('Acquired file hash changed: '+str(folder/name))
    native=np.array(Image.open(folder/'original.native.png'))
    work=np.array(Image.open(folder/'original.png'))
    if np.all((native.min(axis=2)>=230)&(np.ptp(native.astype(np.int16),axis=2)<=20)):
        raise ValueError('No visible non-white tissue in acquired field')
    ref=tifffile.imread(folder/'reference.native.tif');cm=tifffile.imread(folder/'coarse.tif')
    if native.shape!=(*ref.shape,3) or list(work.shape[:2])!=row['working_shape_yx'] or cm.shape!=work.shape[:2]:raise ValueError('Native/working geometry changed')
    if min(ref.shape)<896 or min(work.shape[:2])<896:raise ValueError('Synthetic or insufficient real source capacity')
    if not np.array_equal(cm,coarse_mask(cv2.resize(ref,(work.shape[1],work.shape[0]),interpolation=cv2.INTER_NEAREST))):raise ValueError('Coarse input does not match declared90px majority convention')
    check=inspect(cm,work.shape[:2],policy)
    if not check['eligible']:raise ValueError('Acquired field fails frozen tile/group policy')
    if set(map(int,np.unique(ref)))-{0}-set(map(int,row['class_names'])):raise ValueError('Fine reference class omitted')
    if len(row['stained_observations_by_group'])<2:raise ValueError('Glass masquerades as a second tissue group')
    mpp=row['native_mpp_xy'];wmpp=row['working_mpp_xy']
    if mpp is None:
        if wmpp is not None or work.shape!=native.shape or row['physical_metrics_allowed']:raise ValueError('Unknown calibration was invented')
    else:
        expected=[native.shape[1]*mpp[0]/work.shape[1],native.shape[0]*mpp[1]/work.shape[0]]
        if not np.allclose(expected,wmpp,rtol=0,atol=1e-12):raise ValueError('Physical sampling transform disagrees')
    return dict(case=row['case'],native_capacity_224_yx=[ref.shape[0]//224,ref.shape[1]//224],
        usable_coarse_tiles=check['usable_coarse_tiles'],observed_groups=check['observed_groups'],
        fine_groups=sorted(map(int,np.unique(ref[ref>0]))),file_hashes_checked=4,
        annotated_fraction=float(np.mean(ref>0)),physical_calibration=physical_calibration(row),passed=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);policy=json.loads(POLICY_PATH.read_text())
    allrows=[];datasets=[];pending=[];audits=[];hashes=set()
    for dataset,version in SOURCES.items():
        root=a.data/'results'/f'bora-tissue-expansion-inputs-2026-10-18-{version}'/dataset
        plan=root/'PLAN.json'
        if not plan.exists():
            journal=root/'ACQUISITION_PROGRESS.json'
            n=json.loads(journal.read_text())['accepted'] if journal.exists() else 0
            pending.append(dict(dataset=dataset,accepted=n,quota=40,stage='acquisition_or_qualification_pending'));continue
        value=json.loads(plan.read_text())
        if value['accepted']!=40 or len(value['samples'])!=40 or not value['quota_met']:raise ValueError('Exactly40 verified fields required per included dataset')
        report=[]
        for row in value['samples']:
            folder=root/'cases'/row['case'];check=audit(folder,row,policy)
            digest=row['files_sha256']['original.native.png']
            if digest in hashes:raise ValueError('Duplicate native field in expansion')
            hashes.add(digest);report.append(check);audits.append(check)
            row=dict(row,physical_calibration=check['physical_calibration'],acquired_folder=str(folder),preview='images/'+row['case'])
            preview(folder,row,a.out/row['preview']);allrows.append(row)
        groups=len({r['group'] for r in value['samples']})
        datasets.append(dict(dataset=dataset,images=40,source_groups=groups,group_level=value['samples'][0]['source']['group_level'],plan=str(plan),plan_sha256=sha(plan),pixel_only=sum(r['pixel_only'] for r in value['samples']),physically_calibrated=sum(physical_calibration(r)['eligible'] for r in value['samples']),excluded_candidates=len(value['excluded'])))
    calibrated=[r['case'] for r in allrows if r['physical_calibration']['eligible']]
    manifest=dict(physical_calibration_ready_cases=calibrated,physical_calibration_ready_count=len(calibrated),calibration_pending_cases=[r['case'] for r in allrows if not r['physical_calibration']['eligible']],calibration_policy_sha256=sha(CALIBRATION_POLICY_PATH),stage='acquired_audited_not_encoded_or_scored',datasets=datasets,pending=pending,total=len(allrows),samples=allrows,
        audit=audits,all_passed=all(r['passed'] for r in audits),eligibility_policy_sha256=sha(POLICY_PATH),viewer_source_sha256=sha(__file__),
        source_inclusion='Original public tissue-region semantic annotations only.40 eligible fields per dataset. Fine masks used only for score-blind acquisition and controlled coarse-label generation.',
        independence='ROI, slide, TMA-core and patient IDs are explicitly distinct. All used as development data, including source validation/test fields; no independent test claim.',
        inference_status='No new encoder, RGB-conversion or Bora inference has run. No new segmentation scores exist.',
        remaining_before_cycle11=['Source-case overlap against existing datasets','Reconcile calibrated versus pixel-only sampling','Freeze combined cohort and input hashes','Encode all tissue, including unannotated context','Matched parent baselines on the identical new cohort'])
    (a.out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report=['# Additional tissue annotation datasets','',f'{len(allrows)} native-source fields across{len(datasets)} datasets pass acquisition audit;40 per included dataset.','',
        f'Physical calibration ready: {len(calibrated)}. Unknown-calibration images are acquisition inventory only, excluded from new Virchow2/Bora benchmark inference. No new score or accuracy claim. These are selected development ROIs, not whole-slide evaluations.','',
        '|Dataset|Fields|Calibration ready|Source groups|Pixel-only fields|Rejected candidates|','|---|---:|---:|---:|---:|---:|']
    for d in datasets:report.append(f"|{d['dataset']}|40|{d['physically_calibrated']}|{d['source_groups']}|{d['pixel_only']}|{d['excluded_candidates']}|")
    report+=['','Group meaning is recorded per source; source groups are not automatically independent patients.',
        'Unknown MPP remains unknown. Calibrated working grids target0.5µm/px, with both native and working axes recorded. Native capacity must also reach896pixels each axis; upsampling cannot manufacture eligibility.',
        'At least two usable tissue groups,16 usable coarse observations, and four complete224px fields per axis are required. Glass/ignore cannot be the second group.',
        'All fine classes are retained even when absent from coarse input. Unannotated context is ignored in accuracy.','',
        '## Acquisition still pending','']+[f"- {d['dataset']}: {d['accepted']}/40" for d in pending]
    (a.out/'REPORT.md').write_text('\n'.join(report)+'\n')
    data=json.dumps([dict(case=r['case'],dataset=r['dataset'],native=r['native_shape_yx'],working=r['working_shape_yx'],mpp=r['native_mpp_xy'],wmpp=r['working_mpp_xy'],groups=r['class_names'],palette=r['palette'],tiles=r['eligibility']['usable_coarse_tiles'],source=r['source'],calibrated=r['physical_calibration']['eligible'],physical=r['physical_calibration'],preview=r['preview'],folder=r['acquired_folder'],fov=r['fov_um_xy']) for r in allrows]).replace('</','<\\/')
    page='''<!doctype html><html><head><meta charset="utf-8"><title>Bora: additional tissue datasets</title><style>
body{margin:0;background:#eef1f5;color:#18212d;font:16px system-ui}header{padding:18px 24px;background:#182c42;color:white;position:sticky;top:0;z-index:2}h1{font-size:23px;margin:0 0 7px}select,button{font:inherit;padding:6px;margin:8px 6px 0 0}main{padding:20px}h2{margin:0 0 10px}.meta{line-height:1.6;background:white;padding:14px;margin:12px 0}.grid{display:grid;grid-template-columns:repeat(4,minmax(200px,1fr));gap:10px}figure{margin:0;background:white;padding:8px}img{width:100%;height:auto;display:block;image-rendering:auto}figcaption{font-weight:600;margin-bottom:8px}.legend span{display:inline-block;margin:5px 13px 5px 0}.swatch{width:14px;height:14px;display:inline-block;margin-right:5px}a{color:#1266a3}header a{color:#9dd6ff}small{font-size:13px}.notice{color:#ffdf90}#all{display:none}.sheet{border-top:3px solid #8294a9;margin:30px 0;padding-top:15px}@media(max-width:1000px){.grid{grid-template-columns:repeat(2,1fr)}}
</style></head><body><header><h1>Additional tissue annotation datasets · TOTAL acquired · CALIBRATED calibrated</h1><div class="notice">Acquisition review: original H&E, expert reference and controlled coarse input. No new Bora results or scores.</div><select id="dataset"><option value="">All datasets</option></select><select id="case"></select><button id="prev">← Previous</button><button id="next">Next →</button><button id="mode">Show every field</button> <a href="REPORT.md">Report</a> · <a href="MANIFEST.json">Manifest / provenance</a></header><main><div id="single"></div><div id="all"></div></main><script>
const rows=DATA;let filtered=rows,index=0,allMode=false;
const ds=document.querySelector('#dataset'),cs=document.querySelector('#case'),single=document.querySelector('#single'),all=document.querySelector('#all');
const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
[...new Set(rows.map(r=>r.dataset))].forEach(d=>ds.add(new Option(d,d)));
function panel(r){const mpp=r.mpp?r.mpp.map(v=>v.toFixed(5)).join(' × ')+' µm/px':'Unknown MPP · native pixels preserved';const labels=Object.entries(r.groups).map(([k,v])=>`<span><i class="swatch" style="background:rgb(${r.palette[Number(k)-1]})"></i>${esc(k+': '+v)}</span>`).join('');return `<h2>${esc(r.case)} · ${esc(r.dataset)}</h2><div class="grid">${[['he','Original H&E'],['reference','Reference mask · black = ignored'],['overlay','Reference over H&E · 45% opacity'],['coarse','Input tile mask over H&E · 90px tiles']].map(([f,t])=>`<figure><figcaption>${t}</figcaption><img loading="lazy" src="${r.preview}/${f}.jpg" alt="${esc(r.case+' '+t)}">${r.fov?'<div style="width:'+10000/r.fov[0]+'%;border-top:4px solid #18212d;margin-top:8px"></div><small>100 µm</small>':'<small>Scale unknown</small>'}</figure>`).join('')}</div><div class="meta"><div class="legend">${labels}</div><b>${r.calibrated?'Physical sampling passes encoder preflight':'CALIBRATION PENDING · excluded from Virchow2/Bora benchmark'}</b><br><b>Analysis field: ROI</b> · native ${r.native[1]}×${r.native[0]}px · working ${r.working[1]}×${r.working[0]}px · ${r.tiles} usable coarse observations.<br><b>Native sampling:</b> ${mpp}${r.fov?' · field '+r.fov.map(v=>v.toFixed(1)).join(' × ')+' µm':''}<br><b>Working sampling:</b> ${r.wmpp?r.wmpp.map(v=>v.toFixed(5)).join(' × ')+' µm/px':'Pixels only'}<br><b>224px Virchow2 windows:</b> ${r.calibrated?Object.values(r.physical.scales).map(s=>s.window_um_xy.map(v=>v.toFixed(1)).join(' × ')+' µm').join('; '):'Blocked pending calibration'}<br><b>Base interpolation:</b> ${r.calibrated?r.physical.scales['1'].interpolation_upscale_xy.map(v=>v.toFixed(2)).join(' × ')+' (1.00 = no enlargement; added pixels do not restore optical detail)':'Not defined'}<br><b>Source:</b> <a href="${esc(r.source.record)}">${esc(r.source.id)}</a> · ${esc(r.source.group_level)}<br>${esc(r.source.microscopy)}<br><small>${esc(r.source.license)} · ${esc(r.folder)}</small></div>`}
function render(){if(!filtered.length)return;cs.selectedIndex=index;single.innerHTML=panel(filtered[index]);location.hash=filtered[index].case;if(allMode){all.innerHTML=filtered.map(r=>'<section class="sheet">'+panel(r)+'</section>').join('')}single.style.display=allMode?'none':'block';all.style.display=allMode?'block':'none'}
function filter(){filtered=rows.filter(r=>!ds.value||r.dataset===ds.value);index=0;cs.replaceChildren(...filtered.map(r=>new Option(r.case,r.case)));render()}
ds.onchange=filter;cs.onchange=()=>{index=cs.selectedIndex;render()};document.querySelector('#prev').onclick=()=>{index=(index+filtered.length-1)%filtered.length;render()};document.querySelector('#next').onclick=()=>{index=(index+1)%filtered.length;render()};document.querySelector('#mode').onclick=()=>{allMode=!allMode;document.querySelector('#mode').textContent=allMode?'Show selected field':'Show every field';render()};filter();const hash=decodeURIComponent(location.hash.slice(1));const wanted=filtered.findIndex(r=>r.case===hash);if(wanted>=0){index=wanted;render()}
</script></body></html>'''
    # Restore requested hash before initial render rather than overwriting it.
    page=page.replace('const rows=DATA;','const initialHash=decodeURIComponent(location.hash.slice(1));const rows=DATA;').replace('const hash=decodeURIComponent(location.hash.slice(1));','const hash=initialHash;')
    page=page.replace('TOTAL',str(len(allrows))).replace('CALIBRATED',str(len(calibrated))).replace('DATA',data)
    (a.out/'index.html').write_text(page)
    print(json.dumps(dict(total=len(allrows),datasets=datasets,pending=pending,all_passed=manifest['all_passed'])))


if __name__=='__main__':main()
