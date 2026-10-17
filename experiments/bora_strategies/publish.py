"""Every retained conversion and mask, with synchronized annotation overlays."""
import argparse,json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
import tifffile
from experiments.virchow2_inverse.common import digest,save_json
from experiments.virchow2_inverse.finalize import font
from experiments.virchow2_multi.publish import overlay,tile
from experiments.virchow2_smoothing.publish import save_palette
from .registry import STRATEGIES


def render(root,source,row,strategies=None):
    strategies=STRATEGIES if strategies is None else strategies
    cv2.setNumThreads(2);case=row['case'];out=root/'cases'/case;src=source/'cases'/case
    rgb=np.array(Image.open(src/'original.native.png').convert('RGB'));palette=row['sample']['palette'];truth=tifffile.imread(src/'reference.native.tif')
    support=tifffile.imread(src/'tissue.native.tif')>0;coarse=tifffile.imread(out/'tiles.native.tif')
    save_palette(truth,palette,out/'expert.mask.png');save_palette(coarse,palette,out/'coarse.mask.png')
    coverage=np.zeros((*truth.shape,4),np.uint8);coverage[truth>0]=[30,110,230,255];coverage[support&(truth==0)]=[230,175,35,255];Image.fromarray(coverage).save(out/'coverage.mask.png')
    ratio=min(1.,1536/max(rgb.shape[:2]));shape=tuple(max(1,round(n*ratio)) for n in rgb.shape[:2])
    row['display_mpp_xy']=[row['sample']['native_mpp_xy'][0]*rgb.shape[1]/shape[1],row['sample']['native_mpp_xy'][1]*rgb.shape[0]/shape[0]]
    small=Image.fromarray(rgb);small.thumbnail((360,360));smallrgb=np.array(small)
    for conversion,record in row['conversions'].items():
        folder=out/conversion;sheet=Image.new('RGB',(1120,1010),'white');draw=ImageDraw.Draw(sheet)
        draw.text((12,6),case+' | RGB '+conversion+' | '+row['cohort']+' | native macro IoU',fill='#203744',font=font(17))
        converted=Image.open(src/conversion/'representation.png')
        for i,(im,title) in enumerate([(rgb,'Original H&E'),(overlay(rgb,truth,palette),'Original annotation'),(overlay(rgb,coarse,palette),'Input tiles: IoU %.4f'%row['tiles']['macro_iou']),(converted,'Actual converted RGB')]):tile(sheet,im,i*280,35,title,size=280)
        for i,(key,(title,_)) in enumerate(strategies.items()):
            target=folder/key;mask=tifffile.imread(target/'bora.native.tif');diff=mask!=coarse
            save_palette(mask,palette,target/'mask.png');save_palette(diff,[[245,30,180]],target/'difference.mask.png',binary=True)
            view=cv2.resize(mask,tuple(shape[::-1]),interpolation=cv2.INTER_NEAREST);vd=cv2.resize(diff.astype(np.uint8),tuple(shape[::-1]),interpolation=cv2.INTER_NEAREST)>0
            save_palette(view,palette,target/'view.mask.png');save_palette(vd,[[245,30,180]],target/'view.difference.mask.png',binary=True)
            preview=overlay(smallrgb,cv2.resize(mask,small.size,interpolation=cv2.INTER_NEAREST),palette)
            Image.fromarray(preview).save(target/'overlay.jpg',quality=88)
            values=record['strategies'][key];tile(sheet,preview,(i%4)*280,360+(i//4)*325,key+' '+title[:24],f"IoU {values['score']['macro_iou']:.4f}; delta tiles {values['delta_vs_tiles']:+.4f}",size=280)
        sheet.save(folder/'contact.jpg',quality=90)
    save_json(out/'scores.json',row);print(json.dumps(dict(rendered=case)),flush=True);return row


def report(root,results):
    cases=results['cases'];counts=Counter(r['dataset'] for r in cases);fresh=[r for r in cases if r['cohort']=='new_extension'];newcounts=Counter(r['dataset'] for r in fresh)
    total=len(cases)*len(results['protocol']['conversions'])*len(STRATEGIES)
    lines=[f'# Expanded Bora benchmark: {len(cases)} annotated ROIs','',f'{len(fresh)} newly added ROIs: {dict(newcounts)}. Full corpus: {dict(counts)}. Five retained RGB conversions (#9, #10, #15, #19, equal 9+19); eight retained Bora strategies; {total:,} native final masks.','',
    'Rejected strategies B04, B20, B16, B17, B14, B13, B01, B06, B07, B11, B15 and B18, and RGB conversions #4, #5 (Worst-class geometry), #8 (Stain perturbation), are excluded. Their benchmark implementations have been removed from the active source. Previous results remain an explicitly historical archive.','',
    'B19 is revision 2: equal votes from retained B02/B05/B09/B10, with appearance resolving ties. It contains no rejected solver. Its revised definition is distinct from historical B19. Other retained solvers and RGB conversions retain their prior parameters.','',
    '## Cohorts and selection','',
    'The original 49 cases are retained as previously inspected. Additional TIGER and BCSS groups are absent from that 49-case plan. TIGER uses non-TCGA cases and excludes group 114; BCSS uses TCGA. Each dataset contributes at most one ROI per recorded group. New images were selected deterministically with seed 8811, choosing annotated boundary-containing fields by class entropy, never by strategy scores. All previously available BEETLE cases and nearly all eligible CAMELYON groups were already used; there are no new cases from those datasets in this extension. Group IDs follow provider metadata; complete patient identity verification is unavailable.','',
    'All inputs are ROIs, including crops originating from full WSIs. Working encoder sampling targets 0.5 µm/px; exact actual per-axis sampling, native dimensions, source crop and field of view are recorded in PLAN.json and each SAMPLE.json. Virchow2 uses overlapping 224-pixel windows, 112-pixel stride and 14-pixel spatial token spacing. Nominal scale variants target 1 and 2 µm/px; actual sampling after integer resizing is saved. Optical resolving power and objective magnification are unknown. No resampling creates extra optical detail.','',
    '## Supervision and tissue policy','',
    'Feature selection remains per-image, supervised by the original reference-derived coarse tiles: pair-balanced Fisher separation with a correlation penalty selects 64 of 1280 spatial feature dimensions; #9 further retains 32 with cross-scale persistence. Coarse tiles are 90 working pixels (nominally 45 µm). Fine boundaries are excluded from prediction code. Entire original RGB fields are encoded, including unannotated tissue. Plain selected-feature evidence completes unlabelled coarse tiles identically for all solvers; completed pseudo labels never train the feature selector or class heads.','',
    'The original RGB alone determines tissue/background support. Every detected tissue pixel receives a class. Background is removed after segmentation. The scoring domain is the original reference >0 and is never intersected with detected tissue: unannotated tissue is unscored, while annotated tissue mistakenly removed by the detector remains penalized. All expert reference classes contribute to macro IoU, including classes absent from coarse supervision. CAMELYON non-tumor reference is automatic tissue support, not expert stroma.','',
    '## Ranking','',
    'Native macro IoU is averaged across five conversions within each case, then cases within each dataset, then equally across the datasets represented in that cohort. New-only ranking therefore balances two datasets; full ranking balances four. Baseline is the completed input tile mask with the same final background clipping. Differences and 3000-resample paired within-dataset case bootstrap intervals are relative to that input, with conversions kept together. Boundary F1 at 8 µm is secondary. Intervals are descriptive because coarse labels derive from the reference and method choices followed prior visual inspection.','']
    for title,ranking in [('100 new images',results['new_image_ranking']),('All images',results['ranking'])]:
        lines += ['## '+title,'','| Rank | Strategy | Macro IoU | Delta vs tiles | 95% paired interval | Boundary F1 8 µm |','|---|---|---:|---:|---|---:|']
        for r in ranking:
            lo,hi=r['paired_delta_bootstrap95'];bf=r['boundary_f1_8um'];lines.append(f"| {r['rank']} | {r['strategy']} {r['title']} | {r['dataset_balanced_macro_iou']:.6f} | {r['delta_vs_tiles']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {bf:.6f} |")
    lines+=['','## Interpretation and files','',
    'This is a controlled refinement experiment using reference-derived coarse supervision, not independent whole-slide classification or a test using realistic classifier errors. The new groups extend image diversity but do not remove per-image supervision. Aggregate ranks can conceal class loss or poor boundaries; every retained output and per-class score is visible. Native outputs preserve source geometry; browser mask previews use nearest-neighbor resizing capped at 1536 pixels and disclose display sampling. No diffusion model or photorealistic inverse reconstruction is used.','',
    'gallery.html supplies original H&E, original annotation, completed input tiles, every retained converted RGB and all eight synchronized semi-transparent final masks. all_outputs.html contains every case/conversion sheet. scores.csv and results.json retain all scores, per-class metrics and cohort-specific rankings. PROTOCOL.json, input/prediction freezes, runtime_sources, source hashes and AUDIT.json record reproducibility and coverage. CPU inference and scoring use three low-priority workers with two threads each; Virchow2 uses the cached checkpoint and a bounded GPU batch of four. No desktop settings or unrelated analyses are changed.','']
    (root/'REPORT.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args();results=json.loads((a.root/'results.json').read_text())
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn')) as pool:results['cases']=list(pool.map(render,[a.root]*len(results['cases']),[a.source]*len(results['cases']),results['cases']))
    save_json(a.root/'results.json',results);report(a.root,results)
    (a.root/'gallery.html').write_text(Path(__file__).with_name('gallery.html').read_text().replace('__DATA__',json.dumps(results,allow_nan=False)))
    blocks=[]
    for row in results['cases']:
        blocks.append(f'<section><h2>{row["case"]} · {row["cohort"]}</h2>')
        for m in results['protocol']['conversions']:
            blocks.append(f'<h3>RGB {m} · <a href="gallery.html#{row["case"]}/{m}">Synchronized masks and scores</a></h3><a href="gallery.html#{row["case"]}/{m}"><img loading="lazy" src="cases/{row["case"]}/{m}/contact.jpg"></a>')
        blocks.append('</section>')
    (a.root/'all_outputs.html').write_text('<!doctype html><meta charset="utf-8"><title>Expanded Bora: all outputs</title><style>body{font:15px system-ui;background:#edf3f6;color:#203744;margin:18px}section{background:white;padding:15px;margin:25px 0}img{width:100%;max-width:1120px}a{color:#09688e}</style><h1>'+str(len(results['cases']))+' images · five RGB conversions · eight Bora strategies</h1><p><a href="gallery.html">Detailed viewer</a> · <a href="scores.csv">All scores</a> · <a href="REPORT.md">Report</a></p>'+''.join(blocks))
    files={str(p.relative_to(a.root)):digest(p) for p in a.root.rglob('*') if p.is_file() and 'runtime_sources' not in p.parts and p.suffix in ('.png','.jpg','.html')}
    save_json(a.root/'VISUAL_MANIFEST.json',dict(files_sha256=files))

if __name__=='__main__':main()
