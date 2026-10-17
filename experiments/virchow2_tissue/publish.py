"""Every conversion and mask, with separate tissue and scoring supports."""
import argparse,html,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import tifffile
from experiments.virchow2_inverse.common import save_json
from experiments.virchow2_inverse.finalize import font
from experiments.virchow2_multi.publish import rgba,overlay,preview,tile
from .run import ACTIVE,TITLES

PRIMARY=[*[f'{n:02d}' for n in ACTIVE],'ensemble']


def render(root,source,row):
    out=root/'cases'/row['case'];src=source/'cases'/row['case'];palette=row['sample']['palette']
    rgb=np.array(Image.open(src/'original.native.png').convert('RGB'));tissue=tifffile.imread(out/'tissue.native.tif')>0;valid=tifffile.imread(out/'evaluation.native.tif')>0
    coarse=tifffile.imread(out/'coarse.full.native.tif');Image.fromarray(rgba(coarse,palette)).save(out/'coarse.mask.png',compress_level=3)
    coverage=np.zeros(tissue.shape,np.uint8);coverage[tissue&~valid]=1;coverage[valid]=2
    coverage_palette=[[245,170,20],[30,130,240]]
    Image.fromarray(rgba(coverage,coverage_palette)).save(out/'coverage.mask.png',compress_level=3)
    Image.fromarray(rgba(tissue.astype(np.uint8),[[245,170,20]])).save(out/'analysis.mask.png',compress_level=3)
    thumbs=out/'thumbs';thumbs.mkdir(exist_ok=True)
    preview(rgb,thumbs/'original.jpg');preview(overlay(rgb,coarse,palette),thumbs/'coarse.jpg')
    expert=np.array(Image.open(src/'expert.mask.png'));a=rgb.astype(np.float32);m=expert[...,3]>0;a[m]=.65*a[m]+.35*expert[m,:3];preview(np.rint(a).astype(np.uint8),thumbs/'expert.jpg')
    preview(overlay(rgb,coverage,coverage_palette),thumbs/'coverage.jpg');preview(tissue.astype(np.uint8)*255,thumbs/'tissue.jpg')
    sheet=Image.new('RGB',(9*190,3*240+40),'white');draw=ImageDraw.Draw(sheet);draw.text((10,6),row['case']+' | every tissue pixel labelled; original annotation region alone is scored',fill='#203744',font=font(18))
    for i,(name,title) in enumerate([('original','Original H&E'),('expert','Original reference'),('coarse','Completed tile input'),('coverage','Blue scored; gold unscored'),('tissue','RGB tissue / background')]):tile(sheet,Image.open(thumbs/f'{name}.jpg'),i*190,40,title,size=190)
    for i,key in enumerate(TITLES):
        folder=out/key;mask=tifffile.imread(folder/'bora.native.tif');Image.fromarray(rgba(mask,palette)).save(folder/'mask.png',compress_level=3)
        if (folder/'before.native.tif').exists():Image.fromarray(rgba(tifffile.imread(folder/'before.native.tif'),palette)).save(folder/'before.mask.png',compress_level=3)
        display=overlay(rgb,mask,palette);preview(display,folder/'overlay.jpg',size=1200);preview(display,thumbs/f'{key}.overlay.jpg')
        representation=np.array(Image.open(folder/'representation.png').convert('RGB'));preview(representation,thumbs/f'{key}.input.jpg')
        score=row['controls'][key] if key in ('he','features') else row['methods'][key]['after']
        tile(sheet,representation,i*190,280,key+' converted RGB',TITLES[key][:26],size=190)
        tile(sheet,display,i*190,520,key+' Bora mask',f"IoU {score['macro_iou']:.4f}",size=190)
    sheet.save(out/'contact.jpg',quality=93)
    print(json.dumps(dict(rendered=row['case'])),flush=True)


def report(root,results):
    s=results['summary'];coverage=[r['coverage'] for r in results['cases']];unlabelled=sum(r['unannotated_tissue_pixels'] for r in coverage);scored=sum(r['annotated_pixels'] for r in coverage);outside=sum(r['annotated_pixels_outside_detected_tissue'] for r in coverage)
    lines=['# Full-tissue Virchow2 / Bora rerun','',
      'The same 49 native ROIs are rerun with separate analysis, background and quality supports. Retained methods are 5, 8, 9, 10, 15 and 19, plus the equal-evidence 9 + 19 combination. Algorithm 4 remains rejected. H&E and plain-feature controls use the same completed input and background removal. All 441 final masks and actual RGB inputs are retained.','',
      '## Coverage and quality','',
      f'All RGB-detected tissue pixels receive a positive class label in every final mask; every pixel outside that support is zero. The masks include **{unlabelled:,} unannotated tissue pixels** across the 49 fields. These pixels are displayed but excluded from quality scores. Evaluation uses the original **{scored:,} annotation-valid pixels**, with no intersection with predicted tissue. **{outside:,} annotation-valid pixels fall outside the RGB tissue estimate**; their removal remains penalized. These counts include all datasets, with the CAMELYON caveat below.','',
      'The primary metric is per-image native-reference macro IoU, averaged within each dataset and then equally over four datasets. Every original annotation class remains in scoring, including classes absent from the coarse inputs. Label 0 is ignored solely according to the original reference. Boundary F1 uses native physical spacing and excludes annotation-ignore interfaces.','',
      '| Rank | Representation + Bora | Dataset-balanced native macro IoU |','|---|---|---:|']
    for r in results['ranking']:lines.append(f"| {r['rank']} | {r['method']} {r['title']} | {r['dataset_balanced_macro_iou']:.6f} |")
    lines+=['',f"Completed input tiles: {s['dataset_balanced']['coarse']:.6f}; H&E + Bora: {s['dataset_balanced']['he']:.6f}. Current #9: {s['dataset_balanced']['09']:.6f}; current 9 + 19: {s['dataset_balanced']['ensemble']:.6f}. The paired descriptive 95% bootstrap interval for combination minus #9 is {s['paired_delta_bootstrap95']}. All scores and changes from the prior limited-support protocol are in scores.csv. Both coarse completion and background removal change the pipeline; differences from old outputs cannot be attributed only to an RGB conversion.",'',
      '## Prediction protocol','',
      'Existing SHA-verified Virchow2 caches encode the entire original RGB field, not just annotated tiles. Every token covering the original image is checked to have at least one encoder window. The full-image caches are reused; no redundant encoder forward pass is needed. Feature selection and class heads are fit only to the original nonzero coarse labels. Unknown 90-pixel tiles are filled from mean plain-feature class evidence over detected tissue, or all pixels if the tile contains no detected tissue. Known coarse labels are preserved. All representations and controls receive exactly the same completed tiled input; pseudo labels never retrain their classifiers.','',
      'All selected transformations operate over the whole field. Method 15 now computes its candidate nuclei/lumen/context maps on full RGB-detected tissue, instead of restricting those maps to annotated coarse support. Its class head is still trained only from original labelled tile summaries. Predictions can use only the tissue classes represented in their original coarse training labels; newly labelled unannotated areas are extrapolations without reference validation.','',
      'Bora uses the unchanged internal-wand preset with downsample 2. It runs on the completed full-field labels, including temporary labels over background. Those labels are removed only afterward with the independent native RGB tissue mask. The working output is separately clipped to its working support. The final native mask is projected from the unmasked Bora prediction before native clipping, so fine support pixels are never lost because a resampled working support was zero.','',
      '## RGB tissue/background estimate','',
      'The fixed detector uses mean optical density > 0.025 with RGB saturation > 0.04, or mean optical density > 0.30 for dark neutral pixels. It applies elliptical closing of radius 2 µm, removes connected components smaller than 4 µm² and fills enclosed holes. Enclosed gland lumina and adipose spaces are retained as part of tissue architecture; exterior glass remains excluded. The rule is an algorithmic tissue estimate, not an expert annotation. Each support mask is exposed for visual inspection.','',
      'An initial OD 0.12 neutral threshold misclassified grey scanner glass in two CAMELYON RGB fields. Before any new fine-reference scoring, RGB-only review of all 49 images led to OD 0.30. Preliminary partial predictions are archived under virchow2-full-tissue-support-pilot-2026-10-16. The selected support rule was fixed before the final predictions and their scores. No score-driven support threshold search is performed.','',
      '## Scale, annotation and interpretation','',
      'All inputs are ROIs, not whole-WSI inference. Native RGB, crop coordinates, reference geometry and per-axis pixel spacing are inherited unchanged. Processing uses approximately 0.5 µm/px, 224-pixel encoder windows (about 112 µm), 14-pixel tokens (about 7 µm) and 90-pixel coarse tiles (about 45 µm). Native predictions are nearest-neighbour projections followed by native support clipping. Optical resolving power and objective magnification remain unknown.','',
      'Coarse supervision is reference-derived, and the methods and these cases were previously inspected. This remains exploratory controlled boundary refinement, not independent held-out classification or clinical validation. TIGER and BCSS avoid known TCGA reuse as documented in the original scope. CAMELYON tumor is expert-derived, but its non-tumor tissue reference is automatically detected, not expert stroma. Such automatic non-tumor labels may disagree with the new RGB background estimate; the reference is retained unchanged and those disagreements are visible in coverage counts.','',
      '## Files and views','',
      '- all_outputs.html: all 49 fields; original/reference/original tiled labels/completed tiled input/coverage; every retained converted RGB above its final Bora overlay and native score. H&E and plain-feature controls are also linked.',
      '- gallery.html: synchronized original, reference, completed tiled input, converted RGB and final Bora panels, shared zoom and opacity, native masks, tissue/quality supports and per-class scores.',
      '- all_inputs.jpg / all_masks.jpg: complete thumbnail sheets; cases/<case>/contact.jpg contains every converted input and final mask for that field.',
      '- tissue.native.tif / tissue.working.tif: RGB-derived analysis support; evaluation.native.tif is the original annotation-valid support, created only after freezing predictions.',
      '- coarse.completed.tif: exact working input to Bora, with original labels preserved and unknown tiles filled; coarse.full.native.tif is its native projection; coarse.native.tif is its tissue-clipped comparison.',
      '- <method>/representation.png: actual working-grid conversion supplied to Bora; bora.pre_background.working.tif: output before the final background-removal operation; bora.native.tif: delivered native final class mask.',
      '- results.json / scores.csv: every native score and coverage count; ALL_PREDICTIONS_FROZEN.json and FROZEN.json: immutable prediction hashes. Original reference/images remain in the preceding multi-image folder; keep it alongside this result folder.','',
      'Generated images, data, caches and model weights stay outside Git. Code, protocol and this manuscript-oriented report are saved in the repository.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args();results=json.loads((a.root/'results.json').read_text())
    for row in results['cases']:render(a.root,a.source,row)
    report(a.root,results)
    for template_name,output_name in [('gallery.html','detail.html'),('compare.html','gallery.html')]:
        template=Path(__file__).with_name(template_name).read_text();(a.root/output_name).write_text(template.replace('__DATA__',json.dumps(results,allow_nan=False)))
    blocks=[];source='../'+a.source.name
    inputs=Image.new('RGB',(8*180,len(results['cases'])*220+40),'white');masks=Image.new('RGB',(10*180,len(results['cases'])*220+40),'white')
    for i,row in enumerate(results['cases']):
        case=row['case'];base='cases/'+case;src=source+'/cases/'+case;thumbs=a.root/base/'thumbs';y=40+i*220
        context=[('Original H&E',f'{base}/thumbs/original.jpg'),('Original reference',f'{base}/thumbs/expert.jpg'),('Original labelled tiles',f'{src}/thumbs/coarse.jpg'),('Completed Bora tile input',f'{base}/thumbs/coarse.jpg'),('Blue: scored; gold: unscored tissue',f'{base}/thumbs/coverage.jpg')]
        cells=[]
        for j,key in enumerate(PRIMARY):
            score=row['methods'][key]['after']['macro_iou'];cells.append(f'<td><b>{key}: {html.escape(TITLES[key])}</b><a href="{base}/{key}/representation.png"><img loading="lazy" src="{base}/thumbs/{key}.input.jpg" alt="{case} {key} RGB conversion"></a><small>Converted RGB supplied to Bora</small><a href="gallery.html#{case}/{key}"><img loading="lazy" src="{base}/thumbs/{key}.overlay.jpg" alt="{case} {key} final Bora mask"></a><b>IoU {score:.4f}</b> · <a href="{base}/{key}/bora.native.tif">Native mask TIFF</a></td>')
            tile(inputs,Image.open(thumbs/f'{key}.input.jpg'),(j+1)*180,y,key+': converted RGB',size=180)
            tile(masks,Image.open(thumbs/f'{key}.overlay.jpg'),(j+3)*180,y,key+': Bora mask',f'IoU {score:.4f}',size=180)
        tile(inputs,Image.open(thumbs/'original.jpg'),0,y,case,size=180)
        for j,name in enumerate(['original','expert','coarse']):tile(masks,Image.open(thumbs/f'{name}.jpg'),j*180,y,case if j==0 else name,size=180)
        c=row['coverage'];blocks.append(f'<section id="{case}"><h2>{case} · {html.escape(row["dataset"])} · {html.escape(row["group"])}</h2><p>Detected tissue: {c["detected_tissue_pixels"]:,} pixels; unannotated tissue analysed but unscored: {c["unannotated_tissue_pixels"]:,}; original scored region: {c["annotated_pixels"]:,} pixels. <a href="gallery.html#{case}">Synchronized viewer</a> · <a href="{base}/contact.jpg">All inputs and masks</a> · <a href="{base}/tissue.png">Tissue/background mask</a></p><div class="context">'+''.join(f'<figure><figcaption>{title}</figcaption><img loading="lazy" src="{file}"></figure>' for title,file in context)+'</div><div class="scroll"><table><tr>'+''.join(cells)+'</tr></table></div><details><summary>Matched controls</summary>'+''.join(f'<p>{TITLES[k]}: IoU {row["controls"][k]["macro_iou"]:.4f} · <a href="{base}/{k}/representation.png">RGB input</a> · <a href="{base}/{k}/overlay.jpg">Bora overlay</a> · <a href="{base}/{k}/bora.native.tif">Native mask</a></p>' for k in ('he','features'))+'</details></section>')
    inputs.save(a.root/'all_inputs.jpg',quality=90);masks.save(a.root/'all_masks.jpg',quality=90)
    (a.root/'all_outputs.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Full-tissue Bora: every RGB conversion and mask</title><style>body{font:15px system-ui;background:#edf3f6;color:#213742;margin:16px}section{background:white;border:1px solid #d4dee5;border-radius:8px;padding:15px;margin:20px 0}.context{display:grid;grid-template-columns:repeat(5,minmax(100px,1fr));gap:10px;overflow:auto}figure{margin:0}figcaption{min-height:40px}figure img{width:100%;max-width:256px}.scroll{overflow:auto}table{width:100%;table-layout:fixed}td{border:1px solid #d4dee5;min-width:0;padding:6px;vertical-align:top;font-size:12px;overflow-wrap:anywhere}td img{display:block;width:100%;margin:6px 0}small{display:block;color:#566c79}a{color:#10668b}</style><h1>Every RGB conversion and final full-tissue Bora mask</h1><p>49 ROIs × 7 retained methods/combinations. Each method shows its actual RGB conversion above its final mask over the original H&E. All detected tissue is assigned; background is removed after Bora. Scores use the original annotated pixels only. Algorithm 4 remains rejected.</p><p><a href="gallery.html">Interactive side-by-side comparison</a> · <a href="scores.csv">All scores CSV</a> · <a href="REPORT.md">Protocol and report</a> · <a href="all_inputs.jpg">All conversions sheet</a> · <a href="all_masks.jpg">All masks sheet</a></p>'+''.join(blocks)+'</html>')
    save_json(a.root/'COMPLETE.json',dict(cases=49,methods=list(TITLES),final_masks=49*len(TITLES),primary_conversions=49*len(PRIMARY),all_outputs_rendered=True))

if __name__=='__main__':main()
