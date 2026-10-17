"""Four-panel anatomical comparison, with every case and score retained."""
import argparse,html,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import tifffile
from experiments.virchow2_inverse.common import save_json
from experiments.virchow2_inverse.finalize import font
from experiments.virchow2_multi.publish import rgba,overlay,preview,tile


def render(root,source,row):
    folder=root/'cases'/row['case'];src=source/'cases'/row['case'];palette=row['sample']['palette'];combo_label=row['methods']['ensemble']['title'].removeprefix('Combined ')
    rgb=np.array(Image.open(src/'original.native.png').convert('RGB'));mask=tifffile.imread(folder/'bora.native.tif')
    Image.fromarray(rgba(mask,palette)).save(folder/'mask.png',compress_level=3)
    preview(overlay(rgb,mask,palette),folder/'overlay.jpg',size=1600)
    preview(overlay(rgb,mask,palette),folder/'thumb.jpg',size=320)
    preview(rgb,folder/'original.jpg',size=320)
    for name,file in [('reference','expert.mask.png'),('coarse','coarse.mask.png')]:
        label=np.array(Image.open(src/file).convert('RGBA'));mixed=rgb.astype(np.float32);valid=label[...,3]>0
        mixed[valid]=.65*mixed[valid]+.35*label[valid,:3];preview(np.rint(mixed).astype(np.uint8),folder/f'{name}.jpg',size=320)
    preview(np.array(Image.open(src/'09/overlay.jpg')),folder/'09.jpg',size=320)
    panels=[('Original H&E','Native tissue ROI',folder/'original.jpg'),('Original reference mask','Annotation over H&E',folder/'reference.jpg'),
        ('Input tiled mask',f"IoU {row['controls']['coarse']['macro_iou']:.4f}",folder/'coarse.jpg'),
        ('Bora with algorithm 9',f"IoU {row['methods']['09']['after']['macro_iou']:.4f}",folder/'09.jpg'),
        ('Bora with '+combo_label,f"IoU {row['methods']['ensemble']['after']['macro_iou']:.4f}",folder/'thumb.jpg')]
    sheet=Image.new('RGB',(5*480,560),'white');ImageDraw.Draw(sheet).text((12,8),row['case']+' | identical H&E, colours and 35% opacity',fill='#1d3440',font=font(20))
    # Load larger source images for exported sheets rather than enlarging thumbnails.
    for i,(title,subtitle,path) in enumerate(panels):
        picture=Image.open(path)
        if i==0:picture=Image.fromarray(rgb)
        elif i in (1,2):
            labels=np.array(Image.open(src/('expert.mask.png' if i==1 else 'coarse.mask.png')));a=rgb.astype(np.float32);valid=labels[...,3]>0;a[valid]=.65*a[valid]+.35*labels[valid,:3];picture=Image.fromarray(np.rint(a).astype(np.uint8))
        elif i==3:picture=Image.open(src/'09/overlay.jpg')
        else:picture=Image.open(folder/'overlay.jpg')
        tile(sheet,picture,i*480,40,title,subtitle,size=480)
    sheet.save(folder/'comparison.jpg',quality=93)
    four=Image.new('RGB',(1600,1730),'white');draw=ImageDraw.Draw(four)
    draw.text((12,8),row['case']+' | same original H&E; masks at 35% opacity',fill='#203744',font=font(24))
    for i,index in enumerate([0,1,2,4]):
        title,subtitle,_=panels[index];pic=sheet.crop((index*480,79,(index+1)*480,559))
        # Render a larger matched field from the original or numerical mask.
        if index==0:pic=Image.fromarray(rgb)
        elif index==4:pic=Image.fromarray(overlay(rgb,mask,palette))
        else:
            labels=np.array(Image.open(src/('expert.mask.png' if index==1 else 'coarse.mask.png')));a=rgb.astype(np.float32);valid=labels[...,3]>0;a[valid]=.65*a[valid]+.35*labels[valid,:3];pic=Image.fromarray(np.rint(a).astype(np.uint8))
        tile(four,pic,(i%2)*800,50+(i//2)*835,f'{i+1}. {title}',subtitle,size=800)
    four.save(folder/'four_panel.jpg',quality=94)


def report(root,results):
    s=results['summary'];v=s['dataset_balanced'];ci=s['paired_delta_bootstrap95'];components=results.get('components',[9,19,4]);label=' + '.join(map(str,components))
    lines=['# Equal evidence combination: algorithms '+label,'',
      'Same 49 tissue ROIs as the preceding study: 16 TIGER, 16 BCSS, 7 CAMELYON16 and 10 BEETLE. No case, crop or coarse label was changed. This is ROI processing, not full-WSI inference.','',
      '| Input / method | Dataset-balanced native macro IoU |','|---|---:|',
      f"| Input tiled mask | {v['coarse']:.6f} |",f"| Original H&E + Bora | {v['he']:.6f} |",f"| Algorithm 9 + Bora | {v['09']:.6f} |",f"| Equal combination {label}, then Bora | {v['ensemble']:.6f} |",'',
      f"Ensemble minus algorithm 9: **{s['dataset_balanced_delta']:+.6f}** macro IoU; paired case bootstrap 95% interval [{ci[0]:+.6f}, {ci[1]:+.6f}]. Wins / losses / ties: **{s['wins']} / {s['losses']} / {s['ties']}**. These are descriptive intervals for reused cases, not independent confirmation or clinical efficacy.",'',
      '| Dataset | ROIs | Algorithm 9 | Combined | Difference | Wins / losses / ties |','|---|---:|---:|---:|---:|---|']
    for d,x in s['per_dataset'].items():lines.append(f"| {d} | {x['cases']} | {x['09']:.6f} | {x['ensemble']:.6f} | {x['delta']:+.6f} | {x['wins']} / {x['losses']} / {x['ties']} |")
    lines+=['','## Fixed method and controls','',
      f'Class probabilities from algorithms {label} are aligned by actual class ID and averaged with fixed weights 1/{len(components)} each. The mean is converted to the same class-palette RGB, then processed with the unchanged Bora internal-wand preset (downsample 2). It is a fusion of class evidence, not an H&E photographic reconstruction. Probabilities are averaged before RGB quantization; no mask voting or reference-driven weight search is used.','',
      'All selected component RGB images and pre-Bora masks were reconstructed from the cached features and verified pixel-identical to the preceding experiment for every case. No encoder inference was repeated. All ensemble predictions were frozen and hashed before the scorer opened fine references. Existing algorithm 9 masks were preserved and every baseline score was reproduced exactly.','',
      'The primary score averages native-reference macro IoU within each dataset and gives each of four datasets equal weight. The interval uses 3,000 paired resamples within dataset. Code 0 is ignored according to the fine reference; classes missing from coarse inputs remain in the score. Boundary F1 uses native physical coordinates. Complete per-class IoU/Dice and boundary metrics are in results.json and per-case scores.json.','',
      'Working sampling is approximately 0.5 µm/px with recorded axis-specific values; native sampling and physical field size are inherited unchanged in each sample record. Coarse tiles are 90 working pixels (about 45 µm), Virchow2 windows 224 pixels (about 112 µm), spatial tokens 14 pixels (about 7 µm). Native masks are nearest-neighbour projections of working predictions. Optical resolving power and objective magnification are unknown, not inferred from pixel spacing.','',
      'Coarse labels are majority labels derived from the reference for a controlled boundary-refinement test. The methods were selected on these same cases, so this is exploratory reuse, not a held-out comparison. TIGER/BCSS/BEETLE provide their released tissue annotations; CAMELYON tumor is expert-derived, whereas its non-tumor tissue reference is automatically detected and must not be called expert stroma. Some fields have only partial annotations. Dataset-specific results are retained.','',
      '## Reading the images','', ('Algorithm 4 was rejected by the user after visual inspection and is excluded from this combination and its algorithm selector. Its earlier outputs and the completed three-way trial remain archived.' if 4 not in components else 'Historical three-way trial; the user subsequently rejected algorithm 4.'),'',
      'gallery.html presents exactly four matched panels: original H&E; original annotated reference over H&E; actual input tiled mask over H&E; Bora refinement over H&E. Switch the algorithm to compare any of the retained preceding methods or the new combination. Mask opacity, zoom and crop centre are shared. The input tiled mask is the coarse supervision supplied to Bora; it is not the algorithm probability argmax. The reference is used for scoring and display, not prediction. Transparent areas mean ignore/unlabelled, not a tissue class.','',
      'all_outputs.html shows every ROI with original H&E, reference, tiled input, algorithm 9 refinement and combined refinement. All native numeric TIFF masks, full-resolution coloured masks, actual RGB supplied to Bora and scores are linked. No best-case filtering is applied. Side-by-side figures use a common 35% overlay opacity.','',
      'The prior result folder must remain alongside this one: gallery links use ../virchow2-multicase-2026-10-16/ for the unchanged originals, reference, tiled masks and algorithm outputs. New predictions are in cases/<case>/; PROTOCOL.json and FROZEN manifests record scope, weights, provenance and checksums. Source images and generated results are excluded from Git.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    results=json.loads((a.root/'results.json').read_text());label=' + '.join(map(str,results.get('components',[9,19,4])))
    for row in results['cases']:
        render(a.root,a.source,row);print(json.dumps(dict(rendered=row['case'])),flush=True)
    report(a.root,results)
    template=Path(__file__).with_name('gallery.html').read_text();(a.root/'gallery.html').write_text(template.replace('__DATA__',json.dumps(results,allow_nan=False)))
    rows=[]
    for row in results['cases']:
        case=row['case'];base=f'cases/{case}';m=row['methods'];c=row['controls']['coarse']['macro_iou'];e=m['ensemble']['after']['macro_iou'];b=m['09']['after']['macro_iou']
        panels=[('Original H&E','original.jpg',''),('Original reference mask','reference.jpg','Annotated reference'),('Input tiled mask','coarse.jpg',f'IoU {c:.4f}'),
            ('Bora: algorithm 9','09.jpg',f'IoU {b:.4f}; Δ tiled {b-c:+.4f}'),('Bora: combined '+label,'thumb.jpg',f'IoU {e:.4f}; Δ tiled {e-c:+.4f}; Δ #9 {e-b:+.4f}')]
        rows.append(f'<section><h2>{case} · {html.escape(row["dataset"])} · {html.escape(row["group"])}</h2><a href="gallery.html#{case}">Open synchronized four-panel viewer</a> · <a href="{base}/comparison.jpg">Large comparison image</a><div class="panels">'+''.join(f'<figure><figcaption><b>{title}</b><br>{score}</figcaption><a href="gallery.html#{case}"><img loading="lazy" src="{base}/{file}"></a></figure>' for title,file,score in panels)+'</div></section>')
    (a.root/'all_outputs.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>All 49 matched mask comparisons</title><style>body{font:15px system-ui;margin:20px;color:#213540;background:#f4f7f9}section{background:white;padding:14px;margin:20px 0;border:1px solid #d4dde2;border-radius:8px}.panels{display:grid;grid-template-columns:repeat(5,minmax(230px,1fr));gap:10px;overflow:auto}figure{margin:0}figcaption{height:68px}img{width:100%;height:auto}a{color:#146584}</style><h1>All 49 images: original → reference → tiled input → Bora</h1><p>Every mask is shown over the same H&E at 35% opacity. Score = native macro IoU against the original reference; higher is better. <a href="gallery.html">Four-panel viewer, all algorithms and scores</a> · <a href="scores.csv">All scores CSV</a> · <a href="REPORT.md">Report</a></p>'+''.join(rows)+'</html>')
    save_json(a.root/'COMPLETE.json',dict(cases=len(results['cases']),ensemble_native_masks=len(list((a.root/'cases').glob('*/bora.native.tif'))),all_cases_rendered=True,primary_four_panel_view=True))

if __name__=='__main__':main()
