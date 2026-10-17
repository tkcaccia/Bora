"""All native masks, explicit difference maps, and synchronized comparisons."""
import argparse,json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
import tifffile
from experiments.virchow2_inverse.common import digest,save_json
from experiments.virchow2_multi.publish import rgba,overlay,preview,tile
from experiments.virchow2_inverse.finalize import font
from .run import PRIMARY,METHODS,VARIANTS


def save_palette(mask,colours,path,binary=False):
    """Lossless indexed PNG; class 0 is transparent in browser overlays."""
    palette=np.zeros((256,3),np.uint8);palette[1:len(colours)+1]=np.asarray(colours,np.uint8)
    im=Image.fromarray(np.asarray(mask,np.uint8));im.putpalette(palette.ravel().tolist())
    im.save(path,transparency=0,compress_level=1,bits=1 if binary else (4 if mask.max()<16 else 8))


def editable_native(coarse,native_shape):
    h,w=coarse.shape;small=cv2.resize(coarse,(int(np.ceil(w/2)),int(np.ceil(h/2))),interpolation=cv2.INTER_NEAREST)
    edge=np.zeros(small.shape,bool)
    d=(small[:,1:]!=small[:,:-1])&(small[:,1:]>0)&(small[:,:-1]>0);edge[:,1:]|=d;edge[:,:-1]|=d
    d=(small[1:]!=small[:-1])&(small[1:]>0)&(small[:-1]>0);edge[1:]|=d;edge[:-1]|=d
    band=cv2.distanceTransform((~edge).astype(np.uint8),cv2.DIST_L2,3)<=round(45*.28) if edge.any() else np.zeros(small.shape,bool)
    work=cv2.resize(band.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)
    return cv2.resize(work,tuple(native_shape[::-1]),interpolation=cv2.INTER_NEAREST)>0


def render(root,source,originals,row):
    case=row['case'];out=root/'cases'/case;src=source/'cases'/case;orig=originals/'cases'/case
    rgb=np.array(Image.open(orig/'original.native.png').convert('RGB'));palette=row['sample']['palette']
    tissue=tifffile.imread(src/'tissue.native.tif')>0;coarse=tifffile.imread(src/'coarse.completed.tif')
    band=editable_native(coarse,rgb.shape[:2]);band_rgba=np.zeros((*band.shape,4),np.uint8);band_rgba[band& tissue]=[255,180,0,255]
    Image.fromarray(band_rgba).save(out/'editable.mask.png',compress_level=3)
    full=cv2.resize(coarse,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_NEAREST)
    native_coarse=full.copy();native_coarse[~tissue]=0
    thumb=Image.fromarray(rgb);thumb.thumbnail((340,340),Image.Resampling.LANCZOS);thumb_rgb=np.array(thumb)
    row['diagnostic']=dict(editable_tissue_percent=float(100*band[tissue].mean()),fixed_tissue_percent=float(100*(~band)[tissue].mean()),
        original_tile_pixels=int(coarse.size),kernel9_native_um_xy=[x*18 for x in row['sample']['working_mpp_xy']])
    for m in METHODS:
        baseline=tifffile.imread(out/m/'median9/bora.native.tif');rinput=Image.open(src/m/'representation.png')
        sheet=Image.new('RGB',(5*240,2*285+42),'white');draw=ImageDraw.Draw(sheet)
        draw.text((10,6),case+' | '+m+' | fixed tissue '+f"{row['diagnostic']['fixed_tissue_percent']:.1f}%",fill='#203744',font=font(18))
        context=[(rgb,'Original H&E'),(np.array(Image.open(src/'thumbs/expert.jpg')),'Original annotation'),
            (overlay(rgb,full,palette),'Completed tile input'),(np.array(rinput),'Converted RGB'),(np.array(Image.open(out/'editable.mask.png').convert('RGB')),'Editable band')]
        for i,(image,title) in enumerate(context):tile(sheet,image,i*240,42,title,size=240)
        for i,(v,(title,_)) in enumerate(VARIANTS.items()):
            folder=out/m/v;mask=tifffile.imread(folder/'bora.native.tif')
            small=cv2.resize(mask,(thumb.width,thumb.height),interpolation=cv2.INTER_NEAREST);display=overlay(thumb_rgb,small,palette)
            save_palette(mask,palette,folder/'mask.png')
            preview(display,folder/'overlay.jpg',size=340)
            difference=mask!=baseline
            save_palette(difference,[[245,30,180]],folder/'difference.mask.png',binary=True)
            values=row['methods'][m][v];values['changed_tissue_percent_vs_current']=float(100*difference[tissue].mean())
            values['changed_tissue_percent_vs_tiles']=float(100*(mask[tissue]!=native_coarse[tissue]).mean())
            tile(sheet,display,i*240,327,title,f"IoU {values['score']['macro_iou']:.4f}; delta {values['delta_vs_current']:+.4f}",size=240)
        sheet.save(out/m/'contact.jpg',quality=91)
    save_json(out/'scores.json',row)
    print(json.dumps(dict(rendered=case)),flush=True)
    return row


def report(root,results):
    lines=['# Bora categorical smoothing comparison','',
        'The same 49 ROIs, seven retained Virchow2 representations (5, 8, 9, 10, 15, 19, and 9 + 19), plus H&E and plain-feature controls are compared under five postfilter settings. Algorithm 4 remains rejected. All 2,205 final native masks are retained.','',
        '## Controlled change','',
        'Only the label regularizer changes. Each prediction receives the exact SHA-verified previous converted RGB and completed tile mask. Working downsample 2, LAB preblur sigma 1.4, local color-model averaging, distance penalty, editable band, seed-connectivity cleanup and final native RGB background clipping remain fixed. Thus “no label smoothing” does not remove Gaussian averaging used to estimate the appearance model. No model encoding or feature selection is repeated.','',
        '- median9: original numeric median, window 9 × 9 at the Bora working scale; every one of 441 baselines is reproduced pixel for pixel.',
        '- none: skip the categorical postfilter; keep seed-connected component cleanup.',
        '- median3: reduce the existing numeric median window to 3 × 3.',
        '- majority3: class-identity vote in a 3 × 3 window, unique winners only; ties retain the current class. This avoids imposing numeric order on tissue IDs.',
        '- edge: no median; minimize the original wand class cost plus a contrast-weighted four-neighbor Potts penalty. Fixed weight 0.35, edge sigma 12 in weighted LAB, six deterministic red/black coordinate sweeps, strictly decreasing local energy updates. Changes remain in the same editable band and cannot claim classes with infinite unary cost.','',
        'Parameters were declared before scoring this ablation and were not tuned to these references. Native per-axis sampling is preserved in each case. A kernel footprint in micrometres is reported in the viewer. All inputs remain ROIs rather than full WSI inference.','',
        '## Aggregate results','',
        'Primary summary: average native macro IoU across the seven representations within each case, average cases within each dataset, then weight the four datasets equally. Controls are separate. This is a paired ablation; the seven conversions are not treated as independent specimens. Confidence intervals use 3,000 paired case resamples within dataset, retaining conversions together. Boundary F1 at 8 µm is secondary; 4 and 16 µm are also saved.','',
        '| Setting | Native macro IoU | Change from current | Paired descriptive 95% interval | Boundary F1 at 8 µm |',
        '|---|---:|---:|---|---:|']
    for r in results['ranking']:
        lo,hi=r['paired_delta_bootstrap95'];lines.append(f"| {r['title']} | {r['dataset_balanced_macro_iou']:.6f} | {r['delta_vs_current']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {r['dataset_balanced_boundary_f1_8um']:.6f} |")
    lines+=['','## Per representation','', '| Input | Best measured setting | IoU | Change from current |','|---|---|---:|---:|']
    for m,ranking in results['per_method'].items():
        r=ranking[0];lines.append(f"| {m} | {r['title']} | {r['dataset_balanced_macro_iou']:.6f} | {r['delta_vs_current']:+.6f} |")
    row=next(r for r in results['cases'] if r['case']=='tiger_10');diag=row['diagnostic']
    lines+=['','## Why different RGB inputs can produce similar masks','',
        f"For tiger_10, {diag['fixed_tissue_percent']:.2f}% of detected tissue is outside the editable band. Classifier-derived RGB colors can differ strongly while their class boundaries agree. The viewer reports the fraction changed from the completed tiles and from the current median9 result, and exposes magenta pixel-difference maps. This ablation isolates smoothing; it does not relax the band or spatial prior.",'',
        '## Quality and limitations','',
        'All RGB-detected tissue is assigned; final background is zero. Scoring uses the original annotation-valid pixels only, unchanged from the prior full-tissue experiment. Unannotated tissue is displayed but excluded. Annotated pixels removed by the RGB background detector remain penalized. All original reference classes are scored even when absent from coarse supervision. CAMELYON non-tumor reference is automatically detected tissue, not expert stroma.','',
        'These images and conversions have already been inspected and coarse labels are reference-derived. Results are exploratory boundary-refinement comparisons, not independent held-out validation. A best measured setting is descriptive, not a claim that the setting generalizes. Existing Bora defaults and previous results are preserved.','',
        '## Deliverables','',
        '- gallery.html: original/reference/tile/conversion and all five final masks at synchronized zoom, with pixel-difference toggle and editable-band toggle. Select any of 49 cases and any retained representation/control.',
        '- all_outputs.html: all 49 cases × seven primary representations, all five masks, with linked native TIFFs and per-case scores.',
        '- scores.csv / results.json: all 2,205 scores, boundary metrics and per-class details.',
        '- cases/<case>/<method>/<variant>/: native categorical mask, raw working mask, native RGBA visualization, difference map, overlay thumbnail. contact.jpg compares all settings.',
        '- PROTOCOL.json / FROZEN.json / ALL_PREDICTIONS_FROZEN.json / AUDIT.json: exact parameters, input/output hashes and integrity checks.',
        '- Keep the previous full-tissue and multi-image result folders alongside this folder for original inputs and expert masks.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    results=json.loads((a.root/'results.json').read_text());originals=a.source.parent/results['protocol']['original_folder']
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn')) as pool:
        futures=[pool.submit(render,a.root,a.source,originals,row) for row in results['cases']]
        results['cases']=[future.result() for future in futures]
    save_json(a.root/'results.json',results);report(a.root,results)
    template=Path(__file__).with_name('gallery.html').read_text();(a.root/'gallery.html').write_text(template.replace('__DATA__',json.dumps(results,allow_nan=False)))
    blocks=[]
    for row in results['cases']:
        blocks.append(f"<section><h2>{row['case']} — {row['dataset']}</h2>")
        for m in PRIMARY:
            cells=[]
            for v,(title,_) in VARIANTS.items():
                r=row['methods'][m][v];base=f"cases/{row['case']}/{m}/{v}"
                cells.append(f'<td><a href="gallery.html#{row["case"]}/{m}"><img loading="lazy" src="{base}/overlay.jpg"></a><br>{title}<br>IoU {r["score"]["macro_iou"]:.4f} · Δ {r["delta_vs_current"]:+.4f}<br><a href="{base}/bora.native.tif">Native TIFF</a></td>')
            conversion=f'../{a.source.name}/cases/{row["case"]}/{m}/representation.png'
            blocks.append(f'<h3>Input {m} · <a href="{conversion}">actual converted RGB</a> · <a href="cases/{row["case"]}/{m}/contact.jpg">Full comparison sheet</a></h3><table><tr>'+''.join(cells)+'</tr></table>')
        blocks.append('</section>')
    (a.root/'all_outputs.html').write_text('<!doctype html><meta charset="utf-8"><title>All Bora smoothing outputs</title><style>body{font:15px system-ui;background:#edf3f6;color:#203744;margin:18px}section{background:white;padding:15px;margin:20px 0}table{width:100%;table-layout:fixed}td{vertical-align:top;width:20%;padding:5px}img{width:100%}a{color:#09688e}</style><h1>All 49 images · seven conversions · five Bora regularizers</h1><p><a href="gallery.html">Synchronized detailed viewer</a> · <a href="scores.csv">Every score</a> · <a href="REPORT.md">Report</a></p>'+''.join(blocks))
    files={str(p.relative_to(a.root)):digest(p) for p in a.root.rglob('*') if p.is_file() and p.suffix in ('.png','.jpg','.html')}
    save_json(a.root/'VISUAL_MANIFEST.json',dict(files_sha256=files))


if __name__=='__main__':main()
