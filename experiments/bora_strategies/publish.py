"""Publish every strategy, conversion and case, including all weak results."""
import argparse,json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import tifffile
from experiments.virchow2_inverse.common import digest,save_json
from experiments.virchow2_inverse.finalize import font
from experiments.virchow2_multi.publish import rgba,overlay,preview,tile
from experiments.virchow2_smoothing.publish import editable_native,save_palette
from .registry import STRATEGIES


def render(root,source,originals,row):
    case=row['case'];out=root/'cases'/case;src=source/'cases'/case;orig=originals/'cases'/case
    rgb=np.array(Image.open(orig/'original.native.png').convert('RGB'));palette=row['sample']['palette']
    support=tifffile.imread(src/'tissue.native.tif')>0;coarse=tifffile.imread(src/'coarse.completed.tif')
    import cv2
    cv2.setNumThreads(2)
    native_coarse=cv2.resize(coarse,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_NEAREST)
    band=editable_native(coarse,rgb.shape[:2]);layer=np.zeros((*band.shape,4),np.uint8);layer[band&support]=[255,180,0,255]
    Image.fromarray(layer).save(out/'editable.mask.png',compress_level=3)
    row['diagnostic']=dict(current_editable_tissue_percent=float(100*band[support].mean()),current_fixed_tissue_percent=float(100*(~band)[support].mean()))
    expert=Image.open(src/'thumbs/expert.jpg')
    thumb=Image.fromarray(rgb);thumb.thumbnail((320,320),Image.Resampling.LANCZOS);thumb_rgb=np.array(thumb)
    ratio=min(1.,1536/max(rgb.shape[:2]));view_shape=tuple(max(1,round(n*ratio)) for n in rgb.shape[:2])
    row['diagnostic']['display_shape_yx']=list(view_shape)
    row['diagnostic']['display_mpp_xy']=[row['sample']['native_mpp_xy'][0]*rgb.shape[1]/view_shape[1],row['sample']['native_mpp_xy'][1]*rgb.shape[0]/view_shape[0]]
    for conversion,record in row['conversions'].items():
        folder=out/conversion;baseline=tifffile.imread(folder/'B01/bora.native.tif')
        sheet=Image.new('RGB',(5*220,5*265+40),'white');draw=ImageDraw.Draw(sheet)
        draw.text((10,6),case+' | RGB '+conversion+' | B01–B20 | native IoU, delta vs current',fill='#203744',font=font(18))
        conversion_rgb=Image.open(src/conversion/'representation.png')
        for i,(image,title) in enumerate([(rgb,'Original H&E'),(expert,'Original annotation'),(overlay(rgb,native_coarse,palette),'Completed tile mask'),(conversion_rgb,'Actual converted RGB'),(np.array(Image.open(out/'editable.mask.png').convert('RGB')),'B01 editable band')]):
            tile(sheet,image,i*220,40,title,size=220)
        for i,(key,(title,_)) in enumerate(STRATEGIES.items()):
            target=folder/key;mask=tifffile.imread(target/'bora.native.tif')
            small=cv2.resize(mask,(thumb.width,thumb.height),interpolation=cv2.INTER_NEAREST);display=overlay(thumb_rgb,small,palette)
            save_palette(mask,palette,target/'mask.png');preview(display,target/'overlay.jpg',size=320)
            different=mask!=baseline
            save_palette(different,[[245,30,180]],target/'difference.mask.png',binary=True)
            view_mask=cv2.resize(mask,tuple(view_shape[::-1]),interpolation=cv2.INTER_NEAREST)
            view_diff=cv2.resize(different.astype(np.uint8),tuple(view_shape[::-1]),interpolation=cv2.INTER_NEAREST)>0
            save_palette(view_mask,palette,target/'view.mask.png')
            save_palette(view_diff,[[245,30,180]],target/'view.difference.mask.png',binary=True)
            outside=support&~band;values=record['strategies'][key]
            values['changed_outside_current_band_percent']=float(100*np.mean(mask[outside]!=native_coarse[outside])) if outside.any() else 0.
            score=values['score'];tile(sheet,display,(i%5)*220,305+(i//5)*265,key+' '+title[:21],f"IoU {score['macro_iou']:.4f}; delta {values['delta_vs_current']:+.4f}",size=220)
        sheet.save(folder/'contact.jpg',quality=92)
        save_json(folder/'scores.json',record)
    save_json(out/'scores.json',row);print(json.dumps(dict(rendered=case)),flush=True)
    return row


def report(root,results):
    best=results['ranking'][0];lines=['# Twenty Bora inference strategies on the frozen 49-ROI benchmark','',
        f"Highest measured aggregate macro IoU: **{best['strategy']} {best['title']}**, {best['dataset_balanced_macro_iou']:.6f}; change versus current B01 {best['delta_vs_current']:+.6f}.",'',
        'All 49 ROIs from TIGER (16), BCSS (16), CAMELYON16 (7) and BEETLE (10) are evaluated with the same seven retained RGB conversions: 5, 8, 9, 10, 15, 19, and 9 + 19. The 20 strategies produce **6,860 final native masks**. RGB conversion #4 remains rejected. Strategy IDs B01–B20 are separate from RGB conversion IDs. No unfavorable result is discarded.','',
        '## Distinct hypotheses','',
        '| Strategy | Mechanism |','|---|---|']
    for key,(title,description) in STRATEGIES.items():lines.append(f'| {key} {title} | {description} |')
    lines+=['','These are experimental implementations of distinct inference ideas, not twenty proven methods or twenty claims of methodological novelty. Some share an appearance source or graph representation; their inference variables and objectives differ. B08 uses finite-range local mean-field messages, not a fully connected dense-CRF implementation. B11 is discrete vector region competition inspired by Chan–Vese, not an exact reproduction of the continuous level-set solver. B18 has a smooth bounded displacement and does not guarantee diffeomorphic invertibility.','',
        '## Fixed inputs, supervision and geometry','',
        'Exact SHA-verified previous converted RGBs and original coarse masks are reused. Supervised fits use original coarse-interior labels only. Completed coarse labels initialize geometry or supply a coarse area prior for specified strategies; they are not treated as new supervised training annotations. B11 updates appearance from its evolving region assignments as part of its unsupervised energy optimization. Fine-reference pixels are absent from prediction code. Each scoring pass follows a complete prediction freeze and hash verification.','',
        'B01 copies each previous current native mask pixel for pixel. Its analysis TIFF is a resampled copy for the shared B19 consensus grid; its scored native mask is the exact prior output. New solvers resample each axis to approximately **1 µm/px** using recorded image sampling, then project categorical predictions to the exact native reference coordinates by nearest neighbor. Actual analysis shape and per-axis spacing are saved per conversion. Region target spacing is 15 µm, capped at 4,096 regions. B17 retrieves descriptors at approximately 4 µm spacing; B18 optimizes flow internally on a fourfold coarser grid before interpolating the displacement. All morphology and contextual distances use physical spacing where specified. These are ROIs rather than full-WSI inference; objective magnification and optical resolving power remain unknown.','',
        'Sparse original tile-center anchors, rather than complete fixed class interiors, guide most new solvers. Region-based methods may fix whole regions that contain anchors; elastic deformation retains its shared geometry instead of forcibly resetting anchor pixels. This enables whole-region revisions. The gallery reports what fraction of tissue outside B01’s editable band is relabelled.','',
        'Every detected tissue pixel receives a positive label, including unannotated tissue. The independent previous native RGB tissue mask removes background only after segmentation. The original annotation-valid scoring support is unchanged and never intersected with detected tissue. Therefore unannotated tissue is displayed but unscored, while annotated tissue removed by the detector remains penalized. All original reference classes remain in macro scoring, including those missing from coarse supervision. CAMELYON non-tumor reference is automatic tissue support, not expert stroma.','',
        '## Ranking','',
        'Primary score: mean native macro IoU over the seven conversions within each case, then cases within each dataset, then equal weight over four datasets. Per-conversion rankings are separate. A 3,000-resample paired case bootstrap within dataset keeps all seven correlated conversions together. The intervals are descriptive on reused cases. Boundary F1 at 8 µm is secondary; scores at 4 and 16 µm and per-class IoU/Dice are also retained.','',
        '| Rank | Strategy | Native macro IoU | Change vs B01 | Paired descriptive 95% interval | Boundary F1 at 8 µm |','|---|---|---:|---:|---|---:|']
    for r in results['ranking']:
        lo,hi=r['paired_delta_bootstrap95'];bf='—' if r['boundary_f1_8um'] is None else f"{r['boundary_f1_8um']:.6f}"
        lines.append(f"| {r['rank']} | {r['strategy']} {r['title']} | {r['dataset_balanced_macro_iou']:.6f} | {r['delta_vs_current']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {bf} |")
    lines+=['','## Per-conversion comparison','', '| RGB input | Best measured strategy | IoU | Change vs current B01 |','|---|---|---:|---:|']
    for conversion,ranking in results['per_conversion'].items():
        r=ranking[0];lines.append(f"| {conversion} | {r['strategy']} {r['title']} | {r['dataset_balanced_macro_iou']:.6f} | {r['delta_vs_current']:+.6f} |")
    current=next(r for r in results['ranking'] if r['strategy']=='B01');geodesic=next(r for r in results['ranking'] if r['strategy']=='B09')
    lines+=['',f"The prespecified expert-tissue subset excludes CAMELYON's automatically generated non-tumor reference. On the three remaining datasets, B09 mean IoU is {geodesic['expert_tissue_dataset_balanced_iou']:.6f}, versus B01 {current['expert_tissue_dataset_balanced_iou']:.6f}. B09's means are slightly higher on BCSS, TIGER and BEETLE, but lower on CAMELYON. These are small, descriptive differences on reused cases; the primary four-dataset interval for the B09–B01 IoU difference includes zero. B09 has a higher observed boundary F1 at 8 µm across all four dataset means."]
    lines+=['','## Interpretation and reproducibility','',
        'The same cases and coarse labels were previously inspected, and coarse labels are reference-derived. This is exploratory controlled refinement, not independent held-out classification or clinical validation. Coarse interiors are therefore usually already correct; this design does not independently test recovery from realistic whole-region classification errors. Patient grouping is incompletely verified, so case-bootstrap intervals are descriptive. The best measured strategy is selected after examining these cases; it needs a locked independent test cohort before a generalization claim. The score does not establish that unannotated predictions are correct. Whole-field alternatives may mislabel large regions or lose classes; all such outputs remain visible and receive their full macro-score penalties. B14 topology constraints apply before the common background clipping, which can itself change topology.','',
        'The original smoothing ablation is preserved separately under virchow2-smoothing-2026-10-17. Removing median9 altered only about 0.23% of tissue for RGB #9 on tiger_10. Across seven conversions and four equally weighted datasets, current median9 macro IoU was 0.755587 versus 0.755049 without the label filter, while boundary F1 at 8 µm was 0.492152 versus 0.497364. Thus the ablation showed a small overlap/boundary tradeoff rather than universal harm from smoothing. The expanded study changes inference mechanisms rather than merely retuning that filter.','',
        'All parameters are fixed in PROTOCOL.json. A geometry review found that B17 originally expanded [::step] samples using generic resize centers, which introduced an approximately half-sample offset. A synthetic linear-coordinate identity test exposed the issue. B17 was regenerated for all 343 pairs with interpolation at the actual sampled pixel coordinates. The initial predictions, scores and code are archived under bora-strategies-initial-grid-2026-10-17. The other 19 outputs were retained byte for byte, with unchanged function ASTs verified. Their scores were reused only after native-mask hashes, reference-plan hashes, spacing and evaluator source matched. Final B17 scores were recomputed after the corrected complete prediction set was frozen. CORRECTION_LINEAGE.json records this revision; it was a coordinate correction, not a score-driven parameter search.','',
        'The CNN is a small per-image coarse-only fit with fixed seed, 80 class-balanced patches, original H&E and converted RGB inputs, dilations 1/4/8 and no fine-mask training. Its stage losses and parameter count are saved. B19 combines B01/B02/B06/B10 votes and resolves ties by appearance. Fit code, seeds, input hashes, package versions and frozen output hashes are retained; no pretrained model or model download is introduced.','',
        'Stage times in details.json exclude shared preparation and may reuse cached region graphs. B01 time measures copying an existing baseline, not its original inference. These times are not standalone runtime rankings. CPU processing uses three low-priority workers with two computational threads each on Chiamaka. Initial unscored timing runs are preserved separately.','',
        '## Files and views','',
        '- gallery.html: same original/reference/tile/converted input above all 20 masks, synchronized crop, zoom and opacity, exact magenta differences from B01, native TIFFs, scores and per-class details.',
        '- all_outputs.html: all 49 cases × seven conversions, every 20-strategy comparison sheet. Click into the native-mask viewer for detail.',
        '- cases/<case>/<conversion>/<strategy>/: native class TIFF, analysis-grid TIFF, lossless indexed class PNG, exact difference map, display previews and diagnostics. Display previews are nearest-neighbor resamples capped at 1,536 pixels; native PNG/TIFF files preserve full resolution.',
        '- results.json / scores.csv: all 6,860 native scores and aggregate/per-conversion rankings.',
        '- PROTOCOL.json / FROZEN.json / ALL_PREDICTIONS_FROZEN.json / AUDIT.json: parameters, provenance, inputs, prediction hashes and invariant checks.',
        '- Keep preceding full-tissue and multi-image folders alongside these outputs for the original RGB, annotation masks and actual RGB conversions.','',
        '## Methodological sources','',
        '- [scikit-image segmentation API](https://scikit-image.org/docs/0.25.x/api/skimage.segmentation.html): SLIC and seeded watershed.',
        '- [Boykov, Veksler and Zabih: graph-cut expansion moves](https://www.cs.cornell.edu/rdz/Papers/BVZ-pami01-final.pdf).',
        '- [Shi and Malik: normalized graph partitioning](https://people.eecs.berkeley.edu/~malik/papers/SM-ncut.pdf).',
        '- [Chan, Sandberg and Vese: vector-valued region active contours](https://www.math.ucla.edu/~lvese/PAPERS/JVCIR2000.pdf).',
        '- [Krähenbühl and Koltun: Gaussian-potential CRF inference](https://proceedings.neurips.cc/paper/4296-efficient-inference-in-fully-connected-crfs-with-gaussian-edge-potentials.pdf); B08 is explicitly a simpler finite-range approximation.']
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
        blocks.append(f'<section><h2>{row["case"]} · {row["dataset"]}</h2>')
        for conversion in results['protocol']['conversions']:
            base=f'cases/{row["case"]}/{conversion}'
            blocks.append(f'<h3>Converted RGB {conversion} · <a href="gallery.html#{row["case"]}/{conversion}">Native masks, zoom and scores</a> · <a href="{base}/contact.jpg">Full comparison sheet</a></h3><a href="gallery.html#{row["case"]}/{conversion}"><img loading="lazy" src="{base}/contact.jpg"></a>')
        blocks.append('</section>')
    (a.root/'all_outputs.html').write_text('<!doctype html><meta charset="utf-8"><title>All 6,860 Bora strategy results</title><style>body{font:15px system-ui;background:#edf3f6;color:#203744;margin:18px}section{background:white;padding:15px;margin:25px 0}img{max-width:1100px;width:100%}a{color:#09688e}</style><h1>All 49 benchmark images · seven conversions · 20 Bora strategies</h1><p><a href="gallery.html">Detailed viewer</a> · <a href="scores.csv">All scores</a> · <a href="REPORT.md">Report</a></p>'+''.join(blocks))
    files={str(path.relative_to(a.root)):digest(path) for path in a.root.rglob('*') if path.is_file() and 'runtime_sources' not in path.parts and path.suffix in ('.png','.jpg','.html')}
    save_json(a.root/'VISUAL_MANIFEST.json',dict(files_sha256=files))


if __name__=='__main__':main()
