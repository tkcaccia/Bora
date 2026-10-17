"""Render every output, comparison sheets, rankings and an annotated gallery."""
import argparse,json,shutil
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.virchow2_inverse.common import save_json
from experiments.virchow2_inverse.finalize import font
from .methods import METHODS


def rgba(labels,colours):
    result=np.zeros((*labels.shape,4),np.uint8);valid=labels>0
    result[valid,:3]=np.array(colours,np.uint8)[labels[valid]-1];result[valid,3]=255
    return result


def overlay(image,labels,colours):
    result=image.astype(np.float32);valid=labels>0
    result[valid]=.65*result[valid]+.35*np.array(colours,np.float32)[labels[valid]-1]
    return np.rint(result).astype(np.uint8)


def preview(array,path,size=256):
    im=Image.fromarray(array);im.thumbnail((size,size),Image.Resampling.LANCZOS);im.save(path,quality=88)


def tile(canvas,array,x,y,title,subtitle="",size=180):
    im=Image.fromarray(array) if isinstance(array,np.ndarray) else array.copy()
    im.thumbnail((size,size),Image.Resampling.LANCZOS)
    canvas.paste(im,(x+(size-im.width)//2,y+39+(size-im.height)//2))
    draw=ImageDraw.Draw(canvas);draw.text((x+5,y+3),title,fill="#203040",font=font(14))
    draw.text((x+5,y+21),subtitle,fill="#526271",font=font(11))


def render_case(root,row):
    folder=root/"cases"/row["case"];sample=row["sample"];colours=sample["palette"]
    rgb=np.array(Image.open(folder/"original.native.png").convert("RGB"));truth=tifffile.imread(folder/"reference.native.tif")
    coarse=cv2.resize(tifffile.imread(folder/"coarse.tif"),(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_NEAREST)
    thumbs=folder/"thumbs";thumbs.mkdir(exist_ok=True)
    Image.fromarray(rgba(truth,colours)).save(folder/"expert.mask.png",compress_level=3)
    Image.fromarray(rgba(coarse,colours)).save(folder/"coarse.mask.png",compress_level=3)
    expert=overlay(rgb,truth,colours);coarse_display=overlay(rgb,coarse,colours)
    preview(rgb,thumbs/"original.jpg");preview(expert,thumbs/"expert.jpg");preview(coarse_display,thumbs/"coarse.jpg")
    contact=Image.new("RGB",(8*180,3*220+40),"white");draw=ImageDraw.Draw(contact)
    draw.text((10,5),f"{row['case']} | {row['group']} | native sampling {sample['native_mpp_xy'][0]:.4f} µm/px",fill="#172a35",font=font(18))
    tile(contact,rgb,0,40,"Original H&E","Native ROI, not WSI")
    tile(contact,expert,180,40,"Reference annotation","Mask opacity35%")
    tile(contact,coarse_display,360,40,"Coarse mask",f"IoU {row['controls']['coarse']['macro_iou']:.3f}")
    for name in ("he","features",*[f"{n:02d}" for n in METHODS]):
        target=folder/name;prediction=tifffile.imread(target/"bora.native.tif")
        Image.fromarray(rgba(prediction,colours)).save(target/"mask.png",compress_level=3)
        display=overlay(rgb,prediction,colours);preview(display,target/"overlay.jpg",size=1024);preview(display,thumbs/f"{name}.overlay.jpg")
        input_image=np.array(Image.open(target/"representation.png").convert("RGB"));preview(input_image,thumbs/f"{name}.input.jpg")
        if (target/"before.native.tif").exists():
            raw=tifffile.imread(target/"before.native.tif");Image.fromarray(rgba(raw,colours)).save(target/"before.mask.png",compress_level=3)
            preview(overlay(rgb,raw,colours),thumbs/f"{name}.before.jpg")
        if name=="he":tile(contact,display,540,40,"Bora on H&E",f"IoU {row['controls']['he']['macro_iou']:.3f}")
        if name not in ["he","features"]:
            index=list(METHODS).index(int(name));m=row["methods"][name]
            tile(contact,input_image,index*180,260,f"#{name} input",METHODS[int(name)][:25])
            tile(contact,display,index*180,480,f"#{name} Bora: {m['after']['macro_iou']:.3f}",f"Δ H&E {m['delta_vs_he']:+.3f}; rank {m['rank']}")
        if name=="features":
            tile(contact,input_image,1260,260,"Plain feature RGB","Control")
            tile(contact,display,1260,480,"Plain features + Bora",f"IoU {row['controls']['features']['macro_iou']:.3f}")
    contact.save(folder/"contact.jpg",quality=92)
    print(json.dumps(dict(rendered=row["case"])),flush=True)


def sheets(root,results):
    n=len(results["cases"]);inputs=Image.new("RGB",(8*180,n*220+40),"white");overlays=Image.new("RGB",(10*180,n*220+40),"white")
    ImageDraw.Draw(inputs).text((10,6),f"All {n*7} algorithm inputs | original +4,5,8,9,10,15,19 | thumbnails only",fill="#203040",font=font(18))
    ImageDraw.Draw(overlays).text((10,6),f"All {n*7} Bora overlays | original, reference, H&E Bora, seven methods | opacity35%",fill="#203040",font=font(18))
    for i,row in enumerate(results["cases"]):
        folder=root/"cases"/row["case"]/"thumbs";y=40+i*220
        tile(inputs,Image.open(folder/"original.jpg"),0,y,row["case"],row["group"][:25])
        for col,name in enumerate(["original","expert","he.overlay"]):tile(overlays,Image.open(folder/(name+".jpg")),col*180,y,row["case"] if col==0 else name)
        for j,n in enumerate(METHODS):
            key=f"{n:02d}";m=row["methods"][key]
            tile(inputs,Image.open(folder/f"{key}.input.jpg"),(j+1)*180,y,f"#{key}",f"Bora IoU {m['after']['macro_iou']:.3f}")
            tile(overlays,Image.open(folder/f"{key}.overlay.jpg"),(j+3)*180,y,f"#{key} IoU {m['after']['macro_iou']:.3f}",f"Δ H&E {m['delta_vs_he']:+.3f}")
    inputs.save(root/"all_inputs.jpg",quality=90);overlays.save(root/"all_overlays.jpg",quality=90)


def report(root,results,plan):
    cases=results["cases"];counts=Counter(c["dataset"] for c in cases);ranks=results["ranking"]
    controls={name:float(np.mean([np.mean([c["controls"][name]["macro_iou"] for c in cases if c["dataset"]==d]) for d in counts])) for name in ("coarse","he","features")}
    lines=["# Selected Virchow2 methods across benchmark tissue regions","",
      f"Executed all seven selected algorithms on **{len(cases)} new source-case/slide groups**, yielding **{len(cases)*7} algorithm RGB images and Bora masks**. Every output is available in gallery.html, all_outputs.html, per-case contact.jpg and the two full contact sheets. No original case114 image is included.","",
      "## Ranking","",
      "The primary score is native-reference per-image macro IoU, averaged within each dataset and then equally across datasets. This prevents the larger cohorts from dominating. Before-Bora scores and all per-case/per-class values are retained; no best-case-only selection. Bootstrap intervals describe paired case resampling within dataset, not clinical efficacy.","",
      "| Rank | Method | Dataset-balanced IoU | Δ H&E Bora | Paired bootstrap95% Δ | Wins / losses / ties vs H&E |","|---|---|---:|---:|---|---|"]
    for r in ranks:lines.append(f"| {r['rank']} | #{r['id']:02d} {r['title']} | {r['dataset_balanced_macro_iou']:.4f} | {r['dataset_balanced_delta_vs_he']:+.4f} | [{r['paired_delta_bootstrap95'][0]:+.4f}, {r['paired_delta_bootstrap95'][1]:+.4f}] | {r['wins_vs_he']} / {r['losses_vs_he']} / {r['ties_vs_he']} |")
    lines += ["",f"Matched controls, with the same dataset weighting: coarse **{controls['coarse']:.4f}**, original H&E +Bora **{controls['he']:.4f}**, plain selected-feature RGB +Bora **{controls['features']:.4f}**.","",
      "## Dataset-specific rankings","",
      "| Dataset | Regions/groups | Ordered methods (native macro IoU) |","|---|---:|---|"]
    for d,count in counts.items():
        order=sorted(ranks,key=lambda r:-r["per_dataset"][d]["macro_iou"])
        lines.append(f"| {d} | {count} | "+"; ".join(f"#{r['id']:02d} {r['per_dataset'][d]['macro_iou']:.4f}" for r in order)+" |")
    expert_order=sorted(ranks,key=lambda r:-r["expert_tissue_dataset_balanced_iou"])
    lines += ["", "Excluding CAMELYON's automatically detected non-tumor reference, the equal-weight TIGER/BCSS/BEETLE order is: "+"; ".join(f"#{r['id']:02d} {r['expert_tissue_dataset_balanced_iou']:.4f}" for r in expert_order)+".","",
      "## Exact scope and physical scale","",
      "These are **ROIs**, not full-WSI inference. TIGER and BCSS use released ROI images; CAMELYON16 and BEETLE use native crops from downloaded WSI windows. One conservative case/source-slide group per image is retained. TIGER uses non-TCGA cases, BCSS uses TCGA cases, avoiding their known cross-dataset reuse. Patient identity remains unverified for some source identifiers, especially CAMELYON slides.","",
      "Native RGB and reference masks are preserved in each case directory. Larger fields are cropped to at most about640×640µm; minimum working side896px. Window selection uses annotated-class entropy and coarse-label eligibility, never algorithm scores. This enriches the cohort for boundaries and is not an estimate of clinical tissue prevalence. The input PLAN.json lists every source ID, group, native coordinate, mask hash, exclusion and class mapping.","",
      "Virchow2 inference uses a common approximately0.5µm/px working grid, model224px windows (about112µm),14px spatial tokens (about7µm) and112px window stride. Integer output sizes cause small recorded axis-specific deviations from0.5. Coarse majority tiles are90working px (about45µm). Native spacing is0.5 for TIGER,0.25 for the colour-normalized BCSS mirror, embedded per-axis WSI spacing for CAMELYON, and embedded TIFF X/YResolution with physical units for BEETLE (about0.505µm/px). Do not infer optical resolving power or objective magnification from these values; both are unknown here.","",
      "Bora runs on each algorithm's common-grid RGB output, with the same internal-wand preset and downsample2. Native evaluation masks are obtained by nearest-neighbour projection back to the original cropped reference grid. They preserve native coordinates but do not recover detail lost in resampling. The H&E control undergoes the same calibrated working-grid processing and Bora settings. No reconstructed H&E decoder or GigaTIME markers are used.","",
      "## Methods and before/after interpretation","",
      "Presets are inherited from the original five-class experiment:64 balanced-MRMR coordinates, class-balanced logistic C0.3, NMF12/keep8, max-min RGB geometry350steps, two fixed OD stain transforms, native/2×/4× context, original object scales and four-quadrant ridge completion with4token exclusion. Class handling is generalized to the actual observed coarse classes, retaining numeric IDs and a stable dataset palette. For binary cases, the three-axis geometry initializes an extra null-space axis; the context top-two gate includes both classes. These changes permit the same hypotheses to run across different annotation vocabularies without forcing a five-class task.","",
      "For methods4,8,9,10,15,19, the before-Bora mask is the argmax of that method's class evidence. For method5, its RGB projection has no intrinsic class mask, so a coarse-trained logistic head on the three projection coordinates supplies the explicitly labelled before-Bora control; it never affects the actual RGB/Bora result. Before masks share the same coarse-zero support restriction as Bora. A direct evidence classifier and a local boundary-refinement wand have different constraints, so a stronger before score does not imply the RGB is a photographic reconstruction.","",
      "A regression check against the original five-class example detected a small rounding difference in the initial class head. It was corrected by normalizing the original float64 tile means, matching the earlier implementation. Methods10 and15 and the plain-feature control were regenerated from cached features; the earlier revision, outputs and scores are archived under diagnostics/. Crop selection and method hyperparameters were not changed in response to new-case scores. FIVE_CLASS_REGRESSION.json records the final comparison with all seven original outputs.","",
      "## Annotation and scoring limits","",
      "TIGER supplies manual tissue compartments; BCSS supplies structured crowdsourced tissue labels; BEETLE supplies released expert tissue annotations. CAMELYON tumor is expert-derived, but code1 is automatically detected non-tumor tissue, not expert stroma. Its results are shown separately and an alternative ranking excludes that cohort. Unreconciled SICAP JPEG masks, CoCaHis foreground semantics and uncalibrated RINGS/WSSS sampling are excluded rather than assigned guessed labels or micrometre scales.","",
      "The preparation stage uses fine annotations to select annotated fields and generate majority coarse labels. The prediction module reads only the original RGB, coarse labels and class metadata. All native and working predictions are frozen and hashed for every case before the scorer opens any fine reference. No parameters are selected from the resulting scores. This is controlled boundary reconstruction with reference-derived coarse labels, not independently predicted tissue classification or a prospective patient benchmark. It is a larger follow-up to methods selected on the earlier single ROI; further untouched cases would be needed to confirm a chosen winner.","",
      f"**{sum(bool(c['missing_coarse_classes']) for c in cases)} regions contain reference classes absent from their coarse labels.** Those classes remain in native macro IoU with zero IoU when unpredicted; they are not silently dropped or reassigned. Per-case class coverage is retained in scores.json. Label0 is ignored only where the native reference is0. Boundary F1 is computed per class in native physical coordinates, excluding annotation-ignore interfaces. Missing predicted contours score0 in boundary F1; the supplemental HD95 omits unmatched contours and must not be interpreted alone.","",
      "## Files","",
      "- gallery.html: rankings, every-case contact overview, selectable native expert/coarse/before/after mask overlays with opacity control.",
      "- all_outputs.html: every algorithm image and every Bora overlay, grouped by source region; no hidden failures or only-best subset.",
      "- all_inputs.jpg / all_overlays.jpg: complete49-region sheets (thumbnail overviews).",
      "- scores.csv / ranking.csv / results.json: all native scores, before/after comparisons and rankings.",
      "- cases/<case>/sample.json: source image, class legend, hashes, sampling, ROI/WSI origin and exact crop.",
      "- cases/<case>/original.native.png and reference.native.tif: native reference pair.",
      "- cases/<case>/<method>/representation.png: actual RGB supplied to Bora (working grid).",
      "- cases/<case>/<method>/before.native.tif and bora.native.tif: numeric compact class masks in native coordinates.",
      "- PLAN.json, ALL_PREDICTIONS_FROZEN.json and per-case FROZEN.json: scope and immutable prediction manifests.","",
      "Model: paige-ai/Virchow2, revision3158645804b69e3f3bc4439d4116edddf0840a72, SHA2568d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b. Models, source images, features and generated visual outputs are kept outside Git; code and this report are saved in the repository."]
    text="\n".join(lines)+"\n"
    replacements={"case114":"case 114","bootstrap95%":"bootstrap 95%","about640":"about 640",
        "side896px":"side 896 px","approximately0.5":"approximately 0.5","model224px windows":"224 px model windows",
        "about112µm":"about 112 µm",",14px spatial tokens":", 14 px spatial tokens","about7µm":"about 7 µm",
        "and112px":"and 112 px","from0.5":"from 0.5","are90working px":"are 90 working px","about45µm":"about 45 µm",
        "is0.5":"is 0.5",",0.25":", 0.25","about0.505":"about 0.505","downsample2":"downsample 2",
        "experiment:64":"experiment: 64","C0.3":"C=0.3","NMF12/keep8":"12 NMF components / keep 8",
        "geometry350steps":"geometry with 350 steps","native/2×/4× context":"base, 2× and 4× context at the working sampling",
        "with4token exclusion":"with an exclusion buffer of four tokens","methods4,8,9,10,15,19":"methods 4, 8, 9, 10, 15 and 19",
        "method5":"method 5","Methods10 and15":"Methods 10 and 15","code1":"code 1","Label0":"Label 0",
        "reference is0":"reference is 0","contours score0":"contours score 0","complete49-region":"complete 49-region",
        "revision315864":"revision 315864","SHA2568d6cea":"SHA256 8d6cea"," +Bora":" + Bora"}
    for before,after in replacements.items():text=text.replace(before,after)
    (root/"REPORT.md").write_text(text)
    save_json(root/"control_summary.json",controls)


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);a=p.parse_args()
    results=json.loads((a.root/"results.json").read_text());plan=json.loads((a.root/"PLAN.json").read_text())
    for row in results["cases"]:render_case(a.root,row)
    sheets(a.root,results);report(a.root,results,plan)
    ranks=results["ranking"];fig,ax=plt.subplots(figsize=(10,4.8),layout="constrained")
    y=np.arange(len(ranks));delta=np.array([r["dataset_balanced_delta_vs_he"] for r in ranks]);ci=np.array([r["paired_delta_bootstrap95"] for r in ranks])
    ax.errorbar(delta,y,xerr=np.maximum(np.array([delta-ci[:,0],ci[:,1]-delta]),0),fmt="o",capsize=4,color="#236b85")
    ax.axvline(0,color="#888",ls="--");ax.set_yticks(y,[f"#{r['id']:02d} {r['title']}" for r in ranks]);ax.invert_yaxis()
    ax.set(xlabel="Dataset-balanced macro IoU change versus matched H&E Bora",title=f"Seven methods on {len(results['cases'])} new tissue regions\nPaired within-dataset case bootstrap95% intervals")
    fig.savefig(a.root/"ranking.png",dpi=160);plt.close(fig)
    template=Path(__file__).with_name("gallery.html").read_text()
    (a.root/"gallery.html").write_text(template.replace("__DATA__",json.dumps(results,allow_nan=False)))
    rows=[]
    for row in results["cases"]:
        folder=f"cases/{row['case']}";cells=[f'<td><a href="gallery.html#{row["case"]}/09">{row["case"]}</a><br>{row["group"]}<br><img loading="lazy" src="{folder}/thumbs/original.jpg"><br><a href="{folder}/original.native.png">Native H&E</a></td>',
            f'<td><a href="{folder}/expert.mask.png"><img loading="lazy" src="{folder}/thumbs/expert.jpg"></a><br>Reference</td>']
        for n in METHODS:
            key=f"{n:02d}";m=row["methods"][key]
            cells.append(f'<td><a href="{folder}/{key}/representation.png"><img loading="lazy" src="{folder}/thumbs/{key}.input.jpg"></a><a href="gallery.html#{row["case"]}/{key}"><img loading="lazy" src="{folder}/thumbs/{key}.overlay.jpg"></a><br>IoU {m["after"]["macro_iou"]:.3f}; ΔH&E {m["delta_vs_he"]:+.3f}<br><a href="{folder}/{key}/bora.native.tif">Native mask TIFF</a></td>')
        rows.append('<tr>'+''.join(cells)+'</tr>')
    all_html='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Every Virchow2/Bora output</title><style>body{font:15px system-ui;color:#213540;margin:16px}table{border-collapse:collapse}th{position:sticky;top:0;background:#e7edf1;z-index:2}td,th{border:1px solid #d3dce2;padding:8px;min-width:180px;vertical-align:top}td img{width:180px;height:auto;display:block;margin:4px 0}a{color:#126b91}</style><h1>Every algorithm output and final Bora mask</h1><p>'+str(len(results['cases']))+' regions ×7 algorithms. Each algorithm cell shows its actual input above its final Bora overlay. Click an overlay for adjustable masks, native original image, reference and scores. <a href="gallery.html">Rankings and detailed viewer</a></p><table><thead><tr><th>Original H&E</th><th>Reference</th>'+''.join(f'<th>#{n:02d}<br>{title}</th>' for n,title in METHODS.items())+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></html>'
    (a.root/"all_outputs.html").write_text(all_html)
    save_json(a.root/"COMPLETE.json",dict(cases=len(results["cases"]),methods=list(METHODS),combinations=len(results["cases"])*7,all_outputs_rendered=True,top_method=ranks[0]["id"]))

if __name__=="__main__":main()
