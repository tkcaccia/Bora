"""Freeze all predictions before opening fine masks; score and render the gallery."""
from __future__ import annotations
import argparse,csv,json,shutil,math,textwrap
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage as ndi
import tifffile
from .common import digest,save_json
from .protocol import Context,COLOURS,border,TITLES,rgb_scale

def score(pred,truth):
    valid=truth>0;per_class={}
    for g in range(1,6):
        a=(pred==g)&valid;b=truth==g;i=int((a&b).sum());union=int((a|b).sum());total=int(a.sum()+b.sum())
        per_class[str(g)]={"iou":i/union if union else 0.,"dice":2*i/total if total else 0.,"truth_pixels":int(b.sum()),"prediction_pixels":int(a.sum())}
    pb=border(pred)&valid;tb=border(truth)
    distance_pred=ndi.distance_transform_edt(~tb,sampling=(.5,.5))[pb]
    distance_truth=ndi.distance_transform_edt(~pb,sampling=(.5,.5))[tb]
    bf={}
    for tolerance in (4,8,16):
        precision=float(np.mean(distance_pred<=tolerance)) if len(distance_pred) else 0
        recall=float(np.mean(distance_truth<=tolerance)) if len(distance_truth) else 0
        bf[str(tolerance)]=2*precision*recall/(precision+recall) if precision+recall else 0
    distances=np.r_[distance_pred,distance_truth]
    return {"macro_iou":float(np.mean([r["iou"] for r in per_class.values()])),"macro_dice":float(np.mean([r["dice"] for r in per_class.values()])),
            "worst_class_iou":min(r["iou"] for r in per_class.values()),"pixel_accuracy":float(np.mean(pred[valid]==truth[valid])),"per_class":per_class,
            "boundary_f1_at_um":bf,"hd95_um":float(np.percentile(distances,95)) if len(distances) else None,
            "assd_um":float((distance_pred.mean()+distance_truth.mean())/2) if len(distance_pred) and len(distance_truth) else None,
            "ignore_code":0,"valid_pixels":int(valid.sum()),"boundary_metric_is_class_agnostic":True}

def font(size):
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf","/System/Library/Fonts/Helvetica.ttc"]:
        if Path(path).exists():return ImageFont.truetype(path,size)
    return ImageFont.load_default()

def rgba_mask(labels):
    out=np.zeros((*labels.shape,4),np.uint8)
    for g in range(1,6):out[labels==g]=[*COLOURS[g-1].astype(np.uint8),255]
    return out

def overlay(image,labels,alpha=.35):
    ans=image.astype(float).copy();support=labels>0
    ans[support]=(1-alpha)*ans[support]+alpha*COLOURS[labels[support]-1]
    return np.rint(ans).astype(np.uint8)

def scale_bar(a):
    image=Image.fromarray(a).convert("RGB");d=ImageDraw.Draw(image)
    y=image.height-35;x=image.width-240;d.rectangle((x-12,y-34,x+212,y+16),fill="white")
    d.line((x,y,x+200,y),fill="black",width=7);d.text((x+40,y-29),"100 µm",fill="black",font=font(23))
    return image

def gpu_outputs(ctx):
    for n in (1,2,3):
        folder=ctx.root/f"{n:02d}";info=json.loads((folder/"GPU_READY.json").read_text())
        image=np.asarray(Image.open(folder/info["image"]).convert("RGB"))
        actual=np.load(folder/"stitched_features_seed_42.npy").astype(np.float32)
        z=(actual-ctx.scaler.mean_)/ctx.scaler.scale_
        logits=z.reshape(-1,len(ctx.selected))@ctx.classifier.coef_.T+ctx.classifier.intercept_
        original=(ctx.z.reshape(-1,len(ctx.selected))@ctx.classifier.coef_.T+ctx.classifier.intercept_)
        order=np.argsort(-original,axis=1)
        details={"output_type":"approximate feature-only inversion" if n==1 else ("masked original H&E" if n==2 else "counterfactual H&E intervention"),
            "original_rgb_supplies_detail":n!=1,"steps_per_window":info["steps_per_window"],"window_count":len(info["windows"]),
            "all_encoder_parameters_frozen":True,"selected_dimensions":ctx.selected.tolist(),
            "stitched_normalized_feature_mse":info["stitched_normalized_feature_mse"],"stitched_class_preservation_fraction":float(np.mean(logits.argmax(1)==order[:,0])),
            "stitched_counterfactual_target_fraction":float(np.mean(logits.argmax(1)==order[:,1])),
            "mean_window_initial_primary":float(np.mean([r["history"][0]["primary"] for r in info["windows"]])),
            "mean_window_final_primary":float(np.mean([r["history"][-1]["primary"] for r in info["windows"]])),
            "implementation_sha256":info.get("implementation_sha256") or digest(Path(__file__).with_name("gpu_first_run.py")),
            "pixel_mae_to_original":float(np.mean(abs(image.astype(float)-ctx.image))) }
        extras={}
        if n==1:
            second=np.asarray(Image.open(folder/"optimized_seed_314.png").convert("RGB"))
            details.update(noise_starts=2,between_start_pixel_mae=float(np.mean(abs(image.astype(float)-second))),
                limitation="32 optimization steps per window; feature fitting is incomplete; generated appearance must not be treated as recovered histology")
        elif n==2:
            mask=np.load(folder/"evidence_mask.npy");details["mean_retained_mask"]=float(mask.mean())
            details["mask_objective"]=info.get("mask_objective","per-token KL + area + TV")
            extras["evidence_heatmap"]=rgb_scale(mask)
        else:
            extras["edit_heatmap"]=rgb_scale(np.load(folder/"edit_magnitude.npy"))
            details["limitation"]="counterfactual target is an intentional change to the feature classifier, not a genuine change of tissue or evidence of biological validity"
        ctx.output(n,image,details,extras=extras)

def contact_sheet(root,rows,kind,filename):
    cell_w,cell_h=400,450;canvas=Image.new("RGB",(cell_w*4,cell_h*5+74),"white");d=ImageDraw.Draw(canvas)
    title="20 distinct inputs to Bora" if kind=="representation" else "20 Bora results on original H&E (35% overlay)"
    d.text((16,10),title,fill="black",font=font(26))
    d.text((16,46),"TIGER tissue ROI • native ≈0.5 µm/px • same field in every panel",fill="black",font=font(17))
    for i,row in enumerate(rows):
        x=(i%4)*cell_w;y=(i//4)*cell_h+74
        name="representation.png" if kind=="representation" else "overlay.png"
        image=Image.open(root/f"{row['id']:02d}"/name).convert("RGB");image.thumbnail((390,375))
        canvas.paste(image,(x+5,y+63))
        label=f"{row['id']:02d}  {row['title']}"
        lines=textwrap.wrap(label,width=35)[:2]
        d.multiline_text((x+7,y+3),"\n".join(lines),fill="black",font=font(17),spacing=2)
        caption=f"IoU {row['scores']['macro_iou']:.3f} • Δ H&E {row['delta_macro_iou_vs_he']:+.3f}"
        d.text((x+7,y+43),caption,fill="black",font=font(14))
        kind_label="noise-like inversion" if row["id"]==1 else row["details"]["output_type"]
        d.text((x+7,y+433),kind_label,fill="#40505c",font=font(12))
    canvas.save(root/filename)

def write_gallery(root,rows,controls,protocol):
    payload=json.dumps({"rows":rows,"controls":controls,"protocol":protocol}).replace("</","<\\/")
    html='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Virchow2: 20 experiments for Bora</title>
<style>body{margin:0;background:#f3f4f5;color:#17212b;font:16px system-ui}main{max-width:1550px;margin:auto;padding:24px}h1{font-size:28px;margin:0 0 8px}p{line-height:1.5}.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;background:white;padding:16px;border-radius:10px;position:sticky;top:0;z-index:2}select,button{font:inherit;padding:9px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin-top:20px}figure{margin:0;background:white;padding:10px;border-radius:8px}figure img,canvas{width:100%;height:auto;display:block}figcaption{font-weight:600;margin:0 0 9px}small{font-weight:400;color:#52616e;display:block;line-height:1.4}.notes{background:white;padding:18px;margin-top:18px;border-radius:8px}details{margin:16px 0}pre{white-space:pre-wrap;max-height:350px;overflow:auto;font-size:12px}.legend{display:flex;gap:14px;flex-wrap:wrap;margin:12px 0}.dot{display:inline-block;width:13px;height:13px;margin-right:5px}table{width:100%;border-collapse:collapse;background:white;font-size:14px}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}a{color:#075da5}.tiles{display:flex;gap:8px;overflow:auto}.tiles img{width:224px;height:224px}#extra img{max-width:700px;width:100%}.native{overflow:auto}.native canvas,.native img{width:auto;max-width:none}@media(max-width:900px){.grid{grid-template-columns:1fr}main{padding:12px}}</style></head><body><main>
<h1>Virchow2: 20 distinct experiments for Bora</h1><p>One expert-annotated TIGER breast tissue <b>ROI, not a full WSI</b>. 1224 × 1166 pixels; provider-reported ≈0.5 µm/pixel (612 × 583 µm). Optical resolving power and objective are unknown. Virchow2 tokens cover 14 × 14 native pixels, about 7 × 7 µm.</p>
<div class="controls"><button id="prev">←</button><select id="method"></select><button id="next">→</button><label>Mask opacity <input id="alpha" type="range" min="0" max="100" value="35"><span id="alphav">35%</span></label><label><input type="checkbox" id="native"> Native pixels</label></div>
<div id="legend" class="legend"></div><h2 id="title"></h2><p id="summary"></p>
<div class="grid" id="panels"><figure><figcaption>Original H&E<small>Unmodified image; scale bar 100 µm</small></figcaption><img src="original.scalebar.png"></figure>
<figure><figcaption>Expert annotation<small>Reference used only after all predictions were frozen</small></figcaption><canvas id="reference"></canvas></figure>
<figure><figcaption>Coarse tiling mask<small>90-pixel tiles (45 µm), derived from reference annotations</small></figcaption><canvas id="coarse"></canvas></figure>
<figure><figcaption>Experimental input<small id="inputtype"></small></figcaption><a id="inputlink"><img id="input"></a></figure>
<figure><figcaption>Bora result<small>Original H&E with this experiment’s mask; adjustable opacity</small></figcaption><canvas id="result"></canvas></figure>
<figure><figcaption>H&E Bora control<small>Same fixed internal-wand settings; coarse exterior held fixed</small></figcaption><canvas id="he"></canvas></figure></div>
<div class="notes"><b>Interpretation</b><p id="interpretation"></p><p>The feature RGB images are class evidence or latent structure, not protein measurements or regenerated H&E. Native geometry supplied by original H&E is identified per experiment. A visually attractive image is not proof of better segmentation.</p><a href="contact_inputs.png">All 20 inputs</a> · <a href="contact_overlays.png">All 20 overlays</a> · <a href="scores.csv">Scores CSV</a> · <a href="REPORT.md">Full report</a></div>
<details><summary>Same five 224-pixel crops (112 µm) from this input</summary><div id="tiles" class="tiles"></div></details><details open><summary>Additional evidence / diagnostic</summary><div id="extra"></div></details><details><summary>Method details and limitations</summary><pre id="details"></pre></details>
<h2>Results for every experiment</h2><p>Macro IoU averages all five classes; Δ is relative to the H&E Bora control. Boundary F1 is class-agnostic and uses an 8 µm tolerance. These are descriptive results from one ROI, not patient-level validation.</p><table><thead><tr><th>Experiment</th><th>Macro IoU</th><th>Δ H&E</th><th>Weakest IoU</th><th>Boundary F1</th><th>HD95 µm</th></tr></thead><tbody id="scores"></tbody></table></main>
<script>const DATA=__DATA__;const $=id=>document.getElementById(id);let index=0;const colours=[[220,45,65],[55,165,75],[55,110,230],[200,100,220],[220,180,55]];
DATA.rows.forEach((r,i)=>{const op=document.createElement('option');op.value=i;op.textContent=String(r.id).padStart(2,'0')+' — '+r.title;$('method').append(op)});
Object.entries(DATA.protocol.class_names).forEach(([g,name])=>{if(+g>0){const s=document.createElement('span');s.innerHTML='<i class="dot" style="background:rgb('+colours[+g-1].join(',')+')"></i>'+name;$('legend').append(s)}});
const cache={};function load(path){if(!cache[path])cache[path]=new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=reject;im.src=path});return cache[path]}
async function draw(id,path){const [im,mask]=await Promise.all([load('original.png'),load(path)]);const c=$(id);c.width=im.width;c.height=im.height;const g=c.getContext('2d');g.drawImage(im,0,0);g.globalAlpha=+$('alpha').value/100;g.drawImage(mask,0,0);g.globalAlpha=1;g.fillStyle='white';g.fillRect(im.width-252,im.height-69,232,54);g.fillStyle='black';g.fillRect(im.width-240,im.height-38,200,6);g.font='23px sans-serif';g.fillText('100 µm',im.width-194,im.height-43)}
function redraw(){const r=DATA.rows[index],f=String(r.id).padStart(2,'0');$('alphav').textContent=$('alpha').value+'%';draw('reference','expert.mask.png');draw('coarse','coarse.mask.png');draw('he','he.mask.png');draw('result',f+'/mask.png')}
function select(i){index=(i+DATA.rows.length)%DATA.rows.length;$('method').value=index;const r=DATA.rows[index],f=String(r.id).padStart(2,'0');$('title').textContent=f+' — '+r.title;$('summary').textContent='Macro IoU '+r.scores.macro_iou.toFixed(3)+' | Δ H&E '+(r.delta_macro_iou_vs_he>=0?'+':'')+r.delta_macro_iou_vs_he.toFixed(3)+' | Weakest class IoU '+r.scores.worst_class_iou.toFixed(3);$('input').src=f+'/representation.scalebar.png';$('inputlink').href=f+'/representation.png';$('inputtype').textContent=r.details.output_type+' · original detail: '+(r.details.original_rgb_supplies_detail?'yes':'no');$('details').textContent=JSON.stringify(r.details,null,2);$('interpretation').textContent=r.interpretation;$('tiles').innerHTML='';for(let n=1;n<=5;n++){const im=new Image();im.src=f+'/tiles/'+String(n).padStart(2,'0')+'.png';$('tiles').append(im)}$('extra').innerHTML='';(r.diagnostic_images||[]).forEach(path=>{const a=document.createElement('a');a.href=f+'/'+path;const im=new Image();im.src=a.href;a.append(im);$('extra').append(a)});history.replaceState(null,'','#'+f);redraw()}
$('method').onchange=e=>select(+e.target.value);$('prev').onclick=()=>select(index-1);$('next').onclick=()=>select(index+1);$('alpha').oninput=redraw;$('native').onchange=e=>$('panels').classList.toggle('native',e.target.checked);
DATA.rows.forEach((r,i)=>{const tr=document.createElement('tr');tr.innerHTML='<td><a href="#'+String(r.id).padStart(2,'0')+'">'+String(r.id).padStart(2,'0')+' '+r.title+'</a></td><td>'+r.scores.macro_iou.toFixed(3)+'</td><td>'+r.delta_macro_iou_vs_he.toFixed(3)+'</td><td>'+r.scores.worst_class_iou.toFixed(3)+'</td><td>'+r.scores.boundary_f1_at_um['8'].toFixed(3)+'</td><td>'+r.scores.hd95_um.toFixed(1)+'</td>';tr.querySelector('a').onclick=e=>{e.preventDefault();select(i);window.scrollTo({top:0,behavior:'smooth'})};$('scores').append(tr)});select(Math.max(0,DATA.rows.findIndex(r=>r.id===+(location.hash.slice(1)||'1'))));</script></body></html>'''
    html=html.replace('<div id="legend"', '<small>Mask colours; feature RGB images may use different colours.</small><div id="legend"')
    html=html.replace('<p>One expert-annotated TIGER breast tissue','<details><summary>TIGER tissue ROI · ≈0.5 µm/pixel · microscopy details</summary><p>One expert-annotated TIGER breast tissue',1)
    html=html.replace('about 7 × 7 µm.</p>','about 7 × 7 µm.</p></details>',1)
    html=html.replace('</style>','@media(max-width:600px){.controls{display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:8px;padding:10px}.controls select{width:100%;min-width:0}.controls label{grid-column:span 3}.legend{font-size:13px;gap:8px}h1{font-size:24px}.native figure{overflow:auto}}</style>')
    class_table='<h3>Per-class IoU</h3><table><thead><tr><th>Tissue</th><th>Coarse</th><th>H&E Bora</th><th>This idea</th><th>Δ H&E</th></tr></thead><tbody id="classscores"></tbody></table>'
    html=html.replace('<b>Interpretation</b>',class_table+'<p><b>Interpretation</b></p>')
    extra='<details><summary>Which features distinguish EACH tissue group?</summary><p>Descriptive coarse-tile contrasts; latent dimensions are not biomarkers. No pixel-level significance claim. <a href="feature_contrasts.csv">All contrasts</a>.</p><a href="features_by_class.png"><img src="features_by_class.png" style="max-width:100%;height:auto"></a></details>'
    html=html.replace('<h2>Results for every experiment</h2>',extra+'<h2>Results for every experiment</h2>')
    js="""$('classscores').innerHTML='';for(let g=1;g<=5;g++){const current=r.scores.per_class[g].iou,control=DATA.controls.he_bora.per_class[g].iou;const tr=document.createElement('tr');const delta=current-control;tr.innerHTML='<td>'+DATA.protocol.class_names[g]+'</td><td>'+DATA.controls.coarse.per_class[g].iou.toFixed(3)+'</td><td>'+control.toFixed(3)+'</td><td>'+current.toFixed(3)+'</td><td style="color:'+(delta<0?'#aa2222':'#16733c')+'">'+(delta>=0?'+':'')+delta.toFixed(3)+'</td>';$('classscores').append(tr)};"""
    html=html.replace("$('details').textContent=JSON.stringify(r.details,null,2);","$('details').textContent=JSON.stringify(r.details,null,2);"+js)
    (root/"gallery.html").write_text(html.replace("__DATA__",payload))

def main():
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--root",type=Path,required=True);a=p.parse_args()
    c=Context(a.source,a.root)
    from .feature_report import write_feature_report
    write_feature_report(c)
    for n in (1,2,3):
        if not (a.root/f"{n:02d}/GPU_READY.json").exists():raise RuntimeError(f"GPU experiment {n} is incomplete")
    if not (a.root/"01/gpu_seed_314.json").exists():raise RuntimeError("Second inversion start is incomplete")
    gpu_outputs(c)
    rows=[json.loads((a.root/f"{n:02d}/result.json").read_text()) for n in range(1,21)]
    if any(r["status"]!="completed" for r in rows):raise RuntimeError("Fix or explicitly report failed ideas before full evaluation")
    frozen={str(r["id"]):{"input":digest(a.root/f"{r['id']:02d}/representation.png"),"prediction":digest(a.root/f"{r['id']:02d}/bora.tif")} for r in rows}
    save_json(a.root/"PREDICTIONS_FROZEN.json",{"fine_reference_not_yet_opened":True,"outputs":frozen})
    # First access to fine annotation in this pipeline occurs here.
    truth=tifffile.imread(a.source/"expert_reference.compact.tif")
    tifffile.imwrite(a.root/"expert_reference.compact.tif",truth,compression="deflate")
    protocol=json.loads((a.root/"protocol.json").read_text())
    controls={}
    for name,path in [("coarse","coarse.tif"),("he_bora","control_he_bora.tif"),("feature_bora","control_feature_bora.tif"),("he_barrier","control_he_barrier.tif")]:
        controls[name]=score(tifffile.imread(a.root/path),truth)
    save_json(a.root/"control_scores.json",controls)
    Image.fromarray(rgba_mask(truth)).save(a.root/"expert.mask.png")
    Image.fromarray(rgba_mask(c.coarse)).save(a.root/"coarse.mask.png")
    Image.fromarray(rgba_mask(c.baseline)).save(a.root/"he.mask.png")
    scale_bar(c.image).save(a.root/"original.scalebar.png")
    scale_bar(overlay(c.image,truth)).save(a.root/"expert.overlay.png")
    scale_bar(overlay(c.image,c.coarse)).save(a.root/"coarse.overlay.png")
    scale_bar(overlay(c.image,c.baseline)).save(a.root/"he.overlay.png")
    crops=[(0,0),(448,448),(56,840),(560,840),(840,980)]
    for row in rows:
        n=row["id"];folder=a.root/f"{n:02d}";pred=tifffile.imread(folder/"bora.tif")
        if digest(folder/"bora.tif")!=frozen[str(n)]["prediction"]:raise AssertionError("Prediction changed after reference access")
        row["scores"]=score(pred,truth);row["delta_macro_iou_vs_he"]=row["scores"]["macro_iou"]-controls["he_bora"]["macro_iou"]
        Image.fromarray(rgba_mask(pred)).save(folder/"mask.png")
        scale_bar(overlay(c.image,pred)).save(folder/"overlay.png")
        representation=np.asarray(Image.open(folder/"representation.png").convert("RGB"))
        scale_bar(representation).save(folder/"representation.scalebar.png")
        tiles=folder/"tiles";tiles.mkdir(exist_ok=True)
        for i,(y,x) in enumerate(crops):Image.fromarray(representation[y:y+224,x:x+224]).save(tiles/f"{i+1:02d}.png")
        row["diagnostic_images"]=[p for p in ["optimized_seed_314.png","evidence_heatmap.png","edit_heatmap.png","trust_mask.png","candidate_nuclei.png","candidate_lumens.png","topology_change.png"] if (folder/p).exists()]
        if n==1:row["interpretation"]="The optimized image remains noise-like at this 32-step budget. Lower feature error did not establish recovery of recognizable tissue; this is an unsuccessful reconstruction attempt."
        elif n==2:row["interpretation"]="A soft evidence mask retains part of the original H&E while attempting to preserve class evidence. It is an optimization diagnostic, not an expert segmentation or proof of a globally minimal mask."
        elif n==3:row["interpretation"]="This intentionally changes the feature classifier's evidence. A successful classifier change is not a biological tissue change and need not improve Bora."
        elif n in (6,13,17):row["interpretation"]="This experiments with Bora's boundary decision rule or interface wrapper; the full method differs from an unchanged RGB input substitution."
        elif row["details"]["original_rgb_supplies_detail"]:row["interpretation"]="This hybrid uses original H&E information. Fine detail visible here was supplied by that image, not recovered solely from selected Virchow2 features."
        else:row["interpretation"]="A synthetic RGB representation of selected-feature evidence. It is not regenerated H&E or a measured molecular image."
        save_json(folder/"result.json",row)
    save_json(a.root/"results.json",{"experiments":rows,"controls":controls,"protocol":protocol,"all_predictions_frozen_before_fine_reference":True,"completed":20})
    fields=["id","title","macro_iou","delta_macro_iou_vs_he","macro_dice","worst_class_iou","boundary_f1_8um","hd95_um",*[f"class_{g}_iou" for g in range(1,6)]]
    with (a.root/"scores.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:
            s=r["scores"];writer.writerow({"id":r["id"],"title":r["title"],**{k:s[k] for k in ["macro_iou","macro_dice","worst_class_iou","hd95_um"]},"delta_macro_iou_vs_he":r["delta_macro_iou_vs_he"],"boundary_f1_8um":s["boundary_f1_at_um"]["8"],**{f"class_{g}_iou":s["per_class"][str(g)]["iou"] for g in range(1,6)}})
    contact_sheet(a.root,rows,"representation","contact_inputs.png");contact_sheet(a.root,rows,"overlay","contact_overlays.png")
    write_gallery(a.root,rows,controls,protocol)
    best=max(rows,key=lambda r:r["scores"]["macro_iou"])
    report=["# Twenty Virchow2 experiments for Bora","","Completed on Chiamaka, 16 October 2026. Twenty distinct interventions were run on the same TIGER ROI; this is not a decoder grid.","",
      f"Best macro IoU: **{best['id']:02d} {best['title']}**, {best['scores']['macro_iou']:.4f}; H&E Bora control {controls['he_bora']['macro_iou']:.4f}; coarse mask {controls['coarse']['macro_iou']:.4f}.","",
      "## Scope and physical resolution","","One breast tissue ROI, 1224 × 1166 pixels, provider-reported approximately 0.5 µm/px, field 612 × 583 µm. Not a full WSI. Optical resolving power, scanner and objective are unknown. Virchow2 windows: 224 pixels/112 µm; spatial token: 14 pixels/7 µm. Coarser-scale windows cover approximately 224 and 448 µm. No resampling improves optical resolution.","",
      "Five scored classes: "+", ".join(protocol["class_names"][str(g)] for g in range(1,6))+". Expert code 0 is ignored. Rest is scored.","",
      "## Fair comparison and limits","","All predictions and their SHA-256 hashes were frozen before this pipeline opened the fine reference. Coarse labels are majority tiles of expert annotations, so this is controlled boundary reconstruction, not an independent tissue-classification benchmark. Model selection uses coarse tile interiors only. No fine-mask optimization or tuning was performed. Single ROI; no patient-level inference or significance claim.","",
      "H&E control and ordinary input substitutions use Bora's internal balanced local wand with a 2× working grid (~1 µm/px), tile width 90 native pixels, fixed settings and fixed coarse exterior. Ideas 06, 13 and 17 explicitly change the interface wrapper/decision rule. Idea 13 also has an H&E-gradient watershed control. This is an exploratory comparison, not a claim that every method runs an identical algorithm.","",
      "Class features were selected from 169 coarse tile interiors: 74 tumor, 37 stroma, 12 healthy glands, 12 inflamed stroma, 34 rest. Separate initial spatial-group validation refits selection inside each fold; overlapping encoder context and one ROI limit independence. Reported fine-mask scores evaluate refinement on this ROI, not generalization. Latent dimension IDs are not biomarkers.","",
      "## Results","","| ID | Idea | Macro IoU | Δ H&E | Weakest IoU | Boundary F1, 8 µm | HD95 µm |","|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        s=r["scores"];report.append(f"| {r['id']:02d} | {r['title']} | {s['macro_iou']:.4f} | {r['delta_macro_iou_vs_he']:+.4f} | {s['worst_class_iou']:.4f} | {s['boundary_f1_at_um']['8']:.4f} | {s['hd95_um']:.2f} |")
    report += ["","Boundary F1/HD95 here compare internal boundary locations without class identity; always interpret them alongside class-specific IoU/Dice. Distances use 0.5 µm/pixel.","","## Reconstruction and evidence failures","",
      "Idea 01 performed genuine frozen-encoder pixel optimization from two noise starts, 100 overlapping windows per start, 32 steps/window. It remains noise-like; the available optimization budget did not recover useful histology. This failure does not prove that a better inverse or a learned image prior cannot work.","",
      "Idea 02 uses a soft retained-image mask, not a hard gland/tumor annotation. Its first per-location preservation objective was overly restrictive/unstable; an explicitly logged second preset preserves aggregate evidence for each present class, retains the best iterate, and uses a lower learning rate. Neither claims a globally minimal solution. Original image pixels supply all retained anatomy.","",
      "Idea 03 is an intentional counterfactual classifier intervention. Window-level feature changes do not imply biologically valid tissue changes, and overlap stitching can reduce the counterfactual effect. Both window metrics and re-encoded stitched-image metrics are saved.","",
      "Idea 18 records every donor patch, disjoint target/donor encoder-context boxes, and exact donor coordinates for every output pixel. Donors must lie wholly inside the native image. The mosaic is retrieval, not generation.","",
      "Feature RGB panels show model evidence or structure. H&E detail in 02, 03, 11, 14, 15, 18 and 20 is explicitly supplied by the original image. Candidate nuclei/lumens in 15 are unvalidated image-derived objects, not measured DAPI or manual annotations. Mixture weights in 16 are model assignments, not measured tissue fractions.","",
      "## Files and reproducibility","","Open `gallery.html` for adjustable 35%-default overlays of expert, coarse and refined masks, all 20 representations, full-resolution files, fixed 224-pixel crops and diagnostics. `scores.csv` and `results.json` contain every result; numbered folders contain actual input PNG, output TIFF and method details. `PREDICTIONS_FROZEN.json` records pre-evaluation hashes.","",
      "Code: `experiments/virchow2_inverse/`. Model: [paige-ai/Virchow2](https://huggingface.co/paige-ai/Virchow2), pinned revision `3158645804b69e3f3bc4439d4116edddf0840a72`. Weights, data, outputs and intermediate arrays are not committed to Git.",""]
    (a.root/"REPORT.md").write_text("\n".join(report))
    save_json(a.root/"COMPLETE.json",{"experiments":20,"successful_execution":20,"best_macro_iou_id":best["id"],"gallery":"gallery.html","fine_reference_read_after_freeze":True})
    print(json.dumps({"completed":20,"best":best["id"],"best_macro_iou":best["scores"]["macro_iou"],"he_macro_iou":controls["he_bora"]["macro_iou"],"coarse_macro_iou":controls["coarse"]["macro_iou"]}),flush=True)

if __name__=="__main__":main()
