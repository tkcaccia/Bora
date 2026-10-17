"""Presentation-only updates from frozen results; does not generate predictions."""
import argparse,json,textwrap
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .common import save_json
from .finalize import contact_sheet,write_gallery,font
from .protocol import COLOURS

def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);a=p.parse_args();root=a.root
    data=json.loads((root/"results.json").read_text());rows=data["experiments"];controls=data["controls"];protocol=data["protocol"]
    # Preserve original provider codes as well as the compact evaluation codes.
    protocol["native_label_remap"]={"1":1,"2":2,"4":3,"6":4,"7":5}
    data["protocol"]=protocol;save_json(root/"results.json",data);save_json(root/"protocol.json",protocol)
    best=max(rows,key=lambda r:r["scores"]["macro_iou"])
    if (root/"02/evidence_mask.npy").exists():
        mask=np.load(root/"02/evidence_mask.npy")
        rows[1]["details"]["mask_min_max"]=[float(mask.min()),float(mask.max())]
        rows[1]["details"]["mask_fraction_at_least_0_5"]=float(np.mean(mask>=.5))
        rows[1]["interpretation"]="This did not isolate a small evidence region: every pixel's soft mask remains above 0.5, with mean weight about 0.86. It preserves classifier evidence but fails the sparse-region goal at this budget. The heatmap is contrast-scaled for visibility."
    rows[0]["goal_outcome"]="unsuccessful tissue reconstruction at tested budget"
    rows[1]["goal_outcome"]="unsuccessful sparse evidence localization at tested budget"
    rows[2]["goal_outcome"]="counterfactual classifier intervention, not biological reconstruction"
    for row in rows[:3]:save_json(root/f"{row['id']:02d}/result.json",row)
    data["experiments"]=rows;save_json(root/"results.json",data)
    contact_sheet(root,rows,"representation","contact_inputs.png");contact_sheet(root,rows,"overlay","contact_overlays.png")
    write_gallery(root,rows,controls,protocol)
    width=450;height=505;canvas=Image.new("RGB",(width*3,height*2+145),"white");d=ImageDraw.Draw(canvas)
    d.text((14,8),f"Best average: {best['id']:02d} {best['title']}",font=font(27),fill="black")
    d.text((14,46),f"Macro IoU {best['scores']['macro_iou']:.3f}  |  H&E Bora {controls['he_bora']['macro_iou']:.3f}  |  Coarse {controls['coarse']['macro_iou']:.3f}",font=font(20),fill="black")
    for g in range(1,6):
        x=14+(g-1)*267;d.rectangle((x,84,x+15,99),fill=tuple(COLOURS[g-1].astype(int)))
        d.text((x+21,81),protocol["class_names"][str(g)],font=font(15),fill="black")
    d.text((14,111),"Same TIGER tissue ROI · ≈0.5 µm/pixel · overlays 35% · scale bars 100 µm",font=font(17),fill="#40505c")
    files=[("Original H&E","original.scalebar.png"),("Expert reference","expert.overlay.png"),("Coarse tiling mask","coarse.overlay.png"),
           (f"{best['id']:02d} Experimental feature RGB",f"{best['id']:02d}/representation.scalebar.png"),("H&E Bora control","he.overlay.png"),(f"{best['id']:02d} Feature-driven Bora",f"{best['id']:02d}/overlay.png")]
    for i,(title,path) in enumerate(files):
        x=(i%3)*width;y=(i//3)*height+145;d.text((x+8,y+8),title,font=font(21),fill="black")
        im=Image.open(root/path).convert("RGB");im.thumbnail((width-12,height-44));canvas.paste(im,(x+6,y+43))
    canvas.save(root/"best_comparison.png")
    lines=["## Controls and class tradeoffs","","| Control | Macro IoU | Weakest class IoU | Boundary F1, 8 µm | HD95 µm |","|---|---:|---:|---:|---:|"]
    for key,title in [("coarse","Coarse tiles"),("he_bora","H&E Bora"),("feature_bora","Plain selected-feature RGB Bora"),("he_barrier","H&E barrier wrapper")]:
        v=controls[key];lines.append(f"| {title} | {v['macro_iou']:.4f} | {v['worst_class_iou']:.4f} | {v['boundary_f1_at_um']['8']:.4f} | {v['hd95_um']:.2f} |")
    lines += ["",f"Best average method {best['id']:02d} improves macro IoU by {(best['scores']['macro_iou']-controls['he_bora']['macro_iou'])*100:.2f} percentage points over H&E Bora and {(best['scores']['macro_iou']-controls['feature_bora']['macro_iou'])*100:.2f} points over the plain selected-feature RGB control. This is **not an improvement in every class**.","",
        "| Class | H&E Bora IoU | Best average method IoU | Change, percentage points |","|---|---:|---:|---:|"]
    for g in range(1,6):
        he=controls["he_bora"]["per_class"][str(g)]["iou"];v=best["scores"]["per_class"][str(g)]["iou"]
        lines.append(f"| {protocol['class_names'][str(g)]} | {he:.4f} | {v:.4f} | {(v-he)*100:+.2f} |")
    lines += ["","The best average method gains mainly on tumor and stroma, while healthy-gland and inflamed-stroma IoU decrease. It also has a lower class-agnostic boundary F1 than some other ideas. Selection among methods therefore depends on the tissue classes and boundary outcomes of interest. One ROI does not establish a reliable winner or statistical significance. The best-of-twenty method is selected on this same evaluation image, not on an independent model-selection cohort.","",
        "Provider native-to-compact mask mapping: 1→1 tumor, 2→2 stroma, 4→3 healthy glands, 6→4 inflamed stroma, 7→5 rest. Code 0 is ignored. The expert mask is remapped, not newly annotated.",""]
    lines += ["## Goal failures and classifier sensitivity","",
        "Idea 01 reduced the mean window feature-fitting loss from 4.03 to 2.14, but the stitched output remained noise-like. It is a failed tissue reconstruction at the tested budget, not a recovered slide.","",
        "Idea 02 preserved approximately 98.6% of the stitched-grid classifier decisions, but its soft mask averaged 0.861 and no pixel fell below a mask weight of 0.5. It did not identify a small retained evidence region. Do not interpret the contrast-scaled heatmap as successful sparse localization.","",
        "Idea 03 changed most stitched-grid classifier decisions with a mean RGB change of about 8/255; about 57% reached the intended runner-up class. These diagnostics include the padded border tokens. The almost unchanged H&E appearance and negligible Bora benefit illustrate classifier sensitivity, not a biological tissue transformation.",""]
    report=(root/"REPORT.md").read_text()
    start="## Controls and class tradeoffs";end="## Scope and physical resolution"
    if start in report:report=report[:report.index(start)]+report[report.index(end):]
    report=report.replace(end,"\n".join(lines)+"\n"+end)
    (root/"REPORT.md").write_text(report)
    print(json.dumps({"rendered_only":True,"best":best["id"],"predictions_modified":False}))

if __name__=="__main__":main()
