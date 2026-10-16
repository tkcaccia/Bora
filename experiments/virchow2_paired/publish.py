"""Generate the paired proof-of-concept report and self-contained local gallery."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.virchow2_inverse.common import save_json,digest


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--training",type=Path,required=True);p.add_argument("--previous",type=Path,required=True)
    a=p.parse_args();r=json.loads((a.root/"results.json").read_text());protocol=json.loads((a.root/"PROTOCOL.json").read_text())
    training=json.loads((a.training/"TRAIN_COMPLETE.json").read_text());history=json.loads((a.training/"training_history.json").read_text())
    corpus=json.loads((a.training/"corpus.json").read_text());scope={}
    for split in ("train","validation","test"):
        rows=[p for p in corpus["patches"] if p["split"]==split]
        scope[split]=dict(patches=len(rows),case_proxies=len(set(p["case"] for p in rows)),
                          source_rois=len(set(p["image"] for p in rows)))
    for f in ("corpus.json","TRAIN_PROTOCOL.json","TRAIN_COMPLETE.json","training_history.json","CACHE_READY.json"):
        shutil.copy2(a.training/f,a.root/f)
    save_json(a.root/"DATASET_SCOPE.json",dict(scope=scope,excluded_case="114",dataset=corpus["source"],case_proxy_caveat=True,
        image_type="ROI-derived native224px crops",sampling_um_per_px=.5,roi_fov_um_xy=[612,583],tile_fov_um_xy=[112,112],
        optical_resolution_um=None,objective_magnification=None))
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout="constrained")
    axes[0].plot([x["epoch"] for x in history],[x["train_loss"] for x in history],label="Train")
    axes[0].plot([x["epoch"] for x in history],[x["validation_loss"] for x in history],label="Validation")
    axes[0].axvline(training["best_epoch"],color="grey",ls="--",lw=1,label="Retained checkpoint")
    axes[0].set(xlabel="Epoch",ylabel="RGB + gradient + SSIM loss",title="Case-disjoint decoder training");axes[0].legend()
    axes[1].scatter([x["psnr_db"] for x in r["test_cases"].values()],[x["ssim_rgb"] for x in r["test_cases"].values()],s=30)
    axes[1].set(xlabel="PSNR (dB), mean per case",ylabel="RGB SSIM, mean per case",title=f"{r['test_case_count']} held-out case proxies")
    fig.savefig(a.root/"training_and_test.png",dpi=170);plt.close(fig)
    names={"coarse":"Coarse tiling mask","original":"Original H&E + Bora","full":"Full reconstruction + Bora",
           "selective":"Selective 1.25× + Bora","suppression":"Suppression 0.75× + Bora","stronger":"Selective 1.5× + Bora"}
    lines=["# Virchow2 paired reconstruction proof of concept","",
      "## Question and outcome","",
      "Can the final spatial tokens regenerate H&E, and can a bounded edit improve architectural contrast without changing anatomy? Both outputs use the same learned RGB decoder. This is a separate paired experiment alongside the twenty earlier ideas.","",
      f"Full reconstruction passed the predeclared fidelity screen: **{r['conclusion']['full_reconstruction_passed_fidelity_screen']}**. Selective preprocessing supported by all exploratory criteria: **{r['conclusion']['preprocessing_proof_of_concept_supported']}**.","",
      "A failed screen means this decoder and training budget did not establish a faithful preprocessing method. It does not prove that the tokens lack the missing information, or that another decoder could not recover it.","",
      "## Paper and implementation","",
      "The attached paper, *Computationally Efficient Pathology Segmentation using Knowledge Distillation from Foundation Models* (Lv et al., arXiv:2609.03947v1, 3 September 2026), learns a segmentation decoder over frozen or LoRA-adapted Virchow2 spatial tokens (§4.2.1, Fig.2). It does not demonstrate RGB reconstruction. Our RGB decoder and latent-edit experiment are new exploratory adaptations, not reproduction of a published inverse.","",
      "The frozen, SHA-verified paige-ai/Virchow2 checkpoint supplies 256 final spatial tokens (16×16), each with all 1280 channels. CLS and four registers are excluded. Each 14 px token covers7µm. The decoder projects 1280→256, applies three residual convolution blocks, projects to 588 channels, and pixel-shuffles 14× to 224×224 RGB. There are no source RGB pixels, raw-pixel residuals, image coordinates or reference masks in decoder inputs. No extra generative model is used.","",
      "Training uses RGB L1 +0.2 gradient L1 +0.15(1-local SSIM), AdamW, train-only token normalization and validation checkpoint selection. Encoder weights are frozen. No target-case fine-tuning, target-RGB inversion optimization or test-set checkpoint selection is performed.","",
      f"Retained epoch **{training['best_epoch']}** of **{training['epochs']}**, checkpoint SHA256 `{training['checkpoint_sha256']}`. See TRAIN_PROTOCOL.json and training_history.json.","",
      "## Data and physical scale","",
      "TIGER ROI release: https://zenodo.org/records/6014422. The provider reports images and annotations at **0.5µm/pixel**. These are ROI images, not complete WSIs. Microscopy optical resolving power and objective magnification are unreported; sampling must not be interpreted as either. The selected ROI is 1224×1166px (612×583µm); model crops 224×224px (112×112µm), with no resolution change. TIGER's BCSS-derived ROI subset here uses its released 0.5µm/px sampling, rather than an independent original BCSS mirror's sampling.","",
      "| Decoder split | Native224px tiles | Source-case proxies | Source ROIs |","|---|---:|---:|---:|"]
    for name,row in scope.items():lines.append(f"| {name} | {row['patches']} | {row['case_proxies']} | {row['source_rois']} |")
    lines += ["", "All source identifiers beginning 114, including 114S and 114B, are excluded from every decoder split. TCGA cases use the 12-character patient identifier; other source cases use the conservative leading numeric or TC_Sxx_Pxxxxxx prefix. These are case/patient proxies, not independently verified patient mappings. Related or overlapping ROIs remain within one split. See corpus.json for every crop coordinate, split and image, CACHE_READY.json for source hashes, and DATASET_SCOPE.json for exact scope. No masks are needed for decoder training. Virchow2 pretraining overlap with public source cases cannot be excluded.","",
      "## Same-tile paired intervention","",
      f"Focal direct window index{protocol['focal']['index']}, native top-left(x,y)=({protocol['focal']['x']},{protocol['focal']['y']}). Selected by maximum coarse-class entropy among complete, unpadded windows with at least three coarse classes. The same native224px image is used for both reconstructions; this selection does not read fine annotations.","",
      "The full control changes no token features. The primary selective control amplifies an equal-class mean contrast subspace by 1.25×, learned from the existing 64-dimensional balanced-MRMR pool using only 169 coarse 90 px tile observations. The contrast basis has rank ≤ 4 for five classes. All 1,216 unselected coordinates remain exactly unchanged, and all 1,280 channels remain present. Before capping, the edit lies in the class-contrast subspace; the elementwise cap can introduce a bounded orthogonal component within the 64 selected coordinates. Each edited component is clipped to ±0.5 training standard deviations and the total edit to 5% of the token's normalized norm. Fixed0.75× suppression and 1.5× amplification are sensitivity controls, not new members of the earlier 20-idea study. The decoder is identical and frozen for all four outputs. Latent coordinates are not protein markers.","",
      "The full ROI is decoded from the same 100 original native windows and Hann-weighted overlap-added, then cropped to its native frame. Direct focal-tile results are saved separately so overlap averaging does not conceal single-tile fidelity. No source image detail is blended into any reconstruction.","",
      "## Reconstruction and anatomy checks","",
      "The full fidelity screen was frozen before scoring: RGB SSIM ≥ 0.8 on both the full ROI and direct focal tile, candidate-nucleus matching F1 ≥ 0.8, median matched displacement ≤ 2µm. These exploratory thresholds are not clinical validation criteria. Nuclei are automatic hematoxylin-based candidates, with a threshold fixed from original H&E and one-to-one matching within 2µm. They are not expert nucleus masks and their errors can distort the comparison. Hematoxylin-gradient correlation provides an additional structural check.","",
      "| Reconstruction | ROI SSIM | ROI PSNR dB | H-gradient correlation | Nucleus proxy F1 | Direct-tile SSIM |","|---|---:|---:|---:|---:|---:|"]
    for name,m in r["metrics"].items():
        lines.append(f"| {name} | {m['reconstruction']['ssim_rgb']:.4f} | {m['reconstruction']['psnr_db']:.2f} | {m['reconstruction']['hematoxylin_gradient_correlation']:.4f} | {m['nuclei_proxy']['f1']:.4f} | {m['focal_direct_window']['ssim_rgb']:.4f} |")
    lines += ["",f"On **{r['test_patch_count']} held-out test tiles from{r['test_case_count']} case proxies**, case-macro mean SSIM is **{r['test_summary']['ssim_rgb']['case_macro_mean']:.4f}**, PSNR **{r['test_summary']['psnr_db']['case_macro_mean']:.2f}dB**. Case medians and quartiles are in results.json. The gallery shows the worst,25th,50th,75th and best SSIM test examples, with exact indices, rather than only attractive examples.","",
      "## Architectural contrast and Bora","",
      "Contrast is measured in decoded Lab pixels as class mean separation divided by pooled within-class variance, using the same central-half coarse tiles. The minimum over all ten class pairs tests the weakest distinction. This is a descriptive contrast proxy, not statistical significance or biological identity. The same fixed internal Bora wand configuration refines the same coarse labels for each input; exterior coarse background is held fixed.","",
      "| Input | Worst-pair Lab contrast | Mean-pair Lab contrast |","|---|---:|---:|"]
    for name,c in [("original",r["original_coarse_contrast"]),*[(n,m["coarse_class_contrast"]) for n,m in r["metrics"].items()]]:
        lines.append(f"| {name} | {c['worst_pair_fisher']:.4f} | {c['mean_pair_fisher']:.4f} |")
    lines += ["","| Bora input/control | Macro IoU | Weakest class IoU | Boundary F1 (8µm) |","|---|---:|---:|---:|"]
    for name,m in r["segmentation"].items():lines.append(f"| {names[name]} | {m['macro_iou']:.4f} | {m['worst_class_iou']:.4f} | {m['boundary_f1_at_um']['8']:.4f} |")
    classes=["Invasive tumor","Tumor-associated stroma","Healthy glands","Inflamed stroma","Rest"]
    lines += ["","| Class | Original H&E Bora | Full reconstruction Bora | Selective Bora |","|---|---:|---:|---:|"]
    for i,label in enumerate(classes,1):lines.append("| "+label+" | "+" | ".join(f"{r['segmentation'][n]['per_class'][str(i)]['iou']:.4f}" for n in ("original","full","selective"))+" |")
    lines += ["","The fine expert tissue reference was opened only after all reconstructions and Bora outputs were frozen and SHA-verified. These are expert tissue annotations; code 0 is ignored. Coarse90px tiles were originally derived from these annotations, so this is a controlled boundary-refinement experiment, not an independently predicted coarse-label benchmark. This ROI has also been examined in the previous20 experiments. There are no independent test cases for the selective edit or segmentation claim, no patient-level significance tests, and no claim of universal improvement.","",
      "## Decision","",*[(f"- {k}: **{v}**") for k,v in r["conclusion"].items() if isinstance(v,bool)],"",
      "Only if full fidelity, selective anatomy retention, weakest-pair contrast and Bora improvement all pass should this be carried forward as a candidate preprocessing method. Even then it requires new cases, independently supplied coarse labels, expert morphology review and externally tested gains. Otherwise retain original H&E for Bora and treat the paired output as a diagnostic reconstruction study.","",
      "## Files and earlier experiments","",
      "- gallery.html: paired images, editable mask opacity, all fixed controls, held-out examples.",
      "- paired_tile.png: same native224px tile, original/full/selective.",
      "- paired_roi.png: full native tissue field, original/full/selective.",
      "- segmentation.png: original expert/coarse masks and Bora overlays at 35% opacity.",
      "- full.png, selective.png, suppression.png, stronger.png: native-size outputs.",
      "- bora.*.tif: compact labels1tumor,2stroma,3healthy glands,4inflamed stroma,5rest;0ignore.",
      "- PROTOCOL.json, PREDICTIONS_FROZEN.json, results.json: settings, frozen hashes and complete metrics.",
      "- Remote decoder.pt, tokens.npy and rgb.npy: model and numerical training cache; not committed to Git.",
      "- Earlier20 experiments: ../virchow2-inverse-2026-10-16/gallery.html and REPORT.md; outputs remain unchanged.","",
      "Code: experiments/virchow2_paired/. Reproduction commands and environment are documented in that folder's README.md."]
    if (a.root/"consistency_results.json").exists():
        diagnostic=json.loads((a.root/"consistency_results.json").read_text())
        records=json.loads((a.root/"CONSISTENCY_FROZEN.json").read_text())["records"]
        lines += ["","## Additional full-token consistency diagnostic","",
          "After the primary experiment, both focal reconstructions were optimized for 400 steps against all 1280 target spatial channels, initialized solely from their respective decoder images. Both use the same FP32 forward/gradient computation, learning-rate schedule and weak TV/initialization regularizers. No original RGB pixels or fine masks enter the optimization. This is a single-tile diagnostic, not another full-ROI Bora experiment and not a replacement for the primary frozen comparison.","",
          "Two initial unscaled FP16 optimization pilots were unstable; their protocols, images and logs are preserved under diagnostics/. The final diagnostic uses FP32 to avoid small input-gradient underflow. The change of precision was motivated by feature-objective instability, not fine-mask scoring or a search over visual attractiveness. The actual-model finite-difference audit and FP16/FP32 gradient comparison are saved in GRADIENT_AUDIT.json.", "",
          "| Focal reconstruction | RGB SSIM | PSNR dB | H-gradient correlation | Nucleus proxy F1 |","|---|---:|---:|---:|---:|"]
        for name,m in diagnostic["metrics"].items():lines.append(f"| {name} | {m['ssim_rgb']:.4f} | {m['psnr_db']:.2f} | {m['hematoxylin_gradient_correlation']:.4f} | {m['nuclei_proxy']['f1']:.4f} |")
        for name,record in records.items():
            lines += ["",f"{name}: normalized token MSE {record['history'][0]['normalized_feature_mse']:.5f} initially, {record['history'][-1]['normalized_feature_mse']:.5f} at the final iterate; best objective {record['best_objective']:.5f} at step{record['best_step']}. Optimization uses only token loss to retain an iterate."]
        lines += ["","Closer token matching with little pixel-level improvement would show that latent agreement alone is not a fidelity certificate. It would not establish that accurate inversion is impossible. See paired_tile_consistency.png, CONSISTENCY_PROTOCOL.json and consistency_results.json."]
    (a.root/"REPORT.md").write_text("\n".join(lines)+"\n")
    payload=json.dumps(dict(results=r,protocol=protocol,names=names),allow_nan=False)
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Virchow2 paired reconstruction</title><style>
body{margin:0;background:#f0f3f5;color:#1b2f3b;font:16px system-ui;line-height:1.5}main{max-width:1500px;margin:auto;padding:24px}h1{font-size:30px}img{max-width:100%;height:auto}figure{margin:0;background:white;padding:14px;border-radius:8px}figcaption{font-weight:650}small{display:block;color:#566673}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.controls{display:flex;gap:20px;flex-wrap:wrap;align-items:center;background:#fff;padding:14px;margin:20px 0}select,input,button{font:inherit}canvas{max-width:100%;height:auto}.notice{padding:16px;background:#fff0dc;border-left:4px solid #ba7415}.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:14px;margin:16px 0}.dot{display:inline-block;width:13px;height:13px;margin-right:5px}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{text-align:left;padding:8px;border-bottom:1px solid #dde2e5}.scroll{overflow:auto}.native canvas{max-width:none}details{margin:20px 0}summary{cursor:pointer;font-weight:600}.test{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}@media(max-width:800px){main{padding:12px}.grid,.test{grid-template-columns:1fr}h1{font-size:24px}}</style><main>
<h1>Virchow2: full versus selective reconstruction</h1><p>Same H&E tile · same decoder · all spatial channels retained · no original-pixel shortcut.</p><div class="notice" id="verdict"></div>
<p><a href="REPORT.md">Full report</a> · <a href="results.json">All metrics</a> · <a href="../virchow2-inverse-2026-10-16/gallery.html">The earlier 20 ideas</a></p>
<h2>Same native tile (224 px = 112 µm)</h2><a href="paired_tile.png"><img src="paired_tile.png" alt="Original tile, full-feature reconstruction, selective reconstruction"></a><p id="tileinfo"></p>
<h2>Entire tissue ROI</h2><a href="paired_roi.png"><img src="paired_roi.png" alt="Original and reconstructed tissue ROI"></a>
<details><summary>Resolution, training scope and interpretation</summary><p>TIGER ROI, not WSI: 1224×1166px,612×583µm at provider-reported 0.5µm/px. Objective and optical resolving power are unknown. Decoder inputs: 16×16 spatial tokens with 1280 channels; each token covers 14px=7µm. All114-source images were excluded from decoder training, validation and testing. The public encoder's pretraining overlap cannot be excluded.</p><p>Training 3920 tiles / 152 case proxies; validation 216 tiles / 21proxies; test 216 tiles / 21proxies. Image quality and candidate-nucleus matching are exploratory checks, not expert morphology validation. This one ROI and its reference-derived coarse labels were also used in earlier experiments.</p><p>Primary edit amplifies a coarse-class contrast subspace 1.25×; all other features remain. Component edits ≤ 0.5 training SD; token norm edits ≤ 5%. Sensitivity controls use 0.75×/1.5×. A more separable latent space alone does not establish valid regenerated anatomy.</p></details>
<h2>Expert masks and Bora overlays</h2><div class="controls"><label>Reconstruction <select id="method"><option value="selective">Selective1.25×</option><option value="full">Full features</option><option value="suppression">Suppression0.75×</option><option value="stronger">Amplification1.5×</option></select></label><label>Mask opacity <input id="alpha" type="range" min="0" max="100" value="35"><span id="av">35%</span></label><label><input type="checkbox" id="native">Native pixels</label></div><div class="legend" id="legend"></div><div class="grid" id="panels">
<figure><figcaption>Expert tissue reference</figcaption><small>Annotated tissue classes; code 0 ignored</small><canvas id="expert"></canvas></figure>
<figure><figcaption>Coarse tiling mask</figcaption><small>90 px = 45 µm; reference-derived</small><canvas id="coarse"></canvas></figure>
<figure><figcaption>Bora: original H&E</figcaption><small>Original H&E under mask</small><canvas id="control"></canvas></figure>
<figure><figcaption id="inputtitle">Selective reconstruction</figcaption><small>No original H&E detail added</small><a id="inputlink"><img id="input" alt="Selected native reconstruction"></a></figure>
<figure><figcaption>Bora: selected reconstruction</figcaption><small>Mask over original H&E, for anatomical comparison</small><canvas id="selected"></canvas></figure>
<figure><figcaption>Selected mask over reconstruction</figcaption><small>View the actual input beneath its output mask</small><canvas id="onreconstruction"></canvas></figure></div>
<h2>Fidelity and architectural contrast</h2><div class="scroll"><table><thead><tr><th>Input</th><th>ROI SSIM</th><th>PSNR dB</th><th>Nucleus proxy F1</th><th>Worst-pair contrast</th></tr></thead><tbody id="fidelity"></tbody></table></div><p>Nuclei are automatic candidates, not expert annotations. One-to-one matching within 2µm uses the same original-derived hematoxylin threshold. Colour contrast is measured in decoded Lab pixels on fixed coarse-tile interiors.</p>
<h2>Bora segmentation</h2><div class="scroll"><table><thead><tr><th>Input</th><th>Macro IoU</th><th>Weakest class IoU</th><th>Boundary F1(8µm)</th></tr></thead><tbody id="scores"></tbody></table></div><p>Fine tissue masks were opened after predictions were frozen. No independent segmentation test cohort; changes are descriptive.</p>
<h2>Held-out reconstruction checks</h2><p id="testsummary"></p><img src="training_and_test.png" alt="Training convergence and held-out case reconstruction scores"><p>Worst,25th,50th,75th and best test-tile SSIM examples, respectively:</p><div class="test">'''+''.join(f'<a href="test_example_{i}.png"><img src="test_example_{i}.png" alt="Held-out test example {i}"></a>' for i in range(5))+'''</div><details><summary>Predeclared screen and complete result</summary><pre id="details" style="white-space:pre-wrap"></pre></details></main><script>
const D=__DATA__, $=id=>document.getElementById(id),r=D.results;const colours=['#dc2d41','#37a54b','#376ee6','#c864dc','#dcb437'],classes=['Invasive tumor','Tumor-associated stroma','Healthy glands','Inflamed stroma','Rest'];classes.forEach((v,i)=>$('legend').innerHTML+='<span><i class="dot" style="background:'+colours[i]+'"></i>'+v+'</span>');
$('verdict').textContent=r.conclusion.preprocessing_proof_of_concept_supported?'Exploratory screen passed; independent validation is still required.':'The paired experiment does not establish a faithful architecture-enhancing preprocessing method. Inspect the full-feature reconstruction and anatomy checks before interpreting selective contrast.';
const f=D.protocol.focal;$('tileinfo').textContent='Native top-left(x,y)=('+f.x+','+f.y+'), window'+f.index+'. Selected using coarse labels only; direct decoder outputs shown above, without overlap averaging.';
$('testsummary').textContent=r.test_patch_count+' held-out native tiles from '+r.test_case_count+' case proxies. Case-macro SSIM '+r.test_summary.ssim_rgb.case_macro_mean.toFixed(3)+', PSNR '+r.test_summary.psnr_db.case_macro_mean.toFixed(2)+'dB.';
Object.entries(r.metrics).forEach(([n,m])=>{let tr=document.createElement('tr');tr.innerHTML='<td>'+D.names[n]+'</td><td>'+m.reconstruction.ssim_rgb.toFixed(3)+'</td><td>'+m.reconstruction.psnr_db.toFixed(2)+'</td><td>'+m.nuclei_proxy.f1.toFixed(3)+'</td><td>'+m.coarse_class_contrast.worst_pair_fisher.toFixed(3)+'</td>';$('fidelity').append(tr)});
Object.entries(r.segmentation).forEach(([n,m])=>{let tr=document.createElement('tr');tr.innerHTML='<td>'+D.names[n]+'</td><td>'+m.macro_iou.toFixed(3)+'</td><td>'+m.worst_class_iou.toFixed(3)+'</td><td>'+m.boundary_f1_at_um['8'].toFixed(3)+'</td>';$('scores').append(tr)});
$('details').textContent=JSON.stringify({screen:D.protocol.fidelity_gate,result:r.conclusion},null,2);const cache={};function load(p){return cache[p]||(cache[p]=new Promise((ok,bad)=>{let im=new Image();im.onload=()=>ok(im);im.onerror=bad;im.src=p}))}
async function draw(id,base,mask){const [im,m]=await Promise.all([load(base),load(mask)]),c=$(id);c.width=im.width;c.height=im.height;const g=c.getContext('2d');g.drawImage(im,0,0);g.globalAlpha=+$('alpha').value/100;g.drawImage(m,0,0);g.globalAlpha=1;g.fillStyle='white';g.fillRect(c.width-250,c.height-65,230,50);g.fillStyle='black';g.fillRect(c.width-238,c.height-30,200,5);g.font='22px sans-serif';g.fillText('100µm',c.width-194,c.height-37)}
function redraw(){const n=$('method').value;$('av').textContent=$('alpha').value+'%';$('input').src=n+'.png';$('inputlink').href=n+'.png';$('inputtitle').textContent=D.names[n].replace(' + Bora','');draw('expert','original.png','expert.mask.png');draw('coarse','original.png','coarse.mask.png');draw('control','original.png','bora.original.mask.png');draw('selected','original.png','bora.'+n+'.mask.png');draw('onreconstruction',n+'.png','bora.'+n+'.mask.png')}
$('method').onchange=redraw;$('alpha').oninput=redraw;$('native').onchange=e=>$('panels').classList.toggle('native',e.target.checked);redraw();
</script></html>'''
    if (a.root/"consistency_results.json").exists():
        diagnostic=json.loads((a.root/"consistency_results.json").read_text())
        rows=''.join('<tr><td>'+n+'</td><td>'+f"{m['ssim_rgb']:.3f}"+'</td><td>'+f"{m['nuclei_proxy']['f1']:.3f}"+'</td></tr>' for n,m in diagnostic['metrics'].items())
        section='<h2>Can full-token matching recover lost detail?</h2><p>Additional focal-tile diagnostic: 400 FP32 optimization steps, initialized from decoder images. No original-pixel targets. Both reconstructions use the same procedure. This does not change the primary ROI or Bora results.</p><a href="paired_tile_consistency.png"><img src="paired_tile_consistency.png" alt="Original and full-token consistency reconstructions"></a><table><thead><tr><th>Focal reconstruction</th><th>RGB SSIM</th><th>Nucleus proxy F1</th></tr></thead><tbody>'+rows+'</tbody></table><p><a href="consistency_results.json">Diagnostic details</a>. Closer token agreement alone does not certify preserved anatomy.</p>'
        html=html.replace('<h2>Entire tissue ROI</h2>',section+'<h2>Entire tissue ROI</h2>')
    html=html.replace('.native canvas{max-width:none}', '.native canvas,.native img{max-width:none}.native figure{overflow:auto;max-height:900px}')
    html=html.replace('Selective1.25×','Selective 1.25×').replace('Suppression0.75×','Suppression 0.75×').replace('Amplification1.5×','Amplification 1.5×')
    (a.root/"gallery.html").write_text(html.replace("__DATA__",payload))
    (a.root/"START_HERE.md").write_text("Open gallery.html through a local HTTP server. Read REPORT.md for scientific interpretation and DATASET_SCOPE.json for source splits. Paired outputs have no original-pixel skip. Full numerical caches and decoder weights are stored on Chiamaka under the experiment directory; the Mac copy contains images, masks, audit JSON and report. The earlier 20 outputs are in ../virchow2-inverse-2026-10-16/.\n")
    print(json.dumps(dict(scope=scope,report=str(a.root/"REPORT.md"),conclusion=r["conclusion"])),flush=True)

if __name__=="__main__":main()
