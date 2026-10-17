"""Report-only viewer for the five signed survivors of a completed cycle."""
from html import escape
import json
from pathlib import Path

from .run import digest, save
from .runtime_policy import DEFAULT_POLICY, active_policy, adjusted_score, runtime_ranking


STYLE = """*{box-sizing:border-box}body{margin:0;font:15px system-ui;color:#183040;background:#eef2f4}
main{padding:20px;max-width:2100px;margin:auto}h1{margin:0 0 8px}h2{margin:14px 0 5px}
p{line-height:1.5}.note{color:#4d626e;font-size:13px}table{border-collapse:collapse;background:white}
th,td{text-align:left;padding:9px 14px;border-bottom:1px solid #d8e0e5}th{font-size:13px}
.toolbar{position:sticky;top:0;z-index:5;background:#183040;color:white;padding:12px 20px;
display:flex;gap:12px;align-items:center;flex-wrap:wrap}button,select{font:inherit;padding:7px}
.inputs{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
.outputs{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}
figure{margin:0;background:white;padding:10px;border:1px solid #d8e0e5;border-radius:6px}
figcaption{min-height:46px;font-size:13px;line-height:1.5}.raster{display:block;position:relative;background:#fafafa}
.raster img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}
.raster .mask{opacity:var(--mask-alpha,.55)}.links{font-size:12px;margin-top:7px}
section[hidden]{display:none}.legend{display:flex;gap:12px;flex-wrap:wrap;padding:10px 0}
.swatch{display:inline-block;width:12px;height:12px;margin-right:5px;border:1px solid #999}
@media(max-width:800px){.inputs{grid-template-columns:repeat(2,minmax(0,1fr))}
main{padding:12px}th,td{padding:7px}.scores{overflow-x:auto}}
"""

SCRIPT = """
const sections=[...document.querySelectorAll('section[data-case]')];
const choose=document.querySelector('#case-select');
function showCase(index){
 index=(index+sections.length)%sections.length;
 sections.forEach((s,i)=>s.hidden=i!==index);choose.selectedIndex=index;
 document.querySelector('#position').textContent=(index+1)+' / '+sections.length;
 const key=sections[index].dataset.case;
 if(location.hash.slice(1)!==key)history.replaceState(null,'','#'+key);
}
function fromHash(){const key=location.hash.slice(1).split('/')[0];
 const i=sections.findIndex(s=>s.dataset.case===key);showCase(i<0?0:i);}
choose.addEventListener('change',()=>showCase(choose.selectedIndex));
document.querySelector('#previous').onclick=()=>showCase(choose.selectedIndex-1);
document.querySelector('#next').onclick=()=>showCase(choose.selectedIndex+1);
document.querySelector('#opacity').oninput=e=>{
 document.documentElement.style.setProperty('--mask-alpha',e.target.value);
 document.querySelector('#alpha-value').textContent=Math.round(e.target.value*100)+'%';};
document.addEventListener('keydown',e=>{
 if(['SELECT','INPUT','TEXTAREA'].includes(e.target.tagName))return;
 if(e.key==='ArrowLeft')showCase(choose.selectedIndex-1);
 if(e.key==='ArrowRight')showCase(choose.selectedIndex+1);});
window.addEventListener('hashchange',fromHash);fromHash();
"""


def number(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def summary(entry):
    score = entry["score"]
    small_key = "small_le_400_um2" if "small_le_400_um2" in score["regions"] else "small_le_400_px2"
    small = score["regions"][small_key]
    return dict(iou=score["macro_iou"], dice=score["macro_dice"],
                seconds=entry["median_seconds"], small_recall=small["recall_at_50pct"],
                small_units="µm²" if small_key.endswith("um2") else "pixels²",
                runtime_adjusted_score=adjusted_score(score["macro_iou"], entry["median_seconds"]),
                completely_missed_small=small["completely_missed"],
                annotated_unassigned_pixels=score["annotated_unassigned_pixels"])


def panel(case, title, name, ratio, overlay=False, native=None):
    prefix = "cases/"+case+"/"
    url = escape(prefix+name, quote=True)
    base = (f'<img loading="lazy" alt="Original H&amp;E beneath mask" '
            f'src="{escape(prefix+"original-he.jpg", quote=True)}">') if overlay else ""
    return (f'<figure><figcaption>{title}</figcaption>'
            f'<a class="raster" style="aspect-ratio:{ratio}" href="{url}" target="_blank">'
            +base+f'<img loading="lazy" class="{"mask" if overlay else "image"}" '
            f'alt="{escape(name, quote=True)}" src="{url}"></a>'
            f'<div class="links"><a href="{url}" target="_blank">Full raster</a>'
            +(f' · <a href="{escape(prefix+native, quote=True)}">Native labels TIFF</a>' if native else "")
            +'</div></figure>')


def publish_selected(root):
    root = Path(root)
    selection = json.loads((root/"SELECTION.json").read_text())
    for name, field in (("RESULTS.json", "results_sha256"),
                        ("VISUAL_REVIEW.json", "review_sha256"),
                        ("PROTOCOL.json", "protocol_sha256")):
        if digest(root/name) != selection[field]:
            raise ValueError("Cannot publish changed signed cycle evidence")
    selected = selection["survivors"]
    if len(selected) != 5 or len(set(selected)) != 5:
        raise ValueError("Selected viewer needs exactly five survivors")
    result = json.loads((root/"RESULTS.json").read_text())
    protocol = result["protocol"]
    if protocol != json.loads((root/"PROTOCOL.json").read_text()):
        raise ValueError("Result protocol differs from signed protocol")
    policy = active_policy(protocol) or DEFAULT_POLICY
    ranks = [row for row in runtime_ranking(result, policy) if row["method"] in selected]
    if len(ranks) != 5 or any(not row["complete"] for row in ranks):
        raise ValueError("Selected methods need complete scores")
    # Use the same aggregate runtime-adjusted order across every displayed case.
    order = [row["method"] for row in ranks]
    options, sections, records = [], [], []
    for row in result["cases"]:
        case, sample = row["case"], row["sample"]
        options.append(f'<option value="{escape(case, quote=True)}">{escape(case)}</option>')
        height, width = sample["native_shape_yx"]
        ratio = f'{width}/{height}'
        inputs = ''.join(panel(case, title, name, ratio, overlay) for title, name, overlay in (
            ("Original H&amp;E", "original-he.jpg", False),
            ("Reference annotation", "reference.jpg", True),
            ("Input tile mask", "coarse.jpg", True),
            ("Fixed converted RGB: #9 + #19", "converted.jpg", False)))
        outputs, entries = [], {}
        for key in order:
            entry = row["methods"][key]
            if entry["status"] != "ok":
                raise ValueError("A selected case has a failed prediction")
            info = summary(entry)
            info["runtime_adjusted_score"] = adjusted_score(info["iou"], info["seconds"], policy)
            entries[key] = info
            title = (f'<strong>{escape(key)} · {escape(protocol["roster"][key]["title"])}</strong>'
                     f'<br>IoU {number(info["iou"])} · {number(info["seconds"],3)} s'
                     f' · adjusted {number(info["runtime_adjusted_score"])}'
                     f'<br>Small-region recall (≤400 {info["small_units"]}) {number(info["small_recall"],3)}'
                     f' · fully missed {info["completely_missed_small"]}')
            outputs.append(panel(case, title, key+"/mask.png", ratio, True, key+"/native.tif"))
        legend = ''.join(
            f'<span><i class="swatch" style="background:rgb({",".join(map(str,color))})"></i>'
            f'{escape(sample["class_names"].get(str(i),str(i)))}</span>'
            for i,color in enumerate(sample["palette"],1))
        spacing = (' × '.join(number(v,3) for v in sample['native_mpp_xy'])+' µm/pixel'
                   if sample['native_mpp_xy'] is not None else
                   'Microscopy pixel spacing unknown; numerical scales and area scores are in pixels')
        omitted = row.get('reference_classes_absent_from_coarse', [])
        omitted_note = ('<p class="note">Input tile labels omit these reference classes: '
                        +escape(', '.join(sample['class_names'].get(str(i),str(i)) for i in omitted))
                        +'. Their pixels still count in accuracy and missed-region scores.</p>') if omitted else ''
        sections.append(
            f'<section data-case="{escape(case,quote=True)}" hidden><h2>{escape(case)}</h2>'
            f'<p class="note">{escape(row["dataset"])} · {escape(sample["source_region_type"])} · '
            f'{width} × {height} native pixels · {spacing}. '
            'Pixel spacing is not optical resolving power.</p>'+omitted_note+
            ('<p class="note">Single eligible coarse class: converted RGB is constant. Every method uses the common completed-coarse fallback; an explicit H&amp;E glass rule may still remove background. No RGB refinement is claimed.</p>'
             if row.get('single_eligible_class') else '')+
            ('<p class="note">RINGS is tumor versus outside-tumor, including glass/lumina. Outside-tumor is not expert stroma. Saved background0 is scored as outside-tumor; deleted tumor is still wrong.</p>'
             if row['dataset'] == 'rings-tumor' else '')+
            f'<div class="inputs">{inputs}</div><div class="legend">{legend}</div>'
            f'<div class="outputs">{"".join(outputs)}</div></section>')
        records.append(dict(case=case,dataset=row["dataset"],methods=entries))
    table = ''.join(
        f'<tr><td>{escape(r["method"])} · {escape(protocol["roster"][r["method"]]["title"])}</td>'
        f'<td>{number(r["mean_iou"])}</td><td>{number(r["mean_dice"])}</td>'
        f'<td>{number(r["mean_seconds"],3)}</td>'
        f'<td>{r["images_over_one_second"]}/{len(records)}</td>'
        f'<td>{number(r["mean_runtime_penalty"])}</td>'
        f'<td><strong>{number(r["runtime_adjusted_score"])}</strong></td>'
        f'<td>{number(r["small_region_recall"],3)}</td>'
        f'<td>{r["completely_missed_small_regions"]}</td>'
        +(f'<td>{number(r.get("pixel_small_region_recall"),3)}</td>'
          f'<td>{r.get("pixel_completely_missed_small_regions",0)}</td>'
          f'<td>{number(r["strata"]["historical149"]["mean_iou"])}</td>'
          f'<td>{number(r["strata"]["additional200"]["mean_iou"])}</td>'
          if protocol.get('benchmark_scope') else '')+'</tr>' for r in ranks)
    html = ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Bora cycle {selection["cycle"]}: selected five</title><style>{STYLE}</style>'
            '<div class="toolbar"><button id="previous">← Previous</button>'
            f'<select id="case-select" aria-label="Benchmark image">{"".join(options)}</select>'
            '<button id="next">Next →</button><span id="position"></span>'
            '<label>Mask opacity <input id="opacity" type="range" min="0" max="1" step=".05" '
            'value=".55"></label><span id="alpha-value">55%</span></div><main>'
            f'<h1>Cycle {selection["cycle"]}: the selected five</h1>'
            f'<p>{len(records)} benchmark ROIs · {selection["cycle"]}/20 cycles completed · '
            'fixed #9 + #19 RGB. Only the five selected algorithms are displayed.</p>'
            '<p><strong>Runtime-adjusted ranking:</strong> average of '
            'IoU − 0.05 × max(seconds − 1, 0), calculated per image. '
            'Five IoU percentage points per excess second; no bonus below one second. '
            'The adjusted score is a utility, not a segmentation accuracy.</p>'
            '<p><a href="../runtime_history.html">Completed-cycle accuracy and runtime history</a></p>'
            +('<p class="note">This is a posthoc runtime comparison of the historical five; '
              'their signed selection and raw accuracy scores are unchanged. '
              'Cycle 5 onward uses the penalty as its selection objective.</p>'
              if active_policy(protocol) is None else '')+
            '<p class="note">All images are development data in this repeated-selection study. '
            'Scores count all positive reference pixels; reference 0 is unscored. '
            'Small-region recalls use separate ≤400 µm² and ≤400 pixel² strata when resolution is unknown; recall requires ≥50% correct class coverage. '
            'Opacity affects the display only. Click a panel for its full raster.</p>'
            '<div class="scores"><table><tr><th>Selected algorithm</th><th>Mean IoU</th>'
            '<th>Mean Dice</th><th>Mean seconds</th><th>Images &gt;1s</th>'
            '<th>Mean penalty</th><th>Adjusted score</th><th>Small recall</th><th>Small fully missed</th>'
            +('<th>Pixel-small recall</th><th>Pixel-small missed</th><th>Original149 IoU</th><th>Additional200 IoU</th>'
              if protocol.get('benchmark_scope') else '')+
            f'</tr>{table}</table></div>'+''.join(sections)
            +f'<script>{SCRIPT}</script></main></html>')
    (root/"selected.html").write_text(html)
    save(root/"SELECTED_RESULTS.json", dict(cycle=selection["cycle"],methods=order,
        selection_sha256=digest(root/"SELECTION.json"), ranking=ranks,cases=records,
        predictions_modified=False,scores_modified=False, derived_runtime_scores_added=True,
        runtime_penalty_policy=policy, historical_selection_unchanged=True))
    return dict(selected_gallery="selected.html",selected_gallery_sha256=digest(root/"selected.html"),
                selected_viewer_source_sha256=digest(Path(__file__)),
                selected_results_sha256=digest(root/"SELECTED_RESULTS.json"),
                displayed_methods=order,displayed_cases=len(records))
