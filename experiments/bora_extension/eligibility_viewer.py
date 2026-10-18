"""Render the eligibility inventory separately from signed cycle results."""
import argparse
from html import escape
import json
from pathlib import Path

from .eligibility import digest, save_new


def build(root, cycle_url="../bora-cycles-2026-10-17/cycle-10"):
    root = Path(root)
    report = json.loads((root/"ELIGIBILITY.json").read_text())
    rows = [{k:v for k,v in row.items() if k != 'coarse_observations'} for row in report['cases']]
    data = json.dumps(dict(cases=rows, cycle_url=cycle_url), allow_nan=False).replace('<', '\\u003c')
    table = ''.join(f'<tr><td>{escape(d)}</td><td>{r["total"]}</td>'
        f'<td>{r["eligible"]}</td><td>{r["excluded"]}</td></tr>' for d,r in report['datasets'].items())
    html = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bora benchmark eligibility</title><style>
body{font:15px system-ui;margin:0;color:#183040;background:#eef2f4}main{padding:18px;max-width:1750px;margin:auto}
.toolbar{position:sticky;top:0;background:#fff;padding:12px;display:flex;gap:10px;flex-wrap:wrap;z-index:2;box-shadow:0 1px 4px #aaa}
select,button{padding:8px;font:inherit;max-width:100%}.notice{padding:14px;background:#fff3c6;border-left:5px solid #be8100}
.good{color:#066740}.bad{color:#9e2d12}.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:12px}
figure{margin:0;background:white;padding:10px}figure img{width:100%;max-height:470px;object-fit:contain;background:#f7f7f7}
figcaption{margin-bottom:8px;font-weight:600}td,th{padding:7px;text-align:left;border-bottom:1px solid #ccd5dc}
table{border-collapse:collapse;background:white}h1{font-size:26px}p{line-height:1.5}.legend{display:flex;gap:16px;flex-wrap:wrap}
a{color:#07528c}.counts{font-variant-numeric:tabular-nums}details{margin:14px 0}
</style><div class="toolbar"><select id="scope" aria-label="Inventory">
<option value="eligible">Eligible benchmark (299)</option><option value="diagnostic">Excluded diagnostics (50)</option>
</select><button id="previous">← Previous</button><select id="case" aria-label="Image"></select>
<button id="next">Next →</button><span id="position"></span></div><main>
<h1>Benchmark eligibility: 299 included · 50 excluded</h1>
<p class="notice">The prospective benchmark excludes single-group images and fields with too few real tiles.
The panels below are existing cycle10 images and predictions, displayed for inspection.
No new segmentation, aggregate score or survivor selection has been produced. Signed cycle10 scores remain unchanged.</p>
<p>Required: at least two positive input groups, at least two groups in actual feature-selection observations,
at least four complete 224px encoder fields along each image axis, and at least16 usable labelled 90px coarse tiles.
These are prospective engineering limits. Padding, overlap and repeated tokens add no qualifying fields.
Reference classes missing from coarse input remain errors; a rare group with one usable tile is retained.</p>
<details><summary>Dataset counts and frozen evidence</summary><table><tr><th>Dataset</th><th>Original</th><th>Included</th><th>Diagnostics</th></tr>TABLE</table>
<p><a href="ELIGIBILITY.json">Per-image eligibility and tile counts</a> · <a href="POLICY.json">Frozen rules</a> ·
<a href="PROTOCOL.json">Audit identity</a> · <a href="REPORT.md">Sampling investigation</a></p></details>
<h2 id="heading"></h2><p id="status"></p><p class="counts" id="counts"></p><p id="groups"></p><p id="resolution"></p>
<p id="reason"></p><p><a id="old-link">Open the original signed cycle10 viewer</a></p><div class="panels" id="inputs"></div>
<h2>Existing cycle10 masks — inspection only</h2><div class="panels" id="outputs"></div>
<script>const DATA=PAYLOAD;
const scope=document.querySelector('#scope'), selector=document.querySelector('#case');let subset=[];
const labels={fewer_than_two_input_groups:'Only one input group (ignore/background0 does not count)',
fewer_than_two_feature_selection_groups:'Only one group available to feature selection',
fewer_than_four_real_encoder_fields_on_each_axis:'Insufficient real field of view for a 4×4 block of encoder fields',
fewer_than_sixteen_usable_coarse_tiles:'Fewer than16 usable labelled coarse tiles'};
function text(id,value){document.querySelector('#'+id).textContent=value;}
function panel(title,url){const figure=document.createElement('figure'),caption=document.createElement('figcaption'),a=document.createElement('a'),img=document.createElement('img');caption.textContent=title;a.href=url;img.src=url;img.alt=title;img.loading='lazy';a.append(img);figure.append(caption,a);return figure;}
function render(){let index=Math.max(0,subset.findIndex(r=>r.case===selector.value));let r=subset[index];if(!r)return;
selector.value=r.case;text('position',`${index+1} / ${subset.length}`);text('heading',`${r.case} · ${r.dataset}`);
text('status',r.eligible?'INCLUDED in the prospective benchmark':'EXCLUDED — separate diagnostic only; absent from future averages and rankings');
document.querySelector('#status').className=r.eligible?'good':'bad';
text('counts',`${r.real_unpadded_shape_yx[1]} × ${r.real_unpadded_shape_yx[0]} real pixels · complete base fields ${r.complete_base_fields_yx.join(' × ')} = ${r.complete_base_field_capacity} · usable coarse tiles ${r.usable_coarse_tiles}`);
text('groups',`Input groups ${r.input_groups.length}; feature-selection groups ${r.observed_groups.length}. `+Object.entries(r.coarse_tiles_by_group).map(([g,n])=>`${r.class_names[g]||g}: ${n} usable tiles`).join(' · '));
text('resolution',r.working_mpp_xy?`Working pixel spacing x/y: ${r.working_mpp_xy.map(x=>x.toFixed(6)).join(' / ')} µm/px. This is sampling, not optical resolving power.`:'Microscopy pixel spacing unverified. Native-pixel geometry only; no physical scale is assumed.');
text('reason',r.exclusion_reasons.map(x=>labels[x]).join(' · '));
let base=DATA.cycle_url+'/cases/'+encodeURIComponent(r.case)+'/';document.querySelector('#old-link').href=DATA.cycle_url+'/selected.html#'+encodeURIComponent(r.case);
document.querySelector('#inputs').replaceChildren(...[['Original H&E','original-he.jpg'],['Expert reference mask','reference.jpg'],['Input tiled mask','coarse.jpg'],['Fixed #9 + #19 converted RGB','converted.jpg']].map(([t,f])=>panel(t,base+f)));
document.querySelector('#outputs').replaceChildren(...['J01','M04','N08','N10','C17'].map(k=>panel(k+' · existing cycle10 mask',base+k+'/mask.png')));
history.replaceState(null,'','#'+r.case);}
function populate(wanted){subset=DATA.cases.filter(r=>r.eligible===(scope.value==='eligible'));selector.replaceChildren(...subset.map(r=>{let o=document.createElement('option');o.value=r.case;o.textContent=r.case;return o;}));selector.value=subset.some(r=>r.case===wanted)?wanted:subset[0]?.case;render();}
scope.onchange=()=>populate('');selector.onchange=render;for(let [id,step] of [['previous',-1],['next',1]])document.querySelector('#'+id).onclick=()=>{selector.selectedIndex=(selector.selectedIndex+step+subset.length)%subset.length;render();};
let initial=decodeURIComponent(location.hash.slice(1)), row=DATA.cases.find(r=>r.case===initial);if(row&&!row.eligible)scope.value='diagnostic';populate(initial||'tiger_32');
</script></main></html>'''.replace('TABLE',table).replace('PAYLOAD',data)
    html = html.replace('Eligible benchmark (299)', f'Eligible benchmark ({report["eligible_count"]})')
    html = html.replace('Excluded diagnostics (50)', f'Excluded diagnostics ({report["excluded_count"]})')
    html = html.replace('299 included · 50 excluded', f'{report["eligible_count"]} included · {report["excluded_count"]} excluded')
    save_new(root/'VIEWER_INPUTS.json', dict(eligibility_sha256=digest(root/'ELIGIBILITY.json'),
        historical_cycle_url=cycle_url, predictions_recomputed=False, scores_recomputed=False,
        viewer_source_sha256=digest(__file__)))
    (root/'index.html').write_text(html)
    return dict(html_sha256=digest(root/'index.html'), eligible=report['eligible_count'],diagnostics=report['excluded_count'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--cycle-url',default='../bora-cycles-2026-10-17/cycle-10')
    args=parser.parse_args()
    print(json.dumps(build(args.root,args.cycle_url)))
