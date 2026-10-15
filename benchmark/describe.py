"""Add source-qualified resolution, ground-truth legends and a local annotation viewer."""
from __future__ import annotations

import argparse
import base64
import csv
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from .download import DEFAULT_ROOT, digest

HERE = Path(__file__).parent
COLORS = ["#cbd5e1", "#e53935", "#43a047", "#8e44ad", "#efb936", "#d5803c", "#1976d2", "#95a5a6",
          "#00acc1", "#f2b6d2", "#512da8", "#7e57c2", "#8d6e63", "#26a69a", "#9ccc65", "#607d8b",
          "#f06292", "#a1887f", "#ff7043", "#b71c1c", "#880e4f", "#78909c"]


def label_name(dataset, label, catalog):
    if dataset in catalog and "labels" in catalog[dataset]:
        return catalog[dataset]["labels"][str(label)]
    if dataset in ("glas", "crag", "crag-candidate"):
        return "non-gland / background" if label == 0 else f"gland instance {label}"
    raise ValueError(f"No verified label description for {dataset}: {label}")


def legend_rows(mask, row, labels):
    values, counts = np.unique(mask, return_counts=True)
    valid_count = int(np.count_nonzero(~np.isin(mask, row["ignore_labels"])))
    result = []
    for value, count in zip(values, counts):
        value, count = int(value), int(count)
        ignored = value in row["ignore_labels"]
        # Tissue colors deliberately encode different ontologies; IDs 3-7 differ.
        palette_index = ({3: 20, 4: 13, 5: 4, 6: 3}.get(value, value)
                         if row["dataset"] == "tiger" else value)
        if row["dataset"] == "wsss4luad":
            palette_index = {0: 1, 1: 2, 2: 6, 3: 0}[value]
        result.append({"value": value, "name": label_name(row["dataset"], value, labels),
                       "color": COLORS[palette_index % len(COLORS)], "ignored": ignored,
                       "pixels": count, "percent_of_image": 100 * count / mask.size,
                       "percent_of_included_pixels": None if ignored or not valid_count else 100 * count / valid_count})
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--examples", type=Path, required=True)
    a = p.parse_args(argv)
    resolution = json.loads((HERE / "resolution.json").read_text())
    labels = json.loads((HERE / "labels.json").read_text())
    rows = json.loads((a.examples / "examples.json").read_text())
    viewer = []
    for row in rows:
        image, mask_path = a.examples / row["image"], a.examples / row["mask"]
        if digest(image) != row["image_sha256"] or digest(mask_path) != row["mask_sha256"]:
            raise ValueError("Example bytes changed; regenerate the source copies")
        with Image.open(image) as im:
            width, height = im.size
        with Image.open(mask_path) as im:
            mask = np.asarray(im)
        if mask.shape != (height, width):
            raise ValueError("Example image/mask mismatch")
        row.update(width=width, height=height, resolution=resolution["datasets"][row["dataset"]])
        row["mpp_um"] = row["resolution"]["mpp_um"]
        row["annotation_provenance"] = row["resolution"]["annotation"]
        row["legend"] = legend_rows(mask, row, labels)
        row["field_of_view_um"] = ([width * row["mpp_um"], height * row["mpp_um"]]
                                    if row["mpp_um"] is not None else None)
        # Sampling is for browser display only; original files and measurements remain untouched.
        factor = min(1.0, 1000 / max(width, height))
        shape = (max(1, round(width * factor)), max(1, round(height * factor)))
        small = np.asarray(Image.fromarray(mask.astype(np.uint16)).resize(shape, Image.Resampling.NEAREST), dtype="<u2")
        item = {**row, "preview_width": shape[0], "preview_height": shape[1],
                "grid": base64.b64encode(small.tobytes()).decode("ascii")}
        viewer.append(item)
        (image.parent / (image.stem.replace("-image", "") + "-metadata.json")).write_text(json.dumps(row, indent=2) + "\n")
    (a.examples / "examples.json").write_text(json.dumps(rows, indent=2) + "\n")
    payload = json.dumps(viewer).replace("<", "\\u003c")
    html = HTML.replace("__CASES__", payload)
    (a.examples / "viewer.html").write_text(html)
    lines = ["# Dataset examples", "", "[Open the tissue annotation viewer](viewer.html)", "",
             "Original RGBs and raw annotation masks are retained without resizing. The viewer samples masks for display only.",
             "Colours are a visualization of supplied annotations, not Bora predictions. Hover for a label; use the opacity control to see tissue.",
             "", "Pixel spacing describes sampling in micrometres per pixel, not optical resolving power or screen zoom.",
             "Unknown spacing and unverified semantic mappings are identified explicitly; magnification alone does not calibrate pixels.", "",
             "| Dataset | Example | Pixels (W × H) | µm/pixel | Resolution evidence | Details |", "|---|---|---|---|---|---|"]
    for row in rows:
        key = Path(row["image"]).stem.replace("-image", "")
        details = str(Path(row["image"]).parent / (key + "-metadata.json"))
        spacing = str(row["mpp_um"]) if row["mpp_um"] is not None else "Unknown"
        lines.append(f"| {row['dataset']} | [RGB]({row['image']}) · [mask]({row['mask']}) | {row['width']} × {row['height']} | {spacing} | {row['resolution']['status']} | [legend and provenance]({details}) |")
    lines += ["", "[Resolution and annotation information for all acquired images](all-image-information.csv)"]
    (a.examples / "README.md").write_text("\n".join(lines) + "\n")
    # Cover every acquired image, while preserving the original hashed acquisition inventories.
    fields = ["dataset", "id", "image", "mask", "width", "height", "mpp_um", "resolution_status",
              "resolution_source", "resolution_note", "reported_magnification", "original_release_mpp_um",
              "split", "group_id", "group_level", "source_status", "label_mode", "annotation_provenance", "raw_label_names"]
    count = 0
    with (a.root / "metadata" / "image-information.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for manifest in sorted((a.root / "prepared").glob("*.jsonl")):
            for text in manifest.read_text().splitlines():
                item = json.loads(text)
                info = resolution["datasets"][item["dataset"]]
                writer.writerow({**{k: item[k] for k in ["dataset", "id", "image", "mask", "width", "height", "label_mode"]},
                                 "mpp_um": info["mpp_um"], "resolution_status": info["status"],
                                 "resolution_note": info["note"],
                                 "reported_magnification": info.get("equivalent_magnification") or info.get("table_A1_magnification"),
                                 **{k: item.get(k, "") for k in ("split", "group_id", "group_level", "source_status")},
                                 "resolution_source": info["source"], "original_release_mpp_um": info.get("original_release_mpp_um"),
                                 "annotation_provenance": info["annotation"],
                                 "raw_label_names": "; ".join(f"{v}={label_name(item['dataset'], v, labels)}" for v in item["raw_labels"])})
                count += 1
    (a.root / "metadata" / "resolution-catalog.json").write_text(json.dumps(resolution, indent=2) + "\n")
    shutil.copy2(a.root / "metadata" / "image-information.csv", a.examples / "all-image-information.csv")
    print(f"Described {count} acquired image/mask pairs and {len(rows)} local examples; no Bora execution")


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Histopathology annotation examples</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f4ef;color:#172d35;font:15px/1.55 system-ui,sans-serif}main{max-width:1450px;margin:auto;padding:28px}
h1{font-size:29px;line-height:1.2;margin:0 0 10px}p{margin:10px 0}select,input{font:inherit}select{max-width:100%;padding:9px;background:white;border:1px solid #99a8ac;border-radius:5px}
.bar{display:flex;gap:22px;flex-wrap:wrap;align-items:center;margin:22px 0}.layout{display:grid;grid-template-columns:minmax(0,1.8fr) minmax(300px,1fr);gap:24px}
.image{position:relative;background:white;align-self:start;line-height:0;border:1px solid #bec8c9}.image img{width:100%;height:auto;display:block}.image canvas{position:absolute;inset:0;width:100%;height:100%}
aside{background:white;padding:20px;border:1px solid #d7dfdc;border-radius:6px;overflow-wrap:anywhere}h2{font-size:18px;margin:8px 0}.tag{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:#52736e}
.legend{display:grid;grid-template-columns:16px minmax(0,1fr) auto;gap:8px;margin:9px 0;align-items:center}.swatch{width:14px;height:14px;border:1px solid #a2abaa;border-radius:2px}.muted{color:#5d6a70;font-size:13px}.status{font-weight:600;min-height:27px}
a{color:#17695b}details{margin-top:16px}.legendbox{max-height:380px;overflow:auto}code{font-size:12px}@media(max-width:800px){.layout{grid-template-columns:1fr}main{padding:16px}}
</style><main><div class="tag">Supplied ground-truth annotations · No Bora inference</div><h1>Histopathology example explorer</h1>
<p>Overlay colours identify annotated tissue classes or individual gland IDs. Gland masks do not label tumor versus stroma.</p>
<div class="bar"><select id="choose" aria-label="Example"></select><label>Mask opacity <input id="alpha" type="range" min="0" max="1" step=".05" value=".45"></label></div>
<div class="layout"><section><div class="image"><img id="rgb" alt="Original histopathology image"><canvas id="mask"></canvas></div><p id="hover" class="status">Move over the image to inspect a label.</p>
<p class="muted">Hover reads a nearest-neighbour display sample. Use the original mask for native-pixel coordinates. Percentages below are counted from the complete original mask.</p></section>
<aside><div class="tag" id="dataset"></div><h2>Resolution and provenance</h2><p id="geometry"></p><p id="scale"></p><p id="evidence" class="muted"></p><p id="annotation"></p><p id="grade"></p>
<h2>Annotation legend</h2><p class="muted">Percent of all image pixels; ignored areas are identified explicitly.</p><div class="legendbox" id="legend"></div><details><summary>Original sample and files</summary><p id="sample" class="muted"></p><p><a id="rawimage">Original RGB</a> · <a id="rawmask">Raw mask</a></p></details></aside></div></main>
<script id="data" type="application/json">__CASES__</script><script>
const cases=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id);let current,grid,lookup;
cases.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${c.dataset} / ${c.image.split('/').pop()} — ${c.width} × ${c.height}`;$('choose').append(o)});
function show(){current=cases[Number($('choose').value)];const c=current,raw=atob(c.grid),buf=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)buf[i]=raw.charCodeAt(i);grid=new Uint16Array(buf.buffer);lookup=new Map(c.legend.map(l=>[l.value,l]));
$('rgb').src=c.image;$('dataset').textContent=c.dataset;$('geometry').textContent=`${c.width} × ${c.height} native pixels`;
$('scale').textContent=c.mpp_um===null?`Actual pixel spacing unknown.${c.resolution.original_release_mpp_um?` Original release reference: ${c.resolution.original_release_mpp_um} µm/pixel; unverified for these files.`:''}`:`${c.mpp_um} µm/pixel · field of view approximately ${c.field_of_view_um[0].toFixed(1)} × ${c.field_of_view_um[1].toFixed(1)} µm`;
$('evidence').textContent=`${c.resolution.status}: ${c.resolution.note}`;$('annotation').textContent=c.annotation_provenance;$('grade').textContent=c.image_level_grade?`Image-level grade: ${c.image_level_grade}. This is not a pixel-wise tumor label.`:'';
$('sample').textContent=c.sample_id;$('rawimage').href=c.image;$('rawmask').href=c.mask;$('legend').replaceChildren();
for(const l of c.legend){const r=document.createElement('div');r.className='legend';const s=document.createElement('span');s.className='swatch';s.style.background=l.color;const n=document.createElement('span');n.textContent=`${l.value}: ${l.name}${l.ignored?' (ignored)':''}`;const v=document.createElement('span');v.textContent=`${l.percent_of_image.toFixed(1)}%`;r.append(s,n,v);$('legend').append(r)}
const canvas=$('mask');canvas.width=c.preview_width;canvas.height=c.preview_height;const ctx=canvas.getContext('2d'),pixels=ctx.createImageData(canvas.width,canvas.height);
for(let i=0;i<grid.length;i++){const l=lookup.get(grid[i]),color=l.color;for(let k=0;k<3;k++)pixels.data[i*4+k]=parseInt(color.slice(1+2*k,3+2*k),16);pixels.data[i*4+3]=l.ignored?70:255}ctx.putImageData(pixels,0,0);canvas.style.opacity=$('alpha').value;$('hover').textContent='Move over the image to inspect a label.'}
$('choose').onchange=show;$('alpha').oninput=()=>{$('mask').style.opacity=$('alpha').value};$('mask').onmousemove=e=>{const b=e.target.getBoundingClientRect(),x=Math.min(current.preview_width-1,Math.max(0,Math.floor((e.clientX-b.left)/b.width*current.preview_width))),y=Math.min(current.preview_height-1,Math.max(0,Math.floor((e.clientY-b.top)/b.height*current.preview_height))),v=grid[y*current.preview_width+x];$('hover').textContent=`Annotation ${v}: ${lookup.get(v).name}`};show();
</script></html>'''


if __name__ == "__main__":
    main()
