"""Run a frozen cycle or an explicitly incomplete engineering pilot.

Predictions finish and are hashed before this process imports the scorer.
Selection and cycle advancement are separate review-gated operations.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import tifffile
from threadpoolctl import threadpool_limits
from bora.evidence import predict_evidence, resize
from .methods import predict
from .registry import CYCLE_ONE, roster
from .diagnostics import novelty, white_regions, mask_diagnostics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    os.replace(temporary, path)


def verify(path, files):
    for name, expected in files.items():
        if digest(Path(path)/name) != expected:
            raise RuntimeError(f"Frozen file changed: {Path(path)/name}")


def source_record():
    repo = Path(__file__).resolve().parents[2]
    files = list(Path(__file__).parent.glob("*.py")) + [
        Path(__file__).with_name("protocol.json"),
        repo/"src/bora/evidence.py", repo/"cpp/evidence/kernels.cpp",
        repo/"cpp/evidence/libbora_evidence.so"]
    return {str(p.relative_to(repo)): digest(p) for p in files}


def freeze_case(args):
    source, root, sample, keys, repeats = args
    cv2.setNumThreads(2)
    source, root = Path(source), Path(root)
    src = source/"cases"/sample["case"]
    out = root/"cases"/sample["case"]
    out.mkdir(parents=True, exist_ok=True)
    if (out/"PREDICTIONS.json").exists():
        existing = json.loads((out/"PREDICTIONS.json").read_text())
        verify(out, existing["files_sha256"])
        return sample["case"]
    record = json.loads((src/"INPUTS_FROZEN.json").read_text())
    verify(src, record["files_sha256"])
    rgb = np.array(Image.open(src/"ensemble/representation.png").convert("RGB"))
    he = np.array(Image.open(src/"original.png").convert("RGB"))
    coarse = tifffile.imread(src/"coarse.tif")
    support = tifffile.imread(src/"tissue.working.tif") > 0
    native_support = tifffile.imread(src/"tissue.native.tif") > 0
    records, files = {}, {}
    with threadpool_limits(limits=2):
        for key in keys:
            target = out/key
            target.mkdir(exist_ok=True)
            seconds = []
            try:
                previous = None
                for repeat in range(repeats):
                    started = time.perf_counter()
                    if key in ("B21", "B22"):
                        analysis, details = predict_evidence(
                            rgb, coarse, support, sample["working_mpp_xy"],
                            method="linear" if key == "B21" else "simplex",
                            palette=sample["palette"])
                    else:
                        analysis, details = predict(
                            key, rgb, coarse, support, sample["working_mpp_xy"],
                            palette=sample["palette"], he=he)
                    native = resize(analysis, sample["native_shape_yx"], True)
                    native[~native_support] = 0
                    seconds.append(time.perf_counter()-started)
                    if previous is not None and not np.array_equal(previous, native):
                        raise RuntimeError("Non-deterministic candidate")
                    previous = native
                assert native.shape == tuple(sample["native_shape_yx"])
                if key != "C20":
                    assert np.all(native[native_support] > 0)
                assert np.all(native[~native_support] == 0)
                assert set(np.unique(native)) <= set(np.unique(coarse)) | {0}
                tifffile.imwrite(target/"native.tif", native, compression="deflate")
                files[key+"/native.tif"] = digest(target/"native.tif")
                records[key] = dict(
                    status="ok", seconds=seconds, median_seconds=float(np.median(seconds)),
                    details=details)
            except Exception as exc:
                records[key] = dict(status="failed", exception=repr(exc),
                                    traceback=traceback.format_exc(), seconds=seconds)
            save(target/"DETAILS.json", records[key])
            files[key+"/DETAILS.json"] = digest(target/"DETAILS.json")
        started = time.perf_counter()
        distance, regions, detail = novelty(
            rgb, coarse, support, sample["working_mpp_xy"])
        detail["seconds"] = time.perf_counter()-started
        np.savez_compressed(out/"NOVELTY.npz", distance=distance, regions=regions)
        save(out/"NOVELTY.json", detail)
        for name in ("NOVELTY.npz", "NOVELTY.json"):
            files[name] = digest(out/name)
    save(out/"PREDICTIONS.json", dict(
        case=sample["case"], methods=records, files_sha256=files,
        input_manifest_sha256=digest(src/"INPUTS_FROZEN.json"),
        rgb_sha256=digest(src/"ensemble/representation.png"),
        fine_reference_read=False))
    print(json.dumps(dict(predicted=sample["case"],
                          ok=sum(r["status"] == "ok" for r in records.values()),
                          failed=sum(r["status"] != "ok" for r in records.values()))), flush=True)
    return sample["case"]


def score_case(args):
    from .metrics import RegionEvaluator, discovery_score
    source, root, sample, keys, manifests = args
    cv2.setNumThreads(2)
    src, out = Path(source)/"cases"/sample["case"], Path(root)/"cases"/sample["case"]
    assert digest(out/"PREDICTIONS.json") == manifests[sample["case"]]
    record = json.loads((out/"PREDICTIONS.json").read_text())
    verify(out, record["files_sha256"])
    assert digest(src/"reference.native.tif") == sample["reference_native_sha256"]
    truth = tifffile.imread(src/"reference.native.tif")
    he = np.array(Image.open(src/"original.native.png").convert("RGB"))
    support = tifffile.imread(src/"tissue.native.tif") > 0
    white, exterior, white_detail = white_regions(he)
    evaluator = RegionEvaluator(truth, sample["native_mpp_xy"])
    row = dict(case=sample["case"], dataset=sample["dataset"], sample=sample,
               white=white_detail, methods={})
    with threadpool_limits(limits=2):
        for key in keys:
            entry = dict(record["methods"][key])
            if entry["status"] == "ok":
                mask = tifffile.imread(out/key/"native.tif")
                entry["score"] = evaluator.score(mask)
                entry["background"] = mask_diagnostics(
                    mask, support, white, exterior, sample["native_mpp_xy"])
            row["methods"][key] = entry
    regions = np.load(out/"NOVELTY.npz")["regions"]
    native_regions = resize(regions, truth.shape, True)
    row["novelty"] = discovery_score(native_regions, truth)
    coarse = tifffile.imread(src/"coarse.tif")
    represented = set(np.unique(coarse))-{0}
    row["reference_classes_absent_from_coarse"] = sorted(
        int(g) for g in evaluator.ids if g not in represented)
    save(out/"SCORES.json", row)
    print(json.dumps(dict(scored=sample["case"])), flush=True)
    return row


def aggregate(rows, keys):
    ranking = []
    for key in keys:
        entries = [row["methods"][key] for row in rows]
        failed = [row["case"] for row, entry in zip(rows, entries) if entry["status"] != "ok"]
        if failed:
            ranking.append(dict(method=key, complete=False, failed_cases=failed,
                                cases=len(rows), mean_iou=None))
            continue
        small = [r["score"]["regions"]["small_le_400_um2"]["recall_at_50pct"]
                 for r in entries]
        small = [x for x in small if x is not None]
        ranking.append(dict(
            method=key, complete=True, cases=len(rows),
            mean_iou=float(np.mean([r["score"]["macro_iou"] for r in entries])),
            mean_dice=float(np.mean([r["score"]["macro_dice"] for r in entries])),
            mean_seconds=float(np.mean([r["median_seconds"] for r in entries])),
            small_region_recall=float(np.mean(small)) if small else None,
            small_region_cases=len(small),
            completely_missed_small_regions=sum(
                r["score"]["regions"]["small_le_400_um2"]["completely_missed"]
                for r in entries),
            per_dataset={
                dataset: dict(cases=sum(r["dataset"] == dataset for r in rows),
                              mean_iou=float(np.mean([
                                  r["methods"][key]["score"]["macro_iou"]
                                  for r in rows if r["dataset"] == dataset])))
                for dataset in sorted({r["dataset"] for r in rows})}))
    return sorted(ranking, key=lambda r: (not r["complete"], -(r["mean_iou"] or 0), r["method"]))


def palette_image(labels, colors):
    image = Image.fromarray(np.asarray(labels, np.uint8), "P")
    palette = np.zeros((256, 3), np.uint8)
    palette[0] = [245, 245, 245]
    palette[1:len(colors)+1] = colors
    image.putpalette(palette.ravel().tolist())
    return image.convert("RGB")


def publish(source, root, result):
    keys = result["protocol"]["methods"]
    sections = []
    manifest = {}
    for row in result["cases"]:
        case = row["case"]
        src, out = source/"cases"/case, root/"cases"/case
        colors = row["sample"]["palette"]
        original = Image.open(src/"original.native.png").convert("RGB")
        truth = palette_image(tifffile.imread(src/"reference.native.tif"), colors)
        coarse = resize(tifffile.imread(src/"coarse.tif"), original.size[::-1], True)
        panels = [("Original H&E", original), ("Reference", truth),
                  ("Original coarse mask", palette_image(coarse, colors)),
                  ("Fixed converted RGB", Image.open(src/"ensemble/representation.png").convert("RGB"))]
        tiles = []
        for title, panel in panels:
            name = title.split()[0].lower()+".jpg"
            panel.save(out/name, quality=90)
            tiles.append((title, name, panel))
        for key in keys:
            entry = row["methods"][key]
            if entry["status"] == "ok":
                panel = palette_image(tifffile.imread(out/key/"native.tif"), colors)
                panel.save(out/key/"mask.png")
                title = (f'{key}: IoU {entry["score"]["macro_iou"]:.4f}, '
                         f'{entry["median_seconds"]:.3f}s')
                tiles.append((title, key+"/mask.png", panel))
            else:
                panel = Image.new("RGB", original.size, "#fff0f0")
                ImageDraw.Draw(panel).text((20, 20), "FAILED: "+entry["exception"][:130], fill="black")
                panel.save(out/key/"failed.jpg")
                tiles.append((key+": FAILED", key+"/failed.jpg", panel))
        columns, width, height = 6, 384, 330
        sheet = Image.new("RGB", (columns*width, ((len(tiles)+columns-1)//columns)*height), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (title, _, panel) in enumerate(tiles):
            x, y = (i % columns)*width, (i//columns)*height
            thumb = panel.copy()
            thumb.thumbnail((width-12, height-38))
            sheet.paste(thumb, (x+(width-thumb.width)//2, y+30))
            draw.text((x+6, y+5), title, fill="black")
        sheet.save(out/"contact.jpg", quality=93)
        manifest[case] = digest(out/"contact.jpg")
        sections.append(
            f'<section id="{case}"><h2>{case}</h2><p>{row["dataset"]}</p>'
            f'<a href="cases/{case}/contact.jpg"><img class="sheet" loading="lazy" '
            f'src="cases/{case}/contact.jpg"></a><div class="panels">'+
            "".join(f'<figure><figcaption>{title}</figcaption><a href="cases/{case}/{name}">'
                    f'<img loading="lazy" src="cases/{case}/{name}"></a></figure>'
                    for title, name, _ in tiles)+"</div></section>")
    save(root/"VISUAL_MANIFEST.json", dict(contact_sha256=manifest))
    if not (root/"VISUAL_REVIEW.json").exists():
        save(root/"VISUAL_REVIEW.json", dict(reviewed={}, required_cases=list(manifest),
                                              automated_checks_are_not_visual_review=True))
    ranking = "".join(
        f'<tr><td>{r["method"]}</td><td>{r["mean_iou"]}</td>'
        f'<td>{r.get("mean_seconds")}</td><td>{r.get("small_region_recall")}</td></tr>'
        for r in result["ranking"])
    status = "ENGINEERING PILOT — NOT A COMPLETED CYCLE" if result["protocol"]["pilot"] else "CYCLE AWAITING VISUAL REVIEW"
    html = ('<!doctype html><meta charset="utf-8"><title>Bora improvement cycles</title>'
            '<style>body{font:15px system-ui;background:#eef2f4;margin:20px;color:#183040}'
            '.status{background:#ffe2a1;padding:16px}.sheet{max-width:100%;width:1500px}'
            'section{background:white;margin:24px 0;padding:18px}.panels{display:grid;'
            'grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:12px}'
            'figure{margin:0}figure img{width:100%}table td{padding:6px;border-bottom:1px solid #ccc}'
            '</style><h1>Bora: twenty improvement cycles</h1>'
            f'<p class="status">{status}</p><p>{len(result["cases"])} ROIs · fixed 9+19 RGB · '
            f'{len(keys)} implementations, including controls.</p>'
            '<p>All cases are development data in this repeated-selection study. '
            'Click any image for its full raster. Class-0 reference areas are unscored.</p>'
            '<table><tr><th>Method</th><th>Mean IoU</th><th>Seconds</th>'
            '<th>Small-region recall</th></tr>'+ranking+'</table>'+
            "".join(sections))
    (root/"gallery.html").write_text(html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--keys", nargs="+")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("At least one timed prediction required")
    plan = json.loads((args.source/"PLAN.json").read_text())
    keys = args.keys or ["B21", "B22"]+list(CYCLE_ONE)
    available = {key for key, value in CYCLE_ONE.items() if value.implemented} | {"B21", "B22"}
    if set(keys)-available:
        parser.error("Requested candidate is not implemented: "+str(set(keys)-available))
    samples = plan["samples"]
    if args.cases:
        if not args.pilot:
            parser.error("A full cycle must include every case")
        samples = [sample for sample in samples if sample["case"] in args.cases]
        if set(args.cases) != {s["case"] for s in samples}:
            parser.error("Unknown case")
    if not args.pilot and (len(samples) != 149 or set(keys)-{"B21", "B22"} != set(CYCLE_ONE)):
        parser.error("Full cycle requires all 149 cases and all 20 candidates")
    if not samples or len(set(keys)) != len(keys):
        parser.error("Empty scope or duplicate method")
    args.root.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        cycle=1, pilot=args.pilot, methods=keys, cases=[s["case"] for s in samples],
        roster=roster(), source=str(args.source), source_hashes=source_record(),
        input_plan_sha256=digest(args.source/"PLAN.json"), repeats=args.repeats,
        workers=args.workers, numerical_threads=2, primary_conversion="ensemble",
        study_protocol=json.loads(Path(__file__).with_name("protocol.json").read_text()),
        python=platform.python_version(), host=platform.node())
    if (args.root/"PROTOCOL.json").exists():
        if json.loads((args.root/"PROTOCOL.json").read_text()) != protocol:
            raise RuntimeError("Changed source/protocol requires a fresh output folder")
    else:
        save(args.root/"PROTOCOL.json", protocol)
        repo = Path(__file__).resolve().parents[2]
        for name in protocol["source_hashes"]:
            target = args.root/"runtime_sources"/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo/name, target)
    frozen_path = args.root/"ALL_PREDICTIONS_FROZEN.json"
    if not frozen_path.exists():
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            list(pool.map(freeze_case, [
                (args.source, args.root, sample, keys, args.repeats) for sample in samples]))
        save(frozen_path, dict(
            protocol_sha256=digest(args.root/"PROTOCOL.json"), fine_reference_read=False,
            manifests={s["case"]: digest(args.root/"cases"/s["case"]/"PREDICTIONS.json")
                       for s in samples}))
    frozen = json.loads(frozen_path.read_text())
    if frozen["protocol_sha256"] != digest(args.root/"PROTOCOL.json"):
        raise RuntimeError("Protocol changed after prediction")
    # Check every prediction before any worker opens any fine reference.
    for sample in samples:
        out = args.root/"cases"/sample["case"]
        assert digest(out/"PREDICTIONS.json") == frozen["manifests"][sample["case"]]
        verify(out, json.loads((out/"PREDICTIONS.json").read_text())["files_sha256"])
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        rows = list(pool.map(score_case, [
            (args.source, args.root, sample, keys, frozen["manifests"]) for sample in samples]))
    result = dict(protocol=protocol, cases=rows, ranking=aggregate(rows, keys))
    save(args.root/"RESULTS.json", result)
    publish(args.source, args.root, result)
    save(args.root/"STAGE.json", dict(
        state="pilot_complete" if args.pilot else "awaiting_visual_review",
        cases=len(rows), method_count=len(keys), completed_cycles=0,
        complete_goal=False))
    print(json.dumps(dict(state="pilot_complete" if args.pilot else "awaiting_visual_review",
                          ranking=result["ranking"])), flush=True)


if __name__ == "__main__":
    main()
