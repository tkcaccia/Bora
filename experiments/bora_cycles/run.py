"""Run a frozen cycle or an explicitly incomplete engineering pilot.

Predictions finish and are hashed before this process imports the scorer.
Selection and cycle advancement are separate review-gated operations.
"""
import argparse
import hashlib
from importlib import metadata
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
from threadpoolctl import threadpool_info, threadpool_limits
from bora.evidence import predict_evidence, resize
from .methods import predict
from .registry import (candidates, roster, CHILDREN_TWO, CYCLE_TWO_SURVIVORS,
                       CHILDREN_THREE, CYCLE_THREE_SURVIVORS,
                       CHILDREN_FOUR, CYCLE_FOUR_SURVIVORS,
                       CHILDREN_FIVE, CYCLE_FIVE_SURVIVORS,
                       CHILDREN_SIX, CYCLE_SIX_SURVIVORS,
                       CHILDREN_SEVEN, CYCLE_SEVEN_SURVIVORS,
                       CHILDREN_EIGHT, CYCLE_EIGHT_SURVIVORS,
                       CHILDREN_NINE, CYCLE_NINE_SURVIVORS,
                       CHILDREN_TEN, CYCLE_TEN_SURVIVORS,
                       CHILDREN_ELEVEN, CYCLE_ELEVEN_SURVIVORS,
                       CHILDREN_TWELVE, CYCLE_TWELVE_SURVIVORS)
from .diagnostics import novelty, white_regions, mask_diagnostics,rare_colors
from .runtime_policy import active_policy, runtime_ranking
from .benchmark_scope import load_scope, validate_scope, requires_parent_parity
from experiments.bora_extension.task_adapter import geometry, report_units


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
    folder = Path(__file__).parent
    files = list(folder.glob("*.py"))+list(folder.glob("*.sh"))+list(folder.glob("*.md")) + [
        Path(__file__).with_name("protocol.json"),
        repo/"src/bora/evidence.py", repo/"cpp/evidence/kernels.cpp",
        repo/"cpp/evidence/libbora_evidence.so",
        repo/"experiments/bora_extension/task_adapter.py",
        repo/"experiments/bora_extension/calibrated_cohort.py",
        repo/"experiments/bora_extension/eligibility.py",
        repo/"experiments/bora_extension/eligibility_policy.json",
        repo/"experiments/bora_extension/calibration.py",
        repo/"experiments/bora_extension/CALIBRATION_POLICY.json"]
    return {str(p.relative_to(repo)): digest(p) for p in files}


def environment_record():
    """Distinguish identical package versions built against different BLAS."""
    from . import native
    packages = {}
    for name in ("numpy", "scipy", "opencv-python", "opencv-python-headless",
                 "Pillow", "tifffile", "threadpoolctl"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    cpp_openmp = bool(native.library().bora_evidence_openmp())
    pools = threadpool_info()
    libraries = {entry["filepath"]: digest(entry["filepath"])
                 for entry in pools if entry.get("filepath") and Path(entry["filepath"]).is_file()}
    return dict(packages=packages, numpy_build=getattr(np.__config__, "CONFIG", None),
        threadpools=pools, loaded_numerical_library_sha256=libraries,
        opencv_version=cv2.__version__,
        opencv_build_sha256=hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
        cpp_openmp_available=cpp_openmp,
        note="Requested numerical thread limits do not prove that every vendor runtime honors them; loaded library identity is explicit.")


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
    constant_input = bool(record.get("single_eligible_class", False))
    completed = tifffile.imread(src/"coarse.completed.tif") if constant_input else None
    support = tifffile.imread(src/"tissue.working.tif") > 0
    native_support = tifffile.imread(src/"tissue.native.tif") > 0
    protocol = json.loads((root/"PROTOCOL.json").read_text())
    from .eleven_manifest import SUPPORT_METHODS
    owned_keys=set(keys)&SUPPORT_METHODS if protocol.get('cycle',0)>=11 else set()
    if protocol.get('cycle',0)>=12:
        from .twelve_manifest import NATIVE_CHILDREN
        owned_keys|=set(keys)&NATIVE_CHILDREN
    native_he=(np.array(Image.open(src/'original.native.png').convert('RGB')) if owned_keys else None)
    lineage = protocol.get("lineage")
    coordinates = geometry(sample)
    spacing = coordinates["algorithm_spacing_xy"]
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
                    candidate_support=native_support
                    if key in owned_keys:
                        if key.startswith('Q'):
                            from .cycle_twelve import predict_native
                        else:
                            from .cycle_eleven import predict_native
                        native,candidate_support,details=predict_native(key,rgb,coarse,support,spacing,
                            sample['palette'],he,native_he,geometry(sample,native=True)['algorithm_spacing_xy'])
                    elif constant_input:
                        from .expanded_prediction import constant_evidence_fallback
                        analysis, details = constant_evidence_fallback(
                            key, rgb, coarse, completed, support, spacing, he)
                    elif key in ("B21", "B22"):
                        analysis, details = predict_evidence(
                            rgb, coarse, support, spacing,
                            method="linear" if key == "B21" else "simplex",
                            palette=sample["palette"])
                    else:
                        analysis, details = predict(
                            key, rgb, coarse, support, spacing,
                            palette=sample["palette"], he=he)
                    if key not in owned_keys:
                        native = resize(analysis, sample["native_shape_yx"], True)
                        native[~native_support] = 0
                    seconds.append(time.perf_counter()-started)
                    if previous is not None and not np.array_equal(previous, native):
                        raise RuntimeError("Non-deterministic candidate")
                    previous = native
                assert native.shape == tuple(sample["native_shape_yx"])
                if key not in ("C20", "D12", "F03", "G03", "H03", "J03", "L03", "M10", "M11", "M12", "N10", "N11", "N12"):
                    assert np.all(native[candidate_support] > 0)
                assert np.all(native[~candidate_support] == 0)
                assert set(np.unique(native)) <= set(np.unique(coarse)) | {0}
                if lineage and key in lineage["unchanged_survivors"] and requires_parent_parity(protocol, sample["case"]):
                    if protocol['cycle']>=11:
                        previous_record=json.loads((Path(lineage['previous'])/'cases'/sample['case']/'PREDICTIONS.json').read_text())
                        if previous_record['input_manifest_sha256']!=digest(src/'INPUTS_FROZEN.json'):
                            raise RuntimeError('Parent comparison requires identical frozen case inputs')
                    old_path = Path(lineage["previous"])/"cases"/sample["case"]/key/"native.tif"
                    if not np.array_equal(native, tifffile.imread(old_path)):
                        raise RuntimeError("An unchanged survivor changed its prediction")
                    details["unchanged_survivor_native_parity"] = True
                    details["previous_prediction_sha256"] = digest(old_path)
                if protocol.get("benchmark_scope"):
                    details = report_units(details, coordinates["units"])
                    details["coordinate_geometry"] = coordinates
                tifffile.imwrite(target/"native.tif", native, compression="deflate")
                files[key+"/native.tif"] = digest(target/"native.tif")
                if key in owned_keys:
                    tifffile.imwrite(target/'support.native.tif',candidate_support.astype(np.uint8),compression='deflate')
                    files[key+'/support.native.tif']=digest(target/'support.native.tif')
                    details['native_support']['saved_support_sha256']=files[key+'/support.native.tif']
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
            rgb, coarse, support, spacing)
        if protocol.get("benchmark_scope"):
            detail = report_units(detail, coordinates["units"])
        detail["seconds"] = time.perf_counter()-started
        np.savez_compressed(out/"NOVELTY.npz", distance=distance, regions=regions)
        save(out/"NOVELTY.json", detail)
        for name in ("NOVELTY.npz", "NOVELTY.json"):
            files[name] = digest(out/name)
        if protocol.get('cycle',0)>=11:
            started=time.perf_counter()
            diagnostic_support=support|~white_regions(he)[0]
            rarity,proposals,report=rare_colors(rgb,diagnostic_support,spacing)
            report.update(seconds=time.perf_counter()-started,
                support_rule='Inherited tissue support plus every original-H&E nonwhite working pixel',
                diagnostic_only=True,affects_candidate_labels=False)
            np.savez_compressed(out/'RARITY.npz',score=rarity,regions=proposals)
            save(out/'RARITY.json',report)
            for name in ('RARITY.npz','RARITY.json'):files[name]=digest(out/name)
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
    if len(args) == 7:
        from .expanded_metrics import score_case as expanded_score
        return expanded_score(args)
    from .metrics import RegionEvaluator, discovery_score
    source, root, sample, keys, manifests, metric_backend = args
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
                entry["score"] = evaluator.score(mask, backend=metric_backend)
                entry["background"] = mask_diagnostics(
                    mask, support, white, exterior, sample["native_mpp_xy"], backend=metric_backend)
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


def aggregate(rows, keys, runtime_penalty=None):
    if rows and "coordinate_units" in rows[0]:
        from .expanded_metrics import aggregate as expanded_aggregate
        return expanded_aggregate(rows, keys, runtime_penalty)
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
    if runtime_penalty is not None:
        return runtime_ranking(dict(cases=rows, ranking=ranking), runtime_penalty)
    return sorted(ranking, key=lambda r: (not r["complete"], -(r["mean_iou"] or 0), r["method"]))


def palette_image(labels, colors):
    image = Image.fromarray(np.asarray(labels, np.uint8), "P")
    palette = np.zeros((256, 3), np.uint8)
    palette[0] = [245, 245, 245]
    palette[1:len(colors)+1] = colors
    image.putpalette(palette.ravel().tolist())
    return image.convert("RGB")


def palette_bytes(colors):
    table = np.zeros((256, 3), np.uint8)
    table[0] = [245, 245, 245]
    table[1:len(colors)+1] = colors
    return table.ravel().tolist()


def render_case(args):
    source, root, row, keys = args
    source, root = Path(source), Path(root)
    case = row["case"]
    src, out = source/"cases"/case, root/"cases"/case
    colors = row["sample"]["palette"]
    original = Image.open(src/"original.native.png").convert("RGB")
    truth = palette_image(tifffile.imread(src/"reference.native.tif"), colors)
    coarse = resize(tifffile.imread(src/"coarse.tif"), original.size[::-1], True)
    panels = [("Original H&E", "original-he.jpg", original),
              ("Reference", "reference.jpg", truth),
              ("Original coarse mask", "coarse.jpg", palette_image(coarse, colors)),
              ("Fixed converted RGB", "converted.jpg",
               Image.open(src/"ensemble/representation.png").convert("RGB"))]
    tiles = []
    for title, name, panel in panels:
        panel.save(out/name, quality=90)
        preview = panel.copy()
        preview.thumbnail((372, 292))
        tiles.append((title, name, preview))
    for key in keys:
        entry = row["methods"][key]
        if entry["status"] == "ok":
            indexed = Image.fromarray(tifffile.imread(out/key/"native.tif"), "P")
            indexed.putpalette(palette_bytes(colors))
            indexed.save(out/key/"mask.png", compress_level=1)
            panel = indexed.convert("RGB")
            title = (f'{key}: IoU {entry["score"]["macro_iou"]:.4f}, '
                     f'{entry["median_seconds"]:.3f}s')
            panel.thumbnail((372, 292))
            tiles.append((title, key+"/mask.png", panel))
        else:
            panel = Image.new("RGB", original.size, "#fff0f0")
            ImageDraw.Draw(panel).text((20, 20), "FAILED: "+entry["exception"][:130], fill="black")
            panel.save(out/key/"failed.jpg")
            panel.thumbnail((372, 292))
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
    section = (
        f'<section id="{case}"><h2>{case}</h2><p>{row["dataset"]}</p>'
        f'<a href="cases/{case}/contact.jpg"><img class="sheet" loading="lazy" '
        f'src="cases/{case}/contact.jpg"></a><div class="panels">'+
        "".join(f'<figure><figcaption>{title}</figcaption><a href="cases/{case}/{name}">'
                f'<img loading="lazy" src="cases/{case}/{name}"></a></figure>'
                for title, name, _ in tiles)+"</div></section>")
    print(json.dumps(dict(rendered=case)), flush=True)
    return case, digest(out/"contact.jpg"), section


def publish(source, root, result, workers=1):
    keys = result["protocol"]["methods"]
    tasks = [(source, root, row, keys) for row in result["cases"]]
    if workers == 1:
        rendered = [render_case(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            rendered = list(pool.map(render_case, tasks))
    sections = [section for _, _, section in rendered]
    manifest = {case: sha for case, sha, _ in rendered}
    save(root/"VISUAL_MANIFEST.json", dict(contact_sha256=manifest))
    if not (root/"VISUAL_REVIEW.json").exists():
        save(root/"VISUAL_REVIEW.json", dict(reviewed={}, required_cases=list(manifest),
                                              automated_checks_are_not_visual_review=True))
    penalized = active_policy(result["protocol"]) is not None
    ranking = "".join(
        f'<tr><td>{r["method"]}</td><td>{r["mean_iou"]}</td>'
        f'<td>{r.get("mean_seconds")}</td>'
        +(f'<td>{r.get("images_over_one_second")}</td>'
          f'<td>{r.get("mean_runtime_penalty")}</td>'
          f'<td>{r.get("runtime_adjusted_score")}</td>' if penalized else '')
        +f'<td>{r.get("small_region_recall")}</td></tr>'
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
            +('<p>Primary selection utility: mean over images of '
              'IoU − 0.05 × max(seconds − 1, 0). Raw IoU is reported separately.</p>'
              if penalized else '')+
            '<table><tr><th>Method</th><th>Mean IoU</th><th>Seconds</th>'
            +('<th>Images &gt;1s</th><th>Mean penalty</th><th>Adjusted score</th>' if penalized else '')+
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
    parser.add_argument("--cycle", type=int, choices=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), default=1)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--metric-backend", choices=("numpy", "cpp"), default="cpp")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("At least one timed prediction required")
    plan = json.loads((args.source/"PLAN.json").read_text())
    active = candidates(args.cycle)
    lineage = None
    if args.cycle >= 2:
        # A completed previous inference job is insufficient: actual all-case
        # visual review and a signed five-survivor selection are mandatory.
        from .review import require_review, validate_children
        if args.previous is None:
            parser.error("A new cycle requires the reviewed previous result")
        previous = require_review(args.previous)
        selection = json.loads((args.previous/"SELECTION.json").read_text())
        survivors, children = {2: (CYCLE_TWO_SURVIVORS, CHILDREN_TWO),
                               3: (CYCLE_THREE_SURVIVORS, CHILDREN_THREE),
                               4: (CYCLE_FOUR_SURVIVORS, CHILDREN_FOUR),
                               5: (CYCLE_FIVE_SURVIVORS, CHILDREN_FIVE),
                               6: (CYCLE_SIX_SURVIVORS, CHILDREN_SIX),
                               7: (CYCLE_SEVEN_SURVIVORS, CHILDREN_SEVEN),
                               8: (CYCLE_EIGHT_SURVIVORS, CHILDREN_EIGHT),
                               9: (CYCLE_NINE_SURVIVORS, CHILDREN_NINE),
                               10: (CYCLE_TEN_SURVIVORS, CHILDREN_TEN),
                               11: (CYCLE_ELEVEN_SURVIVORS, CHILDREN_ELEVEN),
                               12: (CYCLE_TWELVE_SURVIVORS, CHILDREN_TWELVE)}[args.cycle]
        if previous["cycle"] != args.cycle-1 or selection["survivors"] != list(survivors):
            parser.error("Implemented children do not match the actual reviewed survivors")
        if digest(args.previous/"VISUAL_REVIEW.json") != selection["review_sha256"]:
            parser.error("Previous visual review changed after selection")
        if digest(args.previous/"RESULTS.json") != selection["results_sha256"]:
            parser.error("Previous scored evidence changed after selection")
        validate_children(selection, children)
        lineage = dict(previous=str(args.previous), selection_sha256=digest(args.previous/"SELECTION.json"),
                       unchanged_survivors=list(survivors), children=children)
    controls=['B21','B22']+(['B23'] if args.cycle>=11 else [])
    keys = args.keys or controls+list(active)
    available = {key for key, value in active.items() if value.implemented} | set(controls)
    if set(keys)-available:
        parser.error("Requested candidate is not implemented: "+str(set(keys)-available))
    samples = plan["samples"]
    if args.cases:
        if not args.pilot:
            parser.error("A full cycle must include every case")
        samples = [sample for sample in samples if sample["case"] in args.cases]
        if set(args.cases) != {s["case"] for s in samples}:
            parser.error("Unknown case")
    scope = load_scope(args.source, args.cycle)
    if not args.pilot and (set(keys)-set(controls) != set(active) or (args.cycle>=11 and not set(controls)<=set(keys))):
        parser.error("Full cycle requires all 20 candidates")
    if not samples or len(set(keys)) != len(keys):
        parser.error("Empty scope or duplicate method")
    args.root.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        cycle=args.cycle, pilot=args.pilot, methods=keys, cases=[s["case"] for s in samples],
        roster=roster(args.cycle), lineage=lineage, source=str(args.source), source_hashes=source_record(),
        input_plan_sha256=digest(args.source/"PLAN.json"), repeats=args.repeats,
        workers=args.workers, numerical_threads=2, primary_conversion="ensemble", metric_backend=args.metric_backend,
        study_protocol=json.loads(Path(__file__).with_name("protocol.json").read_text()),
        python=platform.python_version(), host=platform.node(), environment=environment_record())
    if scope is not None:
        protocol["benchmark_scope"] = scope
        protocol["study_protocol"].update(
            case_scope="All349 frozen ROIs; original149 retained exactly plus200 additions, prospective from cycle8",
            reference_policy="Fine masks are read only after all predictions freeze. All349 images are development data. Reference0 is ignored; RINGS scoring-only removed background is valid outside-tumor.",
            small_region_policy="All reference components scored. Separate <=400 um2 and <=400 px2 summaries; unknown spacing never assigned a physical area.",
            degenerate_evidence_policy="Exactly constant single-eligible-class RGB uses the same frozen coarse-completion fallback for every candidate/control, with only explicitly independent H&E glass removal retained. It is not claimed to be refinement.",
            selection="Review every349 image and select five with the unchanged excess-one-second penalty. Report historical149 and new200 separately.")
        if scope['version']=='calibrated-tissues-v1':
            protocol['diagnostic_controls']=controls
            protocol['study_protocol'].update(
                case_scope=f"All{len(scope['required_cases'])} frozen calibrated tissue ROIs, with prior and newly acquired fields reported separately",
                reference_policy='Fine masks are decoded only after all predictions freeze. Reference0 is ignored; binary negative targets use explicit RGB-white-only background semantics.',
                degenerate_evidence_policy='At least two observed groups are required; single-group and uncalibrated cases remain outside the input plan and every score/runtime denominator.',
                support_policy='P10/P11/P12/B23 own a separately frozen native H&E support raster; all other methods retain inherited support. Every annotated reference pixel is scored.',
                selection='Review every admitted image and all23 outputs, then select five of the20 candidates using the fixed excess-one-second penalty. Controls cannot survive.')
            if args.cycle>=12:
                protocol['study_protocol']['support_policy']='P11/B23 and Q01/Q02/Q03 own frozen native H&E support; other methods retain common support. Only Q01 may remove original nonwhite pixels under its registered affine blank model. Every annotated reference pixel still counts.'
    validate_scope(protocol, pilot=args.pilot)
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
            (args.source, args.root, sample, keys, frozen["manifests"], args.metric_backend)
            + ((scope,) if scope is not None else ()) for sample in samples]))
    result = dict(protocol=protocol, cases=rows,
                  ranking=aggregate(rows, keys, active_policy(protocol)))
    save(args.root/"RESULTS.json", result)
    publish(args.source, args.root, result, workers=args.workers)
    save(args.root/"STAGE.json", dict(
        state="pilot_complete" if args.pilot else "awaiting_visual_review",
        cases=len(rows), method_count=len(keys), completed_cycles=args.cycle-1,
        complete_goal=False))
    print(json.dumps(dict(state="pilot_complete" if args.pilot else "awaiting_visual_review",
                          ranking=result["ranking"])), flush=True)


if __name__ == "__main__":
    main()
