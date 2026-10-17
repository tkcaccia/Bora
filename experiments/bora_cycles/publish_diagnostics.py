"""Supplement frozen predictions with annotation-free rarity visualization.

This publication step never overwrites predictions, original novelty files,
their hashes, scores or timings. Reference data enter only proposal evaluation.
"""
import argparse
from html import escape
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import tifffile
from scipy import ndimage as ndi
from bora.evidence import resize
from .diagnostics import rare_colors
from .metrics import RegionEvaluator
from .run import digest, save, palette_image


def proposal_metrics(selected, truth, coarse_ids, mpp_xy):
    evaluator = RegionEvaluator(truth, mpp_xy)
    coverage = np.bincount(evaluator.components.ravel(), weights=selected.ravel(),
                           minlength=len(evaluator.area_pixels))
    coverage /= np.maximum(evaluator.area_pixels, 1)
    small = (evaluator.area_um2 > 0) & (evaluator.area_um2 <= 400)
    small[0] = False
    absent = ~np.isin(evaluator.component_classes, list(coarse_ids))
    absent[0] = False
    return dict(
        small_components=int(small.sum()),
        small_proposal_coverage_recall50=float((coverage[small] >= .5).mean())
        if small.any() else None,
        absent_coarse_class_components=int(absent.sum()),
        absent_coarse_class_proposal_coverage_recall50=float((coverage[absent] >= .5).mean())
        if absent.any() else None,
        interpretation="Class-agnostic proposal coverage; no semantic class assignment or clinical discovery claim")


def overlay(rgb, selected, color):
    image = np.asarray(rgb).copy()
    image[selected] = np.rint(.45*image[selected]+.55*np.asarray(color)).astype(np.uint8)
    return Image.fromarray(image)


def publish(source, root):
    protocol = json.loads((root/"PROTOCOL.json").read_text())
    frozen = json.loads((root/"ALL_PREDICTIONS_FROZEN.json").read_text())
    samples = {s["case"]: s for s in json.loads((source/"PLAN.json").read_text())["samples"]}
    sections, scores, files = [], [], {}
    output = root/"diagnostics-v2"
    output.mkdir(exist_ok=True)
    for case in protocol["cases"]:
        src, out = source/"cases"/case, output/case
        out.mkdir(exist_ok=True)
        record = root/"cases"/case/"PREDICTIONS.json"
        if digest(record) != frozen["manifests"][case]:
            raise RuntimeError("Prediction provenance changed")
        rgb = np.array(Image.open(src/"ensemble/representation.png").convert("RGB"))
        he = np.array(Image.open(src/"original.native.png").convert("RGB"))
        support = tifffile.imread(src/"tissue.working.tif") > 0
        coarse = tifffile.imread(src/"coarse.tif")
        preview = root/"cases"/case
        Image.fromarray(he).save(preview/"original-he.jpg", quality=90)
        native_coarse = resize(coarse, he.shape[:2], True)
        palette_image(native_coarse, samples[case]["palette"]).save(preview/"coarse.jpg", quality=90)
        rarity, rare_regions, detail = rare_colors(rgb, support, samples[case]["working_mpp_xy"])
        novelty_regions = np.load(root/"cases"/case/"NOVELTY.npz")["regions"]
        combined = (rare_regions > 0) | (novelty_regions > 0)
        regions, count = ndi.label(combined, np.ones((3, 3)))
        np.savez_compressed(out/"RARE_COLOURS.npz", rarity=rarity,
                            rare_regions=rare_regions, combined_regions=regions)
        native_rare = resize((rare_regions > 0).astype(np.uint8), he.shape[:2], True) > 0
        native_unseen = resize((novelty_regions > 0).astype(np.uint8), he.shape[:2], True) > 0
        native_combined = native_rare | native_unseen
        image = overlay(he, native_rare, [255, 210, 0])
        image = overlay(image, native_unseen, [0, 235, 255])
        image.save(out/"overlay.jpg", quality=93)
        Image.fromarray((native_combined*255).astype(np.uint8)).save(out/"proposals.png")
        # Ground truth is opened after the annotation-free proposal is saved.
        truth = tifffile.imread(src/"reference.native.tif")
        if digest(src/"reference.native.tif") != samples[case]["reference_native_sha256"]:
            raise RuntimeError("Reference changed")
        score = proposal_metrics(native_combined, truth, set(np.unique(coarse))-{0},
                                 samples[case]["native_mpp_xy"])
        detail.update(combined_regions=count, score=score)
        save(out/"DETAILS.json", detail)
        scores.append(dict(case=case, **score))
        for name in ("RARE_COLOURS.npz", "overlay.jpg", "proposals.png", "DETAILS.json"):
            files[case+"/"+name] = digest(out/name)
        sections.append(
            f'<section><h2>{case}</h2><p>Yellow: rare observed colours. Cyan: colours '
            f'outside known training support. Regions are proposals with unknown biological identity.</p>'
            f'<a href="diagnostics-v2/{case}/overlay.jpg"><img loading="lazy" '
            f'src="diagnostics-v2/{case}/overlay.jpg"></a>'
            f'<p><a href="gallery.html#{case}">All segmentation outputs</a> · '
            f'<a href="diagnostics-v2/{case}/DETAILS.json">Proposal information</a></p></section>')
    save(output/"MANIFEST.json", dict(
        files_sha256=files, scores=scores, predictions_modified=False,
        segmentation_scores_modified=False, diagnostic_version=2,
        source_sha256={name: digest(Path(__file__).with_name(name))
                       for name in ("publish_diagnostics.py", "diagnostics.py", "metrics.py")}))
    (root/"outliers.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Bora colour-region proposals</title>'
        '<style>body{font:15px system-ui;background:#edf2f4;margin:20px}'
        'section{background:white;margin:20px 0;padding:16px}img{max-width:100%;width:900px}</style>'
        '<h1>Rare and unseen RGB colour regions</h1><p>Diagnostic proposals only. '
        'Fine annotations do not generate these maps. Neither map establishes a new biological class. '
        'Segmentation predictions and their timing remain frozen.</p>'+"".join(sections))
    gallery = root/"gallery.html"
    if gallery.is_file():
        html = gallery.read_text()
        for case in protocol["cases"]:
            old = f"cases/{case}/original.jpg"
            # The earliest renderer used 'Original' for two panel filenames.
            # Fix only publication links; the actual contact sheet was correct.
            html = html.replace(old, f"cases/{case}/original-he.jpg", 2)
            html = html.replace(old, f"cases/{case}/coarse.jpg")
        if 'id="method-catalogue"' not in html:
            catalogue = ('<details id="method-catalogue" open><summary>'
                         'What each algorithm changes</summary><table>'
                         '<tr><th>ID</th><th>Algorithm</th><th>Hypothesis</th><th>Risk</th></tr>')
            for key, entry in protocol["roster"].items():
                catalogue += (f'<tr><td>{key}</td><td>{escape(entry["title"])}</td>'
                              f'<td>{escape(entry["hypothesis"])}</td>'
                              f'<td>{escape(entry["risk"])}</td></tr>')
            catalogue += "</table></details>"
            html = html.replace("</h1>", "</h1>"+catalogue, 1)
        if 'href="outliers.html"' not in html:
            html = html.replace("</h1>", '</h1><p><a href="outliers.html">'
                                'Rare and unseen colour-region proposals</a></p>', 1)
        gallery.write_text(html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    publish(args.source, args.root)


if __name__ == "__main__":
    main()
