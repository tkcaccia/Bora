#!/usr/bin/env python3
"""Render the same selected ridge whole-slide mask in ten registered views."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path("/mnt/sata_ssd/ChatGPT/Bora")
DATA = Path("/mnt/sata_ssd/ChatGPT/Bora-Data")
SITE = Path("/tmp/bora-analysis-bw106-v2")
HERE = Path(__file__).resolve().parent
FINAL = DATA / "benchmark2-supervised-ridge-2026-11-12-v5"
RENDER = DATA / "benchmark-label-reduction-2026-11-11-v6/rendered"
OUT = DATA / "benchmark2-supervised-ridge-2026-11-12-v7/rendered"
CAMPAIGN = ROOT / "migration/benchmark2-unified-wsi-panels-2026-11-11-v17/CAMPAIGN_VERIFIED.json"
ROLES = ROOT / "migration/benchmark1-supervised-virchow2-response-2026-11-10-v1/INPUTS_FROZEN.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def archived_rgb(html: str, name: str) -> tuple[Image.Image, str]:
    pattern = (r"https://raw\.githubusercontent\.com/tkcaccia/Bora-Analysis/"
               r"([0-9a-f]{40})/(site/visium-hd/[^\"<> ]+/" + re.escape(name) + r")")
    matches = set(re.findall(pattern, html))
    if len(matches) != 1:
        raise ValueError("Expected one pinned H&E asset for " + name)
    commit, relative = next(iter(matches))
    payload = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=SITE)
    return Image.open(io.BytesIO(payload)).convert("RGB"), hashlib.sha256(payload).hexdigest()


def support_for(mask: Image.Image, box: list[int], size: tuple[int, int]) -> np.ndarray:
    x, y, w, h = box
    affine = (w / (8 * size[0]), 0, x / 8,
              0, h / (8 * size[1]), y / 8)
    return np.asarray(mask.transform(size, Image.Transform.AFFINE, affine,
                                     resample=Image.Resampling.NEAREST)) > 0


def palette() -> np.ndarray:
    site_data = json.loads((SITE / "site/data.json").read_text())
    roles = json.loads(ROLES.read_text())["classes"]
    colors = [site_data["colours"][0]] + [
        site_data["colours"][site_data["roles"].index(roles[str(i)]) + 1]
        for i in (1, 3, 5, 7, 8, 9, 11, 12)]
    return np.asarray(colors, np.uint8)


def support_path(slug: str) -> tuple[Path, Path]:
    paths = list(DATA.glob(f"benchmark2-unified-wsi-panels-*/shared-bora-support/{slug}/SHARED_SUPPORT_VERIFIED.json"))
    if len(paths) != 1:
        raise ValueError("Expected one shared Bora web support: " + slug)
    receipt_path = paths[0]
    receipt = json.loads(receipt_path.read_text())
    path = receipt_path.parent / "shared-support-4um.png"
    if sha(path) != receipt["support_sha256"]:
        raise ValueError("Shared web support SHA differs")
    return path, receipt_path


def main(slug: str) -> None:
    pages = {r["slug"]: r["page"] for r in json.loads(CAMPAIGN.read_text())["pages"]}
    if slug not in pages or "pm016" in slug.lower():
        raise ValueError("Only thirteen public slides are admitted")
    slide = FINAL / slug
    receipt_path = slide / "FULL_SLIDE_VERIFIED.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt["status"] != "SUPERVISED_BORA_RIDGE_FULL_SLIDE_VECTOR_VERIFIED":
        raise ValueError("Full-slide result is unverified")
    mask_path = slide / "supervised-bora.final-2000um2.uint8.dat"
    if sha(mask_path) != receipt["output_sha256"]["exact_mask"]:
        raise ValueError("Exact global mask SHA differs")
    height, width = receipt["calibrated_shape_yx"]
    mask = np.memmap(mask_path, mode="r", dtype=np.uint8, shape=(height, width))
    previous_path = RENDER / slug / "PANEL_RENDER_VERIFIED.json"
    previous = json.loads(previous_path.read_text())
    if len(previous["views"]) != 10:
        raise ValueError("Ten registered views required")
    html = (SITE / "site/visium-hd" / (pages[slug] + ".html")).read_text()
    support_png, support_receipt = support_path(slug)
    with Image.open(support_png) as image:
        support = image.copy()
    colors = palette()
    out = OUT / slug
    if out.exists():
        raise FileExistsError("Preserve existing render; recover in a new version")
    out.mkdir(parents=True)
    rows = []
    try:
        for view in previous["views"]:
            name, (x, y, w, h) = view["name"], view["source_xywh"]
            size = tuple(view["web_wh"])
            he_name = "he-overview.jpg" if name == "overview" else name + "-he.jpg"
            he, he_hash = archived_rgb(html, he_name)
            if he_hash != view["archived_he_sha256"] or he.size != size:
                raise ValueError("Pinned H&E view identity differs")
            if min(x, y) < 0 or x+w > width or y+h > height:
                raise ValueError("Registered crop exceeds global mask")
            label = np.array(Image.fromarray(np.asarray(mask[y:y+h, x:x+w]), "L")
                             .resize(size, Image.Resampling.NEAREST), dtype=np.uint8, copy=True)
            visible = support_for(support, [x, y, w, h], size)
            label[~visible] = 0
            original = np.asarray(he, np.uint8)
            blended = np.rint(.45 * original.astype(np.float32) +
                               .55 * colors[label].astype(np.float32)).astype(np.uint8)
            blended[label == 0] = original[label == 0]
            path = out / f"{name}-supervised.webp"
            temporary = path.with_suffix(".partial.webp")
            Image.fromarray(blended, "RGB").save(temporary, "WEBP", quality=80, method=4)
            os.replace(temporary, path)
            classes = [int(i) for i in np.unique(label) if i > 0]
            if not set(classes) <= set(receipt["class_ids"]):
                raise AssertionError("View contains a nonretained class")
            rows.append({"name": name, "source_xywh": [x, y, w, h],
                         "web_wh": list(size), "class_ids_on_support": classes,
                         "positive_pixels": int(np.count_nonzero(label)),
                         "asset_sha256": sha(path), "asset_bytes": path.stat().st_size})
            print("RENDERED", slug, name, classes, flush=True)
        result = {"status": "TEN_VIEWS_FROM_ONE_FINAL_GLOBAL_RIDGE_MASK_VERIFIED",
                  "slug": slug, "full_slide_receipt_sha256": sha(receipt_path),
                  "prior_view_receipt_sha256": sha(previous_path),
                  "shared_support_receipt_sha256": sha(support_receipt),
                  "source_code_sha256": sha(Path(__file__)), "views": rows,
                  "whole_slide_expert_accuracy": None, "pm016_included": False}
        (out / "RENDER_VERIFIED.json").write_text(json.dumps(result, indent=2) + "\n")
    except BaseException as exc:
        (out / "FAILED.json").write_text(json.dumps({"error": repr(exc), "views_complete": len(rows)}) + "\n")
        raise


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: render_public_views.py PUBLIC_SLUG")
    main(sys.argv[1])
