"""Export native-pixel review crops without changing predictions or scores.

Reference masks are used only for these post-scoring review figures. Crop
coordinates, pixel spacing and source hashes accompany every exported sheet.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import tifffile

from .run import digest, palette_image, resize, save


def export_zoom(source, root, case, xywh, methods):
    source, root = Path(source), Path(root)
    result = json.loads((root / "RESULTS.json").read_text())
    row = next(r for r in result["cases"] if r["case"] == case)
    sample = row["sample"]
    if not set(methods) <= set(result["protocol"]["methods"]):
        raise ValueError("Unknown method in review crop")
    src, dst = source / "cases" / case, root / "cases" / case
    he_path = src / "original.native.png"
    he = Image.open(he_path).convert("RGB")
    x, y, width, height = map(int, xywh)
    if x < 0 or y < 0 or width < 1 or height < 1 or x + width > he.width or y + height > he.height:
        raise ValueError("Review crop must lie inside the native image")
    box = (x, y, x + width, y + height)
    crop = he.crop(box)
    truth_path, coarse_path = src / "reference.native.tif", src / "coarse.tif"
    truth = tifffile.imread(truth_path)[y:y+height, x:x+width]
    coarse = resize(tifffile.imread(coarse_path), he.size[::-1], True)[y:y+height, x:x+width]
    rgb_path = src / "ensemble" / "representation.png"
    rgb = Image.open(rgb_path).convert("RGB").resize(he.size, Image.Resampling.BILINEAR).crop(box)
    label_panels = [("Original H&E", crop),
                    ("Reference (zero = unscored)", palette_image(truth, sample["palette"])),
                    ("Original coarse labels", palette_image(coarse, sample["palette"])),
                    ("Fixed RGB, resized for display", rgb)]
    overlay_panels = label_panels.copy()
    hashes = {str(p.relative_to(source)): digest(p) for p in (he_path, truth_path, coarse_path, rgb_path)}
    for method in methods:
        path = dst / method / "native.tif"
        labels = tifffile.imread(path)[y:y+height, x:x+width]
        colour = palette_image(labels, sample["palette"])
        blended = np.array(Image.blend(crop, colour, .45))
        blended[labels == 0] = np.asarray(crop)[labels == 0]
        label_panels.append((method, colour))
        overlay_panels.append((method + " (45% mask)", Image.fromarray(blended)))
        hashes[method + "/native.tif"] = digest(path)
    output = dst / "native-review" / f"{x}-{y}-{width}-{height}"
    output.mkdir(parents=True, exist_ok=True)
    for name, panels in (("labels", label_panels), ("overlays", overlay_panels)):
        columns, header = 3, 32
        rows = (len(panels) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * width, rows * (height + header)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (title, panel) in enumerate(panels):
            px, py = index % columns * width, index // columns * (height + header)
            sheet.paste(panel, (px, py + header))
            draw.text((px + 6, py + 8), title, fill="black")
        sheet.save(output / (name + ".png"))
    save(output / "PROVENANCE.json", dict(
        case=case, native_crop_xywh=[x, y, width, height],
        native_mpp_xy=sample["native_mpp_xy"],
        class_names=sample["class_names"], palette=sample["palette"],
        physical_fov_um_xy=([width * sample["native_mpp_xy"][0], height * sample["native_mpp_xy"][1]]
                            if sample["native_mpp_xy"] is not None else None),
        coordinate_units="micrometres" if sample["native_mpp_xy"] is not None else "pixels",
        methods=methods, source_sha256=hashes, display_mask_opacity=.45,
        converted_rgb_resampling="bilinear for display only; no inference",
        predictions_modified=False, scores_modified=False, actual_visual_review_required=True))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--xywh", nargs=4, type=int, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    args = parser.parse_args()
    print(export_zoom(args.source, args.root, args.case, args.xywh, args.methods))


if __name__ == "__main__":
    main()
