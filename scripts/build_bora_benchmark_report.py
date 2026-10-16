#!/usr/bin/env python3
"""Build the four-tissue, three-tile-size Bora benchmark PDF."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
    TableStyle,
)


SHORT = {
    "Visium_HD_11mm_Human_Pancreas_tissue_image": "Pancreas",
    "Visium_HD_3prime_Human_Lymph_Node_tissue_image": "Lymph node",
    "Visium_HD_3prime_Human_Tonsil_tissue_image": "Tonsil",
    "Visium_HD_6p5mm_Human_Brain_Cancer_tissue_image": "Brain cancer",
}
NAVY = colors.HexColor("#17324d")
TEAL = colors.HexColor("#147d82")
PALE = colors.HexColor("#eaf3f4")
MID = colors.HexColor("#e2e8ec")
INK = colors.HexColor("#263645")


def load_runs(bora_root: Path, tiling_root: Path, evaluation_root: Path):
    runs = []
    for size in (56, 90, 216):
        for sample in SHORT:
            folder = bora_root / str(size) / sample
            comparison = json.loads((folder / "comparison.json").read_text())
            vector = json.loads((evaluation_root / str(size) / sample /
                                 "vector_comparison.json").read_text())
            run_report = json.loads((folder / "bora_refined.report.json").read_text())
            tiling = json.loads((tiling_root / sample / "metrics_cpp.json").read_text())
            runs.append({"size": size, "sample": sample, "short": SHORT[sample],
                         "baseline": comparison["baseline"],
                         "raster": comparison["bora"],
                         "vector": vector["vector_vs_ground_truth"],
                         "roundtrip": vector["vector_vs_refined_raster"],
                         "refinement_seconds": comparison["refinement_seconds"],
                         "scoring_seconds": comparison["scoring_seconds"],
                         "vector_scoring_seconds": vector["vector_scoring_seconds"],
                         "tiling_seconds": tiling["total_seconds"],
                         "classes": tiling["classes"],
                         "blocks": run_report["blocks"],
                         "refined_blocks": run_report["refined_blocks"],
                         "annealed_changed_pixels": run_report["annealed_wand_changed_pixels"],
                         "pixels": tiling["width"] * tiling["height"]})
    if len(runs) != 12:
        raise RuntimeError(f"Expected 12 completed runs, found {len(runs)}")
    return runs


def signed(value):
    return f"{value:+.4f}"


def styled_table(rows, widths, *, header=True, font_size=8):
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0,
                  hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, MID),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f9fa")]),
        ]
    table.setStyle(TableStyle(commands))
    return table


def footer(canvas, document):
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(MID)
    canvas.line(18 * mm, 17 * mm, width - 18 * mm, 17 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#61717e"))
    canvas.drawString(18 * mm, 12 * mm, "Bora | four-tissue reconstruction benchmark")
    canvas.drawRightString(width - 18 * mm, 12 * mm, str(document.page))
    canvas.restoreState()


def build(runs, examples, output):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleB", parent=styles["Title"],
                              fontName="Helvetica-Bold", fontSize=19,
                              leading=23, textColor=NAVY, spaceAfter=12))
    styles.add(ParagraphStyle(name="H1B", parent=styles["Heading1"],
                              fontSize=13, leading=16, textColor=NAVY,
                              spaceBefore=12, spaceAfter=7))
    styles.add(ParagraphStyle(name="BodyB", parent=styles["BodyText"],
                              fontSize=9.3, leading=13.5, textColor=INK,
                              spaceAfter=8))
    styles.add(ParagraphStyle(name="SmallB", parent=styles["BodyText"],
                              fontSize=8, leading=11, textColor=INK,
                              spaceAfter=7))
    styles.add(ParagraphStyle(name="CapB", parent=styles["BodyText"],
                              fontSize=8.3, leading=11.5, textColor=INK,
                              spaceAfter=7, alignment=TA_LEFT))
    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(output), pagesize=A4,
                                 leftMargin=18 * mm, rightMargin=18 * mm,
                                 topMargin=18 * mm, bottomMargin=23 * mm,
                                 title="Bora four-tissue multi-size benchmark")
    flow = []
    p = lambda text, style="BodyB": Paragraph(text, styles[style])
    flow += [p("Bora: four-tissue, three-tile-size benchmark", "TitleB"),
             p("14 October 2026 | Native-resolution watershed | 12 whole-slide runs", "SmallB")]
    total_pixels = sum(row["pixels"] for row in runs if row["size"] == 56)
    flow.append(p(
        f"Four H&amp;E tissue images ({total_pixels / 1e9:.2f} billion unique pixels) were "
        "rasterized from the supplied GeoJSON annotations with GDAL. The local "
        "C++ tiling-mask tool "
        "then created majority-vote masks at 56, 90 and 216 pixels. Bora refined "
        "each mask with annealed boundary competition followed by native-resolution "
        "marker-controlled watershed. Its final GeoJSON was rasterized back at the "
        "original dimensions and compared pixel by pixel with the annotation-derived "
        "reference. The runs used 2,048-pixel streaming blocks, a 64-pixel "
        "annealed boundary radius at 4x working downsampling, 16-pixel core "
        "erosion, 24-pixel outer dilation and the accurate GeoJSON profile. "
        "No synthetic border-label corruption was applied."))
    flow.append(p("Summary by tile size", "H1B"))
    means = []
    mean_rows = [["Tile", "Baseline macro IoU", "Bora GeoJSON macro IoU", "Change",
                  "Foreground IoU change"]]
    for size in (56, 90, 216):
        subset = [row for row in runs if row["size"] == size]
        base = mean(row["baseline"]["macro_iou"] for row in subset)
        final = mean(row["vector"]["macro_iou"] for row in subset)
        fg = mean(row["vector"]["foreground_iou"] -
                  row["baseline"]["foreground_iou"] for row in subset)
        mean_rows.append([str(size), f"{base:.4f}", f"{final:.4f}",
                          signed(final - base), signed(fg)])
        means.append((size, base, final, fg))
    flow.append(styled_table(mean_rows, [42, 112, 140, 77, 122], font_size=8.4))
    flow.append(Spacer(1, 10))
    dice_text = []
    for size in (56, 90, 216):
        subset = [row for row in runs if row["size"] == size]
        dice_change = mean(row["vector"]["macro_dice"] -
                           row["baseline"]["macro_dice"] for row in subset)
        accuracy_change = mean(row["vector"]["accuracy"] -
                               row["baseline"]["accuracy"] for row in subset)
        dice_text.append(f"{size}px: macro Dice {signed(dice_change)}, "
                         f"pixel accuracy {signed(accuracy_change)}")
    flow.append(p("Mean changes in other accuracy measures: " +
                  "; ".join(dice_text) + ".", "SmallB"))
    positive_macro = sum(row["vector"]["macro_iou"] > row["baseline"]["macro_iou"]
                         for row in runs)
    positive_fg = sum(row["vector"]["foreground_iou"] > row["baseline"]["foreground_iou"]
                      for row in runs)
    flow.append(p(
        f"Across the 12 tissue-size comparisons, final GeoJSON macro IoU improved "
        f"in {positive_macro} and foreground IoU improved in {positive_fg}. "
        "The table gives an unweighted mean across four tissues; it is descriptive, "
        "not a population estimate."))
    flow.append(p("What was measured", "H1B"))
    flow.append(p(
        "Macro IoU is the arithmetic mean of foreground class IoUs, including "
        "classes present in either prediction or reference. Foreground IoU ignores "
        "class identity and measures the union of tissue labels. Pixel accuracy "
        "includes background. The reference is a rasterization of the same supplied "
        "GeoJSON used to generate the tile masks; this is a controlled reconstruction "
        "test, not independent clinical validation."))
    flow.append(p(
        "The current pancreas annotation contains one foreground class (blood "
        "vessels). The other tissues contain five, seven and four foreground classes "
        "for lymph node, tonsil and brain cancer, respectively. This differs from "
        "earlier Bora handoff benchmarks and is why all masks and scores were regenerated."))
    flow.append(p("Both the Bora and local C++ tiling-mask checkouts contained uncommitted "
                  "implementation changes during this run. The output directories retain "
                  "per-sample logs, metrics and comparison JSON for auditability.", "SmallB"))
    flow.append(PageBreak())

    flow.append(p("Accuracy of every final GeoJSON", "TitleB"))
    rows = [["Tissue", "Tile", "Baseline\nmacro IoU", "GeoJSON\nmacro IoU",
             "IoU\nchange", "GeoJSON\nmacro Dice", "GeoJSON\nforeground IoU",
             "GeoJSON\npixel accuracy", "Bora\nminutes"]]
    for row in runs:
        rows.append([row["short"], str(row["size"]),
                     f"{row['baseline']['macro_iou']:.4f}",
                     f"{row['vector']['macro_iou']:.4f}",
                     signed(row["vector"]["macro_iou"] - row["baseline"]["macro_iou"]),
                     f"{row['vector']['macro_dice']:.4f}",
                     f"{row['vector']['foreground_iou']:.4f}",
                     f"{row['vector']['accuracy']:.4f}",
                     f"{row['refinement_seconds'] / 60:.1f}"])
    flow.append(styled_table(rows, [65, 25, 54, 56, 48, 57, 67, 64, 52], font_size=7.1))
    flow.append(Spacer(1, 9))
    flow.append(p("All GeoJSON columns score the delivered vector result after "
                  "rasterization at native resolution. The TIFF-to-vector agreement "
                  "on the next page separates algorithm changes from geometry "
                  "simplification during export.", "SmallB"))
    flow.append(p("Per-class findings", "H1B"))
    changes = []
    for row in runs:
        base = {item["label"]: item for item in row["baseline"]["per_class"]}
        names = {int(item["value"]): item["name"] for item in row["classes"]}
        for item in row["vector"]["per_class"]:
            old = base.get(item["label"], {"iou": 0, "support": 0})
            changes.append((item["iou"] - old["iou"], row["short"], row["size"],
                            names.get(item["label"], str(item["label"])),
                            item["support"], old["iou"], item["iou"]))
    loss = sorted(changes)[:3]
    gain = sorted(changes, reverse=True)[:3]
    class_rows = [["Direction", "Tissue / tile", "Class", "Reference pixels", "IoU change"]]
    for label, group in (("Largest losses", loss), ("Largest gains", gain)):
        for change, tissue, size, name, support, _, _ in group:
            class_rows.append([label, f"{tissue} / {size}", name[:24],
                               f"{support:,}", signed(change)])
    flow.append(styled_table(class_rows, [75, 101, 140, 104, 73], font_size=7.7))
    lymph_small = next(row for row in runs if row["short"] == "Lymph node" and row["size"] == 56)
    lymph_large = next(row for row in runs if row["short"] == "Lymph node" and row["size"] == 216)
    small_base = {item["label"]: item["iou"] for item in lymph_small["baseline"]["per_class"]}
    large_base = {item["label"]: item["iou"] for item in lymph_large["baseline"]["per_class"]}
    small_losses = sum(item["iou"] < small_base.get(item["label"], 0)
                       for item in lymph_small["vector"]["per_class"])
    large_gains = sum(item["iou"] > large_base.get(item["label"], 0)
                      for item in lymph_large["vector"]["per_class"])
    flow.append(Spacer(1, 8))
    flow.append(p(
        f"Lymph node illustrates the scale effect: at 56 pixels, "
        f"{small_losses} of {len(lymph_small['vector']['per_class'])} classes "
        f"lost IoU and macro IoU changed by "
        f"{signed(lymph_small['vector']['macro_iou'] - lymph_small['baseline']['macro_iou'])}; "
        f"at 216 pixels, {large_gains} of "
        f"{len(lymph_large['vector']['per_class'])} classes gained and macro IoU "
        f"changed by "
        f"{signed(lymph_large['vector']['macro_iou'] - lymph_large['baseline']['macro_iou'])}."
    ))
    flow.append(PageBreak())

    flow.append(p("Runtime and output fidelity", "TitleB"))
    tiling_times = {row["sample"]: row["tiling_seconds"] for row in runs}
    flow.append(p(
        f"The C++ tiling-mask tool produced the pixel reference and all three tile "
        f"sizes in one run per tissue: {mean(tiling_times.values()):.1f} seconds "
        "per tissue on average. Its time is shared across sizes and must not be "
        "counted three times. Bora times below include refinement and its GeoJSON "
        "export; input mask creation and subsequent scoring are separate."))
    speed_rows = [["Tile", "Bora mean", "Bora median", "Bora range", "Vector scoring mean",
                   "TIFF-vector agreement"]]
    for size in (56, 90, 216):
        subset = [row for row in runs if row["size"] == size]
        times = [row["refinement_seconds"] / 60 for row in subset]
        speed_rows.append([str(size), f"{mean(times):.1f} min", f"{median(times):.1f} min",
                           f"{min(times):.1f}-{max(times):.1f} min",
                           f"{mean(row['vector_scoring_seconds'] for row in subset) / 60:.1f} min",
                           f"{mean(row['roundtrip']['accuracy'] for row in subset):.5f}"])
    flow.append(styled_table(speed_rows, [40, 72, 77, 107, 103, 94], font_size=7.8))
    flow.append(Spacer(1, 11))
    flow.append(p("TIFF-vector agreement is pixel accuracy between the refined "
                  "OME-TIFF and a fresh rasterization of the final GeoJSON. Values "
                  "below 1 quantify the vector export's geometry difference. "
                  "Vector scoring time is an audit cost, not part of Bora runtime. "
                  "The three tile-size batches ran concurrently, so their wall "
                  "times include shared CPU, memory and storage contention.", "SmallB"))
    pancreas_216 = next(row for row in runs if row["size"] == 216 and
                        row["short"] == "Pancreas")
    flow.append(p(
        f"For the single-class pancreas at 216 pixels, "
        f"{pancreas_216['refined_blocks']} of {pancreas_216['blocks']} streaming "
        f"blocks required refinement. Annealed label competition changed "
        f"{pancreas_216['annealed_changed_pixels']:,} pixels, so its long runtime "
        "mainly reflects native watershed and image I/O around dispersed vessel boundaries.",
        "SmallB"))
    flow.append(p("Interpretation and concrete improvements", "H1B"))
    findings = []
    if positive_fg > positive_macro:
        findings.append("<b>Separate external and internal boundaries.</b> Tissue "
                        "footprint can improve while semantic labels worsen. Keep "
                        "image-driven watershed on tissue-background borders; apply "
                        "a stronger coarse-label prior and smaller displacement limit "
                        "at interfaces between foreground classes.")
    else:
        findings.append("<b>Gate every boundary update.</b> Compare the image-edge "
                        "evidence with a coarse-label prior, then retain the original "
                        "label when evidence is weak. Tune the gate on development slides.")
    findings.append("<b>Make refinement strength depend on tile size.</b> The lymph "
                    "node result shows losses at 56 pixels and gains at 216 pixels. "
                    "Use a smaller editable band and higher acceptance threshold "
                    "for already accurate fine grids; allow more movement for "
                    "coarse grids only when image evidence is strong.")
    findings.append("<b>Protect small classes.</b> The per-class table identifies "
                    "the largest losses. Make boundary width and overwrite threshold "
                    "depend on class area, and reject changes that erase a region's "
                    "protected core or greatly change its area.")
    findings.append("<b>Adapt the external search band to tile size.</b> The fixed "
                    "16-pixel core erosion and 24-pixel outer dilation can leave "
                    "large 216-pixel stair steps protected. Test a wider, "
                    "image-gated tissue-background band at coarse sizes while "
                    "keeping inter-class motion conservative.")
    if mean(row["roundtrip"]["accuracy"] for row in runs) < 0.999:
        findings.append("<b>Reduce vector-export drift.</b> The GeoJSON differs "
                        "measurably from the refined TIFF. Lower simplification or "
                        "retain an exact raster-linked vector mode when strict "
                        "pixel-level agreement is required.")
    findings.append("<b>Target runtime at the editable band.</b> Profile the "
                    "annealed and watershed stages separately, skip blocks without "
                    "an editable interface, and crop watershed work to the boundary "
                    "envelope. Validate that this preserves output before parallelizing.")
    findings.append("<b>Evaluate boundary quality explicitly.</b> Add boundary IoU "
                    "and distance-tolerant surface Dice for external and internal "
                    "interfaces separately, alongside per-class IoU and changed-core "
                    "pixel counts. Use held-out annotations for parameter selection.")
    for index, finding in enumerate(findings, 1):
        flow.append(p(f"{index}. {finding}", "BodyB"))
    flow.append(PageBreak())

    flow.append(p("Visual comparison at native coordinates", "TitleB"))
    flow.append(p("Illustrative 1,536 x 1,536 pixel regions were selected by "
                  "screening downsampled error maps for strong local accuracy changes. "
                  "Each four-panel image shows the same original H&amp;E crop and "
                  "categorical labels. The crop score is descriptive and was "
                  "not used to compute the whole-slide tables. These deliberately "
                  "selected crops are not representative estimates.", "BodyB"))
    for index, item in enumerate(examples):
        if index:
            flow.append(PageBreak())
        flow.append(p(f"{item['direction'].title()} example: "
                      f"{escape(SHORT[item['sample']])}, {item['tile_size']}-pixel tiles", "H1B"))
        picture = Image(item["file"])
        allowed_width = 165 * mm
        allowed_height = 185 * mm
        scale = min(allowed_width / picture.imageWidth,
                    allowed_height / picture.imageHeight)
        picture.drawWidth = picture.imageWidth * scale
        picture.drawHeight = picture.imageHeight * scale
        flow.append(picture)
        flow.append(Spacer(1, 6))
        flow.append(p(
            f"Location: x={item['x']}, y={item['y']}. "
            f"Tile-mask crop accuracy {item['tile_accuracy']:.3f}; "
            f"Bora GeoJSON crop accuracy {item['bora_vector_accuracy']:.3f}. "
            "Class colors are inherited from tiling-mask; background remains "
            "uncolored over the H&amp;E image.", "CapB"))
    flow.append(Spacer(1, 12))
    flow.append(p("Study limitations", "H1B"))
    flow.append(p("Only four annotated images were assessed, and their GeoJSONs "
                  "were also the source of the coarse masks. This measures recovery "
                  "from tiling, not performance on independent classifier errors. "
                  "No patient-level inference or clinical claim is supported. "
                  "Native-resolution GeoJSON rasterization uses the same pixel "
                  "coordinate frame and categorical class IDs as tiling-mask."))
    document.build(flow, onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bora-root", type=Path, required=True)
    parser.add_argument("--tiling-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = load_runs(args.bora_root, args.tiling_root, args.evaluation_root)
    examples = json.loads((args.examples / "visual_examples.json").read_text())
    build(runs, examples, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
