#!/usr/bin/env python3
"""Summarize one numbered 216-pixel balanced-wand cycle in a PDF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


NAMES = {
    "Visium_HD_11mm_Human_Pancreas_tissue_image": "Pancreas",
    "Visium_HD_3prime_Human_Lymph_Node_tissue_image": "Lymph node",
    "Visium_HD_3prime_Human_Tonsil_tissue_image": "Tonsil",
    "Visium_HD_6p5mm_Human_Brain_Cancer_tissue_image": "Brain cancer",
}


def number(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def styled_table(rows, widths):
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#163b50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef4f5")]),
        ("LINEBELOW", (0, 0), (-1, -1), .3, colors.HexColor("#c6d5da")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--incumbent-cycle", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    root = args.benchmark_root
    folder = root / f"output-bora-cycle-{args.cycle:02d}"
    evaluation = root / f"output-bora-cycle-{args.cycle:02d}-vector-eval"
    records = {}
    for sample, name in NAMES.items():
        output = folder / "216" / sample
        scored = evaluation / "216" / sample
        if not (output / "bora_refined.geojson").is_file():
            raise FileNotFoundError(f"Missing final GeoJSON: {output}")
        records[name] = {
            "comparison": json.loads((output / "comparison.json").read_text()),
            "report": json.loads((output / "bora_refined.report.json").read_text()),
            "vector": json.loads((scored / "vector_comparison.json").read_text()),
            "border": json.loads((scored / "border_comparison_r64.json").read_text()),
            "annotations": len(json.loads((output / "bora_refined.geojson").read_text())["features"]),
        }

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BoraTitle", parent=styles["Title"], fontSize=17,
                              leading=20, textColor=colors.HexColor("#163b50"), spaceAfter=10))
    styles.add(ParagraphStyle(name="BoraBody", parent=styles["BodyText"], fontSize=8.5,
                              leading=12, spaceAfter=7))
    story = [Paragraph(f"Bora cycle {args.cycle:02d}: balanced wand and contour cleanup", styles["BoraTitle"]),
             Paragraph("Four native-resolution H&amp;E whole-slide images; 216-pixel categorical tile masks reused without regeneration. Bora used a symmetric local-color wand for borders between tissue classes, then filled small enclosed holes and smoothed those contours. Final GeoJSONs were rasterized at native resolution before scoring against the original GeoJSON annotations. Accuracy decisions use only borders between different foreground tissue classes.", styles["BoraBody"])]
    story.append(Paragraph("Final annotation features: " + "; ".join(
        f"{name} {item['annotations']}" for name, item in records.items()) +
        ". Every output feature carries its source class name and color in wsiTools annotation metadata.",
        styles["BoraBody"]))
    story.append(Paragraph("Internal tissue-class borders", styles["Heading2"]))
    rows = [["Tissue", "Tile IoU", "Bora IoU", "Tile acc.", "Bora acc.", "Boundary F1 (tile/Bora)"]]
    for name, item in records.items():
        band = item["border"]["internal"]
        baseline, bora = band["tile"], band["bora"]
        rows.append([name, number(baseline["macro_iou"]), number(bora["macro_iou"]),
                     number(baseline["accuracy"]), number(bora["accuracy"]),
                     f"{number(baseline['boundary_f1_8px'])} / {number(bora['boundary_f1_8px'])}"])
    story += [styled_table(rows, [90, 67, 67, 67, 67, 152]), Spacer(1, 5),
              Paragraph("The 64-pixel score band is fixed from the ground-truth or original tile border. Boundary F1 uses an 8-pixel matching tolerance. Pancreas has one foreground class and therefore no internal class border.", styles["BoraBody"])]
    if args.incumbent_cycle is not None:
        previous_root = root / f"output-bora-cycle-{args.incumbent_cycle:02d}-vector-eval"
        story.append(Paragraph(f"Internal borders: cycle {args.incumbent_cycle:02d} to {args.cycle:02d}", styles["Heading2"]))
        rows = [["Tissue", "Previous IoU", "New IoU", "Previous accuracy", "New accuracy", "Boundary F1 before/after"]]
        for sample, name in NAMES.items():
            prior = json.loads((previous_root / "216" / sample /
                                "border_comparison_r64.json").read_text())["internal"]["bora"]
            current = records[name]["border"]["internal"]["bora"]
            rows.append([name, number(prior["macro_iou"]), number(current["macro_iou"]),
                         number(prior["accuracy"]), number(current["accuracy"]),
                         f"{number(prior['boundary_f1_8px'])} / {number(current['boundary_f1_8px'])}"])
        story += [styled_table(rows, [90, 75, 65, 85, 75, 120]), Spacer(1, 6)]
    story.append(Paragraph("Speed and actual tissue-class changes", styles["Heading2"]))
    rows = [["Tissue", "Refine + GeoJSON", "Boundary step", "GeoJSON step", "Class-changed pixels"]]
    for name, item in records.items():
        c, b = item["comparison"], item["border"]
        rows.append([name, f"{c['refinement_seconds']:.1f} s",
                     f"{c['boundary_seconds']:.1f} s", f"{c['geojson_seconds']:.1f} s",
                     f"{b['changed_between_tissue_labels_pixels']:,}"])
    story += [styled_table(rows, [90, 105, 105, 105, 105]), Spacer(1, 5),
              Paragraph("Class-changed pixels count original foreground pixels assigned a different foreground class after final GeoJSON export. Runtime includes the original image-guided pass and the added cleanup/export; it excludes evaluation. Tonsil's original pass was interrupted after its TIFF was saved, so that part of its time is estimated from file timestamps.", styles["BoraBody"])]
    if args.figure and args.figure.is_file():
        story += [Paragraph("Visual comparison", styles["Heading2"]),
                  Image(str(args.figure), width=510, height=170),
                  Paragraph("Lymph-node crop: original tile border, balanced-wand border, and rasterized ground-truth border on the same H&amp;E image. This crop is illustrative; whole-slide scores above are the decision evidence.", styles["BoraBody"])]
    story.append(Paragraph("Decision and next step", styles["Heading2"]))
    decision = json.loads((folder / "216" / "cycle_decision.json").read_text())
    good = decision["accepted"]
    if good:
        conclusion = "The candidate passes the internal tissue-border gate: neither macro IoU nor pixel accuracy falls in any multiclass tissue, all classes remain represented, and at least one score improves. Keep this candidate for external validation."
    else:
        declined = ", ".join(row["sample"].replace("Visium_HD_", "")
                             for row in decision["per_tissue"] if not row["accepted"])
        conclusion = f"The candidate does not meet the per-tissue border-band acceptance rule ({declined}). Keep its numbered results for diagnosis and adjust the local model before accepting it."
    story.append(Paragraph(conclusion, styles["BoraBody"]))
    story.append(Paragraph("The local model estimates each class's LAB color near an interface from protected interior pixels. A symmetric search band allows either class to claim neighboring pixels; a distance cost and connected-core check restrain jumps and isolated islands. Remaining errors should be inspected for weak color separation, annotation ambiguity, and processing-window seams before widening the search band. The same slides were used for tuning, so an independent validation set is needed before claiming general improvement.", styles["BoraBody"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(args.output), pagesize=(612, 792), leftMargin=50,
                            rightMargin=50, topMargin=45, bottomMargin=46)
    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#657984"))
        canvas.drawString(50, 25, "Bora 216-pixel local-color evaluation | 15 October 2026")
        canvas.drawRightString(562, 25, f"Page {document.page}")
        canvas.restoreState()
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(args.output)


if __name__ == "__main__":
    main()
