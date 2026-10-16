#!/usr/bin/env python3
"""Build a compact PDF report from completed 216-pixel Bora cycle folders."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                               TableStyle, Image, PageBreak, KeepTogether)


SAMPLES = {
    "Visium_HD_11mm_Human_Pancreas_tissue_image": "Pancreas",
    "Visium_HD_3prime_Human_Lymph_Node_tissue_image": "Lymph node",
    "Visium_HD_3prime_Human_Tonsil_tissue_image": "Tonsil",
    "Visium_HD_6p5mm_Human_Brain_Cancer_tissue_image": "Brain cancer",
}


def load_cycle(root: Path, number: int) -> dict:
    output = root / f"output-bora-cycle-{number:02d}"
    vectors = root / f"output-bora-cycle-{number:02d}-vector-eval"
    decision = json.loads((output / "216" / "cycle_decision.json").read_text())
    entries = {}
    for sample in SAMPLES:
        folder = output / "216" / sample
        if not (folder / "bora_refined.geojson").is_file():
            raise ValueError(f"Missing GeoJSON: {folder}")
        comparison = json.loads((folder / "comparison.json").read_text())
        vector = json.loads((vectors / "216" / sample / "vector_comparison.json").read_text())
        border = json.loads((vectors / "216" / sample /
                             "border_comparison_r64.json").read_text())
        entries[sample] = {"baseline": comparison["baseline"],
                           "vector": vector["vector_vs_ground_truth"],
                           "border": border,
                           "seconds": comparison["refinement_seconds"],
                           "boundary_seconds": comparison["boundary_seconds"],
                           "geojson_seconds": comparison["geojson_seconds"],
                           "vector_scoring_seconds": vector["vector_scoring_seconds"]}
    return {"number": number, "accepted": decision["accepted"],
            "incumbent": decision["incumbent_root"], "entries": entries}


def fmt(x: float) -> str:
    return f"{x:.4f}" if x is not None else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, nargs="+", required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    cycles = [load_cycle(args.benchmark_root, n) for n in args.cycles]
    accepted = [cycle for cycle in cycles if cycle["accepted"]]
    if not accepted:
        raise ValueError("No accepted cycle")
    best = accepted[-1]
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"],
                              fontName="Helvetica-Bold", fontSize=17,
                              leading=21, textColor=colors.HexColor("#193c51"),
                              spaceAfter=12))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"],
                              fontSize=8.5, leading=12, spaceAfter=7))
    styles.add(ParagraphStyle(name="Caption", parent=styles["BodyText"],
                              fontSize=8, leading=11, alignment=TA_CENTER,
                              textColor=colors.HexColor("#52636e")))
    story = [Paragraph("Bora: 216-pixel boundary refinement cycles",
                       styles["ReportTitle"]),
             Paragraph("Four H&amp;E whole-slide images. Final Bora GeoJSONs were rasterized at native resolution and scored only within 64 pixels of ground-truth or original tile-mask boundaries. Tiling-mask inputs were reused unchanged.", styles["Small"])]
    status = ", ".join(f"{c['number']:02d} {'accepted' if c['accepted'] else 'rejected'}"
                       for c in cycles)
    story.append(Paragraph(f"<b>Cycle decisions:</b> {status}. "
                           f"<b>Selected cycle:</b> {best['number']:02d}.", styles["Small"]))
    story.append(Paragraph("Internal annotation borders: macro IoU", styles["Heading2"]))
    header = ["Tissue", "Tile mask"] + [f"Cycle {c['number']:02d}" for c in cycles]
    data = [header]
    for sample, name in SAMPLES.items():
        data.append([name, fmt(cycles[0]["entries"][sample]["border"]["internal"]["tile"]["macro_iou"])] +
                    [fmt(c["entries"][sample]["border"]["internal"]["bora"]["macro_iou"]) for c in cycles])
    internal_samples = [s for s in SAMPLES if cycles[0]["entries"][s]["border"]["internal"]["tile"]["macro_iou"] is not None]
    data.append(["Internal mean",
                 fmt(sum(cycles[0]["entries"][s]["border"]["internal"]["tile"]["macro_iou"] for s in internal_samples)/len(internal_samples))] +
                [fmt(sum(c["entries"][s]["border"]["internal"]["bora"]["macro_iou"] for s in internal_samples)/len(internal_samples))
                 for c in cycles])
    score_widths = [105] + [395/(len(header)-1)]*(len(header)-1)
    table = Table(data, colWidths=score_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#193c51")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#e4eef1")),
        ("LINEBELOW", (0, 0), (-1, -1), .25, colors.HexColor("#b6c8d0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [table, Spacer(1, 5),
              Paragraph("Pancreas has one annotated tissue class, so it has no internal tissue-class border.", styles["Small"]),
              Paragraph("Tissue-background borders: macro IoU", styles["Heading2"])]
    external_rows = [header]
    for sample, name in SAMPLES.items():
        external_rows.append([name, fmt(cycles[0]["entries"][sample]["border"]["external"]["tile"]["macro_iou"])] +
                             [fmt(c["entries"][sample]["border"]["external"]["bora"]["macro_iou"]) for c in cycles])
    external_table = Table(external_rows, colWidths=score_widths, repeatRows=1)
    external_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#193c51")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("LINEBELOW", (0, 0), (-1, -1), .25, colors.HexColor("#b6c8d0")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [external_table, Spacer(1, 10),
              Paragraph("Mean refinement + GeoJSON runtime per tissue: " +
                        "; ".join(f"cycle {c['number']:02d} "
                                  f"{sum(c['entries'][s]['seconds'] for s in SAMPLES)/4:.1f} s"
                                  for c in cycles) + ".", styles["Small"]),
              Paragraph(f"Selected cycle {best['number']:02d}: accuracy and runtime", styles["Heading2"])]
    rows = [["Tissue", "Internal accuracy", "External accuracy", "Refine + GeoJSON", "Border gain"]]
    for sample, name in SAMPLES.items():
        item = best["entries"][sample]
        region = "internal" if item["border"]["internal"]["bora"]["macro_iou"] is not None else "external"
        change = (item["border"][region]["bora"]["macro_iou"] -
                  item["border"][region]["tile"]["macro_iou"])
        rows.append([name, fmt(item["border"]["internal"]["bora"]["accuracy"]),
                     fmt(item["border"]["external"]["bora"]["accuracy"]),
                     f"{item['seconds']:.1f} s", f"{change:+.4f}"])
    t2 = Table(rows, colWidths=[105, 92, 92, 105, 90], repeatRows=1)
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe7ea")),
        ("LINEBELOW", (0, 0), (-1, -1), .25, colors.HexColor("#b6c8d0")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [t2, Spacer(1, 10),
              Paragraph("Border gain is internal macro IoU gain, except for single-class pancreas where it is external gain. Runtime includes refinement, pyramidal TIFF writing, and GeoJSON export; scoring time is excluded. Some cycles overlapped other processing, so these wall-clock times are descriptive rather than controlled speed comparisons.", styles["Small"]),
              Paragraph("Runtime by cycle (seconds)", styles["Heading2"])]
    speed_rows = [header]
    for sample, name in SAMPLES.items():
        speed_rows.append([name, "-"] +
                          [f"{c['entries'][sample]['seconds']:.1f}" for c in cycles])
    speed_table = Table(speed_rows, colWidths=score_widths, repeatRows=1)
    speed_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe7ea")),
        ("LINEBELOW", (0, 0), (-1, -1), .25, colors.HexColor("#b6c8d0")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [speed_table, Spacer(1, 16),
              Paragraph("Visual review and interpretation", styles["Heading1"])]
    if args.figure and args.figure.is_file():
        story.append(Image(str(args.figure), width=500, height=134))
        story.append(Paragraph("Brain cancer example. Red overlays trace tile, two experimental cycles, and ground-truth boundaries (panel labels are embedded in the image).", styles["Caption"]))
        story.append(Spacer(1, 14))
    story.append(Paragraph("Observed behavior", styles["Heading2"]))
    story.append(Paragraph("The 216-pixel tile masks leave blocky tissue exteriors and coarse class interfaces. Cycle 01 improves both kinds of border in every applicable tissue, but visible processing-block joins remain. Cycle 03 removed a long straight brain-cancer seam (236 seam pixels fell to 6 in the inspected crop). Its lymph-node internal-border pixel accuracy fell slightly, so it was rejected despite a smoother outline. Cycle 05 constrained seam repair further; its lymph-node border IoU fell. These local failures explain why visual smoothness alone is insufficient for adoption.", styles["Small"]))
    story.append(Paragraph("Next algorithm work", styles["Heading2"]))
    for paragraph in (
        "Use a confidence map around each class interface: move boundaries when color or texture evidence is strong, and constrain movement where adjacent tile groups have similar H&amp;E distributions.",
        "Enforce connected seed cores for every annotated class and audit small components before GeoJSON export. This protects rare regions while removing one-pixel islands.",
        "Measure boundary F1 and 95th-percentile Hausdorff distance in addition to region IoU. Evaluate those scores on separate slides to guard against tuning to these four images.",
    ):
        story.append(Paragraph("&#8226; " + paragraph, styles["Small"]))
    story.append(Paragraph("Methods and decision rule", styles["Heading2"]))
    story.append(Paragraph("Each cycle reuses the original H&amp;E image and full-resolution 216-pixel tile mask. Seeded GrabCut and multiclass watershed refine overlapping windows, followed by pyramidal OME-TIFF and GeoJSON export. The 64-pixel evaluation bands are fixed from ground truth and tile-mask borders. Acceptance requires no drop in border macro IoU or pixel accuracy for any tissue, at least one gain, and preservation of every ground-truth class. Pancreas has only an exterior border.", styles["Small"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(args.output), pagesize=(612, 792),
                            leftMargin=50, rightMargin=50,
                            topMargin=48, bottomMargin=48,
                            title="Bora 216-pixel refinement cycles")
    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d4e0e4"))
        canvas.line(50, 34, 562, 34)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#667b86"))
        canvas.drawString(50, 23, "Bora 216-pixel boundary study | 15 October 2026")
        canvas.drawRightString(562, 23, f"Page {document.page}")
        canvas.restoreState()
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(args.output)


if __name__ == "__main__":
    main()
