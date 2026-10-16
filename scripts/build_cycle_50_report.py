#!/usr/bin/env python3
"""Create the 216-pixel cycle-search PDF from numbered decision records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                               LongTable, Table, TableStyle)


DATA = Path("/mnt/sata_ssd/benchmarks/Bora")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "output/pdf/Bora_216_cycles_to_50_report.pdf"


def score(cycle: int, sample: str) -> dict:
    path = (DATA / f"output-bora-cycle-{cycle:02d}-vector-eval" / "216" /
            sample / "border_comparison_r64.json")
    return json.loads(path.read_text())["internal"]["bora"]


def decision(cycle: int) -> dict:
    path = DATA / f"output-bora-cycle-{cycle:02d}/216/cycle_decision.json"
    return json.loads(path.read_text())


def signed(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:+.{digits}f}"


def draw_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawString(18 * mm, 12 * mm, "Bora | 216-pixel tissue interface search")
    canvas.drawRightString(landscape(A4)[0] - 18 * mm, 12 * mm,
                           f"Page {doc.page}")
    canvas.restoreState()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args()
    state = json.loads((DATA / "bora_cycles_13_to_50.json").read_text())
    if not state["finished"] and not args.allow_partial:
        raise SystemExit("Cycle 50 has not completed")
    completed = state["cycles"]
    incumbent = state["quantitative_incumbent"]
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BoraTitle", parent=styles["Title"],
                              textColor=colors.HexColor("#123047"), fontSize=19,
                              leading=22, spaceAfter=8))
    styles.add(ParagraphStyle(name="BoraBody", parent=styles["BodyText"],
                              fontSize=9, leading=13, spaceAfter=6))
    styles.add(ParagraphStyle(name="BoraSmall", parent=styles["BodyText"],
                              fontSize=8, leading=11, spaceAfter=5))
    styles.add(ParagraphStyle(name="BoraSection", parent=styles["Heading2"],
                              textColor=colors.HexColor("#123047"),
                              spaceBefore=13, spaceAfter=6))
    doc = SimpleDocTemplate(str(out), pagesize=landscape(A4),
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=17 * mm, bottomMargin=18 * mm)
    story = [Paragraph("Bora: 216-pixel contour improvement cycles", styles["BoraTitle"]),
             Paragraph(f"Cycles 13-50; completed through {state['completed_through']}; "
                       f"quantitative incumbent: cycle {incumbent:02d}. "
                       f"Status: {'complete' if state['finished'] else 'in progress'}.",
                       styles["BoraBody"]),
             Paragraph("Each cycle produced a GeoJSON for pancreas, lymph node, tonsil, "
                       "and brain cancer. A candidate replaced the incumbent only when "
                       "64-pixel tissue-to-tissue border macro IoU and accuracy did not "
                       "fall in any multiclass tissue, all annotated labels remained "
                       "present, and the absolute count of false tile-edge pixels did "
                       "not rise. At least one measured improvement was required. "
                       "Pancreas has no internal tissue-to-tissue interface. Eight-pixel "
                       "contour F1 is reported separately.", styles["BoraBody"]),
             Paragraph("Cycle 12 was accepted as the new starting point after the user "
                       "reported a strong visual improvement. The original stricter "
                       "decision, which rejected it on narrow contour F1 and a "
                       "denominator-sensitive tile-imprint fraction, remains saved. "
                       "This post hoc change to the acceptance rule must be considered "
                       "when interpreting the search.", styles["BoraSmall"]),
             Paragraph("Cycle decisions", styles["BoraSection"])]
    header = ["Cycle", "Method / parameters", "Accepted", "Minimum IoU change",
              "Minimum accuracy change", "Tile edges removed", "Cumulative time"]
    rows = [header]
    for record in completed:
        number = record["cycle"]
        d = decision(number)
        relevant = [x for x in d["per_tissue"] if x["boundary_metrics"]]
        iou = [x["boundary_metrics"]["internal_macro_iou"]["change"]
               for x in relevant if "internal_macro_iou" in x["boundary_metrics"]]
        acc = [x["boundary_metrics"]["internal_accuracy"]["change"]
               for x in relevant if "internal_accuracy" in x["boundary_metrics"]]
        imprint = [x["boundary_metrics"].get("internal_false_tile_imprint_edge_pixels", {}).get("change", 0)
                   for x in relevant]
        seconds = sum(x["refinement_seconds"] for x in d["per_tissue"])
        spec = record["specification"]
        param = (f"Gaussian {spec['sigma']} / band {spec['band_radius']}"
                 if spec["method"] == "geometry" else
                 f"Pairwise core {spec['core_distance']} / band {spec['search_radius']}")
        rows.append([f"{number:02d}", param,
                     "Yes" if record["accepted"] else "No",
                     signed(min(iou) if iou else None),
                     signed(min(acc) if acc else None),
                     f"{sum(imprint):+,}", f"{seconds / 3600:.2f} h"])
    table = LongTable(rows, repeatRows=1,
                      colWidths=[17*mm, 53*mm, 19*mm, 32*mm, 37*mm, 35*mm, 29*mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123047")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F2F5F7")]),
        ("LINEBELOW", (0, 0), (-1, 0), .5, colors.HexColor("#123047")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story += [table, Spacer(1, 8 * mm),
              Paragraph("Accepted output compared with cycle 07", styles["BoraSection"])]
    samples = ["Visium_HD_3prime_Human_Lymph_Node_tissue_image",
               "Visium_HD_3prime_Human_Tonsil_tissue_image",
               "Visium_HD_6p5mm_Human_Brain_Cancer_tissue_image"]
    labels = ["Lymph node", "Tonsil", "Brain cancer"]
    final_rows = [["Tissue", "IoU (07 to best)", "Accuracy (07 to best)",
                   "False tile edges (07 to best)", "8-px F1 (07 to best)"]]
    for sample, label in zip(samples, labels):
        base, best = score(7, sample), score(incumbent, sample)
        final_rows.append([label,
                           f"{base['macro_iou']:.4f} to {best['macro_iou']:.4f}",
                           f"{base['accuracy']:.4f} to {best['accuracy']:.4f}",
                           f"{base['false_tile_imprint_edge_pixels']:,} to {best['false_tile_imprint_edge_pixels']:,}",
                           f"{base['boundary_f1_8px']:.4f} to {best['boundary_f1_8px']:.4f}"])
    final_table = Table(final_rows, colWidths=[28*mm, 41*mm, 46*mm, 62*mm, 45*mm])
    final_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE7EC")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, 0), .6, colors.HexColor("#123047")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F2F5F7")]),
    ]))
    story += [final_table, Spacer(1, 5*mm),
              Paragraph("This is an exploratory parameter search on the same four slides. "
                        "Its best score is not an unbiased estimate of generalization. "
                        "An independent set of expert-traced tissue interfaces is "
                        "needed before a performance claim.", styles["BoraSmall"])]
    doc.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)
    print(out)


if __name__ == "__main__":
    main()
