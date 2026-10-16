#!/usr/bin/env python3
"""Create the final three-size PDF benchmark from all 200 decision records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import LongTable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = Path("/mnt/sata_ssd/benchmarks/Bora/benchmark-regenerated-2026-10-15")
DEFAULT = ROOT / "output/pdf/Bora_56_90_216_cycles_51_to_250_report.pdf"
NAVY = colors.HexColor("#123047")
PALE = colors.HexColor("#F2F5F7")


def fmt(value, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def signed(value, digits: int = 5) -> str:
    return "n/a" if value is None else f"{value:+.{digits}f}"


def metric(root: Path, size: int, sample: str) -> dict:
    path = Path(str(root)+"-vector-eval") / str(size) / sample / "border_comparison_r64.json"
    return json.loads(path.read_text())["internal"]["bora"]


def timing(root: Path, size: int, sample: str) -> float:
    return json.loads((root / str(size) / sample / "comparison.json").read_text())[
        "refinement_seconds"]


def style_table(table: Table | LongTable, font_size: float = 8) -> None:
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size+2),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))


def footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawString(18*mm, 12*mm, "Bora | regenerated masks | internal tissue borders")
    canvas.drawRightString(landscape(A4)[0]-18*mm, 12*mm,
                           f"Page {document.page}")
    canvas.restoreState()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    state = json.loads((BENCHMARK / "cycles_51_to_250.json").read_text())
    if not state["finished"] and not args.allow_partial:
        raise SystemExit("Cycle 250 is not complete")
    records = state["cycles"]
    incumbent = state["incumbent"]
    best = (BENCHMARK / "baseline-accepted" if incumbent == "baseline" else
            BENCHMARK / f"output-bora-cycle-{incumbent:03d}")
    baseline = BENCHMARK / "baseline-accepted"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BoraTitle250", parent=styles["Title"],
                              textColor=NAVY, fontSize=18, leading=21, spaceAfter=8))
    styles.add(ParagraphStyle(name="BoraBody250", parent=styles["BodyText"],
                              fontSize=9, leading=12, spaceAfter=7))
    styles.add(ParagraphStyle(name="BoraSmall250", parent=styles["BodyText"],
                              fontSize=8, leading=11, spaceAfter=6))
    styles.add(ParagraphStyle(name="BoraSection250", parent=styles["Heading2"],
                              textColor=NAVY, spaceBefore=12, spaceAfter=6))
    story = [
        Paragraph("Bora: triangular-mesh contour search", styles["BoraTitle250"]),
        Paragraph(f"Regenerated 56, 90, and 216-pixel tile masks; 4 slides per size; "
                  f"cycles 51-250; completed through {state['completed_through']}. "
                  f"Best accepted candidate: {incumbent}.", styles["BoraBody250"]),
        Paragraph("The current input GeoJSONs were frozen and hashed before tile-mask "
                  "regeneration. Lymph node now has two foreground classes instead of "
                  "five, and brain has five instead of four. Scores here must not be "
                  "compared numerically with the earlier cycle-12/50 benchmark.",
                  styles["BoraSmall250"]),
        Paragraph("All candidates exported 12 full-resolution GeoJSONs and were scored "
                  "only in a 64-pixel band around internal ground-truth or coarse-tile "
                  "interfaces. Pancreas has a single foreground class and no internal "
                  "tissue interface. The table reports 8/16/32-pixel boundary F1, "
                  "average symmetric surface distance (ASSD), robust 95th-percentile "
                  "Hausdorff distance (HD95), false tile-edge pixels, retained classes, "
                  "and complete experimental pipeline time. The timing includes "
                  "intermediate export for reproducibility.", styles["BoraBody250"]),
        Paragraph("A candidate replaced the incumbent only when internal-border macro "
                  "IoU and accuracy did not decline for any slide/size, neither false "
                  "tile-edge nor off-band internal-edge pixels increased, every truth class "
                  "remained represented, "
                  "and at least one quality or speed value improved. Other contour "
                  "metrics are reported as diagnostics.", styles["BoraSmall250"]),
        Paragraph("Accepted output by tissue and tile size", styles["BoraSection250"]),
    ]
    names = {"Visium_HD_11mm_Human_Pancreas_tissue_image": "Pancreas",
             "Visium_HD_3prime_Human_Lymph_Node_tissue_image": "Lymph node",
             "Visium_HD_3prime_Human_Tonsil_tissue_image": "Tonsil",
             "Visium_HD_6p5mm_Human_Brain_Cancer_tissue_image": "Brain cancer"}
    rows = [["Tissue", "Tile", "IoU base to best", "Accuracy base to best",
             "16-px F1 base to best", "HD95 base to best", "False tile edges", "Time base to best"]]
    for size in (56, 90, 216):
        for sample, label in names.items():
            a, b = metric(baseline, size, sample), metric(best, size, sample)
            old_seconds, new_seconds = timing(baseline, size, sample), timing(best, size, sample)
            rows.append([label, str(size),
                         f"{fmt(a['macro_iou'])} to {fmt(b['macro_iou'])}",
                         f"{fmt(a['accuracy'])} to {fmt(b['accuracy'])}",
                         f"{fmt(a['boundary_f1_16px'])} to {fmt(b['boundary_f1_16px'])}",
                         f"{fmt(a['hausdorff95_pixels'],1)} to {fmt(b['hausdorff95_pixels'],1)}",
                         f"{a['false_tile_imprint_edge_pixels']:,} to {b['false_tile_imprint_edge_pixels']:,}",
                         f"{old_seconds/60:.1f} to {new_seconds/60:.1f} min"])
    table = Table(rows, colWidths=[27*mm, 12*mm, 34*mm, 35*mm,
                                   39*mm, 34*mm, 37*mm, 37*mm])
    style_table(table, 7.2)
    story += [table, Spacer(1, 5*mm),
              Paragraph("All 200 numbered decisions", styles["BoraSection250"])]
    decisions = [["Cycle", "Mesh", "Support", "Band", "RGB", "Round", "Accepted",
                  "Min IoU change", "Min accuracy change", "Speed ratio"]]
    for record in records:
        number = record["cycle"]
        spec = record["specification"]
        decision = json.loads(Path(record["decision"]).read_text())
        deltas = [(r["metrics"]["macro_iou"]["improvement"],
                   r["metrics"]["accuracy"]["improvement"])
                  for r in decision["rows"]]
        iou = [x[0] for x in deltas if x[0] is not None]
        acc = [x[1] for x in deltas if x[1] is not None]
        decisions.append([str(number), str(spec["mesh_spacing"]),
                          str(spec["support_radius"]), str(spec["band_radius"]),
                          fmt(spec["color_weight"], 2),
                          str(spec["post_smooth_radius"]),
                          "Yes" if record["accepted"] else "No",
                          signed(min(iou) if iou else None),
                          signed(min(acc) if acc else None),
                          fmt(decision["total_speedup"], 2)])
    long = LongTable(decisions, repeatRows=1,
                     colWidths=[18*mm, 18*mm, 22*mm, 18*mm, 18*mm,
                                18*mm, 20*mm, 35*mm, 39*mm, 26*mm])
    style_table(long, 7.3)
    story += [long, Spacer(1, 4*mm),
              Paragraph("This is exploratory optimization on the same four slides used "
                        "for evaluation. Independent pathologist-traced interfaces are "
                        "needed to estimate generalization to unseen tissue and stains.",
                        styles["BoraSmall250"])]
    doc = SimpleDocTemplate(str(args.output), pagesize=landscape(A4),
                            leftMargin=18*mm, rightMargin=18*mm,
                            topMargin=17*mm, bottomMargin=18*mm)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(args.output)


if __name__ == "__main__":
    main()
