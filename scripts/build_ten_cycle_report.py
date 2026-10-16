#!/usr/bin/env python3
"""Create the final PDF for the ten 216-pixel internal-border experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


NAMES = {
    "Visium_HD_11mm_Human_Pancreas_tissue_image": "Pancreas",
    "Visium_HD_3prime_Human_Lymph_Node_tissue_image": "Lymph node",
    "Visium_HD_3prime_Human_Tonsil_tissue_image": "Tonsil",
    "Visium_HD_6p5mm_Human_Brain_Cancer_tissue_image": "Brain cancer",
}


def table(rows, widths):
    result = Table(rows, colWidths=widths, repeatRows=1)
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#163b50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#eef4f5")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), .25, colors.HexColor("#c6d5da")),
        ("FONTSIZE", (0, 0), (-1, -1), 7.4),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return result


def fmt(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def read_border(root: Path, number: int, sample: str):
    evaluation = (root / "baseline-pre-island-cleanup-vector-eval" if number == 0
                  else root / f"output-bora-cycle-{number:02d}-vector-eval")
    path = evaluation / "216" / sample / "border_comparison_r64.json"
    return json.loads(path.read_text())["internal"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.benchmark_root
    overview = json.loads((root / "bora_216_island_ten_cycle_summary.json").read_text())
    selected = overview["selected_cycle"]
    if len(overview["cycles"]) != 10:
        raise ValueError("Expected ten completed experiments")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BoraTitle", parent=styles["Title"], fontSize=17,
                              leading=20, textColor=colors.HexColor("#163b50"), spaceAfter=10))
    styles.add(ParagraphStyle(name="BoraBody", parent=styles["BodyText"],
                              fontSize=8.5, leading=12, spaceAfter=7))
    story = [Paragraph("Bora: ten 216-pixel foreground-island cleanup cycles", styles["BoraTitle"]),
             Paragraph("Four original H&amp;E images and the existing 216-pixel tiled masks were reused. "
                       "The baseline was the best balanced local-color wand plus smoothed-contour result "
                       "from the preceding sweep. Each new candidate absorbed small, enclosed foreground "
                       "class islands into a dominant neighboring class, then exported a pyramidal mask "
                       "and final GeoJSON for every tissue. "
                       "The final GeoJSON was rasterized at native image resolution for scoring against the "
                       "original GeoJSON annotations.", styles["BoraBody"]),
             Paragraph("Accuracy was measured only within 64 pixels of a ground-truth or original-tile "
                       "border between different non-background tissue classes, excluding ground-truth "
                       "background. A candidate was accepted only when internal-band macro IoU and pixel "
                       "accuracy did not decrease in any applicable tissue, all ground-truth classes "
                       "remained represented, and at least one score improved. Pancreas has one class, "
                       "so its internal-border score is not applicable.", styles["BoraBody"]),
             Paragraph("Ten candidate decisions", styles["Heading2"])]
    rows = [["Cycle", "Area / surround / round", "Lymph ΔIoU", "Tonsil ΔIoU", "Brain ΔIoU", "Decision"]]
    for item in overview["cycles"]:
        decision = json.loads(Path(item["internal_border_decision"]).read_text())
        deltas = {NAMES[row["sample"]]: row["boundary_metrics"].get("internal_macro_iou", {}).get("change")
                  for row in decision["per_tissue"]}
        rows.append([f"{item['cycle']:02d}",
                     f"{item['max_island_area']} / {item['surround_fraction']:.2f} / "
                     f"{item.get('min_circularity', 0.0):.2f}",
                     fmt(deltas.get("Lymph node"), 5), fmt(deltas.get("Tonsil"), 5),
                     fmt(deltas.get("Brain cancer"), 5),
                     "ACCEPT" if item["accepted"] else "reject"])
    story += [table(rows, [42, 126, 82, 82, 82, 88]), Spacer(1, 8),
              Paragraph("Deltas compare each candidate with the best previously accepted cycle; "
                        "the acceptance gate also checks pixel accuracy for each multiclass tissue.",
                        styles["BoraBody"]),
              Paragraph(f"Selected result: {'baseline' if selected == 0 else f'cycle {selected:02d}'}", styles["Heading2"])]
    rows = [["Tissue", "Tile IoU", "Baseline IoU", "Selected IoU", "Tile acc.", "Selected acc.", "F1 8px"]]
    for sample, name in NAMES.items():
        first = read_border(root, 0, sample)
        final = read_border(root, selected, sample)
        rows.append([name, fmt(first["tile"]["macro_iou"]), fmt(first["bora"]["macro_iou"]),
                     fmt(final["bora"]["macro_iou"]), fmt(first["tile"]["accuracy"]),
                     fmt(final["bora"]["accuracy"]), fmt(final["bora"]["boundary_f1_8px"])] )
    story += [table(rows, [87, 65, 75, 77, 65, 75, 58]), Spacer(1, 8),
              Paragraph("The 8-pixel boundary F1 measures contour location and is diagnostic; "
                        "the predeclared acceptance gate uses internal-band macro IoU and accuracy.",
                        styles["BoraBody"])]
    f1_changes = []
    for sample, name in NAMES.items():
        if name == "Pancreas":
            continue
        before = read_border(root, 0, sample)["bora"]["boundary_f1_8px"]
        after = read_border(root, selected, sample)["bora"]["boundary_f1_8px"]
        f1_changes.append(f"{name} {after-before:+.4f}")
    story += [Paragraph("Change in strict 8-pixel contour F1 versus baseline: " +
                        "; ".join(f1_changes) + ". This measures a different tradeoff from "
                        "region overlap in the fixed border band.", styles["BoraBody"]),
              Paragraph("Runtime and output integrity", styles["Heading2"])]
    rows = [["Tissue", "Baseline time", "Selected time", "Annotations", "Class-changed px", "Timing note"]]
    for sample, name in NAMES.items():
        selected_root = (root / "baseline-pre-island-cleanup" if selected == 0
                         else root / f"output-bora-cycle-{selected:02d}")
        folder = selected_root / "216" / sample
        comparison = json.loads((folder / "comparison.json").read_text())
        source_comparison = json.loads((root / "baseline-pre-island-cleanup" / "216" / sample /
                                        "comparison.json").read_text())
        vector = json.loads((folder / "bora_refined.geojson").read_text())
        eval_root = (root / "baseline-pre-island-cleanup-vector-eval" if selected == 0
                     else root / f"output-bora-cycle-{selected:02d}-vector-eval")
        border = json.loads((eval_root /
                             "216" / sample / "border_comparison_r64.json").read_text())
        note = "estimated" if comparison.get("timing_estimated_from_file_timestamps") else "measured"
        rows.append([name, f"{source_comparison['refinement_seconds']:.1f} s",
                     f"{comparison['refinement_seconds']:.1f} s", len(vector["features"]),
                     f"{border['changed_between_tissue_labels_pixels']:,}", note])
    story += [table(rows, [87, 83, 83, 75, 91, 83]), Spacer(1, 6),
              Paragraph("Runtime includes the prior image-guided refinement, pyramidal export and GeoJSON conversion; "
                        "selected-cycle time additionally includes island cleanup and its final export. "
                        "Evaluation time is excluded. "
                        "Class-changed pixels are foreground pixels assigned a different foreground class "
                        "in the final GeoJSON than in the tile input.", styles["BoraBody"]),
              Paragraph("Interpretation", styles["Heading2"])]
    accepted = [f"{x['cycle']:02d}" for x in overview["cycles"] if x["accepted"]]
    story.append(Paragraph(
        f"Accepted candidate cycles: {', '.join(accepted) if accepted else 'none'}. "
        f"The selected result is {'the baseline' if selected == 0 else f'cycle {selected:02d}'}. "
        "The ten experiments tune on the same four slides "
        "used for reporting, so these numbers describe this benchmark and need independent-slide "
        "validation before generalization. Further work should use local stain-normalized color "
        "likelihoods where adjacent classes overlap in LAB space, with a topology-preserving "
        "regularizer to prevent tiny islands while allowing the internal border to follow image texture.",
        styles["BoraBody"]))
    story += [Paragraph("Next boundary candidate", styles["Heading2"]),
              Paragraph("Residual straight tile steps remain on some internal tissue interfaces. "
                        "The current balanced wand scores each class independently from local LAB color "
                        "and retains a spatial preference for the original tile label. A follow-up "
                        "candidate will instead optimize each adjacent class pair as a single seeded "
                        "interface. Protected cores anchor both tissue labels; local color likelihood "
                        "and H&amp;E edge strength guide the boundary; a contour-length penalty discourages "
                        "small loops and stair steps. The same native-resolution GeoJSON border-band "
                        "acceptance gate should decide whether to use it. A preliminary 1024-pixel "
                        "crop trial of seeded GrabCut improved accuracy in a 64-pixel internal-border "
                        "band in 6/7 lymph-node, 6/9 tonsil and 8/9 brain crops; those selected "
                        "crops are exploratory and "
                        "are not a substitute for full-slide validation.", styles["BoraBody"]),
              Paragraph("Method references: Boykov and Jolly, ICCV 2001, seeded graph cuts; "
                        "Rother et al., SIGGRAPH 2004, GrabCut; Zhu and Yuille, TPAMI 1996, "
                        "region competition. Public starting points: OpenCV GrabCut, PyMaxflow, "
                        "and scikit-image random walker.", styles["BoraBody"])]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(args.output), pagesize=(612, 792), leftMargin=50,
                            rightMargin=50, topMargin=45, bottomMargin=46)
    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#657984"))
        canvas.drawString(50, 25, "Bora 216-pixel study | 15 October 2026")
        canvas.drawRightString(562, 25, f"Page {document.page}")
        canvas.restoreState()
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(args.output)


if __name__ == "__main__":
    main()
