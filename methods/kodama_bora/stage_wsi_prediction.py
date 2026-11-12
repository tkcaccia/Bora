#!/usr/bin/env python3
"""Stage, but do not publish, one Benchmark 2 PLS tissue-type result.

The accepted Benchmark 1 model and the existing Bora polygon geometry must
match exactly. The web derivative retains anonymous group IDs for the viewer;
PLS predictions are added as properties and a table, without refitting.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import html
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
FROZEN = HERE / "TRAINING_SET_FROZEN.json"
sys.path.insert(0, str(HERE.parents[1] / "src"))
from bora.region_identity import geometry_digest


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def web_json(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return read(path)


def stage(site: Path, record_id: str, classified: Path, model_receipt: Path,
          output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    accepted = read(model_receipt)
    frozen = read(FROZEN)
    if (accepted.get("status") != "ACCEPTED" or
            accepted.get("training_set_sha256") != sha(FROZEN) or
            accepted.get("training_images") != frozen["admitted_training_images"] or
            accepted.get("training_groups") != frozen["class_eligible_groups"] or
            accepted.get("fit_scope") != "Benchmark 1 admitted training images only" or
            sha(Path(accepted["model_path"])) != accepted.get("model_sha256")):
        raise ValueError("The exact Benchmark 1 PLS training/model receipt is absent")
    report_path = classified / "report.json"
    named_path = classified / "tissue-annotations.geojson"
    report = read(report_path)
    named = read(named_path)
    if (report.get("format") != "bora-region-identity-v1" or
            report.get("complete") is not True or
            report.get("geometry_unchanged") is not True or
            report.get("input_sha256", {}).get("model") != accepted["model_sha256"] or
            report.get("output_sha256", {}).get("tissue-annotations.geojson") != sha(named_path) or
            report.get("biological_accuracy_qualified") is not False or
            named.get("tissue_identity_model_sha256") != accepted["model_sha256"] or
            named.get("tissue_identity_status") != "unvalidated_model_transfer"):
        raise ValueError("The WSI PLS result is incomplete or uses another model")
    manifest_path = site / "geojson-inspectors.json"
    manifest = read(manifest_path)
    selected = [row for row in manifest["records"] if row["id"] == record_id]
    if len(selected) != 1 or selected[0].get("section") != "visium-hd":
        raise ValueError("One Whole Slide Images inspector record is required")
    record = selected[0]
    if record.get("fastpls_tissue_names_available") is not False:
        raise ValueError("This WSI is already annotated or its baseline changed")
    page_relative = record["page"].split("#", 1)[0]
    page_path = site / page_relative
    page = page_path.read_text()
    phrase = "A fastPLS tissue-name assignment is not available for this saved result."
    anchor = "<h2>Full-slide overview</h2>"
    if page.count(phrase) != 1 or page.count(anchor) != 1:
        raise ValueError("Whole-slide page template or previous annotation changed")
    source_path = site / record["geometry"]
    if sha(source_path) != record["stored_geojson_sha256"]:
        raise ValueError("Saved WSI GeoJSON bytes changed")
    source = web_json(source_path)
    if source_path.suffix == ".gz":
        with gzip.open(source_path, "rb") as stream:
            original_decoded_sha = hashlib.sha256(stream.read()).hexdigest()
    else:
        original_decoded_sha = sha(source_path)
    if original_decoded_sha != record["decoded_geojson_sha256"]:
        raise ValueError("Saved WSI GeoJSON content changed")
    if (source.get("type") != "FeatureCollection" or
            len(source["features"]) != record["source_feature_count"] or
            geometry_digest(source["features"]) != report["geometry_sha256"] or
            geometry_digest(named["features"]) != report["geometry_sha256"] or
            len(named["features"]) != len(source["features"])):
        raise ValueError("PLS result does not describe the published WSI geometry")
    groups = {int(row["source_group_id"]): row for row in report["groups"]}
    if (len(groups) != report["group_count"] or
            set(groups) != {int(row["value"]) for row in record["groups"]}):
        raise ValueError("PLS predictions do not cover every published Bora group")
    derivative = copy.deepcopy(source)
    for original, prediction, target in zip(source["features"], named["features"],
                                             derivative["features"]):
        group_id = int(original["properties"]["value"])
        group = groups[group_id]
        props = prediction["properties"]
        if (original["geometry"] != prediction["geometry"] or
                props.get("source_group_id") != group_id or
                props.get("predicted_tissue_id") != group["prediction_id"] or
                props.get("predicted_tissue") != group["prediction_name"]):
            raise ValueError("Feature-level PLS identity differs from the group report")
        target["properties"].update(predicted_tissue_id=group["prediction_id"],
            predicted_tissue=group["prediction_name"],
            tissue_identity_model_sha256=accepted["model_sha256"],
            tissue_identity_status="unvalidated_model_transfer")
    derivative.update(tissue_identity_model_sha256=accepted["model_sha256"],
        benchmark1_training_set_sha256=accepted["training_set_sha256"],
        tissue_identity_status="unvalidated_model_transfer",
        biological_accuracy_qualified=False)
    if geometry_digest(derivative["features"]) != report["geometry_sha256"]:
        raise ValueError("Website derivative changed Bora geometry")
    relative_geojson = str(Path(record["geometry"]).with_name(
        Path(record["geometry"]).stem.replace(".geojson", "") + "-pls.geojson.gz"))
    # The named artifact is kept separate; the original anonymous GeoJSON remains downloadable.
    target_geojson = output / relative_geojson
    target_geojson.parent.mkdir(parents=True, exist_ok=True)
    with target_geojson.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0,
                           compresslevel=9) as packed:
            packed.write(json.dumps(derivative, separators=(",", ":"),
                                    allow_nan=False).encode())
    with gzip.open(target_geojson, "rb") as packed:
        decoded_sha = hashlib.sha256(packed.read()).hexdigest()
    record["geometry"] = relative_geojson
    record["geojson_bytes"] = target_geojson.stat().st_size
    record["stored_geojson_sha256"] = sha(target_geojson)
    record["decoded_geojson_sha256"] = decoded_sha
    record["label_source"] = "Benchmark 1 trained fastPLS prediction on Bora groups"
    record["label_description"] = (
        "Each numbered Bora group has a PLS tissue-type prediction from the exact "
        "Benchmark 1 training split. Predictions are model transfer, not validated "
        "biological accuracy for this slide. Polygon geometry is unchanged.")
    record["fastpls_tissue_names_available"] = True
    record["tissue_identity_model_sha256"] = accepted["model_sha256"]
    record["benchmark1_training_set_sha256"] = accepted["training_set_sha256"]
    for item in record["groups"]:
        group = groups[int(item["value"])]
        item["name"] = f"Group {int(item['value']):02d} — PLS: {group['prediction_name'].replace('_', ' ')}"
        item["predicted_tissue_id"] = group["prediction_id"]
        item["predicted_tissue"] = group["prediction_name"]
    rows = "".join(
        "<tr><th scope=\"row\">Group %02d</th><td>%s</td><td>%.3f</td></tr>" %
        (group_id, html.escape(group["prediction_name"].replace("_", " ")),
         group["physical_area_um2"] / 1e6)
        for group_id, group in sorted(groups.items()))
    path_from_page = os.path.relpath(relative_geojson, Path(page_relative).parent)
    block = ("<section id=\"pls-tissue-predictions\"><h2>PLS tissue-type predictions</h2>"
        "<p>One prediction per whole Bora group, including disconnected pieces. "
        "The PLS model was trained only on the same 397 admitted training images "
        "used by Benchmark 1. These labels are model predictions for this slide; "
        "they are not a measured tissue reference or a slide-level accuracy claim. "
        "Bora polygon boundaries remain unchanged.</p>"
        "<table><thead><tr><th>Bora group</th><th>Predicted tissue type</th>"
        "<th>Group area (mm²)</th></tr></thead><tbody>" + rows + "</tbody></table>"
        f"<p><a href=\"{html.escape(path_from_page, quote=True)}\" download>Download "
        "GeoJSON with PLS tissue predictions</a>. "
        f"Model SHA-256: <code>{accepted['model_sha256']}</code>; "
        f"Benchmark 1 training-set SHA-256: <code>{accepted['training_set_sha256']}</code>."
        "</p></section>")
    page = page.replace(phrase, "PLS tissue-type predictions are shown below for this saved result.", 1)
    page = page.replace(anchor, block + anchor, 1)
    staged_page = output / page_relative
    staged_page.parent.mkdir(parents=True, exist_ok=True)
    staged_page.write_text(page)
    staged_manifest = output / "geojson-inspectors.json"
    staged_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    receipt = dict(status="STAGED_PLS_WSI", record_id=record_id,
        source_site_manifest_sha256=sha(manifest_path), source_page_sha256=sha(page_path),
        source_geometry_sha256=report["geometry_sha256"],
        model_acceptance_receipt_sha256=sha(model_receipt),
        model_sha256=accepted["model_sha256"],
        benchmark1_training_set_sha256=accepted["training_set_sha256"],
        classification_report_sha256=sha(report_path),
        staged_files_sha256={str(path.relative_to(output)): sha(path)
            for path in (target_geojson, staged_page, staged_manifest)})
    (output / "STAGED_PLS_WSI.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--classified", type=Path, required=True)
    parser.add_argument("--model-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.site, args.record_id, args.classified,
                           args.model_receipt, args.output), indent=2))


if __name__ == "__main__":
    main()
