"""Read-only full-scope audit of a frozen, scored, rendered cycle.

This checks saved evidence; it never regenerates predictions, scores, visual
reviews or selection decisions. A passing audit is not an actual visual review.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile

from .run import aggregate, digest, palette_bytes, verify, resize
from .benchmark_scope import validate_scope, requires_parent_parity


def load(path):
    return json.loads(Path(path).read_text())


def glass_changes(parent, child):
    """A glass-only intervention may remove labels, never replace tissue IDs."""
    if np.any((child != parent) & (child != 0)):
        raise ValueError("Glass child changes nonzero tissue labels")
    return int(np.count_nonzero((parent > 0) & (child == 0)))


def audit(source, root, previous=None, executed_code=None):
    source, root = Path(source), Path(root)
    protocol = load(root/"PROTOCOL.json")
    cases, methods = protocol["cases"], protocol["methods"]
    if len(cases) != len(set(cases)) or len(methods) != len(set(methods)):
        raise ValueError("Duplicate case or method in protocol")
    validate_scope(protocol, pilot=protocol["pilot"])
    if not protocol["pilot"] and (len(protocol["roster"]) != 20
                                  or set(protocol["roster"])-set(methods)):
        raise ValueError("A full-cycle audit requires twenty candidates")
    frozen = load(root/"ALL_PREDICTIONS_FROZEN.json")
    if frozen["protocol_sha256"] != digest(root/"PROTOCOL.json") or frozen["fine_reference_read"]:
        raise ValueError("Prediction freeze/protocol changed or read reference prematurely")
    if set(frozen["manifests"]) != set(cases):
        raise ValueError("Prediction scope differs from protocol")
    if digest(source/"PLAN.json") != protocol["input_plan_sha256"]:
        raise ValueError("Input plan changed")
    samples = {s["case"]: s for s in load(source/"PLAN.json")["samples"]}
    for name, sha in protocol["source_hashes"].items():
        if digest(root/"runtime_sources"/name) != sha:
            raise ValueError("Archived executed source changed: "+name)
        if executed_code is not None and digest(Path(executed_code)/name) != sha:
            raise ValueError("Executed code differs from frozen source: "+name)
    result = load(root/"RESULTS.json")
    if result["protocol"] != protocol:
        raise ValueError("Results describe another protocol")
    rows = {r["case"]: r for r in result["cases"]}
    if len(rows) != len(result["cases"]) or set(rows) != set(cases):
        raise ValueError("Scored scope differs from prediction scope")
    from .runtime_policy import active_policy
    if aggregate(result["cases"], methods, active_policy(protocol)) != result["ranking"]:
        raise ValueError("Saved aggregate differs from per-case scores")
    visual = load(root/"VISUAL_MANIFEST.json")["contact_sha256"]
    if set(visual) != set(cases):
        raise ValueError("Rendered scope differs from protocol")
    lineage = protocol.get("lineage")
    survivors = lineage["unchanged_survivors"] if lineage else []
    if lineage:
        previous = Path(previous or lineage["previous"])
        if digest(previous/"SELECTION.json") != lineage["selection_sha256"]:
            raise ValueError("Previous selection changed")
        selection = load(previous/"SELECTION.json")
        for file, field in (("RESULTS.json", "results_sha256"), ("VISUAL_REVIEW.json", "review_sha256")):
            if digest(previous/file) != selection[field]:
                raise ValueError("Previous selected evidence changed: "+file)
        if survivors != selection["survivors"]:
            raise ValueError("Survivors differ from actual previous selection")
    counts = Counter()
    failures, optimizer, glass = {}, {}, []
    png_hashes = {}
    for case in cases:
        row, sample = rows[case], samples[case]
        src, out = source/"cases"/case, root/"cases"/case
        if row["sample"] != sample or load(out/"SCORES.json") != row:
            raise ValueError("Case scores/sample differ from saved result: "+case)
        if digest(out/"PREDICTIONS.json") != frozen["manifests"][case]:
            raise ValueError("Prediction manifest changed: "+case)
        prediction = load(out/"PREDICTIONS.json")
        if prediction["fine_reference_read"] or set(prediction["methods"]) != set(methods):
            raise ValueError("Invalid prediction scope/reference policy: "+case)
        verify(out, prediction["files_sha256"])
        counts["frozen_files"] += len(prediction["files_sha256"])
        inputs = load(src/"INPUTS_FROZEN.json")
        if digest(src/"INPUTS_FROZEN.json") != prediction["input_manifest_sha256"]:
            raise ValueError("Input manifest changed: "+case)
        verify(src, inputs["files_sha256"])
        counts["input_files"] += len(inputs["files_sha256"])
        if digest(src/"ensemble/representation.png") != prediction["rgb_sha256"]:
            raise ValueError("Fixed RGB changed: "+case)
        if digest(src/"reference.native.tif") != sample["reference_native_sha256"]:
            raise ValueError("Original scoring reference changed: "+case)
        coarse = tifffile.imread(src/"coarse.tif")
        support = tifffile.imread(src/"tissue.native.tif").astype(bool)
        constant_expected = None
        if inputs.get('single_eligible_class') and protocol.get('study_protocol', {}).get('degenerate_evidence_policy'):
            completed = tifffile.imread(src/'coarse.completed.tif')
            constant_expected = resize(completed, sample['native_shape_yx'], True)
            constant_expected[~support] = 0
        allowed = set(np.unique(coarse)) | {0}
        mask_paths = {}
        owned_supports={};native_white=None
        for method in methods:
            saved, executed = row["methods"][method], prediction["methods"][method]
            if any(saved[field] != value for field, value in executed.items()):
                raise ValueError("Executed method record changed: "+case+"/"+method)
            if load(out/method/"DETAILS.json") != executed:
                raise ValueError("Details and prediction record differ: "+case+"/"+method)
            if saved["status"] != "ok":
                failures.setdefault(method, []).append(dict(case=case, exception=saved["exception"]))
                continue
            path = out/method/"native.tif"
            mask = tifffile.imread(path)
            if mask.shape != tuple(sample["native_shape_yx"]) or mask.shape != support.shape:
                raise ValueError("Native geometry changed: "+case+"/"+method)
            from .expanded_metrics import owns_native_support,may_remove_nonwhite
            actual_support=support
            if owns_native_support(method,saved['details']):
                if method+'/support.native.tif' not in prediction['files_sha256']:
                    raise ValueError('Owned support is absent from frozen predictions')
                actual_support=tifffile.imread(out/method/'support.native.tif')>0
                if native_white is None:
                    from .diagnostics import white_regions
                    native_white=white_regions(np.array(Image.open(src/'original.native.png').convert('RGB')))[0]
                if actual_support.shape!=support.shape or (not may_remove_nonwhite(method,saved['details']) and np.any((~native_white)&~actual_support)):
                    raise ValueError('Native support changed geometry or removed non-white tissue')
                if saved['details']['native_support']['saved_support_sha256']!=prediction['files_sha256'][method+'/support.native.tif']:
                    raise ValueError('Reported native support hash differs from frozen raster')
                owned_supports[method]=actual_support
                counts['owned_native_support_contracts']+=1
            if not set(np.unique(mask)) <= allowed or np.any(mask[~actual_support] != 0):
                raise ValueError("Class IDs/background clipping violated: "+case+"/"+method)
            if method not in ("C20", "D12", "F03", "G03", "H03", "J03", "L03", "M10", "M11", "M12", "N10", "N11", "N12") and np.any(mask[actual_support] == 0):
                raise ValueError("Unexpected unassigned tissue: "+case+"/"+method)
            with Image.open(out/method/"mask.png") as display:
                if display.mode != "P" or not np.array_equal(np.asarray(display), mask):
                    raise ValueError("PNG indices differ from native prediction: "+case+"/"+method)
                if display.getpalette() != palette_bytes(sample["palette"]):
                    raise ValueError("PNG palette changed: "+case+"/"+method)
            png_hashes[case+"/"+method] = digest(out/method/"mask.png")
            mask_paths[method] = path
            counts["native_masks"] += 1
            counts["exact_display_rasters"] += 1
            detail = saved["details"]
            if constant_expected is not None:
                if (detail.get('input_contract_fallback') != 'constant-evidence-coarse-completion-v1'
                        or detail.get('refinement_performed') is not False):
                    raise ValueError('Constant RGB incorrectly claims refinement: '+case+'/'+method)
                if method in ('L03', 'M10', 'M11', 'M12', 'N10', 'N11', 'N12'):
                    glass_changes(constant_expected, mask)
                elif not np.array_equal(mask, constant_expected):
                    raise ValueError('Constant-evidence fallback changed coarse completion: '+case+'/'+method)
                counts['constant_evidence_contract_comparisons'] += 1
            numerical = {key: detail[key] for key in (
                "optimizer_success", "iterations", "termination", "objective",
                "gradient_infinity_norm", "projected_gradient_residual",
                "max_kkt_residual", "missing_known_bag_class", "converged",
                "relative_mass_error", "fixed_point_change", "message", "fits",
                "primal_residual", "dual_residual", "lagrangian_projected_residual") if key in detail}
            if numerical:
                optimizer.setdefault(method, []).append(dict(case=case, **numerical))
            if method in survivors and requires_parent_parity(protocol, case):
                old = previous/"cases"/case/method/"native.tif"
                if digest(old) != detail["previous_prediction_sha256"]:
                    raise ValueError("Previous survivor file changed: "+case+"/"+method)
                if not detail["unchanged_survivor_native_parity"] or not np.array_equal(mask, tifffile.imread(old)):
                    raise ValueError("Unchanged survivor prediction differs: "+case+"/"+method)
                counts["unchanged_survivor_comparisons"] += 1
        if 'J01' in mask_paths:
            parent=tifffile.imread(mask_paths['J01'])
            for child_id,own in owned_supports.items():
                child=tifffile.imread(mask_paths[child_id]);overlap=(parent>0)&own
                if not np.array_equal(child[overlap],parent[overlap]):
                    raise ValueError('Support-only hypothesis changed parent tissue labels')
                glass.append(dict(case=case,child=child_id,parent='J01',own_native_support=True,
                    restored_outside_common_support_pixels=int((own&~support).sum()),
                    removed_inside_common_support_pixels=int((support&~own).sum()),
                    overlapping_assigned_labels_unchanged=True))
        for child_id, parent_id in (("F03", "D10"), ("G03", "F01"), ("H03", "F01"), ("J03", "H02"), ("L03", "J01"), ("M10", "J01"), ("M11", "J01"), ("M12", "J01"), ("N10", "J01"), ("N11", "J01"), ("N12", "J01")):
            if child_id not in mask_paths or parent_id not in mask_paths:
                continue
            removed = glass_changes(tifffile.imread(mask_paths[parent_id]),
                                    tifffile.imread(mask_paths[child_id]))
            glass.append(dict(case=case, child=child_id, parent=parent_id,
                newly_unassigned_pixels=removed,
                additional_annotated_unassigned_pixels=row["methods"][child_id]["score"].get(
                    "saved_mask_annotated_unassigned_pixels", row["methods"][child_id]["score"]["annotated_unassigned_pixels"])
                -row["methods"][parent_id]["score"].get("saved_mask_annotated_unassigned_pixels",
                    row["methods"][parent_id]["score"]["annotated_unassigned_pixels"])))
        if digest(out/"contact.jpg") != visual[case]:
            raise ValueError("Contact sheet changed: "+case)
        counts["contact_sheets"] += 1
    review = load(root/"VISUAL_REVIEW.json")["reviewed"]
    return dict(cycle=protocol["cycle"], pilot=protocol["pilot"], cases=len(cases), methods=len(methods),
        **dict(counts), failed_methods=failures, numerical_records=optimizer, glass_child=glass,
        displayed_png_sha256=png_hashes, protocol_sha256=digest(root/"PROTOCOL.json"),
        results_sha256=digest(root/"RESULTS.json"), audit_source_sha256=digest(Path(__file__)),
        actual_review_entries_at_audit=len(review), predictions_modified=False, scores_modified=False,
        visual_reviews_created=False, selection_created=False,
        interpretation="Integrity and numerical-status audit only; not biological correctness or actual visual review.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--executed-code", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh audit output path")
    result = audit(args.source, args.root, args.previous, args.executed_code)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result.get(key) for key in (
        "cycle", "cases", "methods", "native_masks", "exact_display_rasters",
        "unchanged_survivor_comparisons", "failed_methods")}))


if __name__ == "__main__":
    main()
