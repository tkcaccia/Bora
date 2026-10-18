"""Record actual visual inspections and gate five-survivor selection.

Never fabricate reviews from numerical scores or mark unviewed sheets viewed.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import re
from .run import digest, save
from .runtime_policy import active_policy, runtime_ranking


def record_review(root, case, sheet_sha256, observation, observed_methods):
    root = Path(root)
    manifest = json.loads((root/"VISUAL_MANIFEST.json").read_text())["contact_sha256"]
    protocol = json.loads((root/"PROTOCOL.json").read_text())
    if case not in manifest or sheet_sha256 != manifest[case]:
        raise ValueError("Review does not identify the frozen sheet")
    if digest(root/"cases"/case/"contact.jpg") != sheet_sha256:
        raise ValueError("The viewed sheet changed")
    if set(observed_methods) != set(protocol["methods"]):
        raise ValueError("Review must cover every method visible in this case")
    if len(observation.strip()) < 30:
        raise ValueError("A concrete visual observation is required")
    record = json.loads((root/"VISUAL_REVIEW.json").read_text())
    record["reviewed"][case] = dict(
        sheet_sha256=sheet_sha256, observed_methods=observed_methods,
        observation=observation, reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
        review_scope="contact-sheet overview; native-resolution zooms must be noted separately")
    save(root/"VISUAL_REVIEW.json", record)


def record_batch(root, notes, journal):
    """Store explicit, actually observed notes outside a result rsync target.

    This never generates observations or fills unviewed cases. It mirrors the
    signed sheet records so subsequent result synchronization cannot erase work.
    """
    root, journal = Path(root), Path(journal)
    protocol = json.loads((root/"PROTOCOL.json").read_text())
    manifest = json.loads((root/"VISUAL_MANIFEST.json").read_text())["contact_sha256"]
    previous = json.loads(journal.read_text()) if journal.exists() else dict(reviewed={})
    for case, observation in notes.items():
        record_review(root, case, manifest[case], observation, protocol["methods"])
        entry = json.loads((root/"VISUAL_REVIEW.json").read_text())["reviewed"][case]
        previous["reviewed"][case] = entry
    journal.parent.mkdir(parents=True, exist_ok=True)
    save(journal, previous)


def require_review(root):
    root = Path(root)
    protocol = json.loads((root/"PROTOCOL.json").read_text())
    if protocol["pilot"]:
        raise ValueError("An engineering pilot cannot complete a cycle")
    from .benchmark_scope import validate_scope
    validate_scope(protocol)
    candidates = set(protocol["roster"])
    if len(candidates) != 20 or not candidates <= set(protocol["methods"]):
        raise ValueError("Every cycle needs twenty candidates")
    visual = json.loads((root/"VISUAL_MANIFEST.json").read_text())["contact_sha256"]
    reviewed = json.loads((root/"VISUAL_REVIEW.json").read_text())["reviewed"]
    if set(protocol["cases"]) != set(visual) or set(visual) != set(reviewed):
        raise ValueError("Every benchmark case requires an explicit visual inspection")
    for case, expected in visual.items():
        entry = reviewed[case]
        if entry["sheet_sha256"] != expected or digest(root/"cases"/case/"contact.jpg") != expected:
            raise ValueError("Reviewed visual changed")
        if set(entry["observed_methods"]) != set(protocol["methods"]):
            raise ValueError("A case review omitted methods")
    return protocol


def publish_selection_status(root):
    """Refresh report-only status from signed evidence; never rescore masks."""
    root = Path(root)
    selection = json.loads((root/"SELECTION.json").read_text())
    for name, field in (("RESULTS.json", "results_sha256"),
                        ("VISUAL_REVIEW.json", "review_sha256"),
                        ("PROTOCOL.json", "protocol_sha256")):
        if digest(root/name) != selection[field]:
            raise ValueError("Cannot publish changed signed cycle evidence")
    gallery = root/"gallery.html"
    if not gallery.exists():
        return None
    cycle = selection["cycle"]
    html = gallery.read_text()
    html = html.replace("CYCLE AWAITING VISUAL REVIEW",
                        f"CYCLE {cycle} REVIEWED AND SELECTED — {cycle}/20 CYCLES COMPLETE")
    html = re.sub(r"<title>.*?</title>", f"<title>Bora cycle {cycle}: all results</title>", html, count=1)
    protocol=json.loads((root/'PROTOCOL.json').read_text())
    controls=set(protocol['methods'])-set(protocol['roster'])
    control_description='both controls' if len(controls)==2 else f'all {len(controls)} controls'
    info = ('<p id="cycle-selection">Selected: '
            +escape(", ".join(selection["survivors"]))
            +f'. All twenty candidates and {control_description} remain visible. '
            +'<a href="SELECTION.json">All selection and rejection reasons</a>.</p>')
    if 'id="cycle-selection"' in html:
        html = re.sub(r'<p id="cycle-selection">.*?</p>', info, html, count=1)
    else:
        html = html.replace("</h1>", "</h1>"+info, 1)
    gallery.write_text(html)
    from .selected_viewer import publish_selected
    selected_viewer = publish_selected(root)
    record = dict(cycle=cycle, gallery="gallery.html", gallery_sha256=digest(gallery),
        selection_sha256=digest(root/"SELECTION.json"),
        requested_delivery="Open selected.html showing only the five survivors after each completed cycle",
        predictions_modified=False, scores_modified=False, contact_sheets_modified=False,
        **selected_viewer)
    save(root/"VIEWER_READY.json", record)
    return record


def select(root, survivors, tradeoffs):
    root = Path(root)
    protocol = require_review(root)
    candidates = set(protocol["roster"])
    if len(survivors) != 5 or len(set(survivors)) != 5 or not set(survivors) <= candidates:
        raise ValueError("Exactly five distinct candidate survivors are required")
    if set(tradeoffs) != candidates or any(len(s.strip()) < 30 for s in tradeoffs.values()):
        raise ValueError("Record benefits/limitations or a rejection reason for all twenty")
    results = json.loads((root/"RESULTS.json").read_text())
    ranks = {row["method"]: row for row in results["ranking"]}
    if any(not ranks[key]["complete"] for key in survivors):
        raise ValueError("A survivor must have predictions and scores for every case")
    raw_best = max(ranks[key]["mean_iou"] for key in candidates if ranks[key]["complete"])
    raw_leaders = sorted(key for key in candidates if ranks[key]["complete"]
                         and ranks[key]["mean_iou"] == raw_best)
    policy = active_policy(protocol)
    if policy is not None:
        # Recompute from frozen per-case IoU/times rather than trust a stale
        # derived ranking. Historical protocols keep their original rule.
        ranks = {row["method"]: row for row in runtime_ranking(results, policy)}
    metric = "runtime_adjusted_score" if policy is not None else "mean_iou"
    best = max(ranks[key][metric] for key in candidates if ranks[key]["complete"])
    leaders = sorted(key for key in candidates if ranks[key]["complete"]
                     and ranks[key][metric] == best)
    if not set(leaders).intersection(survivors):
        raise ValueError("The preregistered primary selection leader must survive")
    completed = dict(
        cycle=protocol["cycle"], survivors=survivors,
        rejected=sorted(candidates-set(survivors)), tradeoffs=tradeoffs,
        results_sha256=digest(root/"RESULTS.json"),
        review_sha256=digest(root/"VISUAL_REVIEW.json"),
        protocol_sha256=digest(root/"PROTOCOL.json"),
        selection_code_sha256=digest(Path(__file__)), primary_accuracy_leaders=raw_leaders,
        primary_selection_leaders=leaders, primary_selection_metric=metric,
        runtime_penalty_policy=policy,
        no_next_cycle_implemented_by_this_selection=True,
        completed_at_utc=datetime.now(timezone.utc).isoformat())
    save(root/"SELECTION.json", completed)
    save(root/"STAGE.json", dict(state="cycle_reviewed_and_selected",
                                cycle=protocol["cycle"], completed_cycles=protocol["cycle"],
                                complete_goal=False))
    publish_selection_status(root)
    return completed


def validate_children(selection, children):
    if len(children) != 15 or len({c["id"] for c in children}) != 15:
        raise ValueError("Five survivors require exactly fifteen distinct new children")
    expected = {parent: 3 for parent in selection["survivors"]}
    if dict(Counter(child["parent"] for child in children)) != expected:
        raise ValueError("Each of the five survivors requires three innovations")
    if len({child["intervention"] for child in children}) != 15:
        raise ValueError("Children need distinct intervention tags, not parameter variants")
    for child in children:
        for field in ("equation_or_rule", "failure_hypothesis", "difference_from_parent"):
            if len(child[field].strip()) < 30:
                raise ValueError(f"Child needs a concrete {field}")
        if child.get("port_only") or child.get("parameter_sweep"):
            raise ValueError("Ports and parameter sweeps are not new algorithms")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--sheet-sha256", required=True)
    parser.add_argument("--observation", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    args = parser.parse_args()
    record_review(args.root, args.case, args.sheet_sha256, args.observation, args.methods)


if __name__ == "__main__":
    main()
