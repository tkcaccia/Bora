"""Record actual visual inspections and gate five-survivor selection.

Never fabricate reviews from numerical scores or mark unviewed sheets viewed.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from .run import digest, save


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


def require_review(root):
    root = Path(root)
    protocol = json.loads((root/"PROTOCOL.json").read_text())
    if protocol["pilot"]:
        raise ValueError("An engineering pilot cannot complete a cycle")
    if len(protocol["cases"]) != 149:
        raise ValueError("The full 149-image benchmark is required")
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
    completed = dict(
        cycle=protocol["cycle"], survivors=survivors,
        rejected=sorted(candidates-set(survivors)), tradeoffs=tradeoffs,
        results_sha256=digest(root/"RESULTS.json"),
        review_sha256=digest(root/"VISUAL_REVIEW.json"),
        protocol_sha256=digest(root/"PROTOCOL.json"),
        no_next_cycle_implemented_by_this_selection=True,
        completed_at_utc=datetime.now(timezone.utc).isoformat())
    save(root/"SELECTION.json", completed)
    save(root/"STAGE.json", dict(state="cycle_reviewed_and_selected",
                                cycle=protocol["cycle"], complete_goal=False))
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
