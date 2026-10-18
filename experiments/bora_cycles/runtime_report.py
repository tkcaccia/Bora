"""Posthoc runtime comparison of completed cycles, with frozen source hashes."""
import argparse
from html import escape
import json
from pathlib import Path

from .run import digest, save
from .runtime_policy import DEFAULT_POLICY, runtime_ranking


def matched_history_ranking(result):
    """Use the original case cohort when comparing across the scope expansion."""
    scope = result['protocol'].get('benchmark_scope')
    if scope is None:
        return runtime_ranking(result)
    ids = set(scope['historical_cases'])
    rows = [row for row in result['cases'] if row['case'] in ids]
    if {row['case'] for row in rows} != ids or len(rows) != len(ids):
        raise ValueError('Matched historical cohort is incomplete')
    ranks = []
    for record in result['ranking']:
        key = record['method']
        good = all(row['methods'].get(key, {}).get('status') == 'ok' for row in rows)
        subset = dict(method=key, cases=len(rows), complete=good,
            mean_iou=sum(row['methods'][key]['score']['macro_iou'] for row in rows)/len(rows) if good else None,
            mean_seconds=sum(row['methods'][key]['median_seconds'] for row in rows)/len(rows) if good else None)
        ranks.append(subset)
    return runtime_ranking(dict(cases=rows, ranking=ranks))


def build_history(base):
    base = Path(base)
    cycles = {}
    for selection_path in sorted(base.glob("cycle-*/SELECTION.json")):
        root = selection_path.parent
        selection = json.loads(selection_path.read_text())
        for name, field in (("RESULTS.json", "results_sha256"),
                            ("PROTOCOL.json", "protocol_sha256"),
                            ("VISUAL_REVIEW.json", "review_sha256")):
            if digest(root/name) != selection[field]:
                raise ValueError("Historical signed evidence changed: "+str(root/name))
        result = json.loads((root/"RESULTS.json").read_text())
        cycle = selection["cycle"]
        if cycle in cycles:
            raise ValueError("Duplicate selected cycle; choose one authoritative result")
        candidates = set(result["protocol"]["roster"])
        ranking = runtime_ranking(result)
        complete = [r for r in ranking if r["complete"] and r["method"] in candidates]
        if not complete:
            raise ValueError("No complete candidate in selected cycle")
        raw = max(complete, key=lambda r: r["mean_iou"])
        matched = matched_history_ranking(result)
        cycles[cycle] = dict(cycle=cycle, result_folder=root.name,
            source_results_sha256=digest(root/"RESULTS.json"),
            source_selection_sha256=digest(selection_path),
            host=result["protocol"].get("host"), cases=len(result["cases"]),
            historical_survivors=selection["survivors"],
            raw_accuracy_leader=raw["method"], raw_accuracy_leader_row=raw,
            runtime_adjusted_candidate_leader=complete[0]["method"],
            runtime_adjusted_candidate_leader_row=complete[0],
            roster_titles={k:v["title"] for k,v in result["protocol"]["roster"].items()},
            benchmark_scope_version=result['protocol'].get('benchmark_scope', {}).get('version', 'historical149'),
            input_plan_sha256=result['protocol'].get('input_plan_sha256'),
            matched_cohort_cases=len(result['protocol'].get('benchmark_scope', {}).get('historical_cases', result['cases'])),
            matched_historical_ranking=matched,
            ranking=ranking)
    history = dict(runtime_penalty_policy=DEFAULT_POLICY,
        raw_accuracy_modified=False, predictions_modified=False,
        historical_selections_modified=False,
        report_source_sha256=digest(Path(__file__)),
        scope_limitation="Cycles1-7 use149 ROIs,8-10 use349,11 uses469 calibrated ROIs. Overall scores across scopes are not algorithm gains. Each declared prior-cohort subset has its own count; cycle11 changes eligibility and prospectively rescored OCELOT negatives, so it is not an identical historical scoring contract. Elapsed times are not controlled cross-cycle comparisons.",
        timing_limitation="One elapsed measurement per candidate/case in shared low-priority CPU runs; not controlled cross-cycle speedup evidence.",
        cycles=[cycles[k] for k in sorted(cycles)])
    save(base/"RUNTIME_PENALTY_HISTORY.json", history)
    body = []
    for cycle in history["cycles"]:
        raw, adjusted = cycle["raw_accuracy_leader_row"], cycle["runtime_adjusted_candidate_leader_row"]
        body.append(f'<h2>Cycle {cycle["cycle"]}</h2><p>{cycle["cases"]} ROIs. '
                    f'Declared prior-cohort subset: {cycle["matched_cohort_cases"]} ROIs. '
                    f'Raw accuracy leader: {escape(raw["method"])} ({raw["mean_iou"]:.6f}). '
                    f'Runtime-adjusted candidate leader: {escape(adjusted["method"])} '
                    f'({adjusted["runtime_adjusted_score"]:.6f}). '
                    'Historical selection unchanged.</p>')
        body.append('<table><tr><th>Method</th><th>Raw IoU</th><th>Mean seconds</th>'
                    '<th>Images &gt;1s</th><th>Mean penalty</th><th>Adjusted score</th>'
                    '<th>Declared prior-cohort IoU</th></tr>')
        matched = {r['method']:r for r in cycle['matched_historical_ranking']}
        for row in cycle["ranking"]:
            key = row["method"]
            title = cycle["roster_titles"].get(key, "reference control")
            if not row["complete"]:
                body.append(f'<tr><td>{escape(key)} {escape(title)}</td><td colspan="6">Incomplete</td></tr>')
                continue
            body.append(f'<tr><td>{escape(key)} {escape(title)}</td>'
                        f'<td>{row["mean_iou"]:.6f}</td><td>{row["mean_seconds"]:.4f}</td>'
                        f'<td>{row["images_over_one_second"]}/{cycle["cases"]}</td>'
                        f'<td>{row["mean_runtime_penalty"]:.6f}</td>'
                        f'<td>{row["runtime_adjusted_score"]:.6f}</td>'
                        +('<td>'+f'{matched[key]["mean_iou"]:.6f}'+'</td>' if matched[key]['complete'] else '<td>Incomplete</td>')+'</tr>')
        body.append('</table>')
    (base/"runtime_history.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Bora runtime and accuracy history</title>'
        '<style>body{font:16px system-ui;max-width:1200px;margin:25px auto;padding:15px}'
        'table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}'
        'p{line-height:1.5}</style><h1>Runtime-adjusted development results</h1>'
        '<p>Mean over images of IoU − 0.05 × max(seconds − 1, 0). '
        'The penalty is applied before averaging, so occasional slow images count even '
        'when mean time is under one second. This utility is not an accuracy metric.</p>'
        '<p>Cycles1–4 are a posthoc comparison. Historical masks, raw scores and selections '
        'are unchanged. The policy is frozen prospectively from cycle5. '
        'Cycles1–7 use149 images; cycles8–10 use349; cycle11 uses469 calibrated ROIs. Overall scores '
        'across these scopes are not comparable gains. The declared prior-cohort subset '
        'and its count are reported per cycle. Cycle11 also changes eligibility and prospectively '
        'rescored OCELOT broad negatives; its subset is not an identical historical scoring contract. All images are '
        'reused development data. Timing differences between cycles '
        'are not controlled speedup measurements.</p>'+''.join(body))
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    args = parser.parse_args()
    history = build_history(args.base)
    print(json.dumps(dict(cycles=[c["cycle"] for c in history["cycles"]],
                          results="RUNTIME_PENALTY_HISTORY.json", viewer="runtime_history.html")))
