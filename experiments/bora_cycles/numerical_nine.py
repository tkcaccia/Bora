"""Collect finite-solver limitations from frozen execution records only."""
import json
from pathlib import Path
from .run import digest


CHECKS = ('linear_system_max_residual', 'kkt_max_residual', 'equality_max_residual',
          'max_kkt_residual', 'gradient_infinity_norm', 'projected_gradient_residual',
          'primal_residual', 'dual_residual', 'lagrangian_projected_residual',
          'primal_projected_residual','dual_fixed_point_residual','projected_stationarity',
          'order_constraint_violation','projection_change','log_variance_system_residual',
          'simplex_sum_residual','mean_equation_residual','scale_equation_residual',
          'variance_equation_residual','negative_scale_residual','positive_scale_residual',
          'sandwich_probe_equation_residual','constant_reproduction_residual','dual_projected_residual')


def nested(value):
    if isinstance(value, dict):
        yield value
        # Iteration history is evidence, not many separate failed solves.
        for key in ('fits',):
            for item in value.get(key, []):
                yield from nested(item)
        if isinstance(value.get('palette_fit'),dict):yield from nested(value['palette_fit'])


def summarize(root):
    root = Path(root)
    result = json.loads((root/'RESULTS.json').read_text())
    reports = {}
    for key in result['protocol']['methods']:
        maxima, cases, not_converged = {}, [], []
        for row in result['cases']:
            entry = row['methods'][key]
            if entry['status'] != 'ok':
                continue
            detail = entry['details']
            false = False
            for record in nested(detail):
                for name in CHECKS:
                    value = record.get(name)
                    if isinstance(value, list):
                        value = max(value, default=0.)
                    if isinstance(value, (int, float)):
                        maxima[name] = max(maxima.get(name, 0.), abs(value))
                false |= record.get('converged') is False or record.get('optimizer_success') is False
            if false:
                not_converged.append(row['case'])
            if maxima:
                cases.append(row['case'])
        reports[key] = dict(maximum_reported_residuals=maxima,
            explicitly_not_converged_cases=not_converged,
            explicitly_not_converged_count=len(not_converged),
            caveat='No convergence flag is not proof of convergence. Linear residual, stochastic uncertainty error, biological accuracy and finite nonlinear stationarity are different checks.')
    return dict(cycle=result['protocol']['cycle'], cases=len(result['cases']), methods=reports,
        results_sha256=digest(root/'RESULTS.json'), source_sha256=digest(Path(__file__)),
        predictions_modified=False, scores_modified=False,
        visual_review_created=False, selection_created=False)
