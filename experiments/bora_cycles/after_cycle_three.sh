#!/usr/bin/env bash
set -euo pipefail
# One-off dependent checks of this experiment; no prediction/score regeneration.
task_pid="$1"
task_code="$2"
task_python="$3"
task_inputs="$4"
task_results="$5"
task_previous="$6"
task_executed_code="$7"
while kill -0 "$task_pid" 2>/dev/null; do
  task_command="$(ps -p "$task_pid" -o args= || true)"
  task_stat="$(ps -p "$task_pid" -o stat= || true)"
  if [ -z "$task_command" ]; then break; fi
  case "$task_stat" in *Z*) break ;; esac
  case "$task_command" in
    *experiments.bora_cycles.run*"$task_results"*) sleep 10 ;;
    *) echo "PID does not identify the expected running experiment" >&2; exit 2 ;;
  esac
done
cd "$task_code"
export PYTHONPATH=src:.
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
"$task_python" - "$task_results" <<'PY'
import json, sys
from pathlib import Path
r = Path(sys.argv[1])
p = json.loads((r/'PROTOCOL.json').read_text())
s = json.loads((r/'STAGE.json').read_text())
assert not p['pilot'] and p['cycle'] == 3 and len(p['cases']) == 149 and len(p['roster']) == 20
assert s['state'] == 'awaiting_visual_review' and s['cases'] == 149
PY
nice -n 12 "$task_python" -m pytest -q tests/test_bora_cycle_audit.py tests/test_bora_small_region_atlas.py
nice -n 12 "$task_python" -u -m experiments.bora_cycles.audit_cycle \
  --source "$task_inputs" --root "$task_results" --previous "$task_previous" \
  --executed-code "$task_executed_code" --output "$task_results/INTEGRITY_AUDIT.json"
nice -n 12 "$task_python" -u -m experiments.bora_cycles.small_region_atlas \
  --source "$task_inputs" --root "$task_results" --output "$task_results/SMALL_REGION_ATLAS.json"
nice -n 12 "$task_python" -m experiments.bora_cycles.profile_hull \
  --output "$task_results/CPP_HULL_PROFILE_CHIAMAKA.json"
nice -n 12 "$task_python" -m experiments.bora_cycles.publish_diagnostics \
  --source "$task_inputs" --root "$task_results" --reuse "$task_previous"
"$task_python" - "$task_results" <<'PY'
import hashlib, json, sys
from pathlib import Path
r = Path(sys.argv[1])
names = ['INTEGRITY_AUDIT.json', 'SMALL_REGION_ATLAS.json', 'CPP_HULL_PROFILE_CHIAMAKA.json', 'DIAGNOSTICS_REUSE.json']
a = json.loads((r/'INTEGRITY_AUDIT.json').read_text())
record = dict(cases=a['cases'], audited_native_masks=a['native_masks'],
    unchanged_survivor_comparisons=a['unchanged_survivor_comparisons'], failed_methods=a['failed_methods'],
    posthoc_test_cases_passed=6, visual_review_still_required=True, completed_cycles=2,
    complete_goal=False, predictions_modified=False, scores_modified=False,
    artifact_sha256={name:hashlib.sha256((r/name).read_bytes()).hexdigest() for name in names})
(r/'POSTHOC_COMPLETE.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record))
PY
