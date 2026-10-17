#!/usr/bin/env bash
set -euo pipefail
task_pid="$1"
task_code="$2"
task_python="$3"
task_inputs="$4"
task_results="$5"
task_previous="$6"
task_executed_code="$7"
task_profile_root="$8"
task_profile_tag="${9:-CHIAMAKA}"
case "$task_profile_tag" in LOCAL|CHIAMAKA) ;; *) exit 2 ;; esac
while kill -0 "$task_pid" 2>/dev/null; do
  task_command="$(ps -p "$task_pid" -o args= || true)"
  task_stat="$(ps -p "$task_pid" -o stat= || true)"
  if [ -z "$task_command" ]; then break; fi
  case "$task_stat" in *Z*) break ;; esac
  case "$task_command" in
    *experiments.bora_cycles.run*"$task_results"*) sleep 10 ;;
    *) echo "PID does not identify the expected experiment" >&2; exit 2 ;;
  esac
done
cd "$task_code"
export PYTHONPATH=src:.
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
"$task_python" - "$task_results" <<'PY'
import json, sys
from pathlib import Path
r = Path(sys.argv[1])
p = json.loads((r/'PROTOCOL.json').read_text())
s = json.loads((r/'STAGE.json').read_text())
assert not p['pilot'] and p['cycle'] == 7 and len(p['cases']) == 149 and len(p['roster']) == 20
assert s['state'] == 'awaiting_visual_review' and s['cases'] == 149
PY
nice -n 12 "$task_python" -m pytest -q tests/test_bora_cycle_audit.py tests/test_bora_small_region_atlas.py
nice -n 12 "$task_python" -u -m experiments.bora_cycles.audit_cycle \
  --source "$task_inputs" --root "$task_results" --previous "$task_previous" \
  --executed-code "$task_executed_code" --output "$task_results/INTEGRITY_AUDIT.json"
nice -n 12 "$task_python" -u -m experiments.bora_cycles.small_region_atlas \
  --source "$task_inputs" --root "$task_results" --output "$task_results/SMALL_REGION_ATLAS.json"
nice -n 12 "$task_python" -m experiments.bora_cycles.publish_diagnostics \
  --source "$task_inputs" --root "$task_results" --reuse "$task_previous"
"$task_python" - "$task_results" "$task_profile_root" "$task_profile_tag" <<'PY'
import hashlib, json, shutil, sys
from pathlib import Path
r, profiles = map(Path, sys.argv[1:3]); tag = sys.argv[3]
names = ['INTEGRITY_AUDIT.json', 'SMALL_REGION_ATLAS.json', 'DIAGNOSTICS_REUSE.json']
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
for name in [f'CPP_SEVEN_PROFILE_{tag}.json', f'CPP_SEVEN_PROFILE_LARGE_{tag}.json']:
    source = profiles/name
    data = json.loads(source.read_text())
    assert data['cpp_thread_outputs_exact']
    assert data['library_sha256'] == sha(r/'runtime_sources/cpp/evidence/libbora_evidence.so')
    assert data['kernel_source_sha256'] == sha(r/'runtime_sources/cpp/evidence/kernels.cpp')
    assert data['source_sha256'] == sha(r/'runtime_sources/experiments/bora_cycles/profile_seven.py')
    shutil.copy2(source, r/name); names.append(name)
a = json.loads((r/'INTEGRITY_AUDIT.json').read_text())
record = dict(cases=a['cases'], audited_native_masks=a['native_masks'],
    unchanged_survivor_comparisons=a['unchanged_survivor_comparisons'], failed_methods=a['failed_methods'],
    posthoc_test_cases_passed=9, visual_review_still_required=True, completed_cycles=6,
    complete_goal=False, predictions_modified=False, scores_modified=False,
    artifact_sha256={name:sha(r/name) for name in names})
(r/'POSTHOC_COMPLETE.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record))
PY
