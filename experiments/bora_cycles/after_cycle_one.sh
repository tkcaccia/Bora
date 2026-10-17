#!/usr/bin/env bash
set -euo pipefail
# One-off dependent stages of this running experiment, not a scheduled task.
task_job_root="$1"
task_python="$2"
task_inputs="$3"
task_results="$4"
task_code="$5"
task_pid="$(cat "$task_job_root/cycle-01.pid")"
while kill -0 "$task_pid" 2>/dev/null; do
  task_command="$(ps -p "$task_pid" -o args= || true)"
  if [ -z "$task_command" ]; then break; fi
  case "$task_command" in
    *experiments.bora_cycles.run*"$task_results"*) sleep 10 ;;
    *) echo "Original PID no longer identifies the expected experiment" >&2; exit 2 ;;
  esac
done
test -f "$task_results/STAGE.json"
cd "$task_code"
export PYTHONPATH=src:.
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
g++ -O3 -std=c++17 -shared -fPIC -fopenmp \
  cpp/evidence/kernels.cpp -o cpp/evidence/libbora_evidence.so
"$task_python" -m pytest -q test_bora_cycle_foundation.py \
  test_bora_cycle_candidates.py test_bora_cycle_review.py
nice -n 12 "$task_python" -u -m experiments.bora_cycles.audit_compiled \
  --source "$task_inputs" --root "$task_results" \
  --output "$task_results/CPP_PARITY_AUDIT.json" --workers 3
nice -n 12 "$task_python" -m experiments.bora_cycles.profile_native \
  --output "$task_results/CPP_PROFILE_CHIAMAKA.json"
nice -n 12 "$task_python" -m experiments.bora_cycles.publish_diagnostics \
  --source "$task_inputs" --root "$task_results"
"$task_python" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); (p/"POSTHOC_COMPLETE.json").write_text(json.dumps(dict(compiled_parity_checked=True,diagnostics_published=True,visual_review_still_required=True,completed_cycles=0)))' "$task_results"
