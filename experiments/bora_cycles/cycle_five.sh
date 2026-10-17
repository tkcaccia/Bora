#!/usr/bin/env bash
set -euo pipefail
task_code_root="$1"
task_python="$2"
task_input_root="$3"
task_previous_result="$4"
task_output_root="$5"
task_scope="$6"
task_workers="${7:-3}"
cd "$task_code_root"
export PYTHONPATH=src:.
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
case "$task_scope" in
  pilot) set -- --pilot --cases tiger_32 bcss_17 beetle_01 camelyon16-subset_01 tiger_07 tiger_09 tiger_14 bcss_06 ;;
  full) set -- ;;
  *) exit 2 ;;
esac
exec nice -n 12 "$task_python" -u -m experiments.bora_cycles.run \
  --cycle 5 --previous "$task_previous_result" \
  --source "$task_input_root" --root "$task_output_root" \
  --workers "$task_workers" --repeats 1 --metric-backend cpp "$@"
