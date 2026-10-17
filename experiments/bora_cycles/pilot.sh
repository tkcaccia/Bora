#!/usr/bin/env bash
set -euo pipefail
# Engineering smoke run on one case from every dataset, not a cycle.
task_code_root="$1"
task_python="$2"
task_input_root="$3"
task_output_root="$4"
cd "$task_code_root"
export PYTHONPATH=src:.
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
exec nice -n 12 "$task_python" -u -m experiments.bora_cycles.run \
  --source "$task_input_root" --root "$task_output_root" --pilot \
  --cases tiger_32 bcss_17 beetle_01 camelyon16-subset_01 \
  --keys B21 B22 C01 C02 C03 C04 C05 C06 C07 C08 C09 \
  --workers 3 --repeats 2
