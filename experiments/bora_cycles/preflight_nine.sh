#!/usr/bin/env bash
set -euo pipefail
task_code="$1"
task_python="$2"
task_inputs="$3"
task_previous="$4"
task_pilot="$5"
cd "$task_code"
export PYTHONPATH=src:.
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
g++ -O3 -std=c++17 -shared -fPIC -fopenmp cpp/evidence/kernels.cpp -o cpp/evidence/libbora_evidence.so
nice -n 12 "$task_python" -m pytest -q \
  tests/test_bora_cycle_foundation.py tests/test_bora_cycle_candidates.py \
  tests/test_bora_cycle_two.py tests/test_bora_cycle_three.py tests/test_bora_cycle_four.py \
  tests/test_bora_cycle_five.py tests/test_bora_cycle_six.py tests/test_bora_cycle_seven.py \
  tests/test_bora_cycle_eight.py tests/test_bora_cycle_nine.py \
  tests/test_bora_cycle_review.py tests/test_bora_cycle_publish.py tests/test_bora_cycle_audit.py \
  tests/test_bora_runtime_policy.py tests/test_bora_expanded_scope_metrics.py \
  tests/test_bora_extension_task_adapter.py tests/test_bora_extension_conversions.py \
  tests/test_bora_small_region_atlas.py tests/test_bora_input_limits.py
exec nice -n 12 "$task_python" -u -m experiments.bora_cycles.run \
  --cycle 9 --source "$task_inputs" --previous "$task_previous" --root "$task_pilot" \
  --pilot --workers 3 --metric-backend cpp --cases \
  tiger_32 tiger_23 bcss_45 bcss_53 bcss_18 bcss_87 \
  wsss4luad_01 wsss4luad_04 rings-tumor_01 rings-tumor_03 \
  ocelot_01 beetle_01 camelyon16-subset_01
