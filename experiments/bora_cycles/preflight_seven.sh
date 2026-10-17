#!/usr/bin/env bash
set -euo pipefail
task_code="$1"
task_python="$2"
task_inputs="$3"
task_previous="$4"
task_pilot="$5"
task_profiles="$6"
cd "$task_code"
export PYTHONPATH=src:.
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
g++ -O3 -std=c++17 -shared -fPIC -fopenmp cpp/evidence/kernels.cpp -o cpp/evidence/libbora_evidence.so
nice -n 12 "$task_python" -m pytest -q \
  tests/test_bora_cycle_foundation.py tests/test_bora_cycle_candidates.py \
  tests/test_bora_cycle_two.py tests/test_bora_cycle_three.py \
  tests/test_bora_cycle_four.py tests/test_bora_cycle_five.py \
  tests/test_bora_cycle_six.py tests/test_bora_cycle_seven.py \
  tests/test_bora_cycle_review.py tests/test_bora_cycle_publish.py \
  tests/test_bora_cycle_audit.py tests/test_bora_small_region_atlas.py tests/test_bora_runtime_policy.py
nice -n 12 "$task_python" -m experiments.bora_cycles.profile_seven \
  --output "$task_profiles/CPP_SEVEN_PROFILE_CHIAMAKA.json"
nice -n 12 "$task_python" -m experiments.bora_cycles.profile_seven --nodes 32768 \
  --output "$task_profiles/CPP_SEVEN_PROFILE_LARGE_CHIAMAKA.json"
exec bash "$task_code/experiments/bora_cycles/cycle_seven.sh" \
  "$task_code" "$task_python" "$task_inputs" "$task_previous" "$task_pilot" pilot 3
