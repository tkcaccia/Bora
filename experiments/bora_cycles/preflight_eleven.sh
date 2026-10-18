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
  tests/test_bora_cycle_eleven_math.py tests/test_bora_cycle_eleven_candidates.py \
  tests/test_bora_calibrated_cohort.py tests/test_bora_calibrated_scope.py \
  tests/test_bora_expanded_scope_metrics.py tests/test_bora_extension_task_adapter.py \
  tests/test_bora_small_region_atlas.py tests/test_bora_input_limits.py \
  tests/test_bora_cycle_review.py tests/test_bora_cycle_publish.py tests/test_bora_cycle_audit.py
# A pilot only: all23 methods,17 cases covering all10 calibrated datasets,
# the nine requested focus cases, two deterministic timed repeats.
exec nice -n 12 "$task_python" -u -m experiments.bora_cycles.run \
  --cycle 11 --source "$task_inputs" --previous "$task_previous" --root "$task_pilot" \
  --pilot --workers 3 --repeats 2 --metric-backend cpp --cases \
  tiger_32 tiger_33 tiger_36 tiger_37 tiger_38 bcss_17 bcss_18 bcss_19 bcss_24 \
  ocelot_01 beetle_01 camelyon16-subset_01 ignite_01 peso_01 gcss_01 puma_01 gleasonxai_01
