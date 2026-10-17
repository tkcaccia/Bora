"""Measure a C++ port separately; it never creates a new candidate ID."""
import argparse
import os
import platform
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
from . import native
from .run import save


def timed(function, repeats=9):
    function()
    seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        seconds.append(time.perf_counter()-start)
    return dict(seconds=seconds, median_seconds=float(np.median(seconds)),
                min_seconds=min(seconds))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(381)
    comparisons = []
    with threadpool_limits(limits=2):
        for d in (3, 12):
            x = rng.normal(size=(400_000, d)).astype(np.float32)
            coefficient = rng.normal(size=(7, d))
            intercept = rng.normal(size=7)
            expected = (x@coefficient.T+intercept).astype(np.float32)
            one = native.logits(x, coefficient, intercept, threads=1)
            two = native.logits(x, coefficient, intercept, threads=2)
            np.testing.assert_allclose(one, expected, atol=3e-6, rtol=3e-6)
            np.testing.assert_array_equal(one, two)
            comparisons.append(dict(
                operation="linear class logits", pixels=len(x), variables=d, classes=7,
                numpy_float64=timed(lambda: (x@coefficient.T+intercept).astype(np.float32)),
                cpp_one_thread=timed(lambda: native.logits(x, coefficient, intercept, threads=1)),
                cpp_two_threads=timed(lambda: native.logits(x, coefficient, intercept, threads=2)),
                max_absolute_error=float(np.max(np.abs(expected-one))),
                label_disagreements=int((expected.argmax(-1) != one.argmax(-1)).sum())))
    save(args.output, dict(
        host=platform.node(), platform=platform.platform(),
        logical_cpus=os.cpu_count(), openmp=bool(native.library().bora_evidence_openmp()),
        numerical_libraries=threadpool_info(), comparisons=comparisons,
        scope="isolated numerical kernels; not whole-pipeline or clinical performance",
        candidate_count_increment=0))


if __name__ == "__main__":
    main()
