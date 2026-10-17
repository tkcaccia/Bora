"""Synthetic exact-count kernel profile; no image, reference or score I/O."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from threadpoolctl import threadpool_limits

from . import native


def numpy_count(probability, labels):
    """Independent polynomial convolution including all probability derivatives."""
    bags, n, k = probability.shape
    logp, derivative = np.zeros(bags), np.zeros_like(probability)
    for b in range(bags):
        own = labels[b]
        for rival in range(k):
            if rival == own:
                continue
            factors = np.c_[probability[b, :, rival],
                            1-probability[b, :, rival]-probability[b, :, own],
                            probability[b, :, own]]
            forward, backward = [np.ones(1)], [None]*(n+1)
            backward[n] = np.ones(1)
            for row in factors:
                forward.append(np.convolve(forward[-1], row))
            for i in range(n-1, -1, -1):
                backward[i] = np.convolve(backward[i+1], factors[i])
            threshold = int(rival < own)
            p = forward[-1][n+threshold:].sum()
            logp[b] += np.log(max(p, 1e-300))
            if p <= 1e-300:
                continue
            for i in range(n):
                excluded = np.convolve(forward[i], backward[i+1])
                lower, same = (n-1)+threshold-1, (n-1)+threshold
                derivative[b, i, own] -= (excluded[lower] if 0 <= lower < len(excluded) else 0)/p
                derivative[b, i, rival] += (excluded[same] if 0 <= same < len(excluded) else 0)/p
    return logp, derivative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bags", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    rng = np.random.default_rng(1739)
    probability = rng.dirichlet(np.full(6, 1.2), size=(args.bags, 25))
    labels = np.arange(args.bags, dtype=np.int32) % 6
    methods = {"numpy": lambda: numpy_count(probability, labels),
               "cpp_1": lambda: native.categorical_winner(probability, labels, threads=1),
               "cpp_2": lambda: native.categorical_winner(probability, labels, threads=2)}
    with threadpool_limits(limits=2):
        reference = methods["numpy"]()
        one, two = methods["cpp_1"](), methods["cpp_2"]()
        errors = [float(np.max(np.abs(a-b))) for a, b in zip(reference, one)]
        np.testing.assert_allclose(one[0], reference[0], atol=1e-11, rtol=1e-12)
        np.testing.assert_allclose(one[1], reference[1], atol=1e-10, rtol=1e-12)
        for a, b in zip(one, two):
            np.testing.assert_array_equal(a, b)
        times = {k: [] for k in methods}
        for _ in range(args.repeats):
            for key in rng.permutation(list(methods)):
                start = time.perf_counter()
                methods[key]()
                times[key].append(time.perf_counter()-start)
    medians = {k: float(np.median(v)) for k, v in times.items()}
    repo = Path(__file__).resolve().parents[2]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    result = dict(
        scope="synthetic kernel and Python wrapper including allocation; no end-to-end mask speedup claim",
        host=platform.node(), platform=platform.platform(), logical_cpus=os.cpu_count(),
        cpp_openmp_enabled=bool(native.library().bora_evidence_openmp()),
        shape_bags_pixels_classes=list(probability.shape), seed=1739,
        medians_seconds=medians, repetitions_seconds=times,
        cpp_one_to_two_thread_speedup=medians["cpp_1"]/medians["cpp_2"],
        numpy_to_cpp_two_thread_speedup=medians["numpy"]/medians["cpp_2"],
        max_abs_error_logp_and_derivative=errors, cpp_thread_outputs_exact=True,
        input_probability_sha256=hashlib.sha256(probability.tobytes()).hexdigest(),
        source_sha256=sha(Path(__file__)),
        kernel_source_sha256=sha(repo/"cpp/evidence/kernels.cpp"),
        library_sha256=sha(repo/"cpp/evidence/libbora_evidence.so"),
        caveat="No CPU affinity, machine exclusivity or end-to-end scaling measurement.")
    from .run import environment_record
    result["environment"] = environment_record()
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
