"""Controlled kernel-only geometry timings; never counted as an innovation.

Run after prediction jobs finish, against their frozen native library. Save
raw repetitions, numerical parity, source and library hashes. These timings
do not establish end-to-end algorithm or whole-benchmark speedups.
"""
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.spatial import ConvexHull
from threadpoolctl import threadpool_limits
from experiments.bora_cycles import native


def digest(data):
    return hashlib.sha256(data).hexdigest()


def profile(output, pixels=250000, training_points=2000, repeats=7):
    if min(pixels, training_points, repeats) < 1 or training_points < 8:
        raise ValueError("Positive sizes and at least eight hull points required")
    rng = np.random.default_rng(319)
    points = rng.uniform(.05, .95, (training_points, 3))
    x = rng.uniform(0, 1, (pixels, 3)).astype(np.float32)
    planes = np.ascontiguousarray(ConvexHull(points).equations)

    def numpy_stream():
        out = np.empty(len(x), np.float32)
        for start in range(0, len(x), 16384):
            distances = x[start:start+16384]@planes[:, :3].T+planes[:, 3]
            out[start:start+16384] = -distances.max(-1)
        return out

    configurations = {
        "numpy_1": (1, numpy_stream), "numpy_2": (2, numpy_stream),
        "cpp_1": (1, lambda: native.hull_depth(x, planes, threads=1)),
        "cpp_2": (2, lambda: native.hull_depth(x, planes, threads=2)),
    }
    outputs, timings = {}, {key: [] for key in configurations}
    for key, (threads, function) in configurations.items():
        with threadpool_limits(limits=threads):
            outputs[key] = function()  # unmeasured initialization/warm-up
    expected = outputs["numpy_1"]
    error = {key: float(np.max(np.abs(value-expected))) for key, value in outputs.items()}
    for value in outputs.values():
        np.testing.assert_allclose(value, expected, atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(outputs["cpp_1"], outputs["cpp_2"])
    order = list(configurations)
    for repeat in range(repeats):
        # Rotate starting method to limit a systematic first/last cache bias.
        for key in order[repeat % 4:]+order[:repeat % 4]:
            threads, function = configurations[key]
            with threadpool_limits(limits=threads):
                started = time.perf_counter()
                value = function()
                timings[key].append(time.perf_counter()-started)
            np.testing.assert_allclose(value, outputs[key], atol=1e-7, rtol=1e-7)
    library = Path(native.library()._name)
    medians = {key: float(np.median(values)) for key, values in timings.items()}
    result = dict(host=platform.node(), platform=platform.platform(), logical_cpus=os.cpu_count(),
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        pixels=pixels, training_points=training_points, facets=len(planes), repeats=repeats,
        seed=319, numpy_query_chunk=16384, warmup_calls_per_configuration=1,
        cpp_openmp_enabled=bool(native.library().bora_evidence_openmp()),
        raw_seconds=timings, median_seconds=medians, maximum_absolute_error=error,
        cpp_threads_exact_output_parity=True,
        cpp_1_to_2_threads_speedup=medians["cpp_1"]/medians["cpp_2"],
        numpy_2_to_cpp_2_speedup=medians["numpy_2"]/medians["cpp_2"],
        query_sha256=digest(x.tobytes()), planes_sha256=digest(planes.tobytes()),
        cpp_library_sha256=digest(library.read_bytes()),
        profile_source_sha256=digest(Path(__file__).read_bytes()),
        timing_scope="kernel wrapper including output allocation; excludes hull fitting and input generation",
        caveat="Synthetic kernel-only measurement on the recorded host, no CPU affinity or exclusive machine access; not an end-to-end speedup or accuracy result.")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("Use a fresh profile output file")
    output.write_text(json.dumps(result, indent=2))
    source_copy = output.with_suffix(".source.py")
    source_copy.write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(host=result["host"], median_seconds=medians,
        cpp_1_to_2_threads_speedup=result["cpp_1_to_2_threads_speedup"], parity=True)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pixels", type=int, default=250000)
    parser.add_argument("--training-points", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    profile(args.output, args.pixels, args.training_points, args.repeats)


if __name__ == "__main__":
    main()
