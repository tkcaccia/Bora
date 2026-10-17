"""Kernel-only one/two CPU measurements; never claim end-to-end speedup."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import native
from .run import digest, save


def measured(call):
    samples = []
    for _ in range(3):
        started = time.perf_counter()
        output = call()
        samples.append(time.perf_counter()-started)
    return output, dict(seconds=samples, median_seconds=float(np.median(samples)))


def profile(pixels=65536, bags=64):
    rng = np.random.default_rng(619)
    x = rng.normal(size=(pixels, 3)).astype(np.float32)
    means = rng.normal(size=(6, 3))
    matrix = rng.normal(size=(6, 3, 3))
    precision = matrix@matrix.transpose(0, 2, 1)+np.eye(3)
    intercept = rng.normal(size=6)
    one, a = measured(lambda: native.gaussian_logits(x, means, precision, intercept, threads=1))
    two, b = measured(lambda: native.gaussian_logits(x, means, precision, intercept, threads=2))
    delta = x.astype(float)[:, None]-means
    expected = intercept-.5*np.einsum('ngi,gij,ngj->ng', delta, precision, delta)
    gaussian = dict(one_thread=a, two_threads=b, thread_outputs_exact=bool(np.array_equal(one, two)),
        numpy_max_error=float(np.max(np.abs(one-expected))),
        one_to_two_speedup=a['median_seconds']/b['median_seconds'])
    colors = rng.uniform(.1, .9, (7, 3))
    query = rng.uniform(0, 1, (bags, 3)).astype(np.float32)
    prior = rng.dirichlet(np.ones(7), bags).astype(np.float32)
    labels = np.arange(bags)%7
    one, a = measured(lambda: native.dominance_qp(query, prior, colors, labels, threads=1))
    two, b = measured(lambda: native.dominance_qp(query, prior, colors, labels, threads=2))
    dominance = dict(one_thread=a, two_threads=b, thread_outputs_exact=bool(np.array_equal(one[0], two[0])),
        solver=two[1], one_to_two_speedup=a['median_seconds']/b['median_seconds'])
    if not gaussian['thread_outputs_exact'] or not dominance['thread_outputs_exact']:
        raise ValueError('Native CPU thread parity failed')
    repo = Path(__file__).resolve().parents[2]
    return dict(pixels=pixels, bags=bags, gaussian=gaussian, dominance=dominance,
        cpp_thread_outputs_exact=True,
        measurement_scope='Kernel plus wrapper/allocation only; not a full candidate or WSI speedup',
        source_sha256=digest(Path(__file__)),
        library_sha256=digest(repo/'cpp/evidence/libbora_evidence.so'),
        kernel_source_sha256=digest(repo/'cpp/evidence/kernels.cpp'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pixels', type=int, default=65536)
    parser.add_argument('--bags', type=int, default=64)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = profile(args.pixels, args.bags)
    save(args.output, data)
    print(json.dumps(data), flush=True)
