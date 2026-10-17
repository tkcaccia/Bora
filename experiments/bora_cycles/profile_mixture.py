"""Synthetic spatial-mixture wrapper/kernel timing; not an algorithm speedup."""
import argparse
import platform
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from . import native
from .run import digest, save


def profile(pixels=250000, repetitions=7):
    rng = np.random.default_rng(1715)
    x = rng.uniform(0,1,(pixels,3)).astype(np.float32)
    coefficient = rng.normal(size=(2,3))
    intercept = rng.normal(size=2)
    mixing = rng.dirichlet(np.ones(2),size=pixels).astype(np.float32)
    timings = {}
    outputs = {}
    for threads in (1,2):
        native.local_mixture_logits(x,coefficient,intercept,mixing,threads=threads)
        measured = []
        for _ in range(repetitions):
            started = time.perf_counter()
            outputs[threads] = native.local_mixture_logits(x,coefficient,intercept,mixing,threads=threads)
            measured.append(time.perf_counter()-started)
        timings[str(threads)] = measured
    reference = logsumexp(x.astype(float)@coefficient.T+intercept+np.log(mixing.astype(float)),axis=-1)
    repo = Path(__file__).resolve().parents[2]
    return dict(host=platform.node(),pixels=pixels,components=2,repetitions=repetitions,
        elapsed_seconds_by_threads=timings,
        median_seconds_by_threads={k:float(np.median(v)) for k,v in timings.items()},
        one_to_two_thread_ratio=float(np.median(timings['1'])/np.median(timings['2'])),
        cpp_thread_outputs_exact=bool(np.array_equal(outputs[1],outputs[2])),
        max_numpy_error=float(np.max(np.abs(outputs[2]-reference))),
        library_sha256=digest(repo/'cpp/evidence/libbora_evidence.so'),
        kernel_source_sha256=digest(repo/'cpp/evidence/kernels.cpp'),
        source_sha256=digest(Path(__file__)),
        scope='Synthetic kernel plus input validation/allocation wrapper; no whole-Bora CPU scaling claim; CPU affinity/exclusivity uncontrolled')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--pixels',type=int,default=250000)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=profile(args.pixels)
    assert result['cpp_thread_outputs_exact'] and result['max_numpy_error'] < 1e-5
    save(args.output,result)
    print({k:result[k] for k in ('pixels','median_seconds_by_threads','one_to_two_thread_ratio','max_numpy_error')})
