"""Serial synthetic Chebyshev-kernel profile; not whole-mask speed evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from . import native


def run(nodes,repeats=5):
    rng=np.random.default_rng(703)
    colors=rng.uniform(.05,.95,(7,3));x=rng.uniform(.05,.95,(nodes,3));prior=rng.dirichlet(np.ones(7),nodes)
    timings={};outputs=[];details=[]
    for threads in (1,2):
        native.chebyshev_palette(x[:32],prior[:32],colors,threads=threads)
        times=[]
        for _ in range(repeats):
            start=time.perf_counter();out,report=native.chebyshev_palette(x,prior,colors,threads=threads)
            times.append(time.perf_counter()-start)
        timings[str(threads)]=dict(seconds=times,median_seconds=float(np.median(times)))
        outputs.append(out);details.append(report)
    exact=bool(np.array_equal(outputs[0],outputs[1]))
    assert exact and details[0]['max_primal_dual_gap']==details[1]['max_primal_dual_gap']
    repo=Path(__file__).resolve().parents[2];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    return dict(nodes=nodes,classes=7,repeats=repeats,timings=timings,
        one_to_two_thread_ratio=timings['1']['median_seconds']/timings['2']['median_seconds'],
        cpp_thread_outputs_exact=exact,finite_solver=details,
        library_sha256=sha(repo/'cpp/evidence/libbora_evidence.so'),
        kernel_source_sha256=sha(repo/'cpp/evidence/kernels.cpp'),source_sha256=sha(Path(__file__)),
        scope='Serial synthetic independent-colour kernel; no whole-algorithm speedup claim.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--nodes',type=int,default=2048)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=run(args.nodes);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result))
