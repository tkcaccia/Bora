"""Bounded one-GPU child for R training or supervised prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .r_compute import cpu_threads, require_cuda


def run(spec_path: Path) -> dict:
    spec = json.loads(spec_path.read_text())
    if spec.get("format") != "bora-r-method-worker-v1" or \
       spec.get("backend") != "cuda" or spec.get("method") not in ("train", "predict"):
        raise ValueError("Invalid CUDA method worker contract")
    torch = require_cuda()
    torch.set_num_threads(spec["n_cores"])
    torch.cuda.set_per_process_memory_fraction(.65)
    with cpu_threads(spec["n_cores"]):
        if spec["method"] == "train":
            from .r_training import train
            return train(*spec["args"], **spec["kwargs"],
                         backend="cuda", n_cores=spec["n_cores"])
        from .r_prediction import predict_supervised
        return predict_supervised(*spec["args"], **spec["kwargs"],
                                  backend="cuda", n_cores=spec["n_cores"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    args = parser.parse_args()
    result = run(args.spec)
    print(json.dumps({"status": result["status"], "format": result["format"]}),
          flush=True)


if __name__ == "__main__":
    main()
