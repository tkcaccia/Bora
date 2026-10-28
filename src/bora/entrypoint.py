"""Configure the frozen CPU runtime before numerical libraries are imported."""
import os
import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] in {"wsi-fit", "wsi-project", "wsi", "wsi-crop"}:
        # Match the accepted run at library startup. Increasing some OpenBLAS
        # pools after initialization can crash inside the native optimizer.
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[name] = "4"
    from .cli import main as run
    return run()
