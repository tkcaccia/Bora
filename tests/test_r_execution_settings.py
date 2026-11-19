"""Backend and thread controls must fail closed without a CUDA runtime."""

import json
import subprocess

import pytest

from bora import r_api
from bora.r_compute import cpu_threads, validate_execution


@pytest.mark.parametrize("backend,cores", [("gpu", 1), ("cpu", 0),
                                          ("cuda", True), ("cpu", 1.5)])
def test_invalid_execution_settings(backend, cores):
    with pytest.raises(ValueError):
        validate_execution(backend, cores)


def test_cpu_cleanup_never_queries_gpu(monkeypatch):
    monkeypatch.setattr(r_api, "_owned_pids", lambda pgid: [])
    monkeypatch.setattr(r_api, "_gpu_pids", lambda: (_ for _ in ()).throw(
        AssertionError("CPU route queried GPU")))
    assert r_api._cleanup(999999, backend="cpu", window_seconds=0)["status"] == "VERIFIED"


def test_cpu_thread_limit_requests_and_restores_opencv(monkeypatch):
    cv2 = pytest.importorskip("cv2")
    original = cv2.getNumThreads()
    requested = []
    setter = cv2.setNumThreads

    def recorded(value):
        requested.append(value)
        setter(value)

    monkeypatch.setattr(cv2, "setNumThreads", recorded)
    with cpu_threads(1):
        assert requested[0] == 1
    assert requested[-1] == original
    assert cv2.getNumThreads() == original


def test_cuda_method_preflight_preserves_frozen_spec(monkeypatch, tmp_path):
    def unavailable():
        raise subprocess.CalledProcessError(1, "nvidia-smi")

    monkeypatch.setattr(r_api, "_gpu_pids", unavailable)
    output = tmp_path / "predict"
    with pytest.raises(subprocess.CalledProcessError):
        r_api._run_cuda_method("predict", ["features", "model", str(output)],
                               {}, str(output), n_cores=2, max_seconds=60)
    spec = output.with_name(output.name + "-METHOD_FROZEN.json")
    failure = output.with_name(output.name + "-METHOD_FAILED.json")
    assert json.loads(spec.read_text())["n_cores"] == 2
    assert json.loads(failure.read_text())["status"] == "FAILED_PRESERVE_PARTIAL"
    assert not output.exists()
