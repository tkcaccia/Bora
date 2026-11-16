"""Safety checks for worker admission and uncertain cleanup."""
import subprocess

import pytest

from bora import r_api


def test_worker_refuses_existing_gpu_context(monkeypatch, tmp_path):
    monkeypatch.setattr(r_api, "_gpu_pids", lambda: {987654})
    with pytest.raises(RuntimeError, match="GPU compute context exists"):
        r_api._run_worker(tmp_path / "spec.json", 60, tmp_path / "worker.log")
    assert not (tmp_path / "worker.log").exists()


def test_cleanup_requires_successful_gpu_inspection(monkeypatch):
    monkeypatch.setattr(r_api, "_owned_pids", lambda pgid: [])
    monkeypatch.setattr(r_api, "_gpu_pids", lambda: set())
    assert r_api._cleanup(987654, window_seconds=0)["status"] == "VERIFIED"
    def unavailable():
        raise subprocess.TimeoutExpired("nvidia-smi", 20)
    monkeypatch.setattr(r_api, "_gpu_pids", unavailable)
    result = r_api._cleanup(987654, window_seconds=0)
    assert result["status"] == "UNKNOWN_OR_FAILED"
    assert result["observations"][0]["inspection_error"]
