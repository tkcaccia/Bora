"""Offline contract tests for the gated model download helper."""

import hashlib
import json
import sys
import types

import pytest

from bora.r_model_download import download_virchow2


COMMIT = "a" * 40


def _fake_hub(monkeypatch, *, missing=None, denied=False, invalid_config=False):
    def model_info(repo_id, revision):
        assert repo_id == "paige-ai/Virchow2"
        assert revision == "main"
        if denied:
            raise PermissionError("private token secret")
        return types.SimpleNamespace(sha=COMMIT)

    def snapshot_download(**kwargs):
        assert kwargs["revision"] == COMMIT
        assert kwargs["allow_patterns"] == ["config.json", "model.safetensors"]
        for name, content in {"config.json": b'{"architecture":"fake","model_args":{}}', "model.safetensors": b"weights"}.items():
            if name != missing:
                from pathlib import Path
                if name == "config.json" and invalid_config:
                    content = b"{}"
                (Path(kwargs["local_dir"]) / name).write_bytes(content)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(
        HfApi=lambda: types.SimpleNamespace(model_info=model_info),
        snapshot_download=snapshot_download,
    ))


def test_download_records_pinned_model_and_atomic_destination(tmp_path, monkeypatch):
    _fake_hub(monkeypatch)
    directory = tmp_path / "Virchow2"
    receipt_path = download_virchow2(str(directory))
    receipt = json.loads(open(receipt_path, encoding="utf-8").read())
    assert receipt["status"] == "COMPLETE"
    assert receipt["resolved_commit"] == COMMIT
    assert receipt["files"]["model.safetensors"]["sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert json.loads((directory / "config.json").read_text())["architecture"] == "fake"
    assert "token" not in json.dumps(receipt).lower()
    with pytest.raises(FileExistsError):
        download_virchow2(str(directory))


def test_missing_file_cannot_become_model_dir(tmp_path, monkeypatch):
    _fake_hub(monkeypatch, missing="model.safetensors")
    directory = tmp_path / "Virchow2"
    with pytest.raises(RuntimeError, match="missing model.safetensors"):
        download_virchow2(str(directory))
    assert not directory.exists()
    assert len(list(tmp_path.glob(".Virchow2.incomplete-*"))) == 1


def test_wrong_model_config_cannot_become_model_dir(tmp_path, monkeypatch):
    _fake_hub(monkeypatch, invalid_config=True)
    directory = tmp_path / "Virchow2"
    with pytest.raises(RuntimeError, match="lacks encoder settings"):
        download_virchow2(str(directory))
    assert not directory.exists()


def test_gated_access_reports_action_without_token(tmp_path, monkeypatch):
    _fake_hub(monkeypatch, denied=True)
    directory = tmp_path / "Virchow2"
    with pytest.raises(RuntimeError, match="Accept its gated terms") as error:
        download_virchow2(str(directory))
    assert "private token secret" not in str(error.value)
    assert not directory.exists()
