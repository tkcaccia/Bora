"""Download the separately licensed Virchow2 files for the R interface."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4


REPO_ID = "paige-ai/Virchow2"
FILES = ("config.json", "model.safetensors")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_virchow2(model_dir: str, revision: str = "main") -> str:
    """Fetch only the files required by Bora, using the user's Hub credentials.

    The destination appears only after both files and their receipt are complete.
    Failed or interrupted downloads remain in a clearly named staging directory.
    """
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Install huggingface_hub in the RETICULATE_PYTHON environment "
            "(python -m pip install huggingface_hub)."
        ) from exc

    destination = Path(model_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"model_dir already exists: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"model_dir parent does not exist: {destination.parent}")
    if not revision or not isinstance(revision, str):
        raise ValueError("revision must be a nonempty branch, tag or commit")

    # The Hub discovers HF_TOKEN or a locally stored login; credentials are
    # never passed in arguments or included in the receipt.
    try:
        info = HfApi().model_info(REPO_ID, revision=revision)
    except Exception as exc:
        raise RuntimeError(
            "Cannot access paige-ai/Virchow2. Accept its gated terms with "
            "your own institutional Hugging Face account, then set HF_TOKEN "
            "or run 'hf auth login' in the Python environment."
        ) from exc
    commit = info.sha
    if not isinstance(commit, str) or len(commit) != 40:
        raise RuntimeError("The Hub did not return a 40-character model commit")

    staging = destination.parent / f".{destination.name}.incomplete-{uuid4().hex}"
    staging.mkdir()
    try:
        snapshot_download(
            repo_id=REPO_ID, revision=commit, local_dir=str(staging),
            allow_patterns=list(FILES), max_workers=2,
        )
        files = {}
        for name in FILES:
            path = staging / name
            if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
                raise RuntimeError(f"Virchow2 download is missing {name}")
            files[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        config = json.loads((staging / "config.json").read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not {"architecture", "model_args"} <= config.keys():
            raise RuntimeError("Virchow2 config.json lacks encoder settings required by Bora")
        receipt = {
            "format": "bora-r-virchow2-download-v1", "status": "COMPLETE",
            "repo_id": REPO_ID, "requested_revision": revision,
            "resolved_commit": commit, "model_dir": str(destination),
            "files": files,
        }
        with (staging / "DOWNLOAD_COMPLETE.json").open("w", encoding="utf-8") as output:
            json.dump(receipt, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        # A new directory and staging are on the same filesystem, so this
        # final rename makes the complete model visible in one step.
        os.rename(staging, destination)
    except Exception:
        # Retain partial bytes as evidence and for manual diagnosis. Bora will
        # never use this directory because it is not the requested model_dir.
        raise
    return str(destination / "DOWNLOAD_COMPLETE.json")
