"""Retire large whole-slide work arrays after the final result is verified."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def retire_polished_work_arrays(output: Path, complete: dict) -> dict:
    """Keep final TIFF, vectors and receipts; remove verified staging arrays.

    Audit hashes and the separate reviewed tissue support remain available.
    The large solver masks are no longer needed after the final output is
    checksum-bound and are retired by default.
    """
    output = Path(output)
    receipt = output / "INTERMEDIATE_MASKS_RETIRED.json"
    if receipt.exists():
        raise FileExistsError(receipt)
    complete_path = output / "FULL_SLIDE_COMPLETE.json"
    # The in-memory solver receipt can contain tuples (for measured X/Y
    # spacing); JSON persists those as arrays. Compare their serialized form.
    if (complete.get("status") != "COMPLETE" or
            json.loads(complete_path.read_text()) != json.loads(json.dumps(complete))):
        raise ValueError("A matching completed full-slide receipt is required")
    for name, expected in complete["files_sha256"].items():
        if _sha(output / name) != expected:
            raise ValueError(f"Final full-slide asset differs: {name}")
    audit_inputs = (
        "support.npy",
        "global-prepolish-cleaned-mask.npy",
        "contour-smoothed-mask.npy",
        "global-cleaned-mask.npy",
    )
    if not all((output / name).is_file() for name in audit_inputs):
        raise ValueError("Required support or before/after mask evidence is missing")
    contour = json.loads((output / "CONTOUR_POLISH_COMPLETE.json").read_text())
    for name, key in (
        ("global-prepolish-cleaned-mask.npy", "prepolish_mask_sha256"),
        ("contour-smoothed-mask.npy", "smoothed_mask_sha256"),
        ("global-cleaned-mask.npy", "final_cleaned_mask_sha256"),
    ):
        if _sha(output / name) != contour[key]:
            raise ValueError(f"Required mask evidence differs: {name}")
    candidates = [output / name for name in audit_inputs + (
        "diffusion-1um.npy", "smoothed-mask.npy", "global-rasterized-mask.npy",
    )]
    for pattern in ("polished-area-clean-*.npy", "global-raster-clean-*.npy", "global-raster-repaired-*.npy"):
        candidates.extend(sorted(output.glob(pattern)))
    candidates = [path for path in candidates if path.is_file()]
    if len(candidates) != len(set(candidates)):
        raise ValueError("Duplicate work-array candidate")
    removed = {path.name: {"bytes": path.stat().st_size, "sha256": _sha(path)} for path in candidates}
    pending = dict(status="RETIRING", final_receipt_sha256=_sha(complete_path),
                   removed=removed, bytes_to_free=sum(item["bytes"] for item in removed.values()),
                   kept_final_files=dict(complete["files_sha256"]))
    _write_json(output / "INTERMEDIATE_MASKS_RETIRE_PENDING.json", pending)
    for path in candidates:
        item = removed[path.name]
        if path.stat().st_size != item["bytes"] or _sha(path) != item["sha256"]:
            raise ValueError(f"Work array changed during retirement: {path.name}")
        path.unlink()
    if any(path.exists() for path in candidates):
        raise RuntimeError("A superseded work array remains")
    pending["status"] = "RETIRED"
    _write_json(receipt, pending)
    (output / "INTERMEDIATE_MASKS_RETIRE_PENDING.json").unlink()
    return pending
