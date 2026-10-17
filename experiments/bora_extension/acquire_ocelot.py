"""Acquire the original OCELOT 1.0.1 release with immutable provenance.

Only downloads and extracts data. Does not execute downloaded Python files,
run a model, request an account, or modify an existing benchmark.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import stat
import urllib.request
import zipfile

RECORD = "https://zenodo.org/records/8417503"
NAME = "ocelot2023_v1.0.1.zip"
URL = RECORD + "/files/" + NAME + "?download=1"
MD5 = "215230295b0440b4dc356519ef9ff644"
RESERVE = 20 * 2**30


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 2**20), b""):
            h.update(block)
    return h.hexdigest()


def check_members(infos):
    """Reject collisions, links, absolute paths, and traversal before writing."""
    seen = set()
    for info in infos:
        name = info.filename
        parts = Path(name).parts
        if (name in seen or not parts or name.startswith("/") or "\\" in name
                or ".." in parts or ":" in parts[0]
                or stat.S_ISLNK(info.external_attr >> 16)):
            raise ValueError(f"Unsafe or duplicate ZIP member: {name}")
        seen.add(name)


def acquire(root):
    root = Path(root).resolve()
    folder = root / "raw/ocelot"
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / NAME
    part = archive.with_suffix(".zip.part")
    if not archive.exists():
        start = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "Bora-benchmark/0.2"}
        if start:
            headers["Range"] = f"bytes={start}-"
        with urllib.request.urlopen(urllib.request.Request(URL, headers=headers), timeout=60) as response:
            if start and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-")):
                raise ValueError("Server did not honor resume; partial bytes retained")
            if "text/html" in response.headers.get("Content-Type", ""):
                raise ValueError("Received an HTML page rather than the archive")
            with part.open("ab" if start else "wb") as handle:
                for block in iter(lambda: response.read(2**20), b""):
                    if shutil.disk_usage(folder).free < RESERVE + len(block):
                        raise ValueError("Preserve 20 GiB reserve")
                    handle.write(block)
        if digest(part, "md5") != MD5:
            raise ValueError("Publisher archive checksum mismatch; partial bytes retained")
        part.rename(archive)
    if digest(archive, "md5") != MD5:
        raise ValueError("Existing archive differs from pinned publisher release")
    sha = digest(archive)
    extracted = folder / "extracted"
    marker = extracted / ".bora-extraction.json"
    if marker.exists():
        if json.loads(marker.read_text())["archive_sha256"] != sha:
            raise ValueError("Existing extraction belongs to another archive")
    else:
        if extracted.exists() and any(extracted.iterdir()):
            raise ValueError("Incomplete extraction exists; inspect it before recovery")
        with zipfile.ZipFile(archive) as source:
            infos = source.infolist()
            check_members(infos)
            if shutil.disk_usage(folder).free < RESERVE + sum(i.file_size for i in infos):
                raise ValueError("Insufficient space for extraction and reserve")
            extracted.mkdir(exist_ok=True)
            source.extractall(extracted)
        marker.write_text(json.dumps(dict(archive_sha256=sha, members=len(infos)), indent=2) + "\n")
    record = dict(dataset="ocelot", version="1.0.1", source=RECORD, url=URL,
        publisher_md5=MD5, sha256=sha, bytes=archive.stat().st_size,
        archive=str(archive.relative_to(root)), extracted=str(extracted.relative_to(root)),
        license="CC BY-NC 4.0", terms="https://lunit-io.github.io/research/ocelot_tc/",
        downloaded_utc=datetime.now(timezone.utc).isoformat(),
        intended_task="Tissue cancer/non-cancer segmentation; cell point annotations are not benchmark masks",
        no_upstream_code_executed=True, benchmark_not_yet_extended=True,
        acquisition_code_sha256=digest(__file__))
    metadata = root / "metadata/ocelot-acquisition.json"
    metadata.parent.mkdir(exist_ok=True)
    if metadata.exists():
        previous = json.loads(metadata.read_text())
        if previous["sha256"] != sha:
            raise ValueError("Acquisition record collision")
        return previous
    metadata.write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(acquire(args.data_root), indent=2), flush=True)
