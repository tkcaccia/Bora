"""Acquire bounded native SegPath pairs from original Zenodo TAR streams."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.request

import numpy as np
from PIL import Image

from .download import DEFAULT_ROOT, LOCK, digest, fetch, verify
from .prepare import write_once


class PrefixReader:
    """Retain and hash exactly the compressed bytes actually read, up to a hard cap."""
    def __init__(self, response, output, limit, reserve, seed=None, seed_bytes=0):
        self.response, self.output, self.limit, self.reserve = response, output, limit, reserve
        self.seed, self.seed_bytes = seed, seed_bytes
        self.count = 0
        self.sha256 = hashlib.sha256()

    def read(self, size):
        if shutil.disk_usage(Path(self.output.name).parent).free < self.reserve + max(size, 2**20):
            raise ValueError("Insufficient free space while retaining source prefix")
        if self.count >= self.limit:
            return b""
        wanted = min(size, self.limit - self.count)
        data = b""
        if self.seed is not None and self.count < self.seed_bytes:
            data = self.seed.read(min(wanted, self.seed_bytes - self.count))
        if len(data) < wanted:
            fresh = self.response.read(wanted - len(data))
            self.output.write(fresh)
            data += fresh
        self.sha256.update(data)
        self.count += len(data)
        return data


def acquire_target(target, root, count, limit, reserve_gib):
    name = target["class"]
    destination = root / "raw" / "segpath-subset" / name
    destination.mkdir(parents=True, exist_ok=True)
    receipt = destination / "subset-receipt.json"
    if receipt.exists():
        data = json.loads(receipt.read_text())
        if data["source_archive"] != target["archive"] or len(data["pairs"]) != count:
            raise ValueError("Subset settings changed; select a new data root")
        if digest(root / data["prefix_file"]) != data["prefix_sha256"]:
            raise ValueError("Retained source-prefix checksum mismatch")
        for sidecar in target["sidecars"]:
            verify(root / "raw" / "segpath-subset" / sidecar["path"], sidecar)
        for row in data["pairs"]:
            for key in ("image", "mask"):
                if digest(root / row[key]) != row[key + "_sha256"]:
                    raise ValueError("Existing selected pair checksum mismatch")
        return data
    for sidecar in target["sidecars"]:
        fetch(sidecar, root / "raw" / "segpath-subset" / sidecar["path"], reserve_gib)
    archive = target["archive"]
    end = min(limit, archive["size"]) - 1
    prefix = destination / "source-prefix.tar.gz.part"
    existing = prefix.stat().st_size if prefix.exists() else 0
    if existing > end:
        raise ValueError("Existing prefix already reaches the cap; increase the cap to continue")
    request = urllib.request.Request(archive["url"], headers={"Range": f"bytes={existing}-{end}",
                                                           "User-Agent": "Bora-benchmark/0.1"})
    pairs, slides = [], set()
    # Reading a fixed-size seed permits exact byte-range recovery after interruption.
    prefix.touch(exist_ok=True)
    with urllib.request.urlopen(request, timeout=120) as response, prefix.open("ab") as out, prefix.open("rb") as seed:
        expected = f"bytes {existing}-{end}/{archive['size']}"
        if response.status != 206 or response.headers.get("Content-Range") != expected:
            raise ValueError("Source did not honor the bounded byte range")
        etag = response.headers.get("ETag")
        reader = PrefixReader(response, out, end + 1, reserve_gib * 2**30, seed, existing)
        with tempfile.TemporaryDirectory(prefix="selection-", dir=destination) as temporary:
            staging = Path(temporary)
            try:
                with tarfile.open(fileobj=reader, mode="r|gz") as tar:
                    seen = set()
                    for member in tar:
                        path = PurePosixPath(member.name)
                        if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                            raise ValueError("Unsafe source TAR path")
                        if member.isdir():
                            continue
                        if not member.isfile():
                            raise ValueError("Source TAR links/devices are unsupported")
                        if not path.name.endswith(("_HE.png", "_mask.png")):
                            continue
                        if path.name in seen:
                            raise ValueError("Duplicate source basename")
                        seen.add(path.name)
                        if member.size > 16 * 2**20:
                            raise ValueError("Unexpectedly large native SegPath PNG")
                        if shutil.disk_usage(destination).free < reserve_gib * 2**30 + 2 * member.size:
                            raise ValueError("Insufficient space for native PNG staging")
                        data = tar.extractfile(member).read()
                        write_once(staging / path.name, data)
                        base = path.name.removesuffix("_HE.png").removesuffix("_mask.png")
                        slide = base.rsplit("_", 3)[-3]
                        image, mask = staging / (base + "_HE.png"), staging / (base + "_mask.png")
                        if slide in slides or not (image.exists() and mask.exists()):
                            continue
                        with Image.open(image) as im, Image.open(mask) as mk:
                            values = np.asarray(mk)
                            if im.mode not in ("RGB", "RGBA") or im.size != (984, 984) or mk.size != im.size:
                                raise ValueError("Bad native SegPath geometry")
                            if im.mode == "RGBA" and im.getchannel("A").getextrema() != (255, 255):
                                raise ValueError("Source RGBA alpha channel is not fully opaque")
                            if values.ndim != 2 or not set(map(int, np.unique(values))).issubset({0, 1}):
                                raise ValueError("Unexpected SegPath mask values")
                            positive = int(np.count_nonzero(values))
                        if not positive or positive == 984**2:
                            continue  # Require a visible target boundary in this inspection subset.
                        row = {"id": base, "slide_id": slide, "target": name,
                               "positive_pixels": positive}
                        for key, source in (("image", image), ("mask", mask)):
                            dest = destination / "pairs" / source.name
                            write_once(dest, source.read_bytes())
                            row[key], row[key + "_sha256"] = str(dest.relative_to(root)), digest(dest)
                        pairs.append(row)
                        slides.add(slide)
                        if len(pairs) == count:
                            break
            except (tarfile.ReadError, EOFError) as error:
                raise ValueError("Prefix cap reached before enough complete pairs were found") from error
        if len(pairs) != count:
            raise ValueError("Too few native pairs inside the source prefix cap")
    report = {"target": name, "source_record": target["record"], "source_archive": archive,
              "full_archive_checksum_verified": False, "prefix_bytes": prefix.stat().st_size,
              "source_bytes_consumed": reader.count, "prefix_sha256": digest(prefix), "prefix_file": str(prefix.relative_to(root)),
              "response_etag": etag, "pairs": pairs,
              "selection": "First complete nonempty/nonfull pairs from distinct source slide IDs in TAR order",
              "scope": "Convenience inspection subset; not a representative or independent test cohort"}
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{name}: {len(pairs)} native pairs, {reader.count} source-prefix bytes retained", flush=True)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--pairs-per-target", type=int, default=3)
    p.add_argument("--max-prefix-gib", type=float, default=2)
    p.add_argument("--reserve-gib", type=float, default=20)
    a = p.parse_args(argv)
    if min(a.pairs_per_target, a.max_prefix_gib, a.reserve_gib) <= 0:
        p.error("Counts, prefix cap and reserve must be positive")
    root = a.root.resolve()
    source = json.loads(LOCK.read_text())["segpath-subset"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        reports = list(pool.map(lambda t: acquire_target(t, root, a.pairs_per_target,
                                                       int(a.max_prefix_gib * 2**30), a.reserve_gib), source["targets"]))
    path = root / "metadata" / "segpath-subset-acquisition.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"source": source, "targets": reports, "bora_executed": False}, indent=2) + "\n")


if __name__ == "__main__":
    main()
