#!/usr/bin/env python3
"""Download the 50 official paired GigaTIME test patches for reference checks."""
from pathlib import Path
import argparse,hashlib,json,shutil,urllib.request,zipfile

URL="https://www.dropbox.com/scl/fi/8ampg43fs2yowt9y6vvr1/sample_test_data.zip?rlkey=bkg4w183qnvkh2dudqy3d8lsg&dl=1"
SHA256="cd1212b361f7bfe4f7f01779a59f437c691fd43a684d7b5f584c37bf7d0e2601"


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--output",type=Path,default=Path("/mnt/sata_ssd/Bora-benchmark/raw/gigatime-official-test"))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    dest=a.output/"sample_test_data.zip"
    if not dest.exists():
        partial=dest.with_suffix(".part")
        with urllib.request.urlopen(URL,timeout=60) as response,partial.open("wb") as f:
            shutil.copyfileobj(response,f)
        if sha(partial)!=SHA256:raise ValueError("Reference archive checksum mismatch")
        partial.rename(dest)
    if sha(dest)!=SHA256:raise ValueError("Reference archive checksum mismatch")
    root=(a.output/"extracted").resolve()
    with zipfile.ZipFile(dest) as archive:
        for item in archive.infolist():
            if item.is_dir() or item.filename.startswith("__MACOSX"):continue
            target=(root/item.filename).resolve()
            if not target.is_relative_to(root):raise ValueError("Unsafe archive path")
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                with archive.open(item) as source,target.open("wb") as f:shutil.copyfileobj(source,f)
    (a.output/"acquisition.json").write_text(json.dumps({"source_url":URL,"sha256":SHA256,
        "bytes":dest.stat().st_size,"source_repository":"https://github.com/prov-gigatime/GigaTIME",
        "source_commit":"acc49b2567fd15ce95b5a82caaab5f0a295b88f7",
        "scope":"50 official paired test patches; first five sorted H&E files used for diagnostic development, remaining 45 for reference confirmation"},indent=2))


if __name__=="__main__":main()
