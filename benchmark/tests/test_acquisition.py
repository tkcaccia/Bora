import hashlib
from io import BytesIO
import tarfile
import zipfile

import pytest

from benchmark.download import extract_tar, extract_zip, verify


def test_mismatch_is_never_marked_verified(tmp_path):
    file = tmp_path / "data"
    file.write_bytes(b"incorrect")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify(file, {"sha256": hashlib.sha256(b"expected").hexdigest()})


def test_archive_traversal_is_rejected_before_any_extraction(tmp_path):
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("valid.txt", "valid")
        z.writestr("../outside.txt", "outside")
    with pytest.raises(ValueError, match="Unsafe ZIP path"):
        extract_zip(archive, tmp_path / "extracted", reserve_gib=0)
    assert not (tmp_path / "outside.txt").exists()
    assert not (tmp_path / "extracted" / "valid.txt").exists()


def test_verified_archive_can_be_revisited_without_overwriting(tmp_path):
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("nested/image.txt", "original")
    out = tmp_path / "extracted"
    extract_zip(archive, out, reserve_gib=0)
    extract_zip(archive, out, reserve_gib=0)
    assert (out / "nested/image.txt").read_text() == "original"
    assert (out / ".bora-extraction.json").exists()


@pytest.mark.parametrize("unsafe", ["traversal", "symlink"])
def test_tar_rejects_unsafe_members_before_writing_valid_files(tmp_path, unsafe):
    archive = tmp_path / "shard.tar"
    with tarfile.open(archive, "w") as tar:
        valid = tarfile.TarInfo("valid.png")
        valid.size = 5
        tar.addfile(valid, BytesIO(b"image"))
        bad = tarfile.TarInfo("../outside" if unsafe == "traversal" else "link")
        if unsafe == "symlink":
            bad.type, bad.linkname = tarfile.SYMTYPE, "../outside"
        tar.addfile(bad)
    with pytest.raises(ValueError, match="Unsafe TAR|links/devices"):
        extract_tar(archive, tmp_path / "out", reserve_gib=0)
    assert not (tmp_path / "out" / "valid.png").exists()
    assert not (tmp_path / "outside").exists()


def test_paired_tar_shards_share_a_root_and_remain_idempotent(tmp_path):
    for folder in ("images", "masks"):
        archive = tmp_path / (folder + ".tar")
        with tarfile.open(archive, "w") as tar:
            member = tarfile.TarInfo(folder + "/pair.png")
            member.size = 5
            tar.addfile(member, BytesIO(b"bytes"))
        extract_tar(archive, tmp_path / "out", reserve_gib=0)
        extract_tar(archive, tmp_path / "out", reserve_gib=0)
    assert (tmp_path / "out/images/pair.png").read_bytes() == b"bytes"
    assert (tmp_path / "out/masks/pair.png").read_bytes() == b"bytes"
