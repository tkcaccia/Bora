import hashlib
import zipfile

import pytest

from benchmark.download import extract_zip, verify


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
