import io
import zipfile

import pytest

from experiments.bora_extension.http_ranges import RangeFile


def test_seekable_subset_reads_native_zip_and_detects_cache_change(tmp_path,monkeypatch):
    memory=io.BytesIO()
    with zipfile.ZipFile(memory,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('image.png',bytes(range(256))*20)
        z.writestr('mask.png',b'maskbytes'*100)
    data=memory.getvalue()
    fetched=[]
    def fetch(self,start,end):
        fetched.append((start,end))
        return data[start:end+1]
    monkeypatch.setattr(RangeFile,'_fetch',fetch)
    reader=RangeFile('https://example.org/pinned.zip',len(data),tmp_path,block_size=64,reserve=0)
    with zipfile.ZipFile(reader) as z:
        assert z.read('mask.png')==b'maskbytes'*100
        assert z.read('image.png')==bytes(range(256))*20
    reader.seek(0)
    assert reader.read(17)==data[:17]
    reader.seek(-5,io.SEEK_END)
    assert reader.read()==data[-5:]
    assert reader.read()==b''
    assert not reader.receipt()['publisher_full_archive_checksum_verified']
    path=next(tmp_path.glob('*.bin'))
    path.write_bytes(bytes(len(path.read_bytes())))
    reader.seek(int(path.name.split('-')[0]))
    with pytest.raises(ValueError,match='block changed'):
        reader.read(1)


def test_range_identity_and_short_responses_fail(tmp_path,monkeypatch):
    monkeypatch.setattr(RangeFile,'_fetch',lambda self,start,end:b'bad')
    reader=RangeFile('https://example.org/fixed',100,tmp_path,block_size=16,reserve=0)
    with pytest.raises(ValueError,match='block length'):
        reader.read(2)
    with pytest.raises(ValueError,match='different source'):
        RangeFile('https://example.org/other',100,tmp_path,block_size=16,reserve=0)
    with pytest.raises(ValueError,match='Invalid'):
        reader.seek(-1)


@pytest.mark.parametrize('header',['bytes 0-15/100','0-15/100'])
def test_accepts_checked_zenodo_and_gdc_content_range_formats(tmp_path,monkeypatch,header):
    class Response:
        status=206
        headers={'Content-Range':header}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,size):return bytes(range(16))
    monkeypatch.setattr('urllib.request.urlopen',lambda *args,**kwargs:Response())
    reader=RangeFile('https://example.org/fixed',100,tmp_path,block_size=16,reserve=0)
    assert reader.read(16)==bytes(range(16))


def test_two_transfer_prefetch_preserves_exact_bytes_and_cached_integrity(tmp_path,monkeypatch):
    data=bytes(range(256))*5;fetched=[]
    def fetch(self,start,end):
        fetched.append((start,end));return data[start:end+1]
    monkeypatch.setattr(RangeFile,'_fetch',fetch)
    reader=RangeFile('https://example.org/native',len(data),tmp_path,block_size=64,reserve=0,concurrency=2)
    reader.seek(17);assert reader.read(700)==data[17:717]
    count=len(fetched);reader.seek(17);assert reader.read(700)==data[17:717] and len(fetched)==count
    assert len(reader.receipt()['blocks_sha256'])==12
    first=tmp_path/'0-63.bin';first.write_bytes(b'x'*64)
    reader.seek(17)
    with pytest.raises(ValueError,match='block changed'):reader.read(700)
