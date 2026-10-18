"""Checked seekable HTTPS reads for bounded original ZIP/WSI subsets.

Retain downloaded source blocks and their hashes. A subset cannot claim the
publisher's full-archive checksum was verified. ZIP additionally checks member
CRC32; selected extracted files must also get SHA256 acquisition records.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import urllib.request


class RangeFile(io.RawIOBase):
    def __init__(self, url, size, cache, block_size=4*2**20, reserve=20*2**30,concurrency=1):
        super().__init__()
        if not url.startswith('https://') or size <= 0 or block_size <= 0:
            raise ValueError('Pinned HTTPS URL and positive source geometry required')
        self.url,self.size,self.cache=url,int(size),Path(cache)
        self.block_size,self.reserve=int(block_size),int(reserve)
        self.concurrency=int(concurrency)
        if self.concurrency not in (1,2):raise ValueError('At most two bounded transfers')
        self.position=0
        self.cache.mkdir(parents=True,exist_ok=True)
        self.identity=dict(url=url,bytes=self.size,block_size=self.block_size)
        p=self.cache/'SOURCE.json'
        if p.exists() and json.loads(p.read_text()) != self.identity:
            raise ValueError('Range cache belongs to a different source')
        p.write_text(json.dumps(self.identity,indent=2)+'\n')
        receipt=self.cache/'BLOCKS.json'
        self.blocks=json.loads(receipt.read_text()) if receipt.exists() else {}

    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position

    def seek(self,offset,whence=io.SEEK_SET):
        origin={io.SEEK_SET:0,io.SEEK_CUR:self.position,io.SEEK_END:self.size}
        if whence not in origin or origin[whence]+offset < 0:
            raise ValueError('Invalid byte seek')
        self.position=origin[whence]+offset
        return self.position

    def _fetch(self,start,end):
        request=urllib.request.Request(self.url,headers={
            'Range':f'bytes={start}-{end}', 'User-Agent':'Bora-tissue-benchmark/0.3',
            'Accept-Encoding':'identity'})
        with urllib.request.urlopen(request,timeout=60) as response:
            # GDC returns206 with "start-end/total" while Zenodo prefixes
            # "bytes ". Both must encode exactly the requested interval.
            header=response.headers.get('Content-Range','').strip()
            match=re.fullmatch(r'(?:bytes\s+)?(\d+)-(\d+)/(\d+)',header,re.IGNORECASE)
            if response.status != 206 or match is None or tuple(map(int,match.groups())) != (start,end,self.size):
                raise ValueError('Provider did not honor the exact bounded byte range')
            data=response.read(end-start+2)
            if len(data) != end-start+1:
                raise ValueError('Truncated or excessive range response')
            return data

    def _block(self,index):
        start=index*self.block_size
        end=min(self.size,start+self.block_size)-1
        name=f'{start}-{end}.bin'
        path=self.cache/name
        if path.exists():
            data=path.read_bytes()
        else:
            if shutil.disk_usage(self.cache).free < self.reserve+(end-start+1):
                raise ValueError('Preserve20GiB free during subset acquisition')
            data=self._fetch(start,end)
            if len(data) != end-start+1:
                raise ValueError('Incorrect range block length')
            temporary=path.with_suffix('.part')
            temporary.write_bytes(data)
            temporary.rename(path)
        sha=hashlib.sha256(data).hexdigest()
        if len(data) != end-start+1 or name in self.blocks and self.blocks[name] != sha:
            raise ValueError('Retained source block changed')
        self.blocks[name]=sha
        (self.cache/'BLOCKS.json').write_text(json.dumps(self.blocks,indent=2)+'\n')
        return data

    def _prefetch(self,indices):
        missing=[]
        for index in indices:
            start=index*self.block_size;end=min(self.size,start+self.block_size)-1
            if not (self.cache/f'{start}-{end}.bin').exists():missing.append((start,end))
        if not missing:return
        if shutil.disk_usage(self.cache).free < self.reserve+sum(b-a+1 for a,b in missing):
            raise ValueError('Preserve20GiB free during bounded batch acquisition')
        # Only network reads run concurrently. Retained blocks and receipts
        # are committed by this thread, avoiding concurrent receipt writes.
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            payloads=pool.map(lambda r:self._fetch(*r),missing)
            for (start,end),payload in zip(missing,payloads):
                if len(payload)!=end-start+1:raise ValueError('Incorrect prefetched range length')
                name=f'{start}-{end}.bin';path=self.cache/name
                temporary=path.with_suffix('.part');temporary.write_bytes(payload);temporary.rename(path)
                self.blocks[name]=hashlib.sha256(payload).hexdigest()
                (self.cache/'BLOCKS.json').write_text(json.dumps(self.blocks,indent=2)+'\n')

    def read(self,size=-1):
        if size is None or size < 0:
            size=max(0,self.size-self.position)
        end=min(self.size,self.position+size)
        if self.concurrency>1 and end>self.position:
            self._prefetch(range(self.position//self.block_size,(end-1)//self.block_size+1))
        chunks=[]
        while self.position < end:
            index=self.position//self.block_size
            block=self._block(index)
            begin=self.position % self.block_size
            take=min(len(block)-begin,end-self.position)
            chunks.append(block[begin:begin+take])
            self.position+=take
        return b''.join(chunks)

    def readinto(self,buffer):
        data=self.read(len(buffer))
        buffer[:len(data)]=data
        return len(data)

    def receipt(self):
        return dict(**self.identity,blocks_sha256=dict(self.blocks),
            retained_source_bytes=sum(int(n.split('-')[1].split('.')[0])-int(n.split('-')[0])+1 for n in self.blocks),
            publisher_full_archive_checksum_verified=False,
            explanation='Only retained source blocks were downloaded; no full-archive hash claim')
