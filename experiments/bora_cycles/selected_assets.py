"""Exact categorical viewer layers with transparent zero, after prediction freeze.

The TIFF labels, scores and existing published PNGs remain unchanged. For
prediction PNGs, only a PNG transparency chunk is inserted; compressed image
data are copied byte-for-byte, avoiding another native-mask encoding pass.
"""
from pathlib import Path
import json
import struct
import zlib

import numpy as np
from PIL import Image
import tifffile

from .run import digest,palette_bytes,resize

SIGNATURE=b'\x89PNG\r\n\x1a\n'


def transparent_zero_png(data):
    if data[:8]!=SIGNATURE:raise ValueError('Expected an indexed PNG')
    position=8;chunks=[];inserted=False;header=False;ended=False
    payload=b'tRNS\0';transparent=struct.pack('>I',1)+payload+struct.pack('>I',zlib.crc32(payload)&0xffffffff)
    while position<len(data):
        if position+12>len(data):raise ValueError('Truncated PNG chunk')
        size=struct.unpack('>I',data[position:position+4])[0];end=position+12+size
        if end>len(data):raise ValueError('Truncated PNG data')
        kind=data[position+4:position+8];body=data[position+8:position+8+size]
        if zlib.crc32(kind+body)&0xffffffff!=struct.unpack('>I',data[end-4:end])[0]:raise ValueError('Corrupt PNG chunk')
        if kind==b'IHDR':
            if position!=8 or size!=13 or body[8:10]!=bytes([8,3]):raise ValueError('Expected8-bit indexed native PNG')
            header=True
        if kind==b'IDAT' and not inserted:
            if not header:raise ValueError('PNG image data precedes header')
            chunks.append(transparent);inserted=True
        if kind!=b'tRNS':chunks.append(data[position:end])
        position=end
        if kind==b'IEND':
            ended=True
            if position!=len(data):raise ValueError('Trailing PNG bytes')
    if not inserted or not ended:raise ValueError('Incomplete PNG')
    return SIGNATURE+b''.join(chunks)


def save_exact(path,data):
    """Never silently replace an earlier display artifact with different bytes."""
    if path.exists():
        if path.read_bytes()!=data:raise ValueError('Existing display asset changed')
    else:path.write_bytes(data)


def layers(source,root,sample,methods):
    source,root=Path(source),Path(root);case=sample['case'];src=source/'cases'/case;out=root/'cases'/case
    protocol=json.loads((root/'PROTOCOL.json').read_text());frozen=json.loads((root/'ALL_PREDICTIONS_FROZEN.json').read_text())
    if (frozen['protocol_sha256']!=digest(root/'PROTOCOL.json') or frozen['fine_reference_read']
            or set(frozen['manifests'])!=set(protocol['cases']) or case not in protocol['cases']
            or digest(source/'PLAN.json')!=protocol['input_plan_sha256']
            or digest(out/'PREDICTIONS.json')!=frozen['manifests'][case]):
        raise ValueError('Complete matching prediction freeze required before viewer layers')
    predicted=json.loads((out/'PREDICTIONS.json').read_text());inputs=json.loads((src/'INPUTS_FROZEN.json').read_text())
    if digest(src/'INPUTS_FROZEN.json')!=predicted['input_manifest_sha256']:raise ValueError('Viewer input identity changed')
    if (digest(src/'reference.native.tif')!=sample['reference_native_sha256']
            or digest(src/'coarse.tif')!=inputs['files_sha256']['coarse.tif']):
        raise ValueError('Viewer reference/coarse input changed')
    folder=out/'selected-view';folder.mkdir(exist_ok=True)
    palette=palette_bytes(sample['palette']);files={};sources={};paths={}
    labels=tifffile.imread(src/'reference.native.tif')
    coarse=resize(tifffile.imread(src/'coarse.tif'),sample['native_shape_yx'],True)
    from io import BytesIO
    for name,values in [('reference',labels),('coarse',coarse)]:
        if values.shape!=tuple(sample['native_shape_yx']) or values.dtype!=np.uint8:raise ValueError('Categorical native uint8 grid required')
        image=Image.fromarray(values,'P');image.putpalette(palette);stream=BytesIO()
        image.save(stream,format='PNG',compress_level=1)
        path=folder/(name+'.png');save_exact(path,transparent_zero_png(stream.getvalue()))
        files[path.name]=digest(path);paths[name]='selected-view/'+path.name
    for method in methods:
        if method not in protocol['methods']:raise ValueError('Unexecuted method requested')
        native=method+'/native.tif';png=out/method/'mask.png'
        if digest(out/native)!=predicted['files_sha256'][native]:raise ValueError('Native prediction changed')
        with Image.open(png) as image:
            if image.mode!='P' or image.getpalette()!=palette or not np.array_equal(np.array(image),tifffile.imread(out/native)):
                raise ValueError('Source display PNG differs from frozen native labels')
        path=folder/(method+'.png');save_exact(path,transparent_zero_png(png.read_bytes()))
        files[path.name]=digest(path);sources[method]=dict(native_sha256=digest(out/native),original_png_sha256=digest(png))
        paths[method]='selected-view/'+path.name
    record=dict(case=case,methods=methods,files_sha256=files,source_masks=sources,
        prediction_manifest_sha256=digest(out/'PREDICTIONS.json'),input_manifest_sha256=digest(src/'INPUTS_FROZEN.json'),
        reference_sha256=sample['reference_native_sha256'],zero_alpha=0,positive_label_alpha=255,
        prediction_png_compressed_image_data_preserved=True,labels_modified=False,scores_modified=False,
        original_pngs_modified=False,source_sha256=digest(Path(__file__)))
    manifest=folder/'MANIFEST.json'
    encoded=json.dumps(record,indent=2,allow_nan=False).encode()
    save_exact(manifest,encoded)
    return paths,dict(record,manifest_sha256=digest(manifest))
