from io import BytesIO
import json
import struct

import numpy as np
from PIL import Image
import pytest
import tifffile

from experiments.bora_cycles.run import digest,palette_bytes,save
from experiments.bora_cycles.selected_assets import transparent_zero_png,layers


def png(labels):
    image=Image.fromarray(labels,'P');image.putpalette(palette_bytes([[220,40,50],[30,180,70]]))
    stream=BytesIO();image.save(stream,format='PNG');return stream.getvalue()


def idat(data):
    parts=[];pos=8
    while pos<len(data):
        n=struct.unpack('>I',data[pos:pos+4])[0]
        if data[pos+4:pos+8]==b'IDAT':parts.append(data[pos+8:pos+8+n])
        pos+=12+n
    return parts


def test_transparency_changes_no_label_palette_or_compressed_image_data():
    labels=np.array([[0,1,2],[2,0,1]],np.uint8);before=png(labels);after=transparent_zero_png(before)
    with Image.open(BytesIO(after)) as image:
        np.testing.assert_array_equal(np.array(image),labels)
        assert image.getpalette()==Image.open(BytesIO(before)).getpalette()
        np.testing.assert_array_equal(np.array(image.convert('RGBA'))[...,3],(labels>0)*255)
    assert idat(before)==idat(after)
    assert transparent_zero_png(after)==after
    corrupt=bytearray(before);corrupt[-5]^=1
    with pytest.raises(ValueError,match='Corrupt'):transparent_zero_png(bytes(corrupt))
    stream=BytesIO();Image.new('RGB',(3,2)).save(stream,format='PNG')
    with pytest.raises(ValueError,match='indexed'):transparent_zero_png(stream.getvalue())


def fixture(tmp_path):
    source=tmp_path/'input';root=tmp_path/'result';src=source/'cases/c';out=root/'cases/c';src.mkdir(parents=True);(out/'J01').mkdir(parents=True)
    palette=[[220,40,50],[30,180,70]];truth=np.array([[0,1,2],[2,0,1]],np.uint8)
    tifffile.imwrite(src/'reference.native.tif',truth);tifffile.imwrite(src/'coarse.tif',np.array([[0,1,2]],np.uint8))
    save(source/'PLAN.json',dict(samples=['fixture']))
    save(src/'INPUTS_FROZEN.json',dict(files_sha256={'coarse.tif':digest(src/'coarse.tif')}))
    tifffile.imwrite(out/'J01/native.tif',truth);(out/'J01/mask.png').write_bytes(png(truth))
    sample=dict(case='c',native_shape_yx=[2,3],reference_native_sha256=digest(src/'reference.native.tif'),palette=palette)
    save(root/'PROTOCOL.json',dict(cases=['c'],methods=['J01'],input_plan_sha256=digest(source/'PLAN.json')))
    save(out/'PREDICTIONS.json',dict(files_sha256={'J01/native.tif':digest(out/'J01/native.tif')},input_manifest_sha256=digest(src/'INPUTS_FROZEN.json')))
    save(root/'ALL_PREDICTIONS_FROZEN.json',dict(protocol_sha256=digest(root/'PROTOCOL.json'),fine_reference_read=False,manifests={'c':digest(out/'PREDICTIONS.json')}))
    return source,root,sample


def test_viewer_layers_preserve_frozen_native_masks_and_create_only_exact_transparent_derivatives(tmp_path):
    source,root,sample=fixture(tmp_path);out=root/'cases/c'
    originals={name:digest(out/name) for name in ['J01/native.tif','J01/mask.png','PREDICTIONS.json']}
    paths,record=layers(source,root,sample,['J01'])
    assert paths['reference']=='selected-view/reference.png'
    assert record['prediction_png_compressed_image_data_preserved'] and not record['labels_modified']
    for key,path in paths.items():
        with Image.open(out/path) as image:assert image.info['transparency']==0
    assert {name:digest(out/name) for name in originals}==originals
    assert idat((out/'selected-view/J01.png').read_bytes())==idat((out/'J01/mask.png').read_bytes())
    assert layers(source,root,sample,['J01'])==(paths,record)


def test_an_incomplete_prediction_freeze_is_rejected_before_fine_annotation_decoding(tmp_path):
    source,root,sample=fixture(tmp_path)
    (source/'cases/c/reference.native.tif').write_bytes(b'must not be decoded')
    f=root/'ALL_PREDICTIONS_FROZEN.json';record=json.loads(f.read_text());record['manifests']={};save(f,record)
    with pytest.raises(ValueError,match='Complete matching prediction freeze'):layers(source,root,sample,['J01'])
    assert not (root/'cases/c/selected-view').exists()


def test_a_display_png_cannot_silently_change_the_frozen_segmentation(tmp_path):
    source,root,sample=fixture(tmp_path)
    (root/'cases/c/J01/mask.png').write_bytes(png(np.ones((2,3),np.uint8)))
    with pytest.raises(ValueError,match='differs from frozen native'):layers(source,root,sample,['J01'])
