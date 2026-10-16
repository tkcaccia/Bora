from io import BytesIO

import numpy as np
from PIL import Image
import pytest

from benchmark.describe import label_name
from benchmark.download import digest
from benchmark.prepare import entry
from benchmark.segpath_subset import PrefixReader


def test_native_opaque_rgba_bytes_are_retained_and_transparency_is_rejected(tmp_path):
    image, mask = tmp_path / 'image.png', tmp_path / 'mask.png'
    Image.new('RGBA', (3, 2), (120, 80, 60, 255)).save(image)
    Image.fromarray(np.array([[0, 1, 0], [1, 1, 0]], dtype=np.uint8)).save(mask)
    before = digest(image)
    row = entry(image, mask, tmp_path, 'segpath-panck', 'example', 'slide', 'train', 'binary', [], .220818)
    assert row['image_sha256'] == before == digest(image)
    assert row['raw_labels'] == [0, 1]
    Image.new('RGBA', (3, 2), (120, 80, 60, 0)).save(image)
    with pytest.raises(ValueError, match='nonopaque'):
        entry(image, mask, tmp_path, 'segpath-panck', 'example', 'slide', 'train', 'binary', [], .220818)


def test_range_recovery_replays_seed_once_then_appends_only_new_bytes(tmp_path):
    path = tmp_path / 'prefix'
    path.write_bytes(b'abcdef')
    with path.open('ab') as out, path.open('rb') as seed:
        reader = PrefixReader(BytesIO(b'ghijkl'), out, 10, 0, seed, 6)
        assert reader.read(4) == b'abcd'
        assert reader.read(4) == b'efgh'
        assert reader.read(4) == b'ij'
        assert reader.read(4) == b''
    assert path.read_bytes() == b'abcdefghij'
    assert reader.count == 10


def test_sicap_raw_jpeg_values_never_receive_an_invented_gleason_mapping():
    assert label_name('sicapv2', 200, {}) == 'raw JPEG intensity 200 (Gleason mapping unverified)'


@pytest.mark.parametrize('mask', [np.zeros((20, 30), dtype=np.uint16),
                                 np.arange(600, dtype=np.uint16).reshape(20, 30)])
def test_display_grid_encoding_preserves_every_label_for_both_storage_formats(mask):
    import base64
    from benchmark.describe import encode_grid
    encoding, encoded = encode_grid(mask)
    data = base64.b64decode(encoded)
    if encoding == 'rle-u16-u32':
        runs = np.frombuffer(data, dtype=[('value', '<u2'), ('count', '<u4')])
        decoded = np.repeat(runs['value'], runs['count'])
    else:
        decoded = np.frombuffer(data, dtype='<u2')
    np.testing.assert_array_equal(decoded.reshape(mask.shape), mask)
