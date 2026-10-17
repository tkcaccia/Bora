import numpy as np
from PIL import Image
import tifffile
from experiments.bora_cycles.run import publish


def test_original_image_and_original_coarse_mask_use_distinct_preview_assets(tmp_path):
    source, root = tmp_path/"input", tmp_path/"result"
    src, out = source/"cases/example", root/"cases/example"
    (src/"ensemble").mkdir(parents=True)
    (out/"C01").mkdir(parents=True)
    rgb = np.full((48, 64, 3), [190, 120, 160], np.uint8)
    truth = np.ones((48, 64), np.uint8)
    coarse = np.ones((6, 8), np.uint8)
    palette = [[200, 20, 20]]
    Image.fromarray(rgb).save(src/"original.native.png")
    Image.fromarray(np.full((6, 8, 3), palette[0], np.uint8)).save(src/"ensemble/representation.png")
    tifffile.imwrite(src/"reference.native.tif", truth)
    tifffile.imwrite(src/"coarse.tif", coarse)
    tifffile.imwrite(out/"C01/native.tif", truth)
    row = dict(case="example", dataset="synthetic", sample=dict(palette=palette),
               methods=dict(C01=dict(status="ok", score=dict(macro_iou=1), median_seconds=.1)))
    result = dict(protocol=dict(methods=["C01"], pilot=True), cases=[row],
                  ranking=[dict(method="C01", mean_iou=1, mean_seconds=.1, small_region_recall=1)])
    publish(source, root, result)
    he = np.array(Image.open(out/"original-he.jpg"))
    tiles = np.array(Image.open(out/"coarse.jpg"))
    np.testing.assert_allclose(he[20, 20], rgb[20, 20], atol=3)
    np.testing.assert_allclose(tiles[20, 20], palette[0], atol=3)
    html = (root/"gallery.html").read_text()
    assert 'cases/example/original-he.jpg' in html
    assert 'cases/example/coarse.jpg' in html
    assert 'cases/example/original.jpg' not in html
