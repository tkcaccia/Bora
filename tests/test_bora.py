import numpy as np
import tifffile
from bora.backends import WatershedBackend
from bora.geojson import labels_to_geojson
from bora.io import label_dtype, read_image, read_mask, write_ome_mask
from bora.refine import RefineConfig, refine_labels


def synthetic():
    y, x = np.mgrid[:128,:160]
    image = np.full((128,160,3), 240, np.uint8)
    image[(x-48)**2+(y-64)**2 < 31**2] = (130,45,100)
    image[(x-112)**2+(y-64)**2 < 27**2] = (70,110,155)
    mask = np.zeros((128,160), np.uint32); mask[32:96,18:78]=1; mask[38:91,86:140]=70000
    return image, mask


def test_refine_and_geojson():
    image, mask = synthetic()
    result, report = refine_labels(image, mask, WatershedBackend(), RefineConfig(8,12,96,16,10,1))
    assert result.dtype == np.uint32 and set(np.unique(result)) == {0,1,70000}
    assert report["input_label_count"] == 2 and label_dtype(int(result.max())) == np.dtype("uint32")
    geo = labels_to_geojson(result, .5)
    assert {f["properties"]["label"] for f in geo["features"]} == {1,70000}


def test_ome_roundtrip(tmp_path):
    image, mask = synthetic(); ip=tmp_path/"i.ome.tif"; mp=tmp_path/"m.tif"; op=tmp_path/"o.ome.tif"
    tifffile.imwrite(ip, image, ome=True, metadata={"axes":"YXS"}); tifffile.imwrite(mp, mask)
    assert read_image(ip).shape == image.shape and np.array_equal(read_mask(mp), mask)
    write_ome_mask(op, mask); assert np.array_equal(read_mask(op), mask)
