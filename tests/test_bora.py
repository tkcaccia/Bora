import numpy as np
import tifffile
from bora.backends import WatershedBackend
from bora.geojson import labels_to_geojson
from bora.io import label_dtype, read_image, read_mask, write_ome_mask
from bora.refine import RefineConfig, refine_labels
from bora.streaming import refine_streaming, write_geojson_streaming
from bora.annealed_wand import annealed_wand_boundary_competition
from bora.wand import run_annealed_wand
from bora.cellphenotyper import mask_to_geojson
from bora.multicpu_geojson import convert


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


def test_streaming_roundtrip(tmp_path):
    image, mask = synthetic(); ip=tmp_path/"i.tif"; mp=tmp_path/"m.tif"; op=tmp_path/"o.tif"; gp=tmp_path/"o.geojson"
    tifffile.imwrite(ip,image,tile=(32,32),compression="deflate"); tifffile.imwrite(mp,mask)
    report=refine_streaming(ip,mp,op,WatershedBackend(),RefineConfig(4,6,64,8,10,1),64)
    assert report["blocks"] == 6 and read_mask(op).shape == mask.shape
    assert write_geojson_streaming(gp,op,64,.5,10) > 0


def test_annealed_wand_preserves_footprint_and_energy():
    image, _ = synthetic()
    labels = np.zeros(image.shape[:2], np.uint16)
    labels[24:104,20:80] = 1
    labels[24:104,80:140] = 2
    result, metadata = annealed_wand_boundary_competition(
        image, labels, labels > 0, boundary_radius=4, iterations=3)
    assert np.array_equal(result > 0, labels > 0)
    assert metadata["accepted_nonincreasing_energy"]
    assert metadata["schema_version"] == "cellphenotyper.annealed_wand_boundary.v2"


def test_cellphenotyper_wand_wrapper_preserves_full_resolution_footprint():
    image, _ = synthetic()
    labels = np.zeros(image.shape[:2], np.uint16)
    labels[24:104, 20:80] = 1
    labels[24:104, 80:140] = 2
    result, metadata = run_annealed_wand(
        image, labels, downsample=4, boundary_radius=16, iterations=2)
    assert result.shape == labels.shape
    assert np.array_equal(result > 0, labels > 0)
    assert metadata["working_downsample"] == 4
    assert metadata["maximum_boundary_displacement_px"] == 19


def test_cellphenotyper_geojson_converter(tmp_path):
    _, mask = synthetic()
    mp, gp = tmp_path / "mask.tif", tmp_path / "mask.geojson"
    tifffile.imwrite(mp, mask)
    count = mask_to_geojson(
        mp, gp, max_page_side=2048, min_area=10, smooth_buffer=0,
        smooth_passes=1, simplify=0, fill_holes=False)
    geo = __import__("json").loads(gp.read_text())
    assert count == 2
    assert {feature["properties"]["value"] for feature in geo["features"]} == {1, 70000}


def test_multicpu_geojson_converter(tmp_path):
    _, mask = synthetic()
    mp, gp = tmp_path / "mask.tif", tmp_path / "mask.geojson"
    tifffile.imwrite(mp, mask)
    report = convert(mp, gp, page=0, min_area=10, simplify=2, workers=2)
    geo = __import__("json").loads(gp.read_text())
    assert report["features"] == 2
    assert report["workers_requested"] == 2
    assert report["union_workers"] == 2
    assert {feature["properties"]["value"] for feature in geo["features"]} == {1, 70000}
