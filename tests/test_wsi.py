import numpy as np
import pytest
import tifffile

from bora.wsi import WholeSlide, ome_mpp, low_stain_background
from bora.conversion import convert_rgb


def make_slide(path):
    y,x=np.indices((73,101))
    image=np.stack(((x*2)%256,(y*3)%256,(x+y)%256),axis=-1).astype(np.uint8)
    tifffile.imwrite(path,image,ome=True,photometric="rgb",
        metadata={"axes":"YXC","PhysicalSizeX":0.27,"PhysicalSizeY":0.31,
                  "PhysicalSizeXUnit":"µm","PhysicalSizeYUnit":"µm"})
    return image


def test_ome_metadata_and_window_coverage(tmp_path):
    path=tmp_path/"slide.ome.tif"
    make_slide(path)
    shape,mpp=ome_mpp(path)
    assert shape==(101,73)
    np.testing.assert_allclose(mpp,[.27,.31])
    pytest.importorskip("pyvips")
    slide=WholeSlide(path)
    seen=np.zeros((73,101),np.uint8)
    for window in slide.windows(core_pixels=32,halo_pixels=9):
        native,working,working_mpp=slide.read(window)
        assert native.shape==(window.height,window.width,3)
        assert working.shape[-1]==3
        np.testing.assert_allclose(np.asarray(working.shape[1::-1])*working_mpp,
                                   np.asarray(native.shape[1::-1])*mpp)
        np.testing.assert_array_equal(native,
            tifffile.imread(path)[window.y:window.y+window.height,
                                  window.x:window.x+window.width])
        seen[window.core_y:window.core_y+window.core_height,
             window.core_x:window.core_x+window.core_width]+=1
    assert np.all(seen==1)


def test_missing_calibration_rejected(tmp_path):
    path=tmp_path/"uncalibrated.tif"
    tifffile.imwrite(path,np.zeros((40,40,3),np.uint8),photometric="rgb")
    with pytest.raises(ValueError,match="OME"):
        ome_mpp(path)


def test_underpowered_wsi_window_has_explicit_fallback():
    coarse=np.ones((224,224),np.uint8)
    tokens=np.zeros((16,16,1280),np.float16)
    palette=np.array([[20,70,160]],np.uint8)
    with pytest.raises(ValueError,match="16 eligible"):
        convert_rgb(tokens,coarse,palette,{})
    rgb,details=convert_rgb(tokens,coarse,palette,{},allow_local_fallback=True)
    assert rgb.shape==(224,224,3)
    assert np.all(rgb==palette[0])
    assert details["full_RGB9_plus_19_applied"] is False


def test_low_stain_background_excludes_bright_gray_but_keeps_stained_tissue():
    pixels=np.asarray([[[234,233,232],[235,205,220],[100,100,100],[250,250,250]]],np.uint8)
    np.testing.assert_array_equal(low_stain_background(pixels),[[True,False,False,True]])
