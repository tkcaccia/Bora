"""Bounded, physically calibrated access to OME-TIFF whole-slide images.

The source is never materialized as one NumPy array. Windows are specified in
native pixel coordinates and resampled from their exact physical extent to the
Virchow2 working resolution. Native offsets remain explicit for later mask and
GeoJSON stitching.
"""
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import tifffile


def low_stain_background(native_rgb):
    """Flag bright, nearly neutral glass/scanner pixels after Bora refinement.

    This is a final-mask exclusion, not a Virchow2 input filter: the encoder
    still sees all tissue and the decision never reads expert annotations.
    Dark neutral nuclei and calcifications are retained by the brightness
    guard. The thresholds are explicit for audit on a new stain protocol.
    """
    rgb=np.asarray(native_rgb)
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[-1]!=3:
        raise ValueError("Expected native uint8 RGB pixels")
    saturation=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)[:,:,1]
    return (saturation<=10)&(np.min(rgb,axis=2)>210)


@dataclass(frozen=True)
class NativeWindow:
    x: int
    y: int
    width: int
    height: int
    core_x: int
    core_y: int
    core_width: int
    core_height: int

    @property
    def core_slices(self):
        return (slice(self.core_y-self.y,self.core_y-self.y+self.core_height),
                slice(self.core_x-self.x,self.core_x-self.x+self.core_width))


def ome_mpp(path):
    """Read measured µm/pixel, not TIFF's often-irrelevant display DPI."""
    with tifffile.TiffFile(path) as slide:
        page = slide.pages[0]
        if len(page.shape) != 3 or page.shape[-1] != 3 or page.dtype != np.uint8:
            raise ValueError("Expected a uint8 RGB whole-slide image")
        description = page.description
        if not description or "OME" not in description:
            raise ValueError("OME physical pixel sizes are required")
        root = ET.fromstring(description)
        pixels = next((node for node in root.iter() if node.tag.rsplit("}",1)[-1] == "Pixels"), None)
        if pixels is None:
            raise ValueError("OME Pixels metadata are required")
        if (int(pixels.attrib.get("SizeX",-1)) != page.shape[1] or
                int(pixels.attrib.get("SizeY",-1)) != page.shape[0]):
            raise ValueError("OME dimensions disagree with the first TIFF page")
        values=[]
        for axis in ("X","Y"):
            if pixels.attrib.get(f"PhysicalSize{axis}Unit") not in ("µm","um"):
                raise ValueError("OME physical sizes must be in micrometres")
            values.append(float(pixels.attrib[f"PhysicalSize{axis}"]))
        mpp=np.asarray(values,dtype=float)
        if not np.isfinite(mpp).all() or np.any(mpp<=0):
            raise ValueError("Positive finite OME physical pixel sizes required")
        return (page.shape[1],page.shape[0]),mpp


class WholeSlide:
    def __init__(self,path,target_mpp=0.5):
        self.path=Path(path)
        self.shape_xy,self.native_mpp_xy=ome_mpp(self.path)
        self.target_mpp=float(target_mpp)
        if not np.isfinite(self.target_mpp) or self.target_mpp<=0:
            raise ValueError("Positive target micrometres per pixel required")
        try:
            import pyvips
        except ImportError as exc:
            raise ImportError("Whole-slide access requires optional pyvips and libvips") from exc
        self.image=pyvips.Image.new_from_file(str(self.path),access="random")
        if (self.image.width,self.image.height) != self.shape_xy or self.image.bands != 3:
            raise ValueError("libvips and OME image geometry disagree")

    def windows(self,core_pixels=4096,halo_pixels=512):
        """Visit each native pixel once in a non-overlapping core with a halo."""
        if core_pixels<1 or halo_pixels<0:
            raise ValueError("Positive core and nonnegative halo required")
        width,height=self.shape_xy
        for y in range(0,height,core_pixels):
            for x in range(0,width,core_pixels):
                cw,ch=min(core_pixels,width-x),min(core_pixels,height-y)
                x0,y0=max(0,x-halo_pixels),max(0,y-halo_pixels)
                x1,y1=min(width,x+cw+halo_pixels),min(height,y+ch+halo_pixels)
                yield NativeWindow(x0,y0,x1-x0,y1-y0,x,y,cw,ch)

    def read(self,window):
        """Return native RGB, calibrated working RGB, and exact working MPP."""
        if (window.x<0 or window.y<0 or window.width<1 or window.height<1 or
                window.x+window.width>self.shape_xy[0] or
                window.y+window.height>self.shape_xy[1]):
            raise ValueError("Window lies outside source")
        region=self.image.crop(window.x,window.y,window.width,window.height)
        native=np.ndarray((window.height,window.width,3),dtype=np.uint8,
                          buffer=region.write_to_memory()).copy()
        shape_xy=np.maximum(1,np.rint(np.asarray([window.width,window.height])*
                                       self.native_mpp_xy/self.target_mpp).astype(int))
        working=cv2.resize(native,tuple(shape_xy),interpolation=cv2.INTER_AREA)
        working_mpp=self.native_mpp_xy*np.asarray([window.width,window.height])/shape_xy
        return native,working,working_mpp
