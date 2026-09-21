import numpy as np


def rgb_to_lab(rgb):
    """Convert uint8 sRGB to CIE Lab (D65), matching CellPhenotyper."""
    x = np.asarray(rgb, dtype=np.float32) / 255.0
    x = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    xyz = x @ np.array([[.4124564,.3575761,.1804375],[.2126729,.7151522,.0721750],
                        [.0193339,.1191920,.9503041]], dtype=np.float32).T
    xyz /= np.array([.95047,1.,1.08883], dtype=np.float32)
    delta = 6/29
    f = np.where(xyz > delta**3, np.cbrt(xyz), xyz/(3*delta**2)+4/29)
    return np.stack((116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])),axis=-1)
