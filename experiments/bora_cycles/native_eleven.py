"""Compiled adaptive-region primitive; not a separate innovation."""
import ctypes
import numpy as np
from .native import library


def contrast_components(a,b,cost,n,area,scale):
    a,b=[np.ascontiguousarray(x,np.int32) for x in (a,b)]
    cost=np.ascontiguousarray(cost,np.float64)
    if (a.ndim!=1 or a.shape!=b.shape or a.shape!=cost.shape
            or not isinstance(n,(int,np.integer)) or n<1
            or not np.isfinite([area,scale]).all() or area<=0 or scale<=0
            or not np.isfinite(cost).all() or np.any(cost<0)
            or np.any(np.diff(cost)<0) or np.any(a<0) or np.any(a>=n)
            or np.any(b<0) or np.any(b>=n)):
        raise ValueError('Sorted finite region graph required')
    result=np.empty(n,np.int32);f=library().bora_contrast_components
    f.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*2+[ctypes.c_double]*2
    f.restype=ctypes.c_int
    if f(a.ctypes.data,b.ctypes.data,cost.ctypes.data,result.ctypes.data,n,len(a),area,scale):
        raise RuntimeError('Adaptive region partition failed')
    if result.min()<0 or len(np.unique(result))!=result.max()+1:
        raise RuntimeError('Region IDs are not a complete compact partition')
    return result
