"""C++ graph primitives; ports are not separately counted as innovations."""
import ctypes
import numpy as np
from .native import library


def reconstruct(ceiling,marker):
    ceiling,marker=[np.ascontiguousarray(a,np.float64) for a in (ceiling,marker)]
    if ceiling.ndim!=3 or marker.shape!=ceiling.shape or not np.isfinite(ceiling).all() or not np.isfinite(marker).all() or np.any(marker<0) or np.any(marker>ceiling+1e-12):
        raise ValueError('Finite nonnegative marker below gray ceiling required')
    result=np.empty_like(ceiling);h,w,k=ceiling.shape
    f=library().bora_gray_reconstruction;f.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*4;f.restype=ctypes.c_int
    if f(ceiling.ctypes.data,marker.ctypes.data,result.ctypes.data,h,w,k,2):raise RuntimeError('Gray reconstruction failed')
    return result


def bayes_regions(log_probability,support):
    values=np.ascontiguousarray(log_probability,np.float64);support=np.ascontiguousarray(support,np.uint8)
    if values.ndim!=3 or support.shape!=values.shape[:2] or min(values.shape)<1 or values.shape[-1]<2 or not np.isfinite(values).all():
        raise ValueError('Aligned finite class log probability grid required')
    h,w,k=values.shape;result=np.empty((h,w),np.int32);details=np.empty(3,np.float64)
    f=library().bora_bayes_regions;f.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*3+[ctypes.c_void_p];f.restype=ctypes.c_int
    if f(values.ctypes.data,support.ctypes.data,result.ctypes.data,h,w,k,details.ctypes.data):raise RuntimeError('Bayes region operation failed')
    return result,dict(greedy_merges=int(details[0]),regions=int(details[1]),positive_log_bayes_gain=float(details[2]))
