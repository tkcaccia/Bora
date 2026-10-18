"""Validated compiled primitives for distinct cycle-ten graph models."""
import ctypes
import numpy as np
from .native import library


def tree_potts(parent, order, unary, penalty):
    parent,order=[np.ascontiguousarray(x,np.int32) for x in (parent,order)]
    unary,penalty=[np.ascontiguousarray(x,np.float64) for x in (unary,penalty)]
    n,k=unary.shape
    if (parent.shape!=(n,) or order.shape!=(n,) or penalty.shape!=(n,)
            or sorted(order.tolist())!=list(range(n)) or np.any(penalty<0)
            or not np.isfinite(unary).all() or not np.isfinite(penalty).all()):
        raise ValueError('Invalid finite rooted-tree model')
    rank=np.empty(n,int);rank[order]=np.arange(n)
    if np.any(parent[order[1:]]<0) or np.any(parent[order[1:]]>=n) or np.any(rank[parent[order[1:]]]>=rank[order[1:]]):
        raise ValueError('Parent must precede its child')
    labels=np.empty(n,np.int32);energy=np.empty(2,np.float64)
    f=library().bora_tree_potts
    f.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*2
    status=f(parent.ctypes.data,order.ctypes.data,unary.ctypes.data,penalty.ctypes.data,
             labels.ctypes.data,energy.ctypes.data,n,k)
    if status or abs(energy[0]-energy[1])>1e-7*max(1,abs(energy[0])):
        raise RuntimeError('Tree DP energy check failed')
    return labels,dict(dynamic_program_energy=float(energy[0]),recomputed_energy=float(energy[1]),
                       tree_nodes=n,tree_edges=n-1,exact_discrete_tree_optimum=True)


def seeded_forest(a,b,seeds):
    a,b,seeds=[np.ascontiguousarray(x,np.int32) for x in (a,b,seeds)]
    n=len(seeds)
    if a.shape!=b.shape or np.any(a<0) or np.any(a>=n) or np.any(b<0) or np.any(b>=n) or np.any(seeds<0):
        raise ValueError('Invalid seeded forest')
    out=np.empty(n,np.int32);f=library().bora_seeded_forest
    f.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*2
    if f(a.ctypes.data,b.ctypes.data,seeds.ctypes.data,out.ctypes.data,n,len(a)):
        raise RuntimeError('Seeded forest failed')
    if not np.array_equal(out[seeds>0],seeds[seeds>0]):
        raise RuntimeError('Forest changed a fixed marker')
    return out


def biconnected_white(white):
    white=np.ascontiguousarray(white,np.uint8)
    if white.ndim!=2 or min(white.shape)<1 or white.max()>1:
        raise ValueError('Binary white grid required')
    out=np.zeros_like(white);f=library().bora_biconnected_white
    f.argtypes=[ctypes.c_void_p]*2+[ctypes.c_int]*2
    if f(white.ctypes.data,out.ctypes.data,*white.shape):
        raise RuntimeError('Biconnected decomposition failed')
    if np.any(out>white):raise RuntimeError('Non-white exterior assignment')
    return out.astype(bool)
