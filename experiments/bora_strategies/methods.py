"""Dispatch genuinely different inference mechanisms, with no fine-label access."""
import numpy as np
from . import fields,graphs,geometry,student
from .registry import STRATEGIES


def run(key,c):
    if key=='B01':return c.outputs[key]
    if key=='B02':out=c.fix(c.u.argmin(-1))
    elif key=='B03':out=fields.multiscale(c)
    elif key=='B04':out=fields.region_snap(c)
    elif key=='B05':out=fields.flooding(c)
    elif key=='B06':out=graphs.harmonic(c)
    elif key=='B07':out=graphs.graph_cut(c)
    elif key=='B08':out=fields.crf(c)
    elif key=='B09':out=fields.geodesic(c)
    elif key=='B10':out=fields.tv(c)
    elif key=='B11':out=fields.contour(c)
    elif key=='B12':out=graphs.merge(c)
    elif key=='B13':out=graphs.spectral(c)
    elif key=='B14':out=geometry.topology(c)
    elif key=='B15':out=fields.object_context(c)
    elif key=='B16':out=fields.transport(c)
    elif key=='B17':out=fields.nonlocal_vote(c)
    elif key=='B18':out=geometry.elastic(c)
    elif key=='B19':
        votes=sum(np.eye(c.k,dtype=np.float32)[c.outputs[k]] for k in ('B01','B02','B06','B10'))
        maximum=votes.max(-1,keepdims=True);tied=votes==maximum;cost=np.where(tied,c.u,np.inf);out=c.fix(cost.argmin(-1))
    elif key=='B20':out=student.student(c)
    else:raise ValueError(key)
    c.finish(out);c.outputs[key]=out.astype(np.int32)
    return c.outputs[key]
