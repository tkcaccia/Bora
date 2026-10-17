"""Only retained inference mechanisms; no rejected-solver dependencies."""
import numpy as np
from . import fields
from .registry import STRATEGIES,CONSENSUS

def run(key,c):
    if key not in STRATEGIES:raise ValueError('Rejected or unknown strategy: '+key)
    if key=='B02':out=c.fix(c.u.argmin(-1))
    elif key=='B03':out=fields.multiscale(c)
    elif key=='B08':out=fields.crf(c)
    elif key=='B19':
        votes=sum(np.eye(c.k,dtype=np.float32)[c.outputs[k]] for k in CONSENSUS)
        tied=votes==votes.max(-1,keepdims=True);out=c.fix(np.where(tied,c.u,np.inf).argmin(-1))
        c.diagnostics[key]=dict(components=list(CONSENSUS),revision=3)
    c.finish(out);c.outputs[key]=out.astype(np.int32)
    return c.outputs[key]
