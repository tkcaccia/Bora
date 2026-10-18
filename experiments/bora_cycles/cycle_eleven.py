"""Cycle11: fifteen prospectively specified mathematical interventions."""
import cv2
import numpy as np

from bora.evidence import Evidence
from .cycle_six import edge_appearance
from .eleven_appearance import (compositional_palette,fractional_field,negative_tail,
    ordered_field,spatial_palette,stain_regions,uncertain_decision,variance_field)
from .eleven_manifest import CHILDREN,SUPPORT_CONTROL,SUPPORT_METHODS
from .eleven_support import infer_support

SPECS={r['id']:r for r in CHILDREN}
IMPLEMENTATIONS=dict(P01=ordered_field,P02=fractional_field,P03=stain_regions,
    P04=lambda c,p,h:uncertain_decision(c,p,h,'winner'),
    P05=lambda c,p,h:uncertain_decision(c,p,h,'absolute'),
    P06=lambda c,p,h:uncertain_decision(c,p,h,'affinity'),
    P07=lambda c,p,h:variance_field(c,p,h,'matrix'),
    P08=lambda c,p,h:variance_field(c,p,h,'hyvarinen'),
    P09=lambda c,p,h:variance_field(c,p,h,'shared'),
    P13=spatial_palette,P14=compositional_palette,P15=negative_tail)


def context(rgb,coarse,support,spacing,he):
    if he is None or he.shape!=rgb.shape:raise ValueError('Aligned original H&E required')
    c=Evidence(rgb,coarse,support,spacing)
    if c.k<2 or not c.support.any():raise ValueError('At least two tissue groups and nonempty support required')
    return c


def report(key,c,details):
    item=SPECS.get(key)
    return dict(details,analysis_mpp_xy=c.mpp.tolist(),analysis_shape_yx=list(c.shape),
        fine_reference_used=False,hard_anchors=False,fixed_margin=False,candidate=key,
        parent=item['parent'] if item else 'J01',
        mechanism=item['intervention'] if item else SUPPORT_CONTROL['mechanism'])


def predict(key,rgb,coarse,support,mpp_xy,palette,he=None):
    if key in SUPPORT_METHODS:raise ValueError('Native support hypotheses require predict_native')
    if key not in IMPLEMENTATIONS:raise ValueError('Unimplemented cycle11 child')
    c=context(rgb,coarse,support,mpp_xy,he)
    q,details=IMPLEMENTATIONS[key](c,palette,he)
    if q.shape!=(*c.shape,c.k) or not np.isfinite(q).all() or q.min()<-1e-6 or not np.allclose(q.sum(-1),1,atol=1e-5):
        raise ValueError('Invalid normalized cycle11 class evidence')
    return c.finish(q),report(key,c,details)


def predict_native(key,rgb,coarse,support,mpp_xy,palette,he,native_he,native_spacing):
    if key not in SUPPORT_METHODS:raise ValueError('Not a native-support hypothesis')
    a=np.array(rgb.shape[:2][::-1])*np.asarray(mpp_xy)
    b=np.array(native_he.shape[:2][::-1])*np.asarray(native_spacing)
    if not np.allclose(a,b,rtol=1e-8,atol=1e-6):raise ValueError('Native and working physical fields differ')
    c=context(rgb,coarse,support,mpp_xy,he)
    # Exactly the parent's J01 class fields; this intervention changes only
    # final native support. No inheritance of the common final clipping mask.
    q,details=edge_appearance(c,palette,he)
    labels=cv2.resize(c.finish(q),native_he.shape[:2][::-1],interpolation=cv2.INTER_NEAREST)
    own_support,geometry=infer_support(key,native_he,native_spacing)
    labels[~own_support]=0
    details.update(native_support=geometry,support_control=key=='B23')
    return labels,own_support,report(key,c,details)
