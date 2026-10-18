"""Cycle12: one explicit mechanism per child, no fine-reference access."""
import cv2
import numpy as np
from .cycle_eleven import context
from .cycle_six import edge_appearance
from .twelve_manifest import CHILDREN,NATIVE_CHILDREN
from .twelve_support import infer_support
from .twelve_appearance import (residual_likelihood,uncertain_risk,sandwich_uncertainty,
                               advected_appearance,skew_appearance)
from .twelve_regions import fuzzy_regions,tile_plurality,censored_palette,bayesian_regions

SPECS={r['id']:r for r in CHILDREN}
IMPLEMENTATIONS=dict(Q04=lambda c,p,h:residual_likelihood(c,p,h,'scale'),
    Q05=lambda c,p,h:residual_likelihood(c,p,h,'asymmetric'),
    Q06=lambda c,p,h:residual_likelihood(c,p,h,'conditional'),
    Q07=lambda c,p,h:uncertain_risk(c,p,h,'credible'),
    Q08=lambda c,p,h:uncertain_risk(c,p,h,'ball'),Q09=sandwich_uncertainty,
    Q10=advected_appearance,Q11=skew_appearance,Q12=fuzzy_regions,
    Q13=tile_plurality,Q14=censored_palette,Q15=bayesian_regions)


def report(key,c,details):
    item=SPECS[key]
    return dict(details,analysis_mpp_xy=c.mpp.tolist(),analysis_shape_yx=list(c.shape),
        fine_reference_used=False,hard_anchors=False,fixed_margin=False,
        candidate=key,parent=item['parent'],mechanism=item['intervention'])


def predict(key,rgb,coarse,support,mpp_xy,palette,he=None):
    if key not in IMPLEMENTATIONS:raise ValueError('Unknown or native-only cycle12 child')
    c=context(rgb,coarse,support,mpp_xy,he);q,details=IMPLEMENTATIONS[key](c,palette,he)
    if q.shape!=(*c.shape,c.k) or not np.isfinite(q).all() or q.min()<-1e-6 or not np.allclose(q.sum(-1),1,atol=1e-5):
        raise ValueError('Invalid normalized cycle12 class evidence')
    return c.finish(q),report(key,c,details)


def predict_native(key,rgb,coarse,support,mpp_xy,palette,he,native_he,native_spacing):
    if key not in NATIVE_CHILDREN:raise ValueError('Not a cycle12 native-support child')
    if not np.allclose(np.array(rgb.shape[:2][::-1])*np.asarray(mpp_xy),
                       np.array(native_he.shape[:2][::-1])*np.asarray(native_spacing),rtol=1e-8,atol=1e-6):
        raise ValueError('Native and working physical fields differ')
    c=context(rgb,coarse,support,mpp_xy,he);q,details=edge_appearance(c,palette,he)
    labels=cv2.resize(c.finish(q),native_he.shape[:2][::-1],interpolation=cv2.INTER_NEAREST)
    own,geometry=infer_support(key,native_he,native_spacing);labels[~own]=0
    details['native_support']=geometry
    return labels,own,report(key,c,details)
