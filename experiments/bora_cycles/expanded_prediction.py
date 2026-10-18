"""Annotation-free degeneracy contract shared by all expanded candidates.

Exactly constant converted RGB contains no class contrast. Numerical variation
in linear solves must not be interpreted as evidence of a new tissue interface.
This is an explicit input fallback, not another candidate innovation.
"""
import numpy as np


def constant_evidence_fallback(key, rgb, coarse, completed, support, spacing, he):
    if rgb.shape != (*coarse.shape, 3) or not np.all(rgb == rgb[0, 0]):
        raise ValueError('Single-eligible-class fallback requires exactly constant RGB')
    if (completed.shape != coarse.shape or support.shape != coarse.shape
            or np.any(completed == 0) or not np.any(support)):
        raise ValueError('Frozen complete coarse assignment and nonempty support required')
    if (not set(np.unique(completed)) <= set(np.unique(coarse))-{0}
            or not np.array_equal(completed[coarse > 0], coarse[coarse > 0])):
        raise ValueError('Fallback cannot invent a class or change an observed coarse label')
    labels = completed.copy()
    detail = dict(candidate=key, input_contract_fallback='constant-evidence-coarse-completion-v1',
        refinement_performed=False, fine_reference_used=False,
        analysis_shape_yx=list(coarse.shape), analysis_mpp_xy=list(spacing),
        limitation='Converted RGB supplies no class contrast; original coarse labels are retained and original unknowns use the frozen annotation-free completion.')
    if key == 'L03':
        from .cycle_eight import wide_path_glass
        excluded, glass = wide_path_glass(he, spacing)
        labels[excluded] = 0
        detail.update(glass, mechanism='white-space-exterior-bottleneck-connectivity',
            proposed_glass_pixels=int((excluded & support).sum()),
            independent_he_glass_rule_applied=True)
    if key in ('M10', 'M11', 'M12'):
        from .cycle_nine import GLASS_MODES, glass_exclusion
        from .registry import CYCLE_NINE
        excluded, glass = glass_exclusion(he, spacing, GLASS_MODES[key])
        labels[excluded] = 0
        detail.update(glass, mechanism=CYCLE_NINE[key].mechanism,
            independent_he_glass_rule_applied=True,
            proposed_glass_pixels=int((excluded & support).sum()))
    if key in ('N10', 'N11', 'N12'):
        from .cycle_ten import GLASS_MODES, topological_glass
        from .registry import CYCLE_TEN
        excluded, glass = topological_glass(he, spacing, GLASS_MODES[key])
        labels[excluded] = 0
        detail.update(glass, mechanism=CYCLE_TEN[key].mechanism,
            independent_he_glass_rule_applied=True,
            proposed_glass_pixels=int((excluded & support).sum()))
    return labels, detail
