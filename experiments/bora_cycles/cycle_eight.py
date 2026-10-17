"""Fifteen cycle-eight models; fixed input colours and no reference-mask I/O."""
import cv2
import numpy as np
from scipy import ndimage as ndi, sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import splu, spsolve
from scipy.spatial import cKDTree
from scipy.special import expit, logsumexp, ndtr, softmax

from bora.evidence import Evidence, project_simplex
from . import native
from .methods import correct_coarse, fit_linear, linear_prob
from .diagnostics import white_regions
from .cycle_two import colour_nodes
from .cycle_three import balanced
from .cycle_five import field_laplacian, optimizer_report
from .cycle_six import edge_appearance, field_geometry, guide_laplacian
from .cycle_seven import (bag_design, empirical_exponential_objective, gaussian_design,
                          linear_result)


def nonlocal_laplacian(guide, spacing):
    h, w = guide.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    xy = np.c_[xx.ravel()*spacing[0], yy.ravel()*spacing[1]]
    features = np.c_[guide.reshape(-1, 3)/.1, xy/128.]
    distance, neighbours = cKDTree(features).query(features, k=min(7, len(features)))
    i = np.repeat(np.arange(len(features)), neighbours.shape[1]-1)
    j = neighbours[:, 1:].ravel()
    ds = np.sum((xy[i]-xy[j])**2, -1)
    weight = np.exp(-.5*distance[:, 1:].ravel()**2)*256/(ds+64.)
    graph = sparse.coo_matrix((weight, (i, j)), shape=(len(xy), len(xy))).tocsr()
    graph = .5*(graph+graph.T)
    return sparse.diags(np.asarray(graph.sum(1)).ravel())-graph


def field_inputs(c, he, mode):
    _, _, means, covariance = fit_linear(c)
    shape, spacing = field_geometry(c)
    if mode in ('nonlocal', 'uncertain'):
        if he is None:
            raise ValueError('Aligned original H&E required')
        original = cv2.resize(he, shape[::-1], interpolation=cv2.INTER_AREA)
        guide = -np.log((original.astype(float)+1)/256.)
        lap = (nonlocal_laplacian(guide, spacing) if mode == 'nonlocal'
               else guide_laplacian(guide, spacing))
    else:
        base = field_laplacian(shape, spacing)
        lap = base.T@base
    return means, covariance, shape, spacing, lap


def resize_field(values, shape, interpolation=cv2.INTER_AREA):
    # Some OpenCV area kernels accept at most four interleaved channels.
    if values.ndim == 3 and values.shape[-1] > 4:
        return np.stack([cv2.resize(values[..., j], shape[::-1], interpolation=interpolation)
                         for j in range(values.shape[-1])], -1)
    return cv2.resize(values, shape[::-1], interpolation=interpolation)


def class_statistics(c, g, shape, values=None):
    values = c.rgb if values is None else values
    weight = c.weight*(c.labels == g)
    mass = cv2.resize(weight, shape[::-1], interpolation=cv2.INTER_AREA).ravel()
    sums = resize_field(weight[..., None]*values, shape)
    return mass, sums.reshape(len(mass), -1)


def inverse_diagonal(matrix, probes=16):
    """Hutchinson estimate, not an exact marginal-variance certificate."""
    z = np.random.default_rng(812).choice([-1., 1.], size=(matrix.shape[0], probes))
    estimate = np.mean(z*splu(matrix.tocsc()).solve(z), axis=1)
    return np.maximum(estimate, 0), dict(diagonal_probes=probes,
        diagonal_estimate_minimum=float(estimate.min()),
        negative_diagonal_estimates_clipped=int((estimate < 0).sum()))


def guided_field(c, palette, he, mode):
    means, covariance, shape, spacing, lap = field_inputs(c, he, mode)
    precision = np.linalg.inv(covariance)
    scores, reports = [], []
    for g in range(c.k):
        mass, sums = class_statistics(c, g, shape)
        rhs = sums+.015*means[g]
        matrix = lap+sparse.diags(mass+.015)
        fitted = spsolve(matrix.tocsc(), rhs)
        surface = cv2.resize(fitted.reshape(*shape, 3), c.shape[::-1], interpolation=cv2.INTER_LINEAR)
        delta = c.rgb-surface
        quadratic = np.einsum('...i,ij,...j->...', delta, precision, delta)
        report = dict(linear_system_max_residual=float(np.max(np.abs(matrix@fitted-rhs))))
        if mode == 'uncertain':
            diagonal, variance_report = inverse_diagonal(matrix)
            factor = 1+.015*cv2.resize(diagonal.reshape(shape), c.shape[::-1], interpolation=cv2.INTER_LINEAR)
            scores.append(-.5*quadratic/factor-1.5*np.log(factor))
            report.update(variance_report, maximum_predictive_variance_factor=float(factor.max()))
        else:
            scores.append(-.5*quadratic)
        reports.append(report)
    return softmax(np.stack(scores, -1), -1), dict(model=mode+' appearance fields',
        field_shape_yx=list(shape), field_mpp_xy=spacing.tolist(), fits=reports,
        caveat='Weak coarse observations do not establish the biological correctness of transported appearance.')


def wide_path_glass(he, spacing, radius=3.):
    white, _, detail = white_regions(he)
    clearance = ndi.distance_transform_edt(white, sampling=np.asarray(spacing)[::-1])
    core = white & (clearance >= radius)
    seeds = np.zeros_like(core)
    seeds[[0, -1]] = core[[0, -1]]
    seeds[:, [0, -1]] = core[:, [0, -1]]
    accessible = ndi.binary_propagation(seeds, structure=np.ones((3, 3)), mask=core)
    excluded = np.zeros_like(core)
    if accessible.any():
        excluded = white & (ndi.distance_transform_edt(~accessible,
            sampling=np.asarray(spacing)[::-1]) <= radius)
    return excluded, dict(white_policy=detail, route_minimum_radius_um=radius,
        exterior_seed_pixels=int(seeds.sum()), accessible_white_core_pixels=int(accessible.sum()))


def exterior_field(c, palette, he):
    q, detail = edge_appearance(c, palette, he)
    original = cv2.resize(he, c.shape[::-1], interpolation=cv2.INTER_AREA)
    excluded, report = wide_path_glass(original, c.mpp)
    detail.update(report, glass_exclusion=excluded,
        proposed_glass_pixels=int((excluded & c.support).sum()))
    return q, detail


def bounded_system(matrix, rhs):
    shape = rhs.shape
    start = np.clip(spsolve(matrix.tocsc(), rhs), 0, 1)
    def objective(flat):
        u = flat.reshape(shape)
        gradient = matrix@u-rhs
        return float(.5*np.sum(u*(matrix@u))-np.sum(rhs*u)), gradient.ravel()
    fit = minimize(objective, start.ravel(), jac=True, method='L-BFGS-B', bounds=[(0., 1.)]*start.size,
        options=dict(maxiter=120, ftol=1e-12, gtol=1e-6))
    _, gradient = objective(fit.x)
    residual = float(np.max(np.abs(fit.x-np.clip(fit.x-gradient, 0, 1))))
    return fit.x.reshape(shape), dict(**optimizer_report(fit, objective),
        projected_gradient_residual=residual, converged=residual < 1e-5)


def huber_radial(norm, radius=2.5):
    return np.where(norm <= radius, .5*norm**2, radius*norm-.5*radius**2)


def robust_system(lap, mass, target, mean, iterations=8):
    rhs = mass[:, None]*target+.015*mean
    fitted = spsolve((lap+sparse.diags(mass+.015)).tocsc(), rhs)
    def objective(u):
        return float(np.sum(mass*huber_radial(np.linalg.norm(u-target, axis=1)))
            +.5*np.sum(u*(lap@u))+.0075*np.sum((u-mean)**2))
    history = [objective(fitted)]
    for used in range(iterations):
        norms = np.linalg.norm(fitted-target, axis=1)
        weight = mass*np.minimum(1., 2.5/np.maximum(norms, 1e-12))
        fitted = spsolve((lap+sparse.diags(weight+.015)).tocsc(),
                        weight[:, None]*target+.015*mean)
        history.append(objective(fitted))
    residual = fitted-target
    gradient = lap@fitted+.015*(fitted-mean)+mass[:, None]*residual*np.minimum(
        1., 2.5/np.maximum(np.linalg.norm(residual, axis=1), 1e-12))[:, None]
    norm = float(np.max(np.abs(gradient)))
    return fitted, dict(iterations=used+1, objective_path=history,
        gradient_infinity_norm=norm, converged=norm < 1e-5,
        observation_level='coarse-weighted class means at field nodes; no pixel-label expansion')


def bending_field(c, palette, he, mode):
    means, covariance, shape, spacing, lap = field_inputs(c, he, mode)
    precision = np.linalg.inv(covariance)
    scores, reports = [], []
    transform = None
    if mode == 'robust':
        eigen, rotation = np.linalg.eigh(covariance)
        transform = (rotation/np.sqrt(eigen))@rotation.T
    for g in range(c.k):
        values = c.rgb if transform is None else c.rgb@transform
        mass, sums = class_statistics(c, g, shape, values)
        mean = means[g] if transform is None else means[g]@transform
        matrix = lap+sparse.diags(mass+.015)
        rhs = sums+.015*mean
        if mode == 'bounded':
            fitted, report = bounded_system(matrix, rhs)
        elif mode == 'robust':
            target = sums/np.maximum(mass[:, None], 1e-12)
            fitted, report = robust_system(lap, mass, target, mean)
        else:
            outer = (c.rgb[..., :, None]*c.rgb[..., None, :]).reshape(*c.shape, 9)
            _, moment_sums = class_statistics(c, g, shape, outer)
            prior_second = (np.outer(mean, mean)+covariance).reshape(9)
            combined_rhs = np.c_[rhs, moment_sums+.015*prior_second]
            solved = spsolve(matrix.tocsc(), combined_rhs)
            fitted, second = solved[:, :3], solved[:, 3:]
            report = dict(linear_system_max_residual=float(np.max(np.abs(matrix@solved-combined_rhs))))
        surface = cv2.resize(fitted.reshape(*shape, 3), c.shape[::-1], interpolation=cv2.INTER_LINEAR)
        delta = values-surface
        if mode == 'robust':
            score = -huber_radial(np.linalg.norm(delta, axis=-1))
        elif mode == 'covariance':
            moment = resize_field(second.reshape(*shape, 9), c.shape, interpolation=cv2.INTER_LINEAR)
            raw_cov = moment.reshape(*c.shape, 3, 3)-surface[..., :, None]*surface[..., None, :]
            local_cov = .8*raw_cov+.2*covariance
            eigen, rotation = np.linalg.eigh(.5*(local_cov+local_cov.swapaxes(-1, -2)))
            report['covariance_eigenvalues_clipped'] = int((eigen < .0025).sum())
            eigen = np.maximum(eigen, .0025)
            rotated = np.einsum('...i,...ij->...j', delta, rotation)
            score = -.5*np.sum(rotated**2/eigen+np.log(eigen), -1)
        else:
            score = -.5*np.einsum('...i,ij,...j->...', delta, precision, delta)
        scores.append(score)
        reports.append(report)
    return softmax(np.stack(scores, -1), -1), dict(model=mode+' bending appearance',
        field_shape_yx=list(shape), field_mpp_xy=spacing.tolist(), fits=reports,
        caveat='Finite weakly supervised moment/field estimates can create false class islands.')


def hellinger_objective(q, x, prior, colors):
    observed, prediction = x+1/255., q@colors+1/255.
    value = .5*np.sum((np.sqrt(prediction)-np.sqrt(observed))**2)+.006*np.sum((q-prior)**2)
    gradient = .5*(1-np.sqrt(observed/prediction))@colors.T+.012*(q-prior)
    return float(value), gradient


def hellinger_inverse(x, prior, colors, iterations=120):
    q = np.asarray(prior, float).copy()
    value, gradient = hellinger_objective(q, x, prior, colors)
    history, step = [value], 1.
    for used in range(iterations):
        for backtrack in range(40):
            changed = project_simplex(q-step*gradient)
            delta = changed-q
            new, new_gradient = hellinger_objective(changed, x, prior, colors)
            if new <= value+np.sum(gradient*delta)+np.sum(delta*delta)/(2*step)+1e-12:
                break
            step *= .5
        else:
            raise RuntimeError('Hellinger backtracking failed')
        q, value, gradient = changed, new, new_gradient
        history.append(value)
        residual = float(np.max(np.abs(q-project_simplex(q-gradient))))
        if residual < 1e-6:
            break
        step = min(step*1.2, 64.)
    return q, dict(iterations=used+1, objective_path=history,
        projected_gradient_residual=residual, converged=residual < 1e-6)


def palette_energy(q, x, prior, colors, mode):
    error2 = np.sum((q@colors-x)**2, -1)
    if mode == 'student':
        data = .00125*np.log1p(error2/.0025)
    else:
        data = .5*error2-.002*np.sum((q+1e-5)*np.log(q+1e-5), -1)
    return float(np.sum(data)+.006*np.sum((q-prior)**2))


def majorized_inverse(x, prior, colors, mode, iterations=12):
    q, initial = native.simplex_qp(x, prior, colors)
    history = [palette_energy(q, x, prior, colors, mode)]
    h = colors@colors.T
    maximum_kkt = initial['max_kkt_residual']
    for used in range(iterations):
        if mode == 'student':
            weight = .0025/(.0025+np.sum((q@colors-x)**2, -1))
            hessian = weight[:, None, None]*h+.012*np.eye(len(colors))
            linear = weight[:, None]*(x@colors.T)+.012*prior
        else:
            hessian = h+.012*np.eye(len(colors))
            linear = x@colors.T+.012*prior+.002*(np.log(q+1e-5)+1)
        changed, report = native.quadratic_simplex(hessian, linear, q)
        maximum_kkt = max(maximum_kkt, report['max_kkt_residual'])
        difference = float(np.max(np.abs(changed-q)))
        q = changed
        value = palette_energy(q, x, prior, colors, mode)
        if value > history[-1]+max(1e-6, abs(history[-1])*1e-6):
            raise RuntimeError('Palette majorization increased the stated energy')
        history.append(value)
        if difference < 1e-6:
            break
    return q, dict(iterations=used+1, objective_path=history,
        max_probability_change=difference, max_kkt_residual=maximum_kkt,
        converged=difference < 1e-6, nonconvex_local_solution=True)


def palette_model(c, palette, he, mode):
    x, prior, index = colour_nodes(c)
    colors = np.asarray(palette, float)[c.ids-1]/255.
    if mode == 'hellinger':
        q, report = hellinger_inverse(x, prior, colors)
    else:
        q, report = majorized_inverse(x, prior, colors, mode)
    return correct_coarse(q[index], c), dict(model=mode+' palette observation',
        occupied_colour_nodes=len(x), quantization_bins=24, **report)


def max_hinge_objective(flat, design, labels, weight, quadrature, k):
    coefficient = flat.reshape(k, design.shape[-1])
    logits = design@coefficient.T
    margin = 1+logits-logits[np.arange(len(labels)), :, labels][..., None]
    margin[np.arange(len(labels)), :, labels] = -np.inf
    rival = margin.argmax(-1)
    maximum = np.maximum(np.max(margin, -1), 0)
    weighted = weight[:, None]*quadrature[None]
    loss = float(np.sum(weighted*maximum**2))
    derivative = np.zeros_like(logits)
    n, q = np.indices(maximum.shape)
    derivative[n, q, rival] += 2*weighted*maximum
    derivative[n, q, labels[:, None]] -= 2*weighted*maximum
    gradient = np.einsum('nqk,nqd->kd', derivative, design)
    return loss+.0025*np.sum(coefficient**2), (gradient+.005*coefficient).ravel()


def heteroscedastic_hinge(flat, x, labels, weight, covariance, k):
    coefficient = flat.reshape(k, x.shape[1])
    gradient, loss = np.zeros_like(coefficient), 0.
    for own in range(k):
        take = labels == own
        for rival in range(k):
            if rival == own or not take.any():
                continue
            delta = coefficient[rival]-coefficient[own]
            mean = 1+x[take]@delta
            covariance_delta = covariance[take]@delta[:3]
            variance = np.maximum(covariance_delta@delta[:3], 0)+1e-12
            sd = np.sqrt(variance)
            z = mean/sd
            cdf, pdf = ndtr(z), np.exp(-.5*z*z)/np.sqrt(2*np.pi)
            value = (mean**2+variance)*cdf+mean*sd*pdf
            derivative = 2*(mean*cdf+sd*pdf)[:, None]*x[take]
            derivative[:, :3] += 2*cdf[:, None]*covariance_delta
            g = (weight[take, None]*derivative).sum(0)
            gradient[rival] += g
            gradient[own] -= g
            loss += float(weight[take]@value)
    return loss+.0025*np.sum(coefficient**2), (gradient+.005*coefficient).ravel()


def kernel_scores(query, means, labels, k):
    scores = []
    for g in range(k):
        points = means[labels == g]
        self_distance = np.sum((points[:, None]-points[None])**2, -1)
        norm = float(np.mean((3/3.08)**1.5*np.exp(-self_distance/(2*3.08))))
        distance = np.sum((query[:, None]-points[None])**2, -1)
        cross = np.mean((3/3.04)**1.5*np.exp(-distance/(2*3.04)), axis=1)
        scores.append(2*cross-norm)
    return np.stack(scores, -1)


def tile_discriminator(c, palette, he, mode):
    data = bag_design(c)
    if data is None:
        return linear_prob(c), dict(missing_known_bag_class=True)
    bags, labels, transform, center, x = data
    weight = balanced(labels, c.k)
    if mode == 'kernel':
        nodes, _, index = colour_nodes(c)
        values = kernel_scores((nodes-center)@transform, x[:, :3], labels, c.k)
        return softmax(values, -1)[index], dict(model='Gaussian-noise kernel mean class distance',
            training_bags=len(bags), kernel_squared_bandwidth=3., reference_noise_covariance=.04)
    if mode == 'joint':
        design, quadrature = gaussian_design(x)
        objective = lambda a: max_hinge_objective(a, design, labels, weight, quadrature, c.k)
    else:
        residual = (bags[..., :3]-bags[..., :3].mean(1, keepdims=True))@transform
        covariance = np.einsum('nsi,nsj->nij', residual, residual)/625.
        objective = lambda a: heteroscedastic_hinge(a, x, labels, weight, covariance, c.k)
    fit = minimize(objective, np.zeros(c.k*4), jac=True, method='L-BFGS-B',
        options=dict(maxiter=60, ftol=1e-9))
    return linear_result(c, fit.x.reshape(c.k, 4), transform, center), dict(
        model=mode+' Gaussian tile discrimination', training_bags=len(bags),
        **optimizer_report(fit, objective))


def residual_design(bags, transform, x, mode):
    residual = (bags[..., :3]-bags[..., :3].mean(1, keepdims=True))@transform
    if mode == 'block':
        blocks = residual.reshape(len(bags), 5, 5, 3)
        noise = np.concatenate([blocks.mean(1), blocks.mean(2)], axis=1)/np.sqrt(5.)
    else:
        noise = residual/5.
    design = np.repeat(x[:, None], noise.shape[1], axis=1)
    design[..., :3] += noise
    return design


def tile_log_risks(flat, design, labels, k, include_one=False):
    coefficient = flat.reshape(k, design.shape[-1])
    logits = design@coefficient.T
    terms = logits-logits[np.arange(len(labels)), :, labels][..., None]-np.log(design.shape[1])
    terms[np.arange(len(labels)), :, labels] = -np.inf
    value = logsumexp(terms, axis=(1, 2))
    if include_one:
        value = np.logaddexp(0, value)
    derivative = np.exp(terms-value[:, None, None])
    derivative[np.arange(len(labels)), :, labels] = -derivative.sum(-1)
    return value, np.einsum('nqk,nqd->nkd', derivative, design)


def cvar_objective(flat, design, labels, weight, k):
    coefficient, threshold = flat[:-1], flat[-1]
    losses, derivatives = tile_log_risks(coefficient, design, labels, k)
    z = (losses-threshold)/.05
    value = threshold+4*float(weight@(.05*np.logaddexp(0, z)))+.0025*(coefficient@coefficient)
    tail = 4*weight*expit(z)
    gradient = np.einsum('n,nkd->kd', tail, derivatives).ravel()+.005*coefficient
    return value, np.r_[gradient, 1-tail.sum()]


def mean_log_objective(flat, design, labels, weight, k):
    losses, derivatives = tile_log_risks(flat, design, labels, k, include_one=True)
    return float(weight@losses+.0025*(flat@flat)), (
        np.einsum('n,nkd->kd', weight, derivatives).ravel()+.005*flat)


def empirical_model(c, palette, he, mode):
    data = bag_design(c)
    if data is None:
        return linear_prob(c), dict(missing_known_bag_class=True)
    bags, labels, transform, center, x = data
    design = residual_design(bags, transform, x, mode)
    weight = balanced(labels, c.k)
    if mode == 'cvar':
        objective = lambda a: cvar_objective(a, design, labels, weight, c.k)
        start = np.r_[np.zeros(c.k*4), np.log(c.k-1)]
    else:
        objective = lambda a: (empirical_exponential_objective(a, design, labels, weight, c.k)
            if mode == 'block' else mean_log_objective(a, design, labels, weight, c.k))
        start = np.zeros(c.k*4)
    fit = minimize(objective, start, jac=True, method='L-BFGS-B', options=dict(maxiter=60, ftol=1e-9))
    coefficient = fit.x[:-1] if mode == 'cvar' else fit.x
    report = dict(model=mode+' empirical tile risk', residual_points=design.shape[1],
        training_bags=len(bags), **optimizer_report(fit, objective))
    if mode == 'cvar':
        report['tail_threshold'] = float(fit.x[-1])
    return linear_result(c, coefficient.reshape(c.k, 4), transform, center), report


IMPLEMENTATIONS = dict(
    L01=lambda c,p,h: guided_field(c,p,h,'nonlocal'),
    L02=lambda c,p,h: guided_field(c,p,h,'uncertain'), L03=exterior_field,
    L04=lambda c,p,h: bending_field(c,p,h,'bounded'),
    L05=lambda c,p,h: bending_field(c,p,h,'robust'),
    L06=lambda c,p,h: bending_field(c,p,h,'covariance'),
    L07=lambda c,p,h: palette_model(c,p,h,'hellinger'),
    L08=lambda c,p,h: palette_model(c,p,h,'student'),
    L09=lambda c,p,h: palette_model(c,p,h,'sparse'),
    L10=lambda c,p,h: tile_discriminator(c,p,h,'joint'),
    L11=lambda c,p,h: tile_discriminator(c,p,h,'heteroscedastic'),
    L12=lambda c,p,h: tile_discriminator(c,p,h,'kernel'),
    L13=lambda c,p,h: empirical_model(c,p,h,'cvar'),
    L14=lambda c,p,h: empirical_model(c,p,h,'block'),
    L15=lambda c,p,h: empirical_model(c,p,h,'mean-log'))


def predict(key, rgb, coarse, support, mpp_xy, palette, he=None):
    from .registry import CYCLE_EIGHT
    if key not in IMPLEMENTATIONS:
        raise ValueError('Unimplemented cycle-eight child')
    c = Evidence(rgb, coarse, support, mpp_xy)
    if not c.support.any():
        raise ValueError('Empty tissue support')
    if c.k == 1:
        q, detail = np.ones((*c.shape, 1), np.float32), dict(single_class=True)
        if key == 'L03':
            if he is None:
                raise ValueError('Aligned original H&E required')
            excluded, report = wide_path_glass(cv2.resize(he, c.shape[::-1]), c.mpp)
            detail.update(report, glass_exclusion=excluded)
    else:
        q, detail = IMPLEMENTATIONS[key](c, palette, he)
    if (q.shape != (*c.shape, c.k) or not np.isfinite(q).all() or q.min() < -1e-6
            or not np.allclose(q.sum(-1), 1, atol=1e-5)):
        raise ValueError('Invalid cycle-eight posterior')
    labels = c.ids[q.argmax(-1)].astype(np.uint8)
    excluded = detail.pop('glass_exclusion', None)
    if excluded is not None:
        labels[excluded] = 0
    detail.update(analysis_mpp_xy=c.mpp.tolist(), analysis_shape_yx=list(c.shape),
        fine_reference_used=False, hard_anchors=False, fixed_margin=False, candidate=key,
        parent=CYCLE_EIGHT[key].parent, mechanism=CYCLE_EIGHT[key].mechanism)
    return labels, detail
