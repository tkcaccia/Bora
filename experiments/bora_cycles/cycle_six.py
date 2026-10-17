"""Cycle-six hypotheses; immutable RGB and original coarse observations only.

No fine-reference access, encoder inference, pseudo-labelled class expansion,
or voting among solvers. Each child has one explicit model intervention.
"""
import math
from itertools import product

import cv2
import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear, minimize
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from scipy.special import expit, gammaincinv, gammaln, log_ndtr, logsumexp, ndtr, softmax
from scipy.stats import qmc

from bora.evidence import Evidence, normalize, project_simplex
from . import native
from .methods import correct_coarse, fit_linear, linear_prob, palette_active_set, training
from .cycle_two import colour_nodes, tile_bags
from .cycle_three import balanced
from .cycle_five import (blind_palette, field_laplacian, harmonic_prototype, learn_palette,
                         optimizer_report, whitened_bags)


def field_geometry(c, step=8.):
    shape = tuple(max(2, math.ceil(n*v/step)) for n, v in zip(c.shape, c.mpp[::-1]))
    return shape, c.mpp*np.array(c.shape[::-1])/np.array(shape[::-1])


def guide_laplacian(guide, spacing, length=16., contrast=.1, floor=.02):
    """Symmetric physical graph; edge contrast comes from H&E, not labels."""
    shape = guide.shape[:2]
    index = np.arange(np.prod(shape)).reshape(shape)
    rows, cols, values = [], [], []
    for axis in range(2):
        a, b = [slice(None)]*2, [slice(None)]*2
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        weight = ((floor+np.exp(-np.sum((guide[a]-guide[b])**2, -1)/(2*contrast**2)))
                  *(length/spacing[1-axis])**2).ravel()
        i, j = index[a].ravel(), index[b].ravel()
        rows.extend([i, j, i, j]); cols.extend([i, j, j, i])
        values.extend([weight, weight, -weight, -weight])
    return sparse.coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(index.size, index.size)).tocsr()


def edge_appearance(c, palette, he):
    if he is None:
        raise ValueError('Original aligned H&E required')
    _, _, means, covariance = fit_linear(c)
    inverse = np.linalg.inv(covariance)
    shape, spacing = field_geometry(c)
    original = cv2.resize(he, shape[::-1], interpolation=cv2.INTER_AREA)
    guide = -np.log((original.astype(float)+1)/256)
    laplacian = guide_laplacian(guide, spacing)
    scores, residuals = [], []
    for g in range(c.k):
        weight = c.weight*(c.labels == g)
        mass = cv2.resize(weight, shape[::-1], interpolation=cv2.INTER_AREA).ravel()
        rhs = cv2.resize(weight[..., None]*c.rgb, shape[::-1], interpolation=cv2.INTER_AREA)
        rhs = rhs.reshape(-1, 3)+.015*means[g]
        matrix = laplacian+sparse.diags(mass+.015)
        fitted = spsolve(matrix.tocsc(), rhs)
        residuals.append(float(np.max(np.abs(matrix@fitted-rhs))))
        surface = cv2.resize(fitted.reshape(*shape, 3), c.shape[::-1], interpolation=cv2.INTER_LINEAR)
        delta = c.rgb-surface
        scores.append(-.5*np.einsum('...i,ij,...j->...', delta, inverse, delta))
    return softmax(np.stack(scores, -1), -1), dict(
        model='H&E edge-conducted class appearance fields', field_shape_yx=list(shape),
        field_mpp_xy=spacing.tolist(), linear_system_max_residual=residuals,
        conductance_od_scale=.1, conductance_floor=.02,
        caveat='Microscopic stain edges can block legitimate appearance borrowing.')


def shared_field_system(mass, color_sum, laplacian, prior_mean, anchor=.015):
    """Exact convex two-way normal equations: x = class mean + common drift."""
    total = mass.sum(0)
    class_mass = mass.sum(1)
    a = laplacian+sparse.diags(total+anchor)
    d = sparse.diags(class_mass*(1+anchor))
    cross = sparse.csr_matrix(mass)
    matrix = sparse.bmat([[a, cross.T], [cross, d]], format='csc')
    rhs = np.vstack([color_sum.sum(0), color_sum.sum(1)+anchor*class_mass[:, None]*prior_mean])
    fitted = spsolve(matrix, rhs)
    return fitted[:len(total)], fitted[len(total):], dict(
        linear_system_max_residual=float(np.max(np.abs(matrix@fitted-rhs))))


def shared_appearance(c, palette, he):
    _, _, prior, covariance = fit_linear(c)
    shape, spacing = field_geometry(c)
    mass, color_sum = [], []
    for g in range(c.k):
        weight = c.weight*(c.labels == g)
        mass.append(cv2.resize(weight, shape[::-1], interpolation=cv2.INTER_AREA).ravel())
        color_sum.append(cv2.resize(weight[..., None]*c.rgb, shape[::-1],
                                    interpolation=cv2.INTER_AREA).reshape(-1, 3))
    field, means, report = shared_field_system(np.array(mass), np.array(color_sum),
                                               field_laplacian(shape, spacing), prior)
    surface = cv2.resize(field.reshape(*shape, 3), c.shape[::-1], interpolation=cv2.INTER_LINEAR)
    coefficient = means@np.linalg.inv(covariance)
    q = softmax(native.logits(c.rgb-surface, coefficient, -.5*np.sum(coefficient*means, -1)), -1)
    return q, dict(**report, model='joint class intercepts and shared spatial colour nuisance',
        field_shape_yx=list(shape), field_mpp_xy=spacing.tolist(), class_means=means.tolist(),
        field_range=float(np.ptp(field)),
        caveat='A shared field cannot model independent local class appearance changes.')


def glass_committor(he, support, spacing):
    """Dirichlet glass probability with exterior white and stained tissue seeds."""
    od = -np.log((he.astype(float)+1)/256)
    white = (he.min(-1) >= 235) & (np.ptp(he.astype(float), axis=-1) <= 18)
    border = np.zeros(support.shape, bool)
    border[[0, -1]] = True; border[:, [0, -1]] = True
    background = white & (border | ~support)
    foreground = (od.sum(-1) >= .45) & support
    report = dict(background_seeds=int(background.sum()), foreground_seeds=int(foreground.sum()))
    if not background.any() or not foreground.any():
        return np.zeros(support.shape), dict(**report, no_two_sided_seed_evidence=True,
                                           linear_system_max_residual=0.)
    lap = guide_laplacian(od, spacing, length=1., contrast=.12, floor=.0001)
    fixed = (background | foreground).ravel()
    q = background.ravel().astype(float)
    free = ~fixed
    if free.any():
        matrix = lap[free][:, free].tocsc()
        rhs = -(lap[free][:, fixed]@q[fixed])
        q[free] = spsolve(matrix, rhs)
        residual = float(np.max(np.abs(matrix@q[free]-rhs)))
    else:
        residual = 0.
    if q.min() < -1e-7 or q.max() > 1+1e-7:
        raise ValueError('Harmonic glass probability violates the maximum principle')
    return q.reshape(support.shape), dict(**report, no_two_sided_seed_evidence=False,
        linear_system_max_residual=residual)


def harmonic_glass(c, palette, he):
    if he is None:
        raise ValueError('Original aligned H&E required')
    q, report = harmonic_prototype(c, palette, he)
    shape, spacing = field_geometry(c, 4.)
    original = cv2.resize(he, shape[::-1], interpolation=cv2.INTER_AREA)
    support = cv2.resize(c.support.astype(np.uint8), shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
    probability, glass = glass_committor(original, support, spacing)
    probability = cv2.resize(probability, c.shape[::-1], interpolation=cv2.INTER_LINEAR)
    original = cv2.resize(he, c.shape[::-1], interpolation=cv2.INTER_AREA)
    white = (original.min(-1) >= 235) & (np.ptp(original.astype(float), axis=-1) <= 18)
    removed = white & (probability > .95) & c.support
    report.update(glass_model='exterior H&E harmonic committor', glass_fit=glass,
        glass_exclusion=removed, proposed_glass_pixels=int(removed.sum()),
        glass_grid_mpp_xy=spacing.tolist(), glass_threshold=.95,
        caveat='White border adipose is not distinguishable from glass by access alone.')
    return q, report


def simplex_quadrature(k, power=7, alpha=.3):
    """Cyclic symmetry, positive Dirichlet proposal and explicit importance weights."""
    uniform = qmc.Sobol(k, scramble=True, seed=113).random_base2(power)
    uniform = np.clip(uniform, 1e-10, 1-1e-10)
    gamma = gammaincinv(alpha, uniform)
    samples = gamma/gamma.sum(1, keepdims=True)
    samples = np.concatenate([np.roll(samples, j, axis=1) for j in range(k)])
    log_proposal = gammaln(k*alpha)-k*gammaln(alpha)+(alpha-1)*np.log(samples).sum(1)
    return samples, -log_proposal


def posterior_dominance(x, prior, colors, samples, log_weight, variance=.025**2):
    output = np.empty_like(prior, dtype=float)
    means = samples@colors
    winner = samples == samples.max(1, keepdims=True)
    log_ties = np.log(winner.sum(1))
    for start in range(0, len(x), 512):
        at = slice(start, start+512)
        residual = (np.sum(x[at]**2, -1)[:, None]+np.sum(means**2, -1)[None]
                    -2*x[at]@means.T)
        penalty = (np.sum(prior[at]**2, -1)[:, None]+np.sum(samples**2, -1)[None]
                   -2*prior[at]@samples.T)
        log_posterior = -.5*(residual+.012*penalty)/variance+log_weight
        evidence = np.stack([logsumexp(log_posterior[:, winner[:, g]]-log_ties[winner[:, g]], -1)
                             for g in range(len(colors))], -1)
        output[at] = softmax(evidence, -1)
    return output


def integrated_palette(c, palette, he):
    x, pi, index = colour_nodes(c)
    colors = np.asarray(palette, float)[c.ids-1]/255
    samples, log_weight = simplex_quadrature(c.k)
    q = posterior_dominance(x, pi, colors, samples, log_weight)
    return correct_coarse(q[index], c), dict(
        model='integrated dominant-class mass of the palette posterior',
        quadrature_points=len(samples), symmetric_cycles=c.k, proposal_dirichlet_alpha=.3,
        observation_variance=.025**2, prior_ridge=.012,
        caveat='Finite deterministic importance quadrature may miss a narrow or high-dimensional mode.')


def colour_graph(x):
    if len(x) <= 1:
        return sparse.csr_matrix((len(x), len(x))), 0.
    distance, neighbours = cKDTree(x).query(x, k=min(7, len(x)), workers=2)
    distance, neighbours = distance[:, 1:], neighbours[:, 1:]
    scale = max(float(np.median(distance[:, 0])), 1/255)
    rows = np.repeat(np.arange(len(x)), neighbours.shape[1])
    adjacency = sparse.coo_matrix((np.exp(-.5*(distance.ravel()/scale)**2),
                                  (rows, neighbours.ravel())), shape=(len(x), len(x))).tocsr()
    adjacency = (adjacency+adjacency.T)*.5
    degree = np.asarray(adjacency.sum(1)).ravel()
    return sparse.diags(degree)-adjacency, scale


def graph_inverse(c, palette, he):
    x, pi, index = colour_nodes(c)
    colors = np.asarray(palette, float)[c.ids-1]/255
    q, initial = native.simplex_qp(x, pi, colors)
    q = q.astype(float)
    lap, scale = colour_graph(x)
    hessian = colors@colors.T+.012*np.eye(c.k)
    rhs = x@colors.T+.012*pi
    step = 1/(np.linalg.norm(hessian, 2)+.03*float(lap.diagonal().max(initial=0)))
    history = []
    for iteration in range(100):
        gradient = q@hessian-rhs+.015*(lap@q)
        changed = project_simplex(q-step*gradient)
        change = float(np.max(np.abs(changed-q))); q = changed
        history.append(float(.5*np.sum((q@colors-x)**2)+.006*np.sum((q-pi)**2)
                             +.0075*np.sum(q*(lap@q))))
        if change < 1e-7:
            break
    residual = float(np.max(np.abs(q-project_simplex(q-(q@hessian-rhs+.015*(lap@q))))))
    return correct_coarse(q[index], c), dict(model='coupled occupied-colour graph palette inverse',
        initial_solver=initial, occupied_nodes=len(x), graph_nonzeros=lap.nnz,
        graph_bandwidth=scale, graph_regularizer=.015, iterations=iteration+1,
        projected_gradient_residual=residual, converged=residual < 1e-6,
        objective_path=history,
        caveat='Nearby colours may be different tissues; colour coupling can erase real interfaces.')


def feature_bags(c, feature):
    bags, labels = [], []
    h, w = c.shape
    for y in np.arange(0, h, c.tile_y):
        for x in np.arange(0, w, c.tile_x):
            yy = np.clip(np.floor(y+(np.arange(5)+.5)*c.tile_y/5).astype(int), 0, h-1)
            xx = np.clip(np.floor(x+(np.arange(5)+.5)*c.tile_x/5).astype(int), 0, w-1)
            known = c.known[yy[:, None], xx[None, :]]
            if known.mean() < .5:
                continue
            hist = np.bincount(c.labels[yy[:, None], xx[None, :]][known], minlength=c.k)
            bags.append(feature[yy[:, None], xx[None, :]].reshape(25, -1))
            labels.append(int(hist.argmax()))
    return np.asarray(bags), np.asarray(labels, int)


def calibration_objective(flat, x, labels, weight, initial):
    coefficient = flat.reshape(initial.shape)
    logits = x@coefficient
    p = softmax(logits, -1)
    value = float(weight@(logsumexp(logits, -1)-logits[np.arange(len(labels)), labels]))
    p[np.arange(len(labels)), labels] -= 1
    delta = coefficient-initial
    return value+.005*float(np.sum(delta**2)), (x.T@(weight[:, None]*p)+.01*delta).ravel()


def calibrated_palette(c, palette, he):
    q, report = palette_active_set(c, palette)
    log_fraction = np.log(np.maximum(q, 1e-5))
    bags, labels = feature_bags(c, log_fraction)
    if not len(bags) or set(labels) != set(range(c.k)):
        return q, dict(**report, missing_known_bag_class=True)
    x = np.c_[bags.mean(1), np.ones(len(bags))]
    initial = np.vstack([np.eye(c.k), np.zeros((1, c.k))])
    objective = lambda flat: calibration_objective(flat, x, labels, balanced(labels, c.k), initial)
    fit = minimize(objective, initial.ravel(), jac=True, method='L-BFGS-B',
                   options=dict(maxiter=60, ftol=1e-9))
    coefficient = fit.x.reshape(initial.shape)
    result = softmax(native.logits(log_fraction, coefficient[:-1].T, coefficient[-1]), -1)
    return result, dict(**optimizer_report(fit, objective), palette_solver=report,
        model='coarse tile log-fraction decoder', training_bags=len(bags), decoder_ridge=.01,
        caveat='Mean log-fraction calibration can follow coarse contamination rather than anatomy.')


def hinge_objective(flat, x, labels, weight, k):
    coefficient = flat.reshape(k, x.shape[1])
    gradient, loss = np.zeros_like(coefficient), 0.
    for own in range(k):
        take = labels == own
        for rival in range(k):
            if rival == own or not take.any():
                continue
            delta = coefficient[rival]-coefficient[own]
            mean = 1+x[take]@delta
            variance = np.sum(delta[:3]**2)/25+1e-12
            sd = np.sqrt(variance); z = mean/sd
            cdf, pdf = ndtr(z), np.exp(-.5*z*z)/np.sqrt(2*np.pi)
            value = (mean**2+variance)*cdf+mean*sd*pdf
            dm = 2*(mean*cdf+sd*pdf)
            derivative = dm[:, None]*x[take]
            derivative[:, :3] += 2*cdf[:, None]*delta[:3]/25
            g = (weight[take, None]*derivative).sum(0)
            gradient[rival] += g; gradient[own] -= g
            loss += float(weight[take]@value)
    return loss+.0025*float(np.sum(coefficient**2)), (gradient+.005*coefficient).ravel()


def integrated_hinge(c, palette, he):
    data = whitened_bags(c)
    if data is None:
        return linear_prob(c), dict(missing_known_bag_class=True)
    bags, labels, _, _, transform = data
    center = bags[..., :3].mean((0, 1))
    x = np.c_[(bags[..., :3].mean(1)-center)@transform, np.ones(len(bags))]
    objective = lambda flat: hinge_objective(flat, x, labels, balanced(labels, c.k), c.k)
    fit = minimize(objective, np.zeros(c.k*4), jac=True, method='L-BFGS-B',
                   options=dict(maxiter=60, ftol=1e-9))
    coefficient = fit.x.reshape(c.k, 4)
    raw = coefficient[:, :3]@transform.T
    return softmax(native.logits(c.rgb, raw, coefficient[:, 3]-raw@center), -1), dict(
        **optimizer_report(fit, objective), model='Gaussian-integrated squared hinge discrimination',
        training_bags=len(bags), noise_mean_covariance=1/25,
        caveat='Squared hinge growth limits extremes but can underfit genuinely difficult tiles.')


def positive_scatter(matrix):
    values, vectors = np.linalg.eigh((matrix+matrix.T)*.5)
    return (vectors*np.maximum(values, 0))@vectors.T


def denoised_scatter(c, palette, he):
    data = whitened_bags(c)
    if data is None:
        return linear_prob(c), dict(missing_known_bag_class=True)
    bags, labels, means, _, transform = data
    centers = bags[..., :3].mean(1)@transform
    class_mean = means@transform
    residual = centers-class_mean[labels]
    pooled = positive_scatter(residual.T@residual/len(residual)-np.eye(3)/25)
    precision, intercept, reports = [], [], []
    for g in range(c.k):
        values = residual[labels == g]
        observed = values.T@values/len(values)
        latent = positive_scatter(observed-np.eye(3)/25)
        weight = len(values)/(len(values)+5)
        covariance = np.eye(3)+weight*latent+(1-weight)*pooled
        precision.append(transform@np.linalg.inv(covariance)@transform.T)
        intercept.append(-.5*np.linalg.slogdet(covariance)[1])
        reports.append(dict(class_id=int(c.ids[g]), known_tiles=len(values),
            class_scatter_weight=weight, latent_eigenvalues=np.linalg.eigvalsh(latent).tolist()))
    scores = native.gaussian_logits(c.rgb, means, np.asarray(precision), np.asarray(intercept))
    return softmax(scores, -1), dict(model='measurement-error-deconvolved class scatter', fits=reports,
        scatter_noise_subtraction=1/25, pixel_noise_addition=1., sparse_class_prior_tiles=5,
        caveat='Few coarse tiles may not identify class scatter; covariance subtraction is a model assumption.')


def additive_basis(x, knots):
    extra = np.logaddexp(0, (x[..., :, None]-knots)/.5)*.5
    return np.concatenate([np.ones((*x.shape[:-1], 1)), x, extra.reshape(*x.shape[:-1], -1)], -1)


def cubature_risk(flat, design, labels, weight, quadrature_weight, k):
    """Monotone log of positive convex exponential risk plus its ridge."""
    coefficient = flat.reshape(design.shape[-1], k)
    logits = design@coefficient
    own = logits[np.arange(len(labels)), :, labels]
    log_term = logits-own[..., None]+np.log(weight)[:, None, None]+np.log(quadrature_weight)[None, :, None]
    log_term[np.arange(len(labels)), :, labels] = -np.inf
    log_risk = logsumexp(log_term)
    ridge = .0025*np.sum(coefficient**2)
    log_total = np.logaddexp(log_risk, np.log(ridge) if ridge > 0 else -np.inf)
    derivative = np.exp(log_term-log_total)
    derivative[np.arange(len(labels)), :, labels] = -derivative.sum(-1)
    gradient = np.einsum('nqd,nqk->dk', design, derivative)+.005*coefficient*np.exp(-log_total)
    return float(log_total), gradient.ravel()


def additive_discrimination(c, palette, he):
    data = whitened_bags(c)
    if data is None:
        return linear_prob(c), dict(missing_known_bag_class=True)
    bags, labels, _, _, transform = data
    center = bags[..., :3].mean((0, 1))
    mean = (bags[..., :3].mean(1)-center)@transform
    knots = np.quantile(mean, [.2, .4, .6, .8], axis=0).T
    point, qw = np.polynomial.hermite.hermgauss(3)
    offsets = np.array(list(product(point, repeat=3)))*np.sqrt(2/25)
    quadrature_weight = np.array([a*b*d for a,b,d in product(qw, repeat=3)])/np.pi**1.5
    design = additive_basis(mean[:, None]+offsets, knots)
    location = design.mean((0, 1)); scale = np.maximum(design.std((0, 1)), .25)
    location[0] = 0; scale[0] = 1
    design = (design-location)/scale
    objective = lambda flat: cubature_risk(flat, design, labels, balanced(labels, c.k),
                                          quadrature_weight, c.k)
    fit = minimize(objective, np.zeros(design.shape[-1]*c.k), jac=True, method='L-BFGS-B',
                   options=dict(maxiter=60, ftol=1e-9))
    coefficient = fit.x.reshape(design.shape[-1], c.k)
    x, _, index = colour_nodes(c)
    features = (additive_basis((x-center)@transform, knots)-location)/scale
    return softmax(native.logits(features, coefficient.T, np.zeros(c.k)), -1)[index], dict(
        **optimizer_report(fit, objective), model='additive nonlinear colour discrimination under Gaussian cubature',
        training_bags=len(bags), basis_dimensions=design.shape[-1], knots=knots.tolist(), cubature_points=27,
        objective_reporting='log of positive convex risk plus ridge; same minimizer, finite budget',
        caveat='Additivity misses colour interactions; Gaussian cubature is finite and occupied nodes approximate pixels.')


def fit_rigid_palette(x, prior, weight, canonical, iterations=12):
    weight = np.asarray(weight, float); weight /= weight.sum()
    colors = np.asarray(canonical, float).copy()
    objectives, residuals = [], []
    rotation, translation = np.eye(3), np.zeros(3)
    for iteration in range(iterations):
        q, report = native.simplex_qp(x, prior, colors)
        z = q@canonical
        z_mean, x_mean = weight@z, weight@x
        cross = z.T@(weight[:, None]*x)-np.outer(z_mean, x_mean)/1.02+.01*np.eye(3)
        u, _, vt = np.linalg.svd(cross)
        sign = np.ones(3); sign[-1] = np.linalg.det(u@vt)
        rotation = (u*sign)@vt
        translation = (x_mean-z_mean@rotation)/1.02
        fitted = canonical@rotation+translation
        change = float(np.max(np.abs(fitted-colors))); colors = fitted
        objectives.append(float(.5*np.sum(weight[:, None]*(q@colors-x)**2)
            +.006*np.sum(weight[:, None]*(q-prior)**2)
            +.005*np.sum((rotation-np.eye(3))**2)+.01*np.sum(translation**2)))
        residuals.append(report['max_kkt_residual'])
        if change < 1e-5:
            break
    return colors, dict(iterations=iteration+1, objective_path=objectives,
        q_kkt_residuals=residuals, final_palette_change=change, rotation=rotation.tolist(),
        translation=translation.tolist(), rotation_determinant=float(np.linalg.det(rotation)),
        orthogonality_residual=float(np.max(np.abs(rotation.T@rotation-np.eye(3)))))


def rigid_palette(c, palette, he):
    prior = c.lattice()
    data, _, weight = training(c, np.concatenate([c.rgb, prior], -1), limit=4096)
    canonical = np.asarray(palette, float)[c.ids-1]/255
    colors, report = fit_rigid_palette(data[:, :3], data[:, 3:], weight, canonical)
    x, pi, index = colour_nodes(c)
    q, solved = native.simplex_qp(x, pi, colors)
    return correct_coarse(q[index], c), dict(**report, model='rigid latent palette calibration',
        fitted_palette=colors.tolist(), prediction_solver=solved, training_points=len(data),
        fitted_palette_range=[float(colors.min()), float(colors.max())],
        caveat='Rigid geometry can underfit real distortion; latent vertices may lie outside the RGB cube.')


def project_dominant(x, labels):
    """Projection on {q >= 0, sum q = 1, q[label] >= every rival}."""
    result = np.asarray(x, float).copy()
    for row, own in zip(result, labels):
        rivals = np.sort(np.delete(row, own))[::-1]
        pooled = np.r_[row[own], row[own]+np.cumsum(rivals)]/np.arange(1, len(row)+1)
        top = pooled.max()
        row[:] = np.minimum(row, top); row[own] = top
    return project_simplex(result)


def dominance_palette(c, palette, he):
    bags, labels = tile_bags(c)
    if not len(bags) or set(labels) != set(range(c.k)):
        q, report = blind_palette(c, palette, he)
        return q, dict(**report, missing_known_bag_class=True)
    x = bags[..., :3].mean(1)
    weight = balanced(labels, c.k)
    canonical = np.asarray(palette, float)[c.ids-1]/255
    prior = np.full((len(x), c.k), 1/c.k)
    q, colors = prior.copy(), canonical.copy()
    objectives, residuals, unconverged = [], [], []
    for iteration in range(12):
        q, fit = native.dominance_qp(x, prior, colors, labels, initial=q)
        root = np.sqrt(weight)
        design = np.vstack([root[:, None]*q, np.sqrt(.02)*np.eye(c.k)])
        target = np.vstack([root[:, None]*x, np.sqrt(.02)*canonical])
        fitted = np.stack([lsq_linear(design, target[:, j], bounds=(0, 1), method='bvls',
                                      tol=1e-10).x for j in range(3)], -1)
        change = float(np.max(np.abs(fitted-colors))); colors = fitted
        objectives.append(float(.5*np.sum(weight[:, None]*(q@colors-x)**2)
            +.006*np.sum(weight[:, None]*(q-prior)**2)+.01*np.sum((colors-canonical)**2)))
        residuals.append(fit['max_projected_gradient_residual'])
        unconverged.append(fit['unconverged_bags'])
        if change < 1e-5:
            break
    query, pi, index = colour_nodes(c)
    posterior, solved = native.simplex_qp(query, pi, colors)
    return correct_coarse(posterior[index], c), dict(
        model='coarse winner constrained latent palette fitting', training_bags=len(bags),
        iterations=iteration+1, objective_path=objectives, dominance_qp_residuals=residuals,
        dominance_unconverged_bags=unconverged, final_palette_change=change,
        fitted_palette=colors.tolist(), prediction_solver=solved,
        caveat='A tile pixel-count winner need not have the greatest mean latent colour fraction.')


def uncertain_palette(c, palette, he):
    prior = c.lattice()
    data, _, weight = training(c, np.concatenate([c.rgb, prior], -1), limit=4096)
    weight = np.asarray(weight, float); weight /= weight.sum()
    canonical = np.asarray(palette, float)[c.ids-1]/255
    colors, report = learn_palette(data[:, :3], data[:, 3:], weight, canonical)
    latent, fitted = native.simplex_qp(data[:, :3], data[:, 3:], colors)
    variance = max(float(np.sum(weight[:, None]*(latent@colors-data[:, :3])**2)/3), 1/255**2)
    covariance = variance*np.linalg.inv(latent.T@(weight[:, None]*latent)+.02*np.eye(c.k))
    x, pi, index = colour_nodes(c)
    hessian = colors@colors.T+.012*np.eye(c.k)+3*covariance
    posterior, solved = native.quadratic_simplex(hessian, x@colors.T+.012*pi, pi)
    return correct_coarse(posterior[index], c), dict(**report,
        model='expected palette reconstruction risk under calibration covariance',
        fitted_palette=colors.tolist(), residual_channel_variance=variance,
        palette_covariance=covariance.tolist(), final_training_solver=fitted,
        prediction_solver=solved, training_points=len(data),
        caveat='Interior Gaussian calibration covariance ignores active palette box constraints.')


def joint_normal_event(probability, labels, nodes=21):
    """Log winner mass and its probability derivative under independent Gaussian counts."""
    n, _, k = probability.shape
    mean = probability.sum(1)
    raw_variance = (probability*(1-probability)).sum(1)
    variance = np.maximum(raw_variance, .05)
    sd = np.sqrt(variance)
    point, quadrature = np.polynomial.hermite.hermgauss(nodes)
    point, quadrature = point*np.sqrt(2), quadrature/np.sqrt(np.pi)
    own_mean, own_sd = mean[np.arange(n), labels], sd[np.arange(n), labels]
    z = (own_mean[:, None, None]+own_sd[:, None, None]*point[None, :, None]
         -mean[:, None, :])/sd[:, None, :]
    own = np.arange(k)[None, :] == labels[:, None]
    log_cdf = log_ndtr(z)
    terms = np.log(quadrature)[None, :]+np.where(own[:, None], 0, log_cdf).sum(-1)
    log_probability = logsumexp(terms, -1)
    responsibility = np.exp(terms-log_probability[:, None])
    mills = np.exp(-.5*z*z-.5*np.log(2*np.pi)-log_cdf)
    mills[own[:, None, :].repeat(nodes, axis=1)] = 0
    dm = -np.sum(responsibility[..., None]*mills/sd[:, None, :], 1)
    dm[np.arange(n), labels] = -dm.sum(-1)
    dv = -.5*np.sum(responsibility[..., None]*mills*z/variance[:, None, :], 1)
    own_dv = .5*np.sum(responsibility*np.sum(mills/sd[:, None, :], -1)*point,
                        1)/own_sd
    dv[np.arange(n), labels] = own_dv
    dv *= raw_variance > .05
    derivative = dm[:, None, :]+dv[:, None, :]*(1-2*probability)
    return log_probability, derivative


def joint_normal_objective(flat, bags, labels, weight, initial):
    coefficient = flat.reshape(initial.shape)
    probability = softmax(bags@coefficient, -1)
    logp, derivative = joint_normal_event(probability, labels)
    derivative *= -weight[:, None, None]
    logit = probability*(derivative-(probability*derivative).sum(-1, keepdims=True))
    delta = coefficient-initial
    return float(-weight@logp+.0002*np.sum(delta**2)), (
        np.einsum('nsi,nsk->ik', bags, logit)+.0004*delta).ravel()


def conditional_counts(c, palette, he):
    bags, labels = tile_bags(c)
    if not len(bags):
        return linear_prob(c), dict(training_bags=0)
    coefficient, intercept, _, _ = fit_linear(c)
    initial = np.c_[coefficient, intercept].T
    objective = lambda flat: joint_normal_objective(flat, bags, labels, balanced(labels, c.k), initial)
    fit = minimize(objective, initial.ravel(), jac=True, method='L-BFGS-B',
                   options=dict(maxiter=60, ftol=1e-9))
    coefficient = fit.x.reshape(initial.shape)
    return softmax(native.logits(c.rgb, coefficient[:3].T, coefficient[3]), -1), dict(
        **optimizer_report(fit, objective), model='shared-own-count Gaussian winner integral',
        training_bags=len(bags), quadrature_points=21,
        caveat='Independent Gaussian class counts ignore multinomial anticorrelation; finite quadrature is approximate.')


def normal_event(probability, labels):
    """Original pairwise-normal composite event, with an explicit log derivative."""
    derivative = np.zeros_like(probability)
    log_probability = np.zeros(len(labels))
    own = probability[np.arange(len(labels)), :, labels]
    for g in range(probability.shape[-1]):
        rival = probability[:, :, g]
        delta = own-rival
        mean = delta.sum(1)
        raw_variance = (own+rival-delta*delta).sum(1)
        variance = np.maximum(raw_variance, .05)
        z = mean/np.sqrt(variance)
        logcdf = log_ndtr(z)
        active = labels != g
        log_probability += logcdf*active
        mills = np.exp(-.5*z*z-.5*np.log(2*np.pi)-logcdf)*active
        dm = mills/np.sqrt(variance)
        dv = -.5*mills*mean/variance**1.5*(raw_variance > .05)
        derivative[np.arange(len(labels)), :, labels] += dm[:, None]+dv[:, None]*(1-2*delta)
        derivative[:, :, g] += -dm[:, None]+dv[:, None]*(1+2*delta)
    return log_probability, derivative


def contaminated_objective(flat, bags, labels, weight, initial):
    coefficient = flat[:-1].reshape(initial.shape)
    probability = softmax(bags@coefficient, -1)
    k, n = probability.shape[-1], len(labels)
    all_labels = [np.full(n, g, int) for g in range(k)]
    log_event = np.stack([normal_event(probability, y)[0] for y in all_labels], -1)
    event = softmax(log_event, -1)
    fraction = expit(flat[-1]); noise = .49*fraction
    clean = event[np.arange(n), labels]
    observed = (1-noise)*clean+noise/k
    clean_responsibility = (1-noise)*clean/np.maximum(observed, 1e-200)
    event_derivative = event.copy(); event_derivative[np.arange(n), labels] -= 1
    event_derivative *= (weight*clean_responsibility)[:, None]
    derivative = np.zeros_like(probability)
    for g in range(k):
        derivative += event_derivative[:, g, None, None]*normal_event(probability, all_labels[g])[1]
    logit = probability*(derivative-(probability*derivative).sum(-1, keepdims=True))
    delta = coefficient-initial
    gradient = np.einsum('nsi,nsk->ik', bags, logit)+.0004*delta
    noise_gradient = (-np.sum(weight*(1/k-clean)/observed)+.02/(1-noise))*.49*fraction*(1-fraction)
    value = (-weight@np.log(observed)-.02*np.log1p(-noise)+.0002*np.sum(delta**2))
    return float(value), np.r_[gradient.ravel(), noise_gradient]


def contaminated_counts(c, palette, he):
    bags, labels = tile_bags(c)
    if not len(bags):
        return linear_prob(c), dict(training_bags=0)
    coefficient, intercept, _, _ = fit_linear(c)
    initial = np.c_[coefficient, intercept].T
    fraction = .05/.49
    start = np.r_[initial.ravel(), np.log(fraction/(1-fraction))]
    objective = lambda flat: contaminated_objective(flat, bags, labels, balanced(labels, c.k), initial)
    fit = minimize(objective, start, jac=True, method='L-BFGS-B',
                   options=dict(maxiter=60, ftol=1e-9))
    coefficient = fit.x[:-1].reshape(initial.shape)
    return softmax(native.logits(c.rgb, coefficient[:3].T, coefficient[3]), -1), dict(
        **optimizer_report(fit, objective), model='latent contamination of normalized composite tile winners',
        training_bags=len(bags), inferred_contamination=float(.49*expit(fit.x[-1])),
        contamination_upper_bound=.49, clean_prior_log_weight=.02,
        caveat='Coarse corruption and appearance error are not statistically identifiable here.')


def unpack_network(flat, d, k, units):
    end = d*k; hidden_end = end+d*units
    return (flat[:end].reshape(d, k), flat[end:hidden_end].reshape(d, units),
            flat[hidden_end:].reshape(units, k))


def neural_objective(flat, bags, labels, weight, initial, hidden_initial):
    skip, hidden, output = unpack_network(flat, bags.shape[-1], initial.shape[-1], hidden_initial.shape[-1])
    activation = np.tanh(bags@hidden)
    probability = softmax(bags@skip+activation@output, -1)
    logp, derivative = normal_event(probability, labels)
    derivative *= -weight[:, None, None]
    logit = probability*(derivative-(probability*derivative).sum(-1, keepdims=True))
    ds, dh = skip-initial, hidden-hidden_initial
    value = -weight@logp+.0002*np.sum(ds**2)+.0025*(np.sum(dh**2)+np.sum(output**2))
    gs = np.einsum('nsd,nsk->dk', bags, logit)+.0004*ds
    go = np.einsum('nsh,nsk->hk', activation, logit)+.005*output
    back = (logit@output.T)*(1-activation**2)
    gh = np.einsum('nsd,nsh->dh', bags, back)+.005*dh
    return float(value), np.r_[gs.ravel(), gh.ravel(), go.ravel()]


def neural_counts(c, palette, he):
    bags, labels = tile_bags(c)
    if not len(bags):
        return linear_prob(c), dict(training_bags=0)
    center = bags[..., :3].mean((0, 1))
    scale = np.maximum(bags[..., :3].std((0, 1)), .05)
    bags = bags.copy(); bags[..., :3] = (bags[..., :3]-center)/scale
    coefficient, intercept, _, _ = fit_linear(c)
    initial = np.vstack([(coefficient*scale).T, intercept+coefficient@center])
    hidden_initial = .5*np.sin(np.arange(32).reshape(4, 8)+1)
    start = np.r_[initial.ravel(), hidden_initial.ravel(), np.zeros(8*c.k)]
    objective = lambda flat: neural_objective(flat, bags, labels, balanced(labels, c.k),
                                            initial, hidden_initial)
    fit = minimize(objective, start, jac=True, method='L-BFGS-B',
                   options=dict(maxiter=60, ftol=1e-9))
    skip, hidden, output = unpack_network(fit.x, 4, c.k, 8)
    x, _, index = colour_nodes(c)
    features = np.c_[(x-center)/scale, np.ones(len(x))]
    q = softmax(features@skip+np.tanh(features@hidden)@output, -1)
    return q[index], dict(**optimizer_report(fit, objective), model='learned nonlinear colour vote field',
        training_bags=len(bags), hidden_units=8, parameters=len(start),
        caveat='Finite nonconvex coarse-only learning can produce unsupported colour islands.')


IMPLEMENTATIONS = dict(J01=edge_appearance, J02=shared_appearance, J03=harmonic_glass,
    J04=integrated_palette, J05=graph_inverse, J06=calibrated_palette,
    J07=integrated_hinge, J08=denoised_scatter, J09=additive_discrimination,
    J10=rigid_palette, J11=dominance_palette, J12=uncertain_palette,
    J13=conditional_counts, J14=contaminated_counts, J15=neural_counts)


def predict(key, rgb, coarse, support, mpp_xy, palette, he=None):
    from .registry import CYCLE_SIX
    if key not in IMPLEMENTATIONS:
        raise ValueError('Unimplemented cycle-six child')
    c = Evidence(rgb, coarse, support, mpp_xy)
    if not c.support.any():
        raise ValueError('Empty tissue support')
    if c.k == 1 and key != 'J03':
        q, detail = np.ones((*c.shape, 1), np.float32), dict(single_class=True)
    else:
        q, detail = IMPLEMENTATIONS[key](c, palette, he)
    excluded = detail.pop('glass_exclusion', None)
    if (q.shape != (*c.shape, c.k) or not np.isfinite(q).all() or np.min(q) < -1e-6
            or not np.allclose(q.sum(-1), 1, atol=1e-5)):
        raise ValueError('Invalid cycle-six posterior')
    labels = c.ids[q.argmax(-1)].astype(np.uint8)
    if excluded is not None:
        labels[excluded] = 0
    detail.update(analysis_mpp_xy=c.mpp.tolist(), analysis_shape_yx=list(c.shape),
        fine_reference_used=False, hard_anchors=False, fixed_margin=False, candidate=key,
        parent=CYCLE_SIX[key].parent, mechanism=CYCLE_SIX[key].mechanism)
    return labels, detail
