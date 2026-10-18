"""Fifteen cycle-nine hypotheses. No reference masks, encoder calls or RGB edits."""
import cv2
import numpy as np
from scipy import ndimage as ndi, sparse
from scipy.sparse.csgraph import breadth_first_order, dijkstra, maximum_flow
from scipy.sparse.linalg import splu, spsolve
from scipy.special import gammaln, logsumexp, softmax

from bora.evidence import Evidence, project_simplex
from . import native
from .methods import correct_coarse, fit_linear
from .cycle_two import colour_nodes
from .cycle_five import field_laplacian
from .cycle_six import edge_appearance, field_geometry, guide_laplacian
from .cycle_seven import graph_from_edges
from .cycle_eight import class_statistics, field_inputs, inverse_diagonal
from .diagnostics import white_regions


def upsample(values, shape, grid):
    return cv2.resize(values.reshape(*grid, -1), shape[::-1], interpolation=cv2.INTER_LINEAR)


def quadratic_distance(c, surface, precision):
    delta = c.rgb-surface
    return np.einsum('...i,ij,...j->...', delta, precision, delta)


def moment_constrained_system(matrix, rhs, mass, target_mean):
    """Exact equality-constrained positive-definite quadratic via one Schur column."""
    solved = splu(matrix.tocsc()).solve(np.c_[rhs, mass])
    fit, direction = solved[:, :3], solved[:, 3]
    denominator = float(mass@direction)
    if denominator <= 0:
        raise ValueError('A nonempty class must have a positive Schur denominator')
    multiplier = (mass.sum()*target_mean-mass@fit)/denominator
    fit = fit+direction[:, None]*multiplier
    return fit, dict(equality_max_residual=float(np.max(np.abs(mass@fit-mass.sum()*target_mean))),
        kkt_max_residual=float(np.max(np.abs(matrix@fit-rhs-mass[:, None]*multiplier))),
        schur_denominator=denominator)


def series_laplacian(he, shape, spacing, analysis_spacing):
    """Series law along subgrid H&E paths; affine OD matches endpoint contrast."""
    od = -np.log((he.astype(float)+1)/256.)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    coordinates = np.stack([(yy+.5)*he.shape[0]/shape[0]-.5,
                            (xx+.5)*he.shape[1]/shape[1]-.5])
    index = np.arange(np.prod(shape)).reshape(shape)
    ii, jj, ww = [], [], []
    for axis in (0, 1):
        a, b = [slice(None)]*2, [slice(None)]*2
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        begin, end = coordinates[(slice(None),)+a], coordinates[(slice(None),)+b]
        intervals = max(2, int(np.ceil(spacing[1-axis]/analysis_spacing[1-axis])))
        previous = np.stack([ndi.map_coordinates(od[..., ch], begin, order=1, mode='nearest')
                             for ch in range(3)], -1)
        resistance = np.zeros(previous.shape[:-1])
        for fraction in np.linspace(0, 1, intervals+1)[1:]:
            point = begin+fraction*(end-begin)
            current = np.stack([ndi.map_coordinates(od[..., ch], point, order=1, mode='nearest')
                                for ch in range(3)], -1)
            contrast = np.sum(((current-previous)*intervals)**2, -1)
            resistance += 1/(.02+np.exp(-contrast/.02))
            previous = current
        weight = intervals/resistance*(16/spacing[1-axis])**2
        ii.append(index[a].ravel()); jj.append(index[b].ravel()); ww.append(weight.ravel())
    return graph_from_edges(index.size, ii, jj, ww)


def field_child(c, palette, he, mode):
    means, covariance, shape, spacing, lap = field_inputs(c, he, 'uncertain')
    if mode == 'series':
        original = cv2.resize(he, c.shape[::-1], interpolation=cv2.INTER_AREA)
        lap = series_laplacian(original, shape, spacing, c.mpp)
    precision = np.linalg.inv(covariance)
    scores, reports = [], []
    for g in range(c.k):
        mass, sums = class_statistics(c, g, shape)
        matrix, rhs = lap+sparse.diags(mass+.015), sums+.015*means[g]
        if mode == 'moment':
            fitted, report = moment_constrained_system(matrix, rhs, mass, means[g])
        else:
            fitted = spsolve(matrix.tocsc(), rhs)
            report = dict(linear_system_max_residual=float(np.max(np.abs(matrix@fitted-rhs))))
        scores.append(-.5*quadratic_distance(c, upsample(fitted, c.shape, shape), precision))
        reports.append(report)
    return softmax(np.stack(scores, -1), -1), dict(model=mode+' appearance field',
        field_shape_yx=list(shape), field_mpp_xy=spacing.tolist(), fits=reports)


def em_energy(fields, mass, observations, means, lap, channel):
    delta = observations[:, :, None, :]-fields[:, None, :, :]
    logp = -.5*np.sum(delta*delta, -1)+np.log(channel)[None]
    return float(-np.sum(mass*logsumexp(logp, -1))
        +.5*sum(np.sum(fields[:, g]*(lap@fields[:, g])) for g in range(fields.shape[1]))
        +.0075*np.sum((fields-means[None])**2))


def latent_fields(c, palette, he, iterations=8):
    means, covariance, shape, spacing, lap = field_inputs(c, he, 'uncertain')
    eigen, rotation = np.linalg.eigh(covariance)
    whitening = (rotation/np.sqrt(eigen))@rotation.T
    means = means@whitening
    stats = [class_statistics(c, g, shape, c.rgb@whitening) for g in range(c.k)]
    mass = np.stack([v[0] for v in stats], -1)
    sums = np.stack([v[1] for v in stats], 1)
    observations = sums/np.maximum(mass[..., None], 1e-30)
    fields = np.stack([spsolve((lap+sparse.diags(mass[:, g]+.015)).tocsc(),
                              sums[:, g]+.015*means[g]) for g in range(c.k)], 1)
    channel = .8*np.eye(c.k)+.2/c.k
    history = [em_energy(fields, mass, observations, means, lap, channel)]
    residuals = []
    for used in range(iterations):
        delta = observations[:, :, None, :]-fields[:, None, :, :]
        responsibility = softmax(-.5*np.sum(delta*delta, -1)+np.log(channel)[None], -1)
        assigned = mass[:, :, None]*responsibility
        new_mass = assigned.sum(1)
        new_sum = np.einsum('noh,noc->nhc', assigned, observations)
        changed = []
        for g in range(c.k):
            matrix = lap+sparse.diags(new_mass[:, g]+.015)
            rhs = new_sum[:, g]+.015*means[g]
            fitted = spsolve(matrix.tocsc(), rhs)
            residuals.append(float(np.max(np.abs(matrix@fitted-rhs))))
            changed.append(fitted)
        fields = np.stack(changed, 1)
        history.append(em_energy(fields, mass, observations, means, lap, channel))
        if history[-1] > history[-2]+1e-7*max(1., abs(history[-2])):
            raise RuntimeError('EM increased the original aggregated-observation energy')
    scores = []
    for g in range(c.k):
        difference = c.rgb@whitening-upsample(fields[:, g], c.shape, shape)
        scores.append(-.5*np.sum(difference*difference, -1))
    return softmax(np.stack(scores, -1), -1), dict(model='latent true-class appearance fields',
        field_shape_yx=list(shape), field_mpp_xy=spacing.tolist(), iterations=used+1,
        objective_path=history, observation_channel=channel.tolist(),
        linear_system_max_residual=max(residuals),
        approximation='Original coarse-class weighted means per field node are the Gaussian observations; no semantic labels from unknown tissue.')


def sampling_variance(matrix, mass, rhs, probes=16):
    z = np.random.default_rng(904).choice([-1., 1.], (len(mass), probes))
    all_rhs = np.c_[rhs, np.sqrt(mass)[:, None]*z]
    solved = splu(matrix.tocsc()).solve(all_rhs)
    variance = .015*np.mean(solved[:, 3:]**2, 1)
    return solved[:, :3], variance, dict(diagonal_probes=probes,
        variance_minimum=float(variance.min()), variance_maximum=float(variance.max()),
        linear_system_max_residual=float(np.max(np.abs(matrix@solved-all_rhs))))


def effective_tile_counts(c, g):
    yy, xx = np.indices(c.shape)
    columns = int(np.ceil(c.shape[1]/c.tile_x))
    tile = (yy/c.tile_y).astype(int)*columns+(xx/c.tile_x).astype(int)
    mass = np.bincount(tile.ravel(), weights=(c.weight*(c.labels == g)).ravel())
    return float(mass.sum()**2/max(np.sum(mass**2), 1e-30))


def expected_distance_scores(distance, variance):
    """E[(x-M)' Sigma^-1 (x-M)] for M~N(mean, variance*Sigma)."""
    return -(distance+3*variance)


def uncertain_child(c, palette, he, mode):
    means, covariance, shape, spacing, lap = field_inputs(c, he, 'uncertain')
    precision = np.linalg.inv(covariance)
    scores, reports, distances, variances = [], [], [], []
    for g in range(c.k):
        mass, sums = class_statistics(c, g, shape)
        matrix, rhs = lap+sparse.diags(mass+.015), sums+.015*means[g]
        if mode == 'sampling':
            fitted, v, report = sampling_variance(matrix, mass, rhs)
        else:
            fitted = spsolve(matrix.tocsc(), rhs)
            diagonal, report = inverse_diagonal(matrix)
            v = .015*diagonal
            report['linear_system_max_residual'] = float(np.max(np.abs(matrix@fitted-rhs)))
        variance = cv2.resize(v.reshape(shape), c.shape[::-1], interpolation=cv2.INTER_LINEAR)
        factor = 1+variance
        distance = quadratic_distance(c, upsample(fitted, c.shape, shape), precision)
        if mode == 'student':
            nu = 3+effective_tile_counts(c, g)
            normalizer = gammaln((nu+3)/2)-gammaln(nu/2)-1.5*np.log(nu*np.pi)
            scores.append(normalizer-1.5*np.log(factor)-.5*(nu+3)*np.log1p(distance/(nu*factor)))
            report['degrees_of_freedom'] = nu
        else:
            scores.append(-.5*distance/factor-1.5*np.log(factor))
        distances.append(distance); variances.append(variance); reports.append(report)
    if mode == 'expected':
        scores = expected_distance_scores(np.stack(distances, -1), np.stack(variances, -1))
    else:
        scores = np.stack(scores, -1)
    return softmax(scores, -1), dict(model=mode+' predictive appearance',
        field_shape_yx=list(shape), field_mpp_xy=spacing.tolist(), fits=reports,
        calibration='Assumed node-noise model; no expert-label calibration. Score softmax need not be a calibrated probability.')


def thin_plate(shape, spacing, length=16.):
    h, w = shape
    def second(n, scale):
        if n < 3:
            return sparse.csr_matrix((0, n))
        return scale*sparse.diags([np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)],
                                 [0, 1, 2], shape=(n-2, n))
    def first(n):
        return sparse.diags([-np.ones(n-1), np.ones(n-1)], [0, 1], shape=(n-1, n))
    dxx = sparse.kron(sparse.eye(h), second(w, length**2/spacing[0]**2))
    dyy = sparse.kron(second(h, length**2/spacing[1]**2), sparse.eye(w))
    dxy = sparse.kron(first(h), first(w))*(length**2/np.prod(spacing))
    return (dxx.T@dxx+2*dxy.T@dxy+dyy.T@dyy).tocsr()


def hermite_terms(c, mass, sums, shape):
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    columns = int(np.ceil(c.shape[1]/c.tile_x))
    tile = (((yy+.5)*c.shape[0]/h/c.tile_y).astype(int)*columns
            +((xx+.5)*c.shape[1]/w/c.tile_x).astype(int))
    index = np.arange(h*w).reshape(shape)
    ii, jj, weights = [], [], []
    for axis in (0, 1):
        a, b = [slice(None)]*2, [slice(None)]*2
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        i, j = index[a].ravel(), index[b].ravel()
        valid = (tile[a] == tile[b]).ravel() & (mass[i] > 0) & (mass[j] > 0)
        ii.append(i[valid]); jj.append(j[valid])
        weights.append(.25*2*mass[i[valid]]*mass[j[valid]]/(mass[i[valid]]+mass[j[valid]]))
    lap = graph_from_edges(h*w, ii, jj, weights)
    target = sums/np.maximum(mass[:, None], 1e-30)
    return lap, lap@target, sum(len(v) for v in ii)


def repulsion_objective(flat, shape, lap, mass, sums, means, strength=.015):
    fields = flat.reshape(shape)
    gradient = np.empty_like(fields)
    value = 0.
    for g in range(shape[1]):
        u = fields[:, g]
        gradient[:, g] = lap@u+(mass[:, g, None]+.015)*u-sums[:, g]-.015*means[g]
        value += .5*np.sum(u*(lap@u))+.5*np.sum(mass[:, g, None]*u*u)-np.sum(sums[:, g]*u)+.0075*np.sum((u-means[g])**2)
    for g in range(shape[1]):
        for h in range(g):
            delta = fields[:, g]-fields[:, h]
            penalty = strength*np.exp(-.5*np.sum(delta*delta, -1))
            value += penalty.sum()
            gradient[:, g] -= penalty[:, None]*delta
            gradient[:, h] += penalty[:, None]*delta
    return float(value), gradient.ravel()


def repulsive_system(fields, lap, mass, sums, means, iterations=80):
    """Quadratic majorization with a global upper Hessian bound.

    Pair positive curvature is at most2*exp(-1.5)*strength; the complete-class
    graph's largest eigenvalue is K. strength*K I is a conservative bound.
    """
    factors = [splu((lap+sparse.diags(mass[:, g]+.015+.015*fields.shape[1])).tocsc())
               for g in range(fields.shape[1])]
    objective = lambda flat: repulsion_objective(flat, fields.shape, lap, mass, sums, means)
    value, gradient = objective(fields.ravel())
    history = [value]
    for used in range(iterations):
        gradient = gradient.reshape(fields.shape)
        if np.max(np.abs(gradient)) < 1e-6:
            break
        step = np.stack([factor.solve(gradient[:, g]) for g, factor in enumerate(factors)], 1)
        fields = fields-step
        value, gradient = objective(fields.ravel())
        if value > history[-1]+1e-8*max(1., abs(history[-1])):
            raise RuntimeError('Repulsive majorization increased original energy')
        history.append(value)
    residual = float(np.max(np.abs(gradient)))
    return fields, dict(iterations=used+1, objective_path=history,
        gradient_infinity_norm=residual, converged=residual < 1e-6,
        nonconvex_local_solution=True, curvature_majorizer=.015*fields.shape[1])


def bending_child(c, palette, he, mode):
    _, _, means, covariance = fit_linear(c)
    shape, spacing = field_geometry(c)
    base = field_laplacian(shape, spacing)
    lap = thin_plate(shape, spacing) if mode == 'plate' else base.T@base
    precision = np.linalg.inv(covariance)
    if mode == 'repulsion':
        eigen, rotation = np.linalg.eigh(covariance)
        transform = (rotation/np.sqrt(eigen))@rotation.T
    else:
        transform = np.eye(3)
    means = means@transform
    stats = [class_statistics(c, g, shape, c.rgb@transform) for g in range(c.k)]
    mass = np.stack([v[0] for v in stats], -1)
    sums = np.stack([v[1] for v in stats], 1)
    fields, reports = [], []
    for g in range(c.k):
        matrix, rhs = lap+sparse.diags(mass[:, g]+.015), sums[:, g]+.015*means[g]
        report = {}
        if mode == 'hermite':
            extra, extra_rhs, count = hermite_terms(c, mass[:, g], sums[:, g], shape)
            matrix, rhs = matrix+extra, rhs+extra_rhs
            report['derivative_observation_edges'] = count
        fitted = spsolve(matrix.tocsc(), rhs)
        report['linear_system_max_residual'] = float(np.max(np.abs(matrix@fitted-rhs)))
        fields.append(fitted); reports.append(report)
    fields = np.stack(fields, 1)
    if mode == 'repulsion':
        fields, detail = repulsive_system(fields, lap, mass, sums, means)
        reports.append(detail)
    scores = []
    for g in range(c.k):
        surface = upsample(fields[:, g], c.shape, shape)
        delta = c.rgb@transform-surface
        scores.append(-.5*(np.sum(delta*delta, -1) if mode == 'repulsion'
                           else np.einsum('...i,ij,...j->...', delta, precision, delta)))
    return softmax(np.stack(scores, -1), -1), dict(model=mode+' bending fields',
        field_shape_yx=list(shape), field_mpp_xy=spacing.tolist(), fits=reports)


def ray_glass(white):
    visible = np.logical_and.accumulate(white, axis=0) | np.logical_and.accumulate(white[::-1], axis=0)[::-1]
    visible |= np.logical_and.accumulate(white, axis=1) | np.logical_and.accumulate(white[:, ::-1], axis=1)[:, ::-1]
    for reverse_y in (False, True):
        work = white[::-1] if reverse_y else white
        for dx in (-1, 1):
            reach = np.zeros_like(work)
            reach[0] = work[0]
            for y in range(1, len(work)):
                previous = np.roll(reach[y-1], dx)
                previous[0 if dx == 1 else -1] = True
                reach[y] = work[y] & previous
            visible |= reach[::-1] if reverse_y else reach
    return visible


def stain_access(he, spacing):
    white, _, detail = white_regions(he)
    h, w = white.shape; index = np.arange(h*w).reshape(h, w)
    burden = np.maximum(-np.log((he.astype(float)+1)/256.).mean(-1)-.03, 0)/.1
    ii, jj, ww = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = (slice(0, h-dy), slice(max(0, -dx), min(w, w-dx)))
        b = (slice(dy, h), slice(max(0, dx), min(w, w+dx)))
        length = np.hypot(dx*spacing[0], dy*spacing[1])
        weight = .5*(burden[a]+burden[b])*length/3
        ii.extend([index[a].ravel(), index[b].ravel()])
        jj.extend([index[b].ravel(), index[a].ravel()])
        # Explicit stored zeros are legitimate cost-free glass edges.
        ww.extend([weight.ravel(), weight.ravel()])
    graph = sparse.coo_matrix((np.concatenate(ww), (np.concatenate(ii), np.concatenate(jj))), shape=(h*w, h*w)).tocsr()
    border = np.zeros_like(white); border[[0, -1]] = True; border[:, [0, -1]] = True
    seeds = np.flatnonzero(white & border)
    if not len(seeds):
        return np.zeros_like(white), dict(white_policy=detail, exterior_seed_pixels=0)
    distance = dijkstra(graph, directed=False, indices=seeds, min_only=True, limit=1.)
    return white & (distance.reshape(h, w) <= 1.), dict(white_policy=detail,
        exterior_seed_pixels=len(seeds), maximum_accumulated_stain_cost=1.)


def binary_cut_glass(he, spacing):
    white, _, detail = white_regions(he)
    h, w = white.shape; n = h*w; source, sink = n, n+1
    index = np.arange(n).reshape(h, w)
    x = he.astype(float)/255
    saturation = np.ptp(x, -1)/np.maximum(x.max(-1), 1e-6)
    logit = (x.min(-1)-.96)/.015-saturation/.02
    # Negative log likelihoods computed without unstable log(sigmoid()).
    bg = np.logaddexp(0, -logit)*np.prod(spacing)
    fg = np.logaddexp(0, logit)*np.prod(spacing)
    scale, infinity = 1000., 1_000_000_000
    cap_bg = np.maximum(1, np.rint(scale*bg)).astype(np.int64)
    cap_fg = np.maximum(1, np.rint(scale*fg)).astype(np.int64)
    cap_bg[~white] = infinity; cap_fg[~white] = 0
    border = np.zeros_like(white); border[[0, -1]] = True; border[:, [0, -1]] = True
    cap_fg[white & border] = infinity; cap_bg[white & border] = 0
    rows = [np.full(n, source), index.ravel()]
    cols = [index.ravel(), np.full(n, sink)]
    values = [cap_fg.ravel(), cap_bg.ravel()]
    edges = []
    for axis in (0, 1):
        a, b = [slice(None)]*2, [slice(None)]*2
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        i, j = index[tuple(a)].ravel(), index[tuple(b)].ravel()
        capacity = int(round(scale*2*spacing[axis]))
        rows.extend([i, j]); cols.extend([j, i]); values.extend([np.full(len(i), capacity)]*2)
        edges.append((i, j, capacity))
    graph = sparse.coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=(n+2, n+2)).tocsr()
    fitted = maximum_flow(graph, source, sink)
    residual = graph-fitted.flow
    residual.eliminate_zeros()
    reached = breadth_first_order(residual, source, directed=True, return_predecessors=False)
    cut = np.zeros(n+2, bool); cut[reached] = True
    energy = np.where(cut[:n], cap_bg.ravel(), cap_fg.ravel()).sum()
    energy += sum(capacity*np.count_nonzero(cut[i] != cut[j]) for i, j, capacity in edges)
    if int(energy) != fitted.flow_value:
        raise RuntimeError('Maxflow/mincut energy certificate failed')
    return cut[:n].reshape(h, w) & white, dict(white_policy=detail,
        cut_energy_integer=int(energy), maxflow_integer=int(fitted.flow_value),
        capacity_scale=scale, perimeter_weight_um=2., cut_grid_mpp_xy=list(spacing),
        no_exterior_access_requirement=True)


GLASS_MODES = {'M10': 'stain', 'M11': 'ray', 'M12': 'cut'}


def glass_exclusion(he, spacing, mode):
    if mode == 'stain':
        # Work at4 units for bounded graph memory, using the same sampling
        # convention as the cut. The original white mask gates all removals.
        grid = tuple(max(2, int(np.ceil(n*s/4))) for n, s in zip(he.shape[:2], np.asarray(spacing)[::-1]))
        original = cv2.resize(he, grid[::-1], interpolation=cv2.INTER_AREA)
        step = np.asarray(spacing)*np.asarray(he.shape[1::-1])/np.asarray(grid[::-1])
        excluded, detail = stain_access(original, step)
        detail['access_grid_mpp_xy'] = step.tolist()
    elif mode == 'cut':
        grid = tuple(max(2, int(np.ceil(n*s/4))) for n, s in zip(he.shape[:2], np.asarray(spacing)[::-1]))
        original = cv2.resize(he, grid[::-1], interpolation=cv2.INTER_AREA)
        step = np.asarray(spacing)*np.asarray(he.shape[1::-1])/np.asarray(grid[::-1])
        excluded, detail = binary_cut_glass(original, step)
    else:
        white, _, report = white_regions(he)
        return ray_glass(white), dict(white_policy=report, lattice_directions=8)
    white, _, report = white_regions(he)
    return white & cv2.resize(excluded.astype(np.uint8), he.shape[1::-1], interpolation=cv2.INTER_NEAREST).astype(bool), dict(detail, original_white_policy=report)


def glass_child(c, palette, he, mode):
    q, detail = edge_appearance(c, palette, he)
    original = cv2.resize(he, c.shape[::-1], interpolation=cv2.INTER_AREA)
    excluded, report = glass_exclusion(original, c.mpp, mode)
    detail.update(report, glass_exclusion=excluded, proposed_glass_pixels=int((excluded & c.support).sum()))
    return q, detail


def interval_objective(q, x, prior, colors):
    residual = q@colors-x
    distance = np.sign(residual)*np.maximum(np.abs(residual)-.5/255, 0)
    return float(.5*np.sum(distance**2)+.006*np.sum((q-prior)**2)), distance@colors.T+.012*(q-prior)


def robust_palette_objective(q, x, prior, colors, radius=.03):
    """Smooth reference for derivative/optimum tests; main ADMM is unsmoothed."""
    residual = q@colors-x
    error = np.sqrt(np.sum(residual**2, -1)+1e-12)
    fraction = np.sqrt(np.sum(q*q, -1)+1e-12)
    worst = error+radius*fraction
    gradient = (worst/error)[:, None]*(residual@colors.T)
    gradient += (radius*worst/fraction)[:, None]*q+.012*(q-prior)
    return float(.5*np.sum(worst**2)+.006*np.sum((q-prior)**2)), gradient


def two_norm_prox(a, b, rho, radius=.03):
    """Exact rowwise prox of0.5*(||a||+radius*||b||)^2 / rho."""
    ra, rb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    threshold = (ra+radius*rb)/(rho+1+radius**2)
    only_b = ra <= threshold
    threshold[only_b] = radius*rb[only_b]/(rho+radius**2)
    only_a = rb <= radius*threshold
    threshold[only_a] = ra[only_a]/(rho+1)
    return (a*np.maximum(1-threshold/np.maximum(ra, 1e-30), 0)[:, None],
            b*np.maximum(1-radius*threshold/np.maximum(rb, 1e-30), 0)[:, None])


def robust_palette_admm(x, prior, colors, iterations=600, rho=.1):
    """Simplex quadratic plus exact two-norm prox; explicit finite residuals."""
    q = np.asarray(prior, float).copy()
    u, v = q@colors-x, q.copy()
    du, dv = np.zeros_like(u), np.zeros_like(v)
    hessian = rho*(colors@colors.T)+(rho+.012)*np.eye(len(colors))
    maximum_kkt, history, rho_changes = 0., [], []
    for used in range(iterations):
        linear = rho*((x+u-du)@colors.T+v-dv)+.012*prior
        q, report = native.quadratic_simplex(hessian, linear, q)
        maximum_kkt = max(maximum_kkt, report['max_kkt_residual'])
        old_u, old_v = u, v
        observed = q@colors-x
        u, v = two_norm_prox(observed+du, q+dv, rho)
        du, dv = du+observed-u, dv+q-v
        primal = float(max(np.max(np.abs(observed-u)), np.max(np.abs(q-v))))
        dual = float(np.max(np.abs(rho*((u-old_u)@colors.T+v-old_v))))
        gradient = .012*(q-prior)+rho*(du@colors.T+dv)
        stationarity = float(np.max(np.abs(q-project_simplex(q-gradient))))
        if used % 20 == 0 or used == iterations-1:
            energy = .5*np.sum((np.linalg.norm(observed, axis=1)+.03*np.linalg.norm(q, axis=1))**2)+.006*np.sum((q-prior)**2)
            history.append(dict(iteration=used+1, objective=float(energy),
                primal_residual=primal, dual_residual=dual,
                lagrangian_projected_residual=stationarity, rho=rho))
        if max(primal, dual, stationarity) < 1e-6:
            break
        if used % 20 == 19:
            new_rho = rho*(2 if primal > 10*dual else .5 if dual > 10*primal else 1)
            new_rho = float(np.clip(new_rho, 1e-4, 1e4))
            if new_rho != rho:
                du, dv = du*(rho/new_rho), dv*(rho/new_rho)
                rho_changes.append(dict(iteration=used+1, old=rho, new=new_rho))
                rho = new_rho
                hessian = rho*(colors@colors.T)+(rho+.012)*np.eye(len(colors))
    return q, dict(iterations=used+1, residual_history=history,
        primal_residual=primal, dual_residual=dual,
        lagrangian_projected_residual=stationarity,
        max_kkt_residual=maximum_kkt, converged=max(primal, dual, stationarity) < 1e-6,
        rho=rho, rho_changes=rho_changes, exact_unsmoothed_norm_objective=True)


def projected_inverse(objective, prior, iterations=180):
    q = np.asarray(prior, float).copy()
    value, gradient = objective(q)
    history, step = [value], 1.
    for used in range(iterations):
        for _ in range(50):
            changed = project_simplex(q-step*gradient)
            delta = changed-q
            new, new_gradient = objective(changed)
            if new <= value+np.sum(gradient*delta)+np.sum(delta*delta)/(2*step)+1e-12:
                break
            step *= .5
        else:
            raise RuntimeError('Palette backtracking failed')
        q, value, gradient = changed, new, new_gradient
        history.append(value)
        residual = float(np.max(np.abs(q-project_simplex(q-gradient))))
        if residual < 1e-6:
            break
        step = min(64., step*1.2)
    return q, dict(iterations=used+1, objective_path=history,
        projected_gradient_residual=residual, converged=residual < 1e-6)


def necessity_scores(x, prior, colors):
    def hull(vertices):
        k = len(vertices)
        initial = np.full((len(x), k), 1/k)
        q, report = native.quadratic_simplex(vertices@vertices.T+1e-7*np.eye(k), x@vertices.T, initial)
        return np.sum((q@vertices-x)**2, -1), report
    baseline, full = hull(colors)
    evidence, residual = [], [full['max_kkt_residual']]
    for g in range(len(colors)):
        reduced, report = hull(colors[np.arange(len(colors)) != g])
        evidence.append(np.maximum(reduced-baseline, 0)+1e-10*prior[:, g])
        residual.append(report['max_kkt_residual'])
    return np.stack(evidence, -1), dict(max_kkt_residual=max(residual),
        numerical_hull_tie_ridge=1e-7, interpretation='Counterfactual reconstruction necessity, not a calibrated tissue posterior.')


def palette_child(c, palette, he, mode):
    x, prior, index = colour_nodes(c)
    colors = np.asarray(palette, float)[c.ids-1]/255.
    if mode == 'necessity':
        scores, detail = necessity_scores(x, prior, colors)
        q = softmax(scores, -1)[index]
        # Prior already serves only as a tie break in the counterfactual score;
        # a second coarse correction would change the stated decision rule.
    else:
        if mode == 'interval':
            solved, detail = projected_inverse(lambda q: interval_objective(q, x, prior, colors), prior)
        else:
            solved, detail = robust_palette_admm(x, prior, colors)
        q = correct_coarse(solved[index], c)
    return q, dict(detail, model=mode+' palette observation', occupied_colour_nodes=len(x),
        branch_difference='All occupied24-bin nodes use an averaged coarse prior; the affine parent uses a direct pixel inverse, so this is not an isolated loss ablation.')


IMPLEMENTATIONS = dict(
    M01=latent_fields, M02=lambda c,p,h: field_child(c,p,h,'moment'),
    M03=lambda c,p,h: field_child(c,p,h,'series'),
    M04=lambda c,p,h: uncertain_child(c,p,h,'sampling'),
    M05=lambda c,p,h: uncertain_child(c,p,h,'student'),
    M06=lambda c,p,h: uncertain_child(c,p,h,'expected'),
    M07=lambda c,p,h: bending_child(c,p,h,'plate'),
    M08=lambda c,p,h: bending_child(c,p,h,'hermite'),
    M09=lambda c,p,h: bending_child(c,p,h,'repulsion'),
    M10=lambda c,p,h: glass_child(c,p,h,'stain'),
    M11=lambda c,p,h: glass_child(c,p,h,'ray'),
    M12=lambda c,p,h: glass_child(c,p,h,'cut'),
    M13=lambda c,p,h: palette_child(c,p,h,'interval'),
    M14=lambda c,p,h: palette_child(c,p,h,'necessity'),
    M15=lambda c,p,h: palette_child(c,p,h,'robust'))


def predict(key, rgb, coarse, support, mpp_xy, palette, he=None):
    from .registry import CYCLE_NINE
    if key not in IMPLEMENTATIONS:
        raise ValueError('Unimplemented cycle-nine child')
    c = Evidence(rgb, coarse, support, mpp_xy)
    if not c.support.any():
        raise ValueError('Empty tissue support')
    if c.k == 1:
        q, detail = np.ones((*c.shape, 1), np.float32), dict(single_class=True)
        if key in GLASS_MODES:
            if he is None:
                raise ValueError('Aligned original H&E required')
            excluded, report = glass_exclusion(cv2.resize(he, c.shape[::-1], interpolation=cv2.INTER_AREA), c.mpp, GLASS_MODES[key])
            detail.update(report, glass_exclusion=excluded)
    else:
        q, detail = IMPLEMENTATIONS[key](c, palette, he)
    if (q.shape != (*c.shape, c.k) or not np.isfinite(q).all() or q.min() < -1e-6
            or not np.allclose(q.sum(-1), 1, atol=1e-5)):
        raise ValueError('Invalid cycle-nine normalized scores')
    labels = c.ids[q.argmax(-1)].astype(np.uint8)
    excluded = detail.pop('glass_exclusion', None)
    if excluded is not None:
        labels[excluded] = 0
    detail.update(analysis_mpp_xy=c.mpp.tolist(), analysis_shape_yx=list(c.shape),
        fine_reference_used=False, hard_anchors=False, fixed_margin=False, candidate=key,
        parent=CYCLE_NINE[key].parent, mechanism=CYCLE_NINE[key].mechanism)
    return labels, detail
