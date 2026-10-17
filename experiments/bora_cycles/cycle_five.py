"""Fifteen cycle-five models, original coarse supervision and fixed RGB only.

These are separate models, not parameter variants or a voting ensemble.
Unknown tissue can inform class prevalence, but never supplies semantic
training labels. Fine annotations are not available to this module.
"""
from itertools import combinations
import math

import cv2
import numpy as np
from scipy import ndimage as ndi, sparse
from scipy.optimize import minimize, lsq_linear
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from scipy.special import logsumexp, softmax

from bora.evidence import Evidence, normalize, project_simplex
from . import native
from .methods import fit_linear, linear_prob, training, correct_coarse
from .cycle_two import colour_nodes, tile_bags
from .cycle_three import uncertain_prototype, balanced
from .cycle_four import bag_statistics


def optimizer_report(fit, objective):
    value, gradient = objective(fit.x)
    return dict(optimizer_success=bool(fit.success), iterations=int(fit.nit),
                objective=float(value), gradient_infinity_norm=float(np.max(np.abs(gradient))),
                termination=str(fit.message))


def spatial_mixture(c, palette, he):
    _, _, global_mean, covariance = fit_linear(c)
    inverse = np.linalg.inv(covariance)
    ry, rx = [max(1, round(48/v)) for v in c.mpp[::-1]]
    scores, reports = [], []
    for g in range(c.k):
        take = c.known & (c.labels == g)
        x, w = c.rgb[take].astype(float), c.weight[take].astype(float)
        if len(x) > 4096:
            index = np.linspace(0, len(x)-1, 4096, dtype=int)
            x, w = x[index], w[index]
        # Deterministic PCA initialization, two latent appearances per known class.
        centered = x-global_mean[g]
        _, vec = np.linalg.eigh((centered.T*w)@centered/w.sum())
        direction = vec[:, -1]
        spread = max(float(np.sqrt(np.average((centered@direction)**2, weights=w))), .01)
        means = global_mean[g]+np.array([-1, 1])[:, None]*spread*direction
        mixing = np.full(2, .5)
        for iteration in range(20):
            coefficient = means@inverse
            intercept = -.5*np.sum(coefficient*means, -1)
            responsibility = softmax(x@coefficient.T+intercept+np.log(mixing), -1)
            mass = (w[:, None]*responsibility).sum(0)+1e-10
            new_means = (responsibility.T@(w[:, None]*x))/mass[:, None]
            change = float(np.max(np.abs(new_means-means)))
            means, mixing = new_means, mass/mass.sum()
            if change < 1e-5:
                break
        coefficient = means@inverse
        intercept = -.5*np.sum(coefficient*means, -1)
        local_resp = softmax(native.logits(c.rgb, coefficient, intercept)+np.log(mixing), -1)
        local = cv2.boxFilter(((c.weight*(c.labels == g))[..., None]*local_resp).astype(np.float32),
                             cv2.CV_32F, (2*rx+1, 2*ry+1), normalize=True)
        local = normalize(local+.015*mixing)
        scores.append(native.local_mixture_logits(c.rgb, coefficient, intercept, local))
        reports.append(dict(class_id=int(c.ids[g]), means=means.tolist(),
                            global_mixing=mixing.tolist(), iterations=iteration+1,
                            last_mean_change=change, training_points=len(x)))
    return softmax(np.stack(scores, -1), -1), dict(
        fits=reports, local_mixing_radius_um=48, components_per_class=2,
        model="Gaussian latent appearances with spatial mixture probabilities",
        caveat="Coarse contamination can create false subclasses; finite EM is not a global optimum.")


def field_laplacian(shape, mpp, length_um=16.):
    ny, nx = shape
    def one(n, factor):
        if n == 1:
            return sparse.csr_matrix((1, 1))
        degree = np.full(n, 2.); degree[[0, -1]] = 1
        return factor*sparse.diags([-np.ones(n-1), degree, -np.ones(n-1)], [-1, 0, 1])
    return (sparse.kron(one(ny, (length_um/mpp[1])**2), sparse.eye(nx))
            +sparse.kron(sparse.eye(ny), one(nx, (length_um/mpp[0])**2))).tocsr()


def harmonic_prototype(c, palette, he):
    _, _, means, covariance = fit_linear(c)
    inverse = np.linalg.inv(covariance)
    shape = tuple(max(2, math.ceil(n*v/8)) for n, v in zip(c.shape, c.mpp[::-1]))
    spacing = c.mpp*np.array(c.shape[::-1])/np.array(shape[::-1])
    laplacian = field_laplacian(shape, spacing)
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
        model="screened-Laplace class appearance field", field_shape_yx=list(shape),
        field_mpp_xy=spacing.tolist(), diffusion_length_um=16, prior_precision=.015,
        linear_system_max_residual=residuals,
        caveat="A coarse field can smooth genuine drift; only appearances, not labels, are diffused.")


def interface_cavities(he, probability, support, mpp):
    white = (he.min(-1) >= 235) & (np.ptp(he.astype(float), axis=-1) <= 18)
    components, count = ndi.label(white, np.ones((3, 3)))
    if not count or probability.shape[-1] < 2:
        return np.zeros_like(white), dict(white_components=count, accepted_components=0)
    distance, nearest = ndi.distance_transform_edt(~white, sampling=mpp[::-1], return_indices=True)
    near_component = components[tuple(nearest)]
    ring = (distance > 0) & (distance <= 2.) & support
    mass = np.bincount(near_component[ring], minlength=count+1)
    fractions = np.stack([np.bincount(near_component[ring], weights=probability[..., g][ring],
                                     minlength=count+1)/np.maximum(mass, 1)
                          for g in range(probability.shape[-1])], -1)
    second = np.sort(fractions, -1)[:, -2]
    area = np.bincount(components.ravel(), minlength=count+1)*np.prod(mpp)
    accepted = (second >= .25) & (area >= 32.) & (mass > 0)
    accepted[0] = False
    return accepted[components], dict(white_components=count, accepted_components=int(accepted.sum()),
        ring_width_um=2., min_area_um2=32., min_second_class_ring_probability=.25,
        caveat="Lumina at real class interfaces can still be deleted; class posterior ambiguity can fool the rule.")


def interface_glass(c, palette, he):
    if he is None:
        raise ValueError("Aligned original H&E is required")
    q, detail = uncertain_prototype(c, palette, he)
    original = cv2.resize(he, c.shape[::-1], interpolation=cv2.INTER_AREA)
    excluded, report = interface_cavities(original, q, c.support, c.mpp)
    detail.update(report, glass_exclusion=excluded,
                  proposed_glass_pixels=int((excluded & c.support).sum()))
    return q, detail


def learn_palette(x, prior, weight, canonical, iterations=15):
    weight = np.asarray(weight, float); weight = weight/weight.sum()
    colors, objectives, residuals = np.asarray(canonical, float).copy(), [], []
    root = np.sqrt(weight)
    for iteration in range(iterations):
        q, report = native.simplex_qp(x, prior, colors)
        design = np.vstack([root[:, None]*q, np.sqrt(.02)*np.eye(len(colors))])
        target = np.vstack([root[:, None]*x, np.sqrt(.02)*canonical])
        new = np.stack([lsq_linear(design, target[:, j], bounds=(0, 1),
                                  method='bvls', tol=1e-10).x for j in range(3)], -1)
        objective = (.5*np.sum(weight[:, None]*(q@new-x)**2)
                     +.006*np.sum(weight[:, None]*(q-prior)**2)
                     +.01*np.sum((new-canonical)**2))
        objectives.append(float(objective)); residuals.append(report['max_kkt_residual'])
        change = float(np.max(np.abs(new-colors))); colors = new
        if change < 1e-5:
            break
    return colors, dict(iterations=iteration+1, objective_path=objectives,
                        final_palette_change=change, q_kkt_residuals=residuals)


def blind_palette(c, palette, he):
    prior = c.lattice()
    data, _, weight = training(c, np.concatenate([c.rgb, prior], -1), limit=4096)
    colors, detail = learn_palette(data[:, :3], data[:, 3:], weight,
                                   np.asarray(palette, float)[c.ids-1]/255)
    x, pi, index = colour_nodes(c)
    q, solved = native.simplex_qp(x, pi, colors)
    detail.update(fitted_palette=colors.tolist(), training_points=len(data),
        semantic_training="original known coarse tissue only; no unknown-tile pseudo-labels",
        prediction_solver=solved,
        caveat="Alternating constrained mixture fitting is nonconvex and coarse mixtures can bias the learned palette.")
    return correct_coarse(q[index], c), detail


def profiled_plane_cost(x, color):
    matrix = np.array([[color@color+.01, color.sum()], [color.sum(), 3+.01]])
    rhs = np.stack([x@color+.01, x.sum(-1)], -1)
    ab = np.linalg.solve(matrix, rhs.T).T
    negative = ab[:, 0] < 0
    ab[negative, 0] = 0
    ab[negative, 1] = x[negative].sum(-1)/3.01
    residual = x-ab[:, :1]*color-ab[:, 1:]
    return np.sum(residual**2, -1)+.01*((ab[:, 0]-1)**2+ab[:, 1]**2)


def affine_nuisance(c, palette, he):
    x, _, index = colour_nodes(c)
    colors = np.asarray(palette, float)[c.ids-1]/255
    scores = []
    for color in colors:
        value = profiled_plane_cost(x, color)
        scores.append(-value/(2*.025**2))
    return correct_coarse(softmax(np.stack(scores, -1), -1)[index], c), dict(
        model="profile likelihood on nonnegative brightness/neutral-offset class planes",
        gain_offset_ridge=.01, residual_scale=.025,
        caveat="Different class planes overlap; brightness nuisance can erase genuine class evidence.")


def cardinal_coordinates(x, colors):
    unique, inverse = np.unique(colors, axis=0, return_inverse=True)
    if len(unique) == 1:
        return np.full((len(x), len(colors)), 1/len(colors)), dict(unique_vertices=1)
    squared = np.sum((unique[:, None]-unique[None, :])**2, -1)
    length2 = float(np.median(squared[squared > 0]))
    kernel = np.exp(-squared/(2*length2))
    system = np.block([[kernel+1e-7*np.eye(len(unique)), np.ones((len(unique), 1))],
                       [np.ones((1, len(unique))), np.zeros((1, 1))]])
    coefficient = np.linalg.solve(system, np.vstack([np.eye(len(unique)), np.zeros((1, len(unique)))]))
    query = np.exp(-np.sum((np.asarray(x)[:, None]-unique[None, :])**2, -1)/(2*length2))
    coordinate = np.c_[query, np.ones(len(x))]@coefficient
    coordinate = project_simplex(coordinate)
    return coordinate[:, inverse]/np.bincount(inverse)[inverse], dict(
        unique_vertices=len(unique), bandwidth_squared=length2,
        model="constant-reproducing Gaussian cardinal palette interpolation",
        duplicated_vertex_policy="split indistinguishable classes equally",
        caveat="Nonlinear cardinal coordinates need not recover the true conversion mixture.")


def nonlinear_palette(c, palette, he):
    x, _, index = colour_nodes(c)
    q, detail = cardinal_coordinates(x, np.asarray(palette, float)[c.ids-1]/255)
    return correct_coarse(q[index], c), detail


def trimmed_objective(flat, bags, labels, weight, initial):
    coefficient = flat.reshape(initial.shape)
    logits = bags@coefficient
    count = math.ceil(bags.shape[1]/initial.shape[1])
    selected = np.argsort(logits, axis=1, kind='stable')[:, -count:, :]
    pooled = np.take_along_axis(logits, selected, axis=1).mean(1)
    p = softmax(pooled, -1)
    loss = float(np.sum(weight*(logsumexp(pooled, -1)-pooled[np.arange(len(labels)), labels])))
    derivative = p.copy(); derivative[np.arange(len(labels)), labels] -= 1
    gradient = np.zeros_like(logits)
    for g in range(initial.shape[1]):
        gradient[np.arange(len(labels))[:, None], selected[..., g], g] = (
            weight*derivative[:, g])[:, None]/count
    delta = coefficient-initial
    return loss+.0002*np.sum(delta**2), (
        np.einsum('nsp,nsk->pk', bags, gradient)+.0004*delta).ravel()


def trimmed_tiles(c, palette, he):
    bags, labels = tile_bags(c)
    if not len(bags):
        return linear_prob(c), dict(training_bags=0)
    coefficient, intercept, _, _ = fit_linear(c)
    initial = np.c_[coefficient, intercept].T
    objective = lambda flat: trimmed_objective(flat, bags, labels, balanced(labels, c.k), initial)
    fit = minimize(objective, initial.ravel(), jac=True, method='L-BFGS-B',
                   options=dict(maxiter=60, ftol=1e-9))
    coefficient = fit.x.reshape(initial.shape)
    return softmax(native.logits(c.rgb, coefficient[:3].T, coefficient[3]), -1), dict(
        **optimizer_report(fit, objective), training_bags=len(bags),
        required_support_pixels=math.ceil(25/c.k),
        caveat="Minimal plural support is only a necessary count bound, not the true joint winner likelihood.")


def channel_objective(flat, bags, labels, weight, initial, channel):
    coefficient = flat.reshape(initial.shape)
    p = softmax(bags@coefficient, -1)
    mean = p.mean(1)
    observed = mean@channel
    py = np.maximum(observed[np.arange(len(labels)), labels], 1e-200)
    dp = -channel[:, labels].T*(weight/py)[:, None]
    gradient_logits = p*(dp[:, None]-(p*dp[:, None]).sum(-1, keepdims=True))/bags.shape[1]
    delta = coefficient-initial
    return float(-np.sum(weight*np.log(py))+.0002*np.sum(delta**2)), (
        np.einsum('nsp,nsk->pk', bags, gradient_logits)+.0004*delta).ravel()


def noisy_tiles(c, palette, he):
    bags, labels = tile_bags(c)
    if not len(bags):
        return linear_prob(c), dict(training_bags=0)
    fractions, _ = native.simplex_qp(bags[..., :3].mean(1), np.full((len(bags), c.k), 1/c.k),
                                    np.asarray(palette, float)[c.ids-1]/255)
    counts = fractions.T@np.eye(c.k)[labels]
    channel = .5*np.eye(c.k)+.5*(counts+1e-8)/(counts.sum(1, keepdims=True)+c.k*1e-8)
    coefficient, intercept, _, _ = fit_linear(c)
    initial = np.c_[coefficient, intercept].T
    objective = lambda flat: channel_objective(flat, bags, labels, balanced(labels, c.k), initial, channel)
    fit = minimize(objective, initial.ravel(), jac=True, method='L-BFGS-B',
                   options=dict(maxiter=60, ftol=1e-9))
    coefficient = fit.x.reshape(initial.shape)
    return softmax(native.logits(c.rgb, coefficient[:3].T, coefficient[3]), -1), dict(
        **optimizer_report(fit, objective), training_bags=len(bags), observation_channel=channel.tolist(),
        caveat="The coarse winner channel is estimated from palette fractions, not observed true pixel labels.")


def information_objective(flat, x, labels, weight, k, initial):
    # Last class is the reference coordinate, removing the softmax gauge.
    coefficient = flat.reshape(k-1, x.shape[1])
    logits = np.c_[x@coefficient.T, np.zeros(len(x))]
    p = softmax(logits, -1); nonref = p[:, :-1]
    w = np.eye(k-1)[None]*nonref[:, :, None]-nonref[:, :, None]*nonref[:, None, :]
    n = len(x)
    information = n*np.einsum('n,ngh,np,nq->gphq', weight, w, x, x).reshape(flat.size, flat.size)
    information += 1e-4*np.eye(flat.size)
    inverse = np.linalg.inv(information).reshape(k-1, x.shape[1], k-1, x.shape[1])
    v = np.einsum('np,gphq,nq->ngh', x, inverse, x)
    r = np.diagonal(v, axis1=1, axis2=2)-2*np.einsum('ngh,nh->ng', v, nonref)
    correction = nonref*(r-(nonref*r).sum(-1, keepdims=True))
    derivative = p.copy(); derivative[np.arange(n), labels] -= 1
    gradient = (weight[:, None]*(derivative[:, :-1]-.5*correction)).T@x
    delta = coefficient-initial
    value = (np.sum(weight*(logsumexp(logits, -1)-logits[np.arange(n), labels]))
             -.5/n*np.linalg.slogdet(information)[1]+.0002*np.sum(delta**2))
    return float(value), (gradient+.0004*delta).ravel()


def information_tiles(c, palette, he):
    bags, labels = tile_bags(c)
    if not len(bags):
        return linear_prob(c), dict(training_bags=0)
    coefficient, intercept, _, _ = fit_linear(c)
    full = np.c_[coefficient, intercept]
    initial = full[:-1]-full[-1]
    x = bags.mean(1)
    objective = lambda flat: information_objective(flat, x, labels, balanced(labels, c.k), c.k, initial)
    fit = minimize(objective, initial.ravel(), jac=True, method='L-BFGS-B',
                   options=dict(maxiter=60, ftol=1e-9))
    coefficient = np.vstack([fit.x.reshape(initial.shape), np.zeros(4)])
    return softmax(native.logits(c.rgb, coefficient[:, :3], coefficient[:, 3]), -1), dict(
        **optimizer_report(fit, objective), training_bags=len(bags), information_ridge=1e-4,
        model="regularized Firth-type multinomial information-volume correction",
        caveat="Uses tile mean logits and a ridged information determinant; not an exact count model or exact invariant Jeffreys prior.")


def whitened_bags(c):
    data = bag_statistics(c)
    if data is None:
        return None
    bags, labels, means, _, covariance = data
    covariance = .8*covariance+.2*np.diag(np.diag(covariance))+.0025*np.eye(3)
    values, vectors = np.linalg.eigh(covariance)
    transform = vectors@np.diag(1/np.sqrt(values))@vectors.T
    return bags, labels, means, covariance, transform


def noise_objective(flat, x, labels, weight, k):
    coefficient = flat.reshape(k, x.shape[1])
    gradient, loss = np.zeros_like(coefficient), 0.
    for own in range(k):
        take = labels == own
        for rival in range(k):
            if own == rival or not take.any():
                continue
            delta = coefficient[rival]-coefficient[own]
            z = x[take]@delta+.5*np.sum(delta[:3]**2)/25
            exponential = np.exp(z)
            loss += float(weight[take]@exponential)
            derivative = x[take].copy(); derivative[:, :3] += delta[:3]/25
            g = (weight[take]*exponential)@derivative
            gradient[rival] += g; gradient[own] -= g
    return float(loss+.0025*np.sum(coefficient**2)), (gradient+.005*coefficient).ravel()


def noise_marginal(c, palette, he):
    data = whitened_bags(c)
    if data is None:
        return linear_prob(c), dict(missing_known_bag_class=True)
    bags, labels, _, _, transform = data
    center = bags[..., :3].mean((0, 1))
    x = np.c_[(bags[..., :3].mean(1)-center)@transform, np.ones(len(bags))]
    objective = lambda flat: noise_objective(flat, x, labels, balanced(labels, c.k), c.k)
    fit = minimize(objective, np.zeros(c.k*4), jac=True, method='L-BFGS-B',
                   options=dict(maxiter=60, ftol=1e-9))
    coefficient = fit.x.reshape(c.k, 4)
    raw = coefficient[:, :3]@transform.T
    return softmax(native.logits(c.rgb, raw, coefficient[:, 3]-raw@center), -1), dict(
        **optimizer_report(fit, objective), training_bags=len(bags), noise_covariance_in_whitened_mean=1/25,
        model="analytically noise-marginalized all-rival exponential risk",
        caveat="Independent Gaussian tile-mean error is an approximation to correlated tissue.")


def fit_prevalence(log_density, count, iterations=100):
    count = np.asarray(count, float); count = count/count.sum()
    probability = np.full(log_density.shape[1], 1/log_density.shape[1])
    # A weak symmetric Dirichlet keeps every original class admissible.
    pseudo = 1e-4/log_density.shape[1]
    for iteration in range(iterations):
        responsibility = softmax(log_density+np.log(probability), -1)
        new = (count[:, None]*responsibility).sum(0)+pseudo
        new /= new.sum()
        change = float(np.max(np.abs(new-probability))); probability = new
        if change < 1e-8:
            break
    value = float(np.sum(count*logsumexp(log_density+np.log(probability), -1))
                  +pseudo*np.log(probability).sum())
    return probability, dict(iterations=iteration+1, last_change=change,
                            converged=change < 1e-8, regularized_log_likelihood=value)


def prevalence(c, palette, he):
    data = whitened_bags(c)
    if data is None:
        return linear_prob(c), dict(missing_known_bag_class=True)
    _, _, means, covariance, _ = data
    coefficient = np.linalg.solve(covariance, means.T).T
    intercept = -.5*np.sum(coefficient*means, -1)
    x, _, index = colour_nodes(c)
    count = np.bincount(index[c.support], minlength=len(x))
    log_density = x@coefficient.T+intercept
    probability, report = fit_prevalence(log_density, count)
    return softmax(native.logits(c.rgb, coefficient, intercept+np.log(probability)), -1), dict(
        **report, inferred_prevalence=probability.tolist(),
        semantic_densities="fixed from original known coarse tiles only",
        unlabeled_role="estimate mixture proportions, never refit semantic appearances",
        caveat="Label shift assumes each known density remains valid across the whole tissue.")


def geometric_median_objective(point, data, epsilon=1e-4):
    delta = point-data
    distance = np.sqrt(np.sum(delta**2, -1)+epsilon**2)
    return float(distance.mean()), np.mean(delta/distance[:, None], 0)


def robust_centroids(c, palette, he):
    data = whitened_bags(c)
    if data is None:
        return linear_prob(c), dict(missing_known_bag_class=True)
    bags, labels, _, _, transform = data
    x = bags[..., :3].mean(1)@transform
    means, reports = [], []
    for g in range(c.k):
        samples = x[labels == g]
        objective = lambda point: geometric_median_objective(point, samples)
        fit = minimize(objective, samples.mean(0), jac=True, method='L-BFGS-B',
                       options=dict(maxiter=80, ftol=1e-11, gtol=1e-7))
        means.append(fit.x); reports.append(optimizer_report(fit, objective))
    means = np.asarray(means)
    return softmax(native.logits(c.rgb, means@transform.T, -.5*np.sum(means**2, -1)), -1), dict(
        fits=reports, model="whitened smoothed spatial median of original tile centroids",
        smoothing_epsilon=1e-4,
        caveat="A minority genuine appearance can be downweighted as coarse-tile contamination.")


def chow_liu_density(histogram):
    p = np.asarray(histogram, float); p = p/p.sum()
    marginal = [p.sum(tuple(j for j in range(3) if j != i)) for i in range(3)]
    pairs, information = {}, []
    for a, b in combinations(range(3), 2):
        pair = p.sum(tuple(j for j in range(3) if j not in (a, b)))
        pairs[a, b] = pair
        independent = marginal[a][:, None]*marginal[b][None]
        mi = float(np.sum(pair*np.log(np.maximum(pair, 1e-300)/np.maximum(independent, 1e-300))))
        information.append((mi, a, b))
    edges = sorted(information, reverse=True)[:2]
    root = max(range(3), key=lambda i: -float(np.sum(marginal[i]*np.log(np.maximum(marginal[i], 1e-300)))))
    coordinates = np.indices(p.shape)
    density = marginal[root][coordinates[root]]
    visited = {root}; directed = []
    while len(visited) < 3:
        for _, a, b in edges:
            if (a in visited) == (b in visited):
                continue
            parent, child = (a, b) if a in visited else (b, a)
            pair = pairs[min(a,b), max(a,b)]
            joint = pair[coordinates[min(a,b)], coordinates[max(a,b)]]
            density *= joint/np.maximum(marginal[parent][coordinates[parent]], 1e-300)
            visited.add(child); directed.append([parent, child])
    return density, dict(root=root, directed_edges=directed,
                         pair_mutual_information=[[a,b,mi] for mi,a,b in information],
                         discrete_normalization=float(density.sum()))


def graphical_density(c, palette, he):
    bins = 24
    cell = np.minimum((c.rgb*bins).astype(int), bins-1)
    code = (cell[..., 0]*bins+cell[..., 1])*bins+cell[..., 2]
    scores, reports = [], []
    for g in range(c.k):
        hist = np.bincount(code.ravel(), weights=(c.weight*(c.labels == g)).ravel(),
                           minlength=bins**3).reshape((bins,)*3)
        hist = ndi.gaussian_filter(hist, .7)+max(hist.sum(), 1)*1e-7
        density, report = chow_liu_density(hist)
        scores.append(np.log(np.maximum(density.ravel()[code], 1e-30)))
        reports.append(report)
    return softmax(np.stack(scores, -1), -1), dict(fits=reports, bins=bins,
        model="class-specific Chow-Liu RGB dependence trees",
        caveat="A two-edge graph cannot express three-way interactions; discretized weak labels remain contaminated.")


def neighbour_density(c, palette, he):
    x, _, index = colour_nodes(c)
    scores, outside, calibration = [], [], []
    for g in range(c.k):
        take = c.known & (c.labels == g)
        values, weights = c.rgb[take], c.weight[take]
        cumulative = np.cumsum(weights); cumulative /= cumulative[-1]
        count = min(2048, len(values))
        sample = values[np.searchsorted(cumulative, (np.arange(count)+.5)/count)]
        tree = cKDTree(sample)
        k = min(8, count)
        radius = tree.query(x, k=k, workers=2)[0]
        radius = radius if k == 1 else radius[:, -1]
        scores.append(np.log(k/count)-3*np.log(np.maximum(radius, 1/255)))
        own = tree.query(sample, k=min(k+1,count), workers=2)[0]
        own = own if own.ndim == 1 else own[:, -1]
        threshold = max(float(np.quantile(own, .95)), 1/255)
        outside.append(radius > threshold); calibration.append(threshold)
    novel = np.all(outside, axis=0)[index] & c.support
    components, number = ndi.label(novel, np.ones((3,3)))
    sizes = np.bincount(components.ravel())[1:]*np.prod(c.mpp)
    slices = ndi.find_objects(components)
    proposals = []
    for at in np.argsort(sizes)[::-1][:50]:
        sy, sx = slices[at]
        proposals.append(dict(component=int(at+1), area_um2=float(sizes[at]),
            analysis_bbox_xywh=[sx.start,sy.start,sx.stop-sx.start,sy.stop-sy.start],
            semantic_identity="unknown"))
    return softmax(np.stack(scores, -1), -1)[index], dict(
        neighbours=8, max_training_points_per_class=2048,
        candidate_outlier_pixels=int(novel.sum()), unknown_identity_regions=number,
        unknown_region_areas_um2=sorted(sizes.tolist(), reverse=True)[:50],
        candidate_outlier_regions=proposals,
        calibration_radius_95pct=calibration,
        caveat="Finite samples and a one-level RGB distance floor limit density estimation; novelty does not name a biological tissue class.")


def bernstein_basis(x, degree=3):
    x = np.clip(np.asarray(x, float), 0, 1)
    univariate = np.stack([(degree+1)*math.comb(degree,j)*x**j*(1-x)**(degree-j)
                           for j in range(degree+1)], -1)
    return np.einsum('na,nb,nc->nabc', univariate[:,0], univariate[:,1],
                     univariate[:,2]).reshape(len(x), -1)


def fit_bernstein(basis, weight, iterations=40):
    weight = np.asarray(weight, float); weight = weight/weight.sum()
    theta = np.full(basis.shape[1], 1/basis.shape[1]); pseudo = .001/basis.shape[1]
    path = []
    for iteration in range(iterations):
        density = np.maximum(basis@theta, 1e-30)
        new = theta*(basis.T@(weight/density))+pseudo
        new /= new.sum()
        change = float(np.max(np.abs(new-theta))); theta = new
        path.append(float(np.sum(weight*np.log(np.maximum(basis@theta, 1e-30)))
                          +pseudo*np.log(theta).sum()))
        if change < 1e-6:
            break
    return theta, dict(iterations=iteration+1, last_change=change,
                       converged=change < 1e-6, regularized_log_likelihood_path=path)


def bernstein_density(c, palette, he):
    x, _, index = colour_nodes(c); basis = bernstein_basis(x)
    scores, reports = [], []
    for g in range(c.k):
        weight = np.bincount(index.ravel(), weights=(c.weight*(c.labels == g)).ravel(), minlength=len(x))
        theta, report = fit_bernstein(basis, weight)
        scores.append(np.log(np.maximum(basis@theta, 1e-30))); reports.append(report)
    return softmax(np.stack(scores, -1), -1)[index], dict(
        fits=reports, normalized_beta_components=64, polynomial_degree=3,
        model="positive joint Bernstein RGB density mixture",
        caveat="A low-order continuous density can erase rare modes; occupied-colour representatives approximate training RGB.")


IMPLEMENTATIONS = dict(H01=spatial_mixture, H02=harmonic_prototype, H03=interface_glass,
    H04=blind_palette, H05=affine_nuisance, H06=nonlinear_palette, H07=trimmed_tiles,
    H08=noisy_tiles, H09=information_tiles, H10=noise_marginal, H11=prevalence,
    H12=robust_centroids, H13=graphical_density, H14=neighbour_density, H15=bernstein_density)


def predict(key, rgb, coarse, support, mpp_xy, palette, he=None):
    from .registry import CYCLE_FIVE
    if key not in IMPLEMENTATIONS:
        raise ValueError("Unimplemented cycle-five child")
    c = Evidence(rgb, coarse, support, mpp_xy)
    if not c.support.any():
        raise ValueError("Empty tissue support")
    if c.k == 1 and key != 'H03':
        q, detail = np.ones((*c.shape, 1), np.float32), dict(single_class=True)
    else:
        q, detail = IMPLEMENTATIONS[key](c, palette, he)
    excluded = detail.pop('glass_exclusion', None)
    if q.shape != (*c.shape, c.k) or not np.isfinite(q).all() or np.min(q) < -1e-6:
        raise ValueError("Invalid child posterior")
    labels = c.ids[q.argmax(-1)].astype(np.uint8)
    if excluded is not None:
        labels[excluded] = 0
    detail.update(analysis_mpp_xy=c.mpp.tolist(), analysis_shape_yx=list(c.shape),
        fine_reference_used=False, hard_anchors=False, fixed_margin=False, candidate=key,
        parent=CYCLE_FIVE[key].parent, mechanism=CYCLE_FIVE[key].mechanism)
    return labels, detail
