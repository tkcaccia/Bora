"""Fifteen cycle-four hypotheses; fixed RGB, original coarse supervision only.

No reference-mask I/O, encoder calls, pseudo-labelled semantic training or
solver voting. Numerical budgets and failed convergence remain explicit.
"""
import warnings
from itertools import product

import cv2
import numpy as np
from scipy import ndimage as ndi, sparse
from scipy.optimize import minimize, linprog, OptimizeWarning
from scipy.special import betaln, expit, log_ndtr, logsumexp, ndtri, softmax

from bora.evidence import Evidence, normalize, project_simplex
from . import native
from .cycle_two import colour_nodes, tile_bags
from .cycle_three import prototype_system, uncertain_prototype
from .methods import correct_coarse, fit_linear, linear_prob, palette_active_set, shifted


def beta_prototype(c, palette, he):
    _, _, means, covariance = fit_linear(c)
    observed = np.clip(c.rgb, .5/255, 1-.5/255)
    scores, clipped = [], []
    for g in range(c.k):
        matrix, rhs = prototype_system(c, g)
        rhs[..., 0, :] += .015*means[g]
        local = np.linalg.solve(matrix, rhs)[..., 0, :]
        mean = np.clip(local, .5/255, 1-.5/255)
        inflation = 1+.015*np.linalg.inv(matrix)[..., 0, 0]
        variance = np.maximum(inflation[..., None]*np.diag(covariance), 1e-6)
        precision = mean*(1-mean)/variance-1
        clipped.append(float(np.mean(precision < .05)))
        precision = np.clip(precision, .05, 10000)
        a, b = mean*precision, (1-mean)*precision
        scores.append(np.sum((a-1)*np.log(observed)+(b-1)*np.log1p(-observed)-betaln(a, b), -1))
    return softmax(np.stack(scores, -1), -1), dict(
        observation_model="independent bounded-channel Beta densities",
        variance_projection_fraction=clipped, local_mean_radius_um=48,
        caveat="Moment-match the parent predictive variance into the feasible Beta family; not an exact latent regression integral.")


def fit_lad(design, value, weight, slope_penalty=.01):
    """Exact sparse LP for weighted |y-X beta| plus L1 spatial slopes."""
    design, value, weight = map(np.asarray, (design, value, weight))
    n, d = design.shape
    weight = weight/weight.sum()
    slope = np.eye(d)[1:]
    a = sparse.bmat([
        [sparse.csr_matrix(design), -sparse.eye(n), sparse.csr_matrix((n, d-1))],
        [-sparse.csr_matrix(design), -sparse.eye(n), sparse.csr_matrix((n, d-1))],
        [sparse.csr_matrix(slope), sparse.csr_matrix((d-1, n)), -sparse.eye(d-1)],
        [-sparse.csr_matrix(slope), sparse.csr_matrix((d-1, n)), -sparse.eye(d-1)]], format="csr")
    cost = np.r_[np.zeros(d), weight, np.full(d-1, slope_penalty)]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=OptimizeWarning,
                                message="Unrecognized options detected.*")
        fit = linprog(cost, A_ub=a, b_ub=np.r_[value, -value, np.zeros(2*(d-1))],
                      bounds=[(None, None)]*d+[(0, None)]*(n+d-1),
                      method="highs-ds", options=dict(threads=2))
    if not fit.success:
        raise RuntimeError("Spatial LAD fit failed: "+fit.message)
    return fit.x[:d], dict(success=bool(fit.success), iterations=int(fit.nit),
                          objective=float(fit.fun), message=fit.message)


def laplace_spatial(c, palette, he):
    yy, xx = np.indices(c.shape)
    design = np.stack([np.ones(c.shape), 2*xx/max(c.shape[1]-1, 1)-1,
                       2*yy/max(c.shape[0]-1, 1)-1], -1)
    score, reports = [], []
    for g in range(c.k):
        index = np.flatnonzero((c.known & (c.labels == g)).ravel())
        index = index[np.linspace(0, len(index)-1, min(512, len(index)), dtype=int)]
        x, rgb, w = design.reshape(-1, 3)[index], c.rgb.reshape(-1, 3)[index], c.weight.ravel()[index]
        coefficient, scale, fit_report = [], [], []
        for j in range(3):
            beta, report = fit_lad(x, rgb[:, j], w)
            coefficient.append(beta)
            scale.append(max(float(np.average(np.abs(rgb[:, j]-x@beta), weights=w)), .01))
            fit_report.append(report)
        delta = c.rgb-design@np.asarray(coefficient).T
        score.append(-np.sum(np.abs(delta)/scale+np.log(2*np.asarray(scale)), -1))
        reports.append(dict(class_id=int(c.ids[g]), training_points=len(index), fits=fit_report,
                            laplace_scales=scale))
    return softmax(np.stack(score, -1), -1), dict(
        model="class-specific spatial-affine Laplace appearance", fits=reports,
        spatial_slope_L1_penalty=.01, training_limit_per_class=512,
        difference="One global spatial surface per class, not moving-window Gaussian fits.")


def clearance_glass(he, mpp, radius_um=4.):
    white = (he.min(-1) >= 235) & (np.ptp(he.astype(float), axis=-1) <= 18)
    distance = ndi.distance_transform_edt(np.pad(white, 1), sampling=[mpp[1], mpp[0]])[1:-1, 1:-1]
    seeds = distance >= radius_um
    excluded = ndi.binary_propagation(seeds, structure=np.ones((3, 3)), mask=white)
    return excluded, dict(clearance_radius_um=radius_um, white_seed_pixels=int(seeds.sum()),
        reconstruction="entire 8-connected bright component containing an inscribed physical disk",
        caveat="Large genuine gland lumina or adipocytes can also contain such a disk and be removed.")


def clearance_prototype(c, palette, he):
    if he is None:
        raise ValueError("Original aligned H&E is required")
    q, detail = uncertain_prototype(c, palette, he)
    original = cv2.resize(he, c.shape[::-1], interpolation=cv2.INTER_AREA)
    excluded, report = clearance_glass(original, c.mpp)
    detail.update(report, glass_exclusion=excluded,
                  proposed_glass_pixels=int((excluded & c.support).sum()))
    return q, detail


def poisson_inverse(c, palette, he):
    x, prior, index = colour_nodes(c)
    colors = np.maximum(np.asarray(palette, float)[c.ids-1]/255, 1e-5)
    q, z, acceleration = prior.astype(float).copy(), prior.astype(float).copy(), 1.
    # Backtracking on the single convex Poisson observation objective.
    def value(a):
        intensity = np.maximum(a@colors, 1e-12)
        return float(np.sum(intensity-x*np.log(intensity))+.006*np.sum((a-prior)**2))
    lipschitz = 1.
    for iteration in range(120):
        intensity = np.maximum(z@colors, 1e-12)
        gradient = (1-x/intensity)@colors.T+.012*(z-prior)
        old = value(z)
        for _ in range(50):
            new = project_simplex(z-gradient/lipschitz)
            delta = new-z
            if value(new) <= old+float(np.sum(gradient*delta))+.5*lipschitz*np.sum(delta**2)+1e-10:
                break
            lipschitz *= 2
        change = float(np.max(np.abs(new-q)))
        next_acceleration = (1+np.sqrt(1+4*acceleration**2))/2
        z = new+(acceleration-1)/next_acceleration*(new-q)
        # Extrapolation outside the simplex can create negative intensities.
        # Restart there; this changes machinery, not the convex objective.
        if np.any(z@colors <= 0):
            z, next_acceleration = new.copy(), 1.
        q, acceleration = new, next_acceleration
        if change < 1e-6:
            break
    gradient = (1-x/np.maximum(q@colors, 1e-12))@colors.T+.012*(q-prior)
    residual = float(np.max(np.abs(q-project_simplex(q-gradient))))
    return correct_coarse(q[index], c), dict(
        observation="nonnegative Poisson/I-divergence palette mean", iterations=iteration+1,
        projected_gradient_residual=residual, iteration_budget=120,
        objective=value(q), quantization_bins=24,
        caveat="Synthetic RGB is not a photon count; this is a tested observation-loss hypothesis.")


def mass_projection(probability, target, tolerance=1e-7):
    """KL projection with row sums one and prescribed class totals, dual Newton."""
    probability = np.asarray(probability, float)
    target = np.asarray(target, float)
    n, k = probability.shape
    if np.any(target <= 0) or not np.isclose(target.sum(), n):
        raise ValueError("Positive target masses must sum to the number of rows")
    logq = np.log(np.maximum(probability, 1e-12))
    bias = np.zeros(k)
    def state(b):
        q = softmax(logq+b, -1)
        value = float(logsumexp(logq+b, axis=-1).sum()-target@b)
        return value, q
    for iteration in range(50):
        value, q = state(bias)
        gradient = q.sum(0)-target
        error = float(np.max(np.abs(gradient))/n)
        if error < tolerance:
            break
        hessian = np.diag(q.sum(0))-q.T@q
        direction = np.r_[np.linalg.solve(hessian[:-1, :-1]+1e-10*np.eye(k-1), gradient[:-1]), 0.]
        slope, step = float(gradient@direction), 1.
        for _ in range(40):
            candidate = bias-step*direction
            new_value, _ = state(candidate)
            if new_value <= value-1e-4*step*slope:
                break
            step *= .5
        bias = candidate
    _, q = state(bias)
    error = float(np.max(np.abs(q.sum(0)-target))/n)
    return q, dict(iterations=iteration+1, relative_mass_error=error,
                  converged=error < tolerance, dual_bias=bias.tolist())


def area_constrained(c, palette, he):
    q, detail = palette_active_set(c, palette)
    mass = np.bincount(c.labels[c.known], weights=c.weight[c.known], minlength=c.k)
    target = mass/mass.sum()*c.support.sum()
    projected, report = mass_projection(q[c.support], target)
    q[c.support] = projected
    detail.update(report, constraint="known coarse weighted class proportions extrapolated to all tissue",
                  target_class_pixels=target.tolist(),
                  caveat="A tile's winning class does not establish its area; this global constraint may be wrong.")
    return q, detail


def ordinal_palette(c, palette, he):
    x, prior, index = colour_nodes(c)
    colors = np.asarray(palette, float)[c.ids-1]/255
    pairs = [(0, 1), (0, 2), (1, 2)]
    a = np.stack([colors[:, i]-colors[:, j] for i, j in pairs], -1)/.05
    sign = np.stack([np.sign(x[:, i]-x[:, j]) for i, j in pairs], -1)
    def value(q):
        return float(np.sum(np.logaddexp(0, -sign*(q@a)))+.006*np.sum((q-prior)**2))
    def gradient(q):
        margin = sign*(q@a)
        return (-sign*expit(-margin))@a.T+.012*(q-prior)
    q = prior.astype(float).copy()
    history = [value(q)]
    for iteration in range(40):
        grad = gradient(q)
        residual = float(np.max(np.abs(q-project_simplex(q-grad))))
        if residual < 1e-6:
            break
        probability = expit(-sign*(q@a))
        curvature = sign**2*probability*(1-probability)
        hessian = np.einsum("ni,ki,li->nkl", curvature, a, a)+.012*np.eye(c.k)
        linear = np.einsum("nij,nj->ni", hessian, q)-grad
        quadratic, _ = native.quadratic_simplex(hessian, linear, q)
        direction = quadratic-q
        slope, step = float(np.sum(grad*direction)), 1.
        for _ in range(40):
            candidate = q+step*direction
            objective = value(candidate)
            if objective <= history[-1]+1e-4*step*slope+1e-10:
                break
            step *= .5
        q = candidate
        history.append(objective)
    residual = float(np.max(np.abs(q-project_simplex(q-gradient(q)))))
    return correct_coarse(q[index], c), dict(
        observation="three within-pixel channel-order logistic comparisons, amplitudes discarded",
        iterations=iteration+1, iteration_budget=40, projected_gradient_residual=residual,
        converged=residual < 1e-6, objective_history=history,
        solver="feasible damped Newton, checked C++ simplex quadratic subproblems",
        ordering_temperature=.05, quantization_bins=24,
        caveat="Only six strict channel orders exist; classes sharing an order need coarse prior evidence.")


def count_objective(flat, bags, labels, weight, initial, design_effect=1., exact=False):
    k = initial.shape[1]
    coefficient = flat.reshape(4, k)
    p = softmax(bags@coefficient, -1)
    derivative = np.zeros_like(p)
    if exact:
        logp, derivative = native.categorical_winner(p, labels)
        value = -float(logp@weight)
        derivative *= weight[:, None, None]
    else:
        value = 0.
        own = p[np.arange(len(labels)), :, labels]
        for g in range(k):
            other = p[..., g]
            difference = own-other
            mean = difference.sum(1)
            raw_variance = (own+other-difference*difference).sum(1)
            variance = np.maximum(raw_variance, .05)*design_effect
            z = mean/np.sqrt(variance)
            active = labels != g
            logcdf = log_ndtr(z)
            mills = np.exp(np.clip(-.5*z*z-.5*np.log(2*np.pi)-logcdf, -300, 300))
            dm = -mills/np.sqrt(variance)*weight*active
            dv = .5*mills*mean/variance**1.5*design_effect*weight*active*(raw_variance > .05)
            derivative[np.arange(len(labels)), :, labels] += dm[:, None]+dv[:, None]*(1-2*difference)
            derivative[..., g] += -dm[:, None]+dv[:, None]*(1+2*difference)
            value -= float(np.sum(weight*active*logcdf))
    derivative = p*(derivative-(p*derivative).sum(-1, keepdims=True))
    penalty = coefficient-initial
    gradient = np.einsum("nsi,nsk->ik", bags, derivative)+.0004*penalty
    return value+.0002*float(np.sum(penalty**2)), gradient.ravel()


def count_model(c, exact):
    bags, labels = tile_bags(c)
    if not len(bags):
        return linear_prob(c), dict(training_bags=0, fallback="no observed bags")
    coef, intercept, _, _ = fit_linear(c)
    initial = np.c_[coef, intercept].T
    p = softmax(bags@initial, -1)
    means = p.mean(1)
    within = float(np.mean((p-means[:, None])**2))
    residual = np.zeros_like(means)
    for g in range(c.k):
        if np.any(labels == g):
            residual[labels == g] = means[labels == g]-means[labels == g].mean(0)
    between = float(np.mean(residual**2))
    effect = 1. if exact else float(np.clip(bags.shape[1]*between/max(within, 1e-12), 1, bags.shape[1]))
    weight = 1/np.bincount(labels, minlength=c.k)[labels]
    weight /= weight.sum()
    objective = lambda x: count_objective(x, bags, labels, weight, initial, effect, exact)
    fit = minimize(objective, initial.ravel(), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=100, ftol=1e-9, gtol=1e-6))
    coefficient = fit.x.reshape(4, c.k)
    return softmax(native.logits(c.rgb, coefficient[:3].T, coefficient[3]), -1), dict(
        observation="exact pairwise signed categorical count composite likelihood" if exact else "normal count composite likelihood with estimated intrabag design effect",
        training_bags=len(bags), design_effect=effect, effective_pixels_per_bag=25/effect,
        optimizer_success=bool(fit.success), iterations=int(fit.nit), message=str(fit.message),
        gradient_infinity_norm=float(np.max(np.abs(objective(fit.x)[1]))),
        caveat="Exact pairwise comparisons are not the joint multiclass winner probability; neither model establishes independence in actual tissue.")


def correlated_counts(c, palette, he):
    return count_model(c, False)


def exact_counts(c, palette, he):
    return count_model(c, True)


def auc_contrast(c, palette, he):
    bags, labels = tile_bags(c)
    if set(labels) != set(range(c.k)):
        return linear_prob(c), dict(fallback="known class has no complete bag")
    x = bags[..., :3].mean(1)
    coefficient, intercept = [], []
    for g in range(c.k):
        positive, negative = x[labels == g], x[labels != g]
        mean_p, mean_n = positive.mean(0), negative.mean(0)
        a, b = positive-mean_p, negative-mean_n
        difference = mean_p-mean_n
        second = a.T@a/len(a)+b.T@b/len(b)+np.outer(difference, difference)
        beta = np.linalg.solve(second+.0025*np.eye(3), difference)
        coefficient.append(beta)
        intercept.append(-float(beta@(.5*(mean_p+mean_n))))
    return softmax(native.logits(c.rgb, coefficient, intercept), -1), dict(
        objective="one-versus-rest mean squared pairwise ranking loss, ridge 0.0025",
        intercept_identification="equal score at positive/negative mean midpoint",
        training_bags=len(bags), analytic=True,
        caveat="Ranking loss does not identify calibrated multiclass probabilities.")


def bag_statistics(c, transform=None):
    bags, labels = tile_bags(c)
    if not len(bags) or set(labels) != set(range(c.k)):
        return None
    data = bags[..., :3] if transform is None else transform(bags[..., :3])
    means = data.mean(1)
    class_mean = np.asarray([means[labels == g].mean(0) for g in range(c.k)])
    residual = (data-means[:, None]).reshape(-1, 3)
    covariance = residual.T@residual/len(residual)
    return bags, labels, class_mean, residual, covariance


def tyler_scatter(residual, shrinkage=.2):
    residual = np.asarray(residual, float)
    residual = residual[np.linalg.norm(residual, axis=1) > 1e-9]
    shape = np.eye(3)
    if not len(residual):
        return shape, dict(iterations=0, fixed_point_change=0., informative_residuals=0)
    for iteration in range(100):
        d = np.einsum("ni,ij,nj->n", residual, np.linalg.inv(shape), residual)
        update = (1-shrinkage)*3*(residual.T/np.maximum(d, 1e-20))@residual/len(residual)+shrinkage*np.eye(3)
        update *= 3/np.trace(update)
        change = float(np.max(np.abs(update-shape)))
        shape = update
        if change < 1e-7:
            break
    return shape, dict(iterations=iteration+1, fixed_point_change=change,
                      informative_residuals=len(residual), shrinkage=shrinkage)


def angular_covariance(c, palette, he):
    stats = bag_statistics(c)
    if stats is None:
        return linear_prob(c), dict(fallback="missing complete-bag class")
    _, labels, means, residual, covariance = stats
    shape, detail = tyler_scatter(residual)
    covariance = shape*max(float(np.trace(covariance))/3, 1e-6)+.0025*np.eye(3)
    coef = np.linalg.solve(covariance, means.T).T
    detail.update(covariance="regularized Tyler angular shape with empirical common radial scale",
                  training_bags=len(labels))
    return softmax(native.logits(c.rgb, coef, -.5*np.sum(coef*means, -1)), -1), detail


def rank_geometry(c, palette, he):
    grid = (np.arange(96)+.5)/96
    tables = []
    for j in range(3):
        hist, _ = np.histogram(c.rgb[..., j][c.known], bins=96, range=(0, 1), weights=c.weight[c.known])
        hist = hist.astype(float)+1e-5
        cdf = (np.cumsum(hist)-hist/2)/hist.sum()
        tables.append(ndtri(np.clip(cdf, .001, .999)))
    def transform(x):
        return np.stack([np.interp(x[..., j], grid, tables[j]) for j in range(3)], -1)
    stats = bag_statistics(c, transform)
    if stats is None:
        return linear_prob(c), dict(fallback="missing complete-bag class")
    _, labels, means, _, covariance = stats
    covariance = .8*covariance+.2*np.diag(np.diag(covariance))+.0025*np.eye(3)
    coef = np.linalg.solve(covariance, means.T).T
    return softmax(native.logits(transform(c.rgb), coef, -.5*np.sum(coef*means, -1)), -1), dict(
        observation="pooled coarse-weighted marginal normal scores, shared within-bag Gaussian discriminant",
        marginal_bins=96, training_bags=len(labels),
        distinction="One pooled monotone geometry, not per-class marginal/copula density.")


def matched_contrast(c, palette, he):
    stats = bag_statistics(c)
    if stats is None:
        return linear_prob(c), dict(fallback="missing complete-bag class")
    bags, labels, means, _, covariance = stats
    observed = {}
    counter = 0
    for iy, y in enumerate(np.arange(0, c.shape[0], c.tile_y)):
        for ix, x in enumerate(np.arange(0, c.shape[1], c.tile_x)):
            yy = np.clip(np.floor(y+(np.arange(5)+.5)*c.tile_y/5).astype(int), 0, c.shape[0]-1)
            xx = np.clip(np.floor(x+(np.arange(5)+.5)*c.tile_x/5).astype(int), 0, c.shape[1]-1)
            if c.known[yy[:, None], xx[None, :]].mean() >= .5:
                observed[iy, ix] = counter
                counter += 1
    assert counter == len(bags)
    laplacian, rhs, edges = .1*np.eye(c.k), .1*means.copy(), 0
    tile_mean = bags[..., :3].mean(1)
    for (iy, ix), i in observed.items():
        for neighbour in ((iy+1, ix), (iy, ix+1)):
            j = observed.get(neighbour)
            if j is None or labels[i] == labels[j]:
                continue
            direction = np.eye(c.k)[labels[i]]-np.eye(c.k)[labels[j]]
            laplacian += np.outer(direction, direction)
            rhs += direction[:, None]*(tile_mean[i]-tile_mean[j])
            edges += 1
    prototypes = np.linalg.solve(laplacian, rhs)
    covariance = .8*covariance+.2*np.diag(np.diag(covariance))+.0025*np.eye(3)
    coefficient = np.linalg.solve(covariance, prototypes.T).T
    return softmax(native.logits(c.rgb, coefficient, -.5*np.sum(coefficient*prototypes, -1)), -1), dict(
        observation="matched adjacent unlike-label tile colour contrasts",
        matched_edges=edges, anchor_precision=.1,
        scope="Class-contrast normal equations, not spatial label propagation.",
        caveat="True class appearance may vary between boundaries and interiors; disconnected contrasts use coarse-trained anchors.")


def weighted_quantile(value, weight, quantile):
    order = np.argsort(value, kind="stable")
    value, weight = value[order], weight[order]
    position = (np.cumsum(weight)-.5*weight)/weight.sum()
    return np.interp(quantile, position, value)


def sliced_transport(c, palette, he):
    directions = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                             [1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]], float)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    quantile = (np.arange(9)+.5)/9
    offsets = [max(1, round(4/v)) for v in c.mpp]
    energy = np.zeros((*c.shape, c.k))
    for direction in directions:
        projection = c.rgb@direction
        local = np.sort(np.stack([shifted(projection, dy*offsets[1], dx*offsets[0])
                                 for dy, dx in product((-1, 0, 1), repeat=2)], -1), -1)
        for g in range(c.k):
            take = c.known & (c.labels == g)
            target = weighted_quantile(projection[take], c.weight[take], quantile)
            energy[..., g] += np.mean((local-target)**2, -1)/len(directions)
    return softmax(-energy/.01, -1), dict(
        model="seven-projection squared sliced-Wasserstein patch-to-class colour distance",
        patch_samples=9, spacing_um=4, quantiles=9,
        caveat="Finite fixed projections and nine-point empirical patches approximate full distribution transport; coarse class mixtures contaminate target distributions.")


def density_basins(histogram):
    """Acyclic 26-neighbour ascent in colour space, deterministic plateau ties."""
    histogram = np.asarray(histogram, float)
    coords = np.indices(histogram.shape).reshape(3, -1).T
    parent = np.arange(histogram.size)
    flat = histogram.ravel()
    best = flat.copy()
    for delta in product((-1, 0, 1), repeat=3):
        neighbour = np.clip(coords+delta, 0, np.asarray(histogram.shape)-1)
        index = np.ravel_multi_index(neighbour.T, histogram.shape)
        value = flat[index]
        choose = (value > best) | ((value == best) & (index < parent))
        parent[choose], best[choose] = index[choose], value[choose]
    for _ in range(32):
        new = parent[parent]
        if np.array_equal(new, parent):
            return parent.reshape(histogram.shape)
        parent = new
    raise RuntimeError("Colour ascent did not reach its roots")


def modal_colours(c, palette, he):
    bins = 24
    cell = np.minimum((c.rgb*bins).astype(int), bins-1)
    code = (cell[..., 0]*bins+cell[..., 1])*bins+cell[..., 2]
    hist = np.bincount(code[c.support], minlength=bins**3).reshape((bins,)*3)
    roots = density_basins(ndi.gaussian_filter(hist.astype(float), .8, mode="constant"))
    basin = roots.ravel()[code]
    counts = np.stack([np.bincount(basin.ravel(), weights=(c.weight*(c.labels == g)).ravel(),
                                  minlength=bins**3) for g in range(c.k)], -1)
    probability = (counts+.5)/(counts.sum(-1, keepdims=True)+.5*c.k)
    mass = np.bincount(basin[c.support], minlength=bins**3)
    unknown = np.flatnonzero((mass > 0) & (counts.sum(-1) == 0))
    proposals = [dict(colour_mode_rgb=((np.asarray(np.unravel_index(i, (bins,)*3))+.5)/bins).tolist(),
                      pixels=int(mass[i]), area_um2=float(mass[i]*c.mpp.prod()), semantic_identity="unknown")
                 for i in sorted(unknown, key=lambda x: -mass[x])]
    unsupported = (counts.sum(-1)[basin] == 0) & c.support
    components, count = ndi.label(unsupported, np.ones((3, 3)))
    areas = np.bincount(components.ravel())[1:]*c.mpp.prod()
    return probability[basin], dict(
        model="shared full-tissue colour-density modal basins with coarse-only Dirichlet class counts",
        occupied_basins=int((mass > 0).sum()), unsupported_colour_modes=proposals,
        unsupported_spatial_components=count, unsupported_component_areas_um2=areas.tolist(),
        unsupported_pixels=int(mass[unknown].sum()), semantic_new_classes_assigned=False,
        caveat="An unsupported colour basin supplies no semantic class evidence; equal Dirichlet counts tie and the display uses the lowest known class ID.")


def legendre_features(rgb):
    z = 2*np.asarray(rgb)-1
    basis = np.stack([np.ones_like(z), z, .5*(3*z*z-1)], -1)
    return np.stack([basis[..., 0, i]*basis[..., 1, j]*basis[..., 2, k]
                     for i, j, k in product(range(3), repeat=3) if (i, j, k) != (0, 0, 0)], -1)


def maxent_objective(theta, features, moment, ridge=.02):
    logits = features@theta
    logz = logsumexp(logits)-np.log(len(features))
    probability = softmax(logits)
    return (float(logz-moment@theta+.5*ridge*np.sum(theta*theta)),
            features.T@probability-moment+ridge*theta)


def maximum_entropy(c, palette, he):
    axis = (np.arange(12)+.5)/12
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    basis = legendre_features(grid)
    query = legendre_features(c.rgb).astype(np.float32)
    coefficient, intercept, report = [], [], []
    for g in range(c.k):
        take = c.known & (c.labels == g)
        moment = np.average(query[take], axis=0, weights=c.weight[take])
        objective = lambda v: maxent_objective(v, basis, moment)
        fit = minimize(objective, np.zeros(26), jac=True, method="L-BFGS-B",
                       options=dict(maxiter=100, ftol=1e-10, gtol=1e-7))
        logz = float(logsumexp(basis@fit.x)-np.log(len(basis)))
        coefficient.append(fit.x)
        intercept.append(-logz)
        report.append(dict(class_id=int(c.ids[g]), success=bool(fit.success),
                           iterations=int(fit.nit), message=str(fit.message),
                           gradient_infinity_norm=float(np.max(np.abs(objective(fit.x)[1])))))
    return softmax(native.logits(query, coefficient, intercept), -1), dict(
        model="joint maximum-entropy RGB density with tensor degree-two Legendre moments",
        features=26, normalization_quadrature=12**3, fits=report, ridge=.02,
        caveat="Normalization uses midpoint cubature on the unit RGB cube; this is not an exact continuous integral.")


IMPLEMENTATIONS = dict(G01=beta_prototype, G02=laplace_spatial, G03=clearance_prototype,
    G04=poisson_inverse, G05=area_constrained, G06=ordinal_palette,
    G07=correlated_counts, G08=exact_counts, G09=auc_contrast,
    G10=angular_covariance, G11=rank_geometry, G12=matched_contrast,
    G13=sliced_transport, G14=modal_colours, G15=maximum_entropy)


def predict(key, rgb, coarse, support, mpp_xy, palette, he=None):
    from .registry import CYCLE_FOUR
    if key not in IMPLEMENTATIONS:
        raise ValueError("Unimplemented cycle-four child")
    c = Evidence(rgb, coarse, support, mpp_xy)
    if not c.support.any():
        raise ValueError("Empty tissue support")
    if c.k == 1 and key != "G03":
        q, detail = np.ones((*c.shape, 1), np.float32), dict(single_class=True)
    else:
        q, detail = IMPLEMENTATIONS[key](c, palette, he)
    excluded = detail.pop("glass_exclusion", None)
    if q.shape != (*c.shape, c.k) or not np.isfinite(q).all() or np.min(q) < -1e-6:
        raise ValueError("Invalid child posterior")
    labels = c.ids[q.argmax(-1)].astype(np.uint8)
    if excluded is not None:
        labels[excluded] = 0
    detail.update(analysis_mpp_xy=c.mpp.tolist(), analysis_shape_yx=list(c.shape),
        fine_reference_used=False, hard_anchors=False, fixed_margin=False, candidate=key,
        parent=CYCLE_FOUR[key].parent, mechanism=CYCLE_FOUR[key].mechanism)
    return labels, detail
