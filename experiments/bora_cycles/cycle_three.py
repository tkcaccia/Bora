"""Cycle three: fifteen distinct, coarse-only interventions on fixed RGB.

No reference I/O, encoder calls, solver ensembles or input modifications.
Shared numerical routines are machinery, not extra experimental candidates.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from scipy.special import logsumexp, ndtri, softmax

from bora.evidence import Evidence, normalize, project_simplex
from . import native
from .cycle_two import local_affine, tile_bags
from .methods import correct_coarse, fit_linear


def rectangle_sum(a, y0, y1, x0, x1):
    """Clipped rectangle sums, with arbitrary trailing dimensions."""
    integral = np.pad(a.cumsum(0, dtype=float).cumsum(1),
                      [(1, 0), (1, 0)]+[(0, 0)]*(a.ndim-2))
    return integral[y1, x1]-integral[y0, x1]-integral[y1, x0]+integral[y0, x0]


def prototype_system(c, g, exclude_tile=False):
    """Local affine sufficient statistics, optionally excluding the query tile."""
    radius, ridge = 48., .015
    yy, xx = np.indices(c.shape)
    x, y = xx*c.mpp[0]/radius, yy*c.mpp[1]/radius
    rx, ry = [max(1, round(radius/v)) for v in c.mpp]
    w = c.weight*(c.labels == g)
    raw = np.stack([w, w*x, w*y, w*x*x, w*x*y, w*y*y]+[
        w*v*c.rgb[..., j] for v in (np.ones(c.shape), x, y) for j in range(3)], -1)
    stats = cv2.boxFilter(raw, cv2.CV_64F, (2*rx+1, 2*ry+1), normalize=True,
                          borderType=cv2.BORDER_CONSTANT if exclude_tile else cv2.BORDER_REFLECT)
    if exclude_tile:
        # Match the tile geometry used by tile_bags: floor starting coordinate,
        # with half-open integer-pixel intervals. The full own tile is excluded,
        # not merely the query pixel or a fixed boundary margin.
        bx, by = np.floor(xx/c.tile_x), np.floor(yy/c.tile_y)
        x0 = np.maximum(xx-rx, np.ceil(bx*c.tile_x).astype(int)).clip(0, c.shape[1])
        x1 = np.minimum(xx+rx+1, np.ceil((bx+1)*c.tile_x).astype(int)).clip(0, c.shape[1])
        y0 = np.maximum(yy-ry, np.ceil(by*c.tile_y).astype(int)).clip(0, c.shape[0])
        y1 = np.minimum(yy+ry+1, np.ceil((by+1)*c.tile_y).astype(int)).clip(0, c.shape[0])
        stats -= rectangle_sum(raw, y0, y1, x0, x1)/((2*rx+1)*(2*ry+1))
    m, mx, my, mxx, mxy, myy = np.moveaxis(stats[..., :6], -1, 0)
    matrix = np.empty((*c.shape, 3, 3))
    matrix[..., 0, 0] = m
    matrix[..., 0, 1] = matrix[..., 1, 0] = mx-x*m
    matrix[..., 0, 2] = matrix[..., 2, 0] = my-y*m
    matrix[..., 1, 1] = mxx-2*x*mx+x*x*m
    matrix[..., 2, 2] = myy-2*y*my+y*y*m
    matrix[..., 1, 2] = matrix[..., 2, 1] = mxy-x*my-y*mx+x*y*m
    matrix += ridge*np.eye(3)
    b0, bx, by = stats[..., 6:9], stats[..., 9:12], stats[..., 12:15]
    rhs = np.stack([b0, bx-x[..., None]*b0, by-y[..., None]*b0], -2)
    return matrix, rhs


def uncertain_prototype(c, palette, he):
    _, _, means, covariance = fit_linear(c)
    inv = np.linalg.inv(covariance)
    scores, inflation_range = [], []
    for g in range(c.k):
        matrix, rhs = prototype_system(c, g)
        rhs[..., 0, :] += .015*means[g]
        local = np.linalg.solve(matrix, rhs)[..., 0, :]
        # Gaussian regression posterior predictive: integrate the uncertain
        # intercept instead of treating a sparse extrapolation as certain.
        inflation = 1+.015*np.linalg.inv(matrix)[..., 0, 0]
        delta = c.rgb-local
        d = np.einsum("...i,ij,...j->...", delta, inv, delta)
        scores.append(-.5*(d/inflation+3*np.log(inflation)))
        inflation_range.append([float(inflation.min()), float(inflation.max())])
    return softmax(np.stack(scores, -1), -1), dict(
        posterior="Gaussian integrated intercept", predictive_variance_factor=inflation_range,
        window_radius_um=48, prior_precision=.015)


def honest_prototype(c, palette, he):
    _, _, means, covariance = fit_linear(c)
    inv, scores = np.linalg.inv(covariance), []
    for g in range(c.k):
        matrix, rhs = prototype_system(c, g, exclude_tile=True)
        rhs[..., 0, :] += .015*means[g]
        local = np.linalg.solve(matrix, rhs)[..., 0, :]
        delta = c.rgb-local
        scores.append(-.5*np.einsum("...i,ij,...j->...", delta, inv, delta))
    return softmax(np.stack(scores, -1), -1), dict(
        local_training="exclude the entire query coarse tile", global_prior="all known coarse tissue",
        radius_um=48, boundary="zero extension, with exact clipped tile subtraction",
        caveat="Global prior still uses the query tile; only the local fit is held out.")


def cleft_components(he, mpp):
    """A region shape hypothesis; brightness alone never deletes a region."""
    white = (he.min(-1) >= 235) & (np.ptp(he.astype(float), axis=-1) <= 18)
    components, n = ndi.label(white, np.ones((3, 3)))
    yy, xx = np.indices(white.shape)
    x, y = xx*mpp[0], yy*mpp[1]
    index = components.ravel()
    count = np.bincount(index, minlength=n+1)
    mean = [np.bincount(index, weights=v.ravel(), minlength=n+1)/np.maximum(count, 1)
            for v in (x, y, x*x, x*y, y*y)]
    covariance = np.empty((n+1, 2, 2))
    covariance[:, 0, 0] = np.maximum(mean[2]-mean[0]**2, 0)
    covariance[:, 1, 1] = np.maximum(mean[4]-mean[1]**2, 0)
    covariance[:, 0, 1] = covariance[:, 1, 0] = mean[3]-mean[0]*mean[1]
    eig = np.maximum(np.linalg.eigvalsh(covariance), 0)
    length = np.sqrt(12*eig[:, 1])
    ratio = np.sqrt((eig[:, 1]+.25)/(eig[:, 0]+.25))
    area = count*np.prod(mpp)
    width = area/np.maximum(length, 1e-8)
    accepted = (area >= 20) & (length >= 16) & (ratio >= 4) & (width <= 10)
    accepted[0] = False
    return accepted[components], dict(white_components=n, cleft_components=int(accepted.sum()),
        min_area_um2=20, min_length_um=16, min_axis_ratio=4, max_effective_width_um=10,
        caveat="Elongated gland lumina can be mistaken for tears; broad round cavities are not removed.")


def geometric_cleft(c, palette, he):
    if he is None:
        raise ValueError("Aligned original H&E is required for the cleft hypothesis")
    q, detail = local_affine(c, palette, he)
    original = cv2.resize(he, c.shape[::-1], interpolation=cv2.INTER_AREA)
    excluded, shape = cleft_components(original, c.mpp)
    detail.update(shape, glass_exclusion=excluded,
                  proposed_glass_pixels=int((excluded & c.support).sum()),
                  mechanism_scope="component geometry, independent of annotation and border connectivity")
    return q, detail


def blur_inverse(c, palette, he):
    colors = np.asarray(palette, float)[c.ids-1]/255
    prior = c.lattice().astype(float)
    sigma = (2/c.mpp[1], 2/c.mpp[0], 0)
    def blur(v):
        return ndi.gaussian_filter(v, sigma, mode="reflect")
    def gradient(q):
        return blur(blur(q@colors)-c.rgb)@colors.T+.012*(q-prior)
    q, z, acceleration = prior.copy(), prior.copy(), 1.
    step = .99/(np.linalg.norm(colors, 2)**2+.012)
    for iteration in range(160):
        new = project_simplex(z-step*gradient(z))
        change = float(np.max(np.abs(new-q)))
        next_acceleration = (1+np.sqrt(1+4*acceleration**2))/2
        z = new+(acceleration-1)/next_acceleration*(new-q)
        q, acceleration = new, next_acceleration
        if change < 2e-5:
            break
    residual = float(np.max(np.abs(q-project_simplex(q-step*gradient(q))))/step)
    error = blur(q@colors)-c.rgb
    return normalize(q), dict(forward_blur_sigma_um=2, iterations=iteration+1,
        projected_gradient_residual=residual, iteration_budget=160,
        objective=float(.5*np.sum(error**2)+.006*np.sum((q-prior)**2)),
        additive_coarse_correction=False, observation="RGB after a symmetric Gaussian point-spread operator")


def calibrated_palette(c, palette, he):
    colors = np.asarray(palette, float)[c.ids-1]/255
    _, _, means, _ = fit_linear(c)
    design = np.c_[colors, np.ones(c.k)]
    delta = np.linalg.solve(design.T@design+.05*np.eye(4), design.T@(means-colors))
    observed = np.clip(colors+design@delta, 0, 1)
    q, detail = native.simplex_qp(c.rgb, c.lattice(), observed)
    detail.update(sensor_residual_affine=delta.tolist(), fitted_palette=observed.tolist(),
                  calibration_prior_precision=.05, conversion_bytes_unchanged=True)
    return correct_coarse(q, c), detail


def tessellated_weights(x, colors):
    """Piecewise affine decoding on a nonoverlapping intrinsic colour mesh."""
    colors = np.asarray(colors, float)
    unique, inverse = np.unique(colors, axis=0, return_inverse=True)
    center = unique.mean(0)
    _, singular, vh = np.linalg.svd(unique-center, full_matrices=False)
    rank = int(np.sum(singular > 1e-8))
    q = np.zeros((len(x), len(unique)))
    outside = np.zeros(len(x), bool)
    if rank == 0:
        q[:] = 1
    else:
        coordinates = (unique-center)@vh[:rank].T
        query = (x-center)@vh[:rank].T
        if rank == 1:
            order = np.argsort(coordinates[:, 0])
            grid = coordinates[order, 0]
            at = np.searchsorted(grid, query[:, 0]).clip(1, len(grid)-1)
            t = np.clip((query[:, 0]-grid[at-1])/(grid[at]-grid[at-1]), 0, 1)
            q[np.arange(len(x)), order[at-1]] = 1-t
            q[np.arange(len(x)), order[at]] = t
            outside = (query[:, 0] < grid[0]) | (query[:, 0] > grid[-1])
        else:
            triangulation = Delaunay(coordinates)
            simplex = triangulation.find_simplex(query)
            inside = simplex >= 0
            transforms = triangulation.transform[simplex[inside]]
            bary = np.einsum("nij,nj->ni", transforms[:, :rank], query[inside]-transforms[:, rank])
            bary = np.c_[bary, 1-bary.sum(1)]
            q[np.flatnonzero(inside)[:, None], triangulation.simplices[simplex[inside]]] = bary
            outside = ~inside
            nearest = cKDTree(coordinates).query(query[outside])[1]
            q[np.flatnonzero(outside), nearest] = 1
    # A duplicated palette vertex supplies no evidence to distinguish its IDs.
    fractions = q[:, inverse]/np.bincount(inverse)[inverse]
    return normalize(fractions), dict(intrinsic_rank=rank, unique_palette_vertices=len(unique),
        outside_hull_fraction=float(outside.mean()), outside_policy="nearest palette vertex")


def palette_mesh(c, palette, he):
    q, detail = tessellated_weights(c.rgb.reshape(-1, 3), np.asarray(palette)[c.ids-1]/255)
    return correct_coarse(q.reshape(*c.shape, c.k), c), detail


def balanced(labels, k):
    weight = 1/np.maximum(np.bincount(labels, minlength=k)[labels], 1)
    return weight/weight.sum()


def mean_logit_objective(theta, x, y, weight, ridge=.005, hessian=False):
    p = softmax(x@theta, -1)
    loss = -float(np.sum(weight*np.log(np.maximum(p[np.arange(len(y)), y], 1e-300))))
    target = np.eye(theta.shape[1])[y]
    gradient = x.T@(weight[:, None]*(p-target))+ridge*theta
    value = loss+.5*ridge*np.sum(theta*theta)
    if not hessian:
        return value, gradient.ravel()
    s = np.einsum("na,ab->nab", p, np.eye(theta.shape[1]))-p[:, :, None]*p[:, None, :]
    h = np.einsum("n,ni,nj,nab->iajb", weight, x, x, s).reshape(theta.size, theta.size)
    return value, gradient.ravel(), h+ridge*np.eye(theta.size)


def mean_logit(c, palette, he):
    bags, y = tile_bags(c)
    if not len(bags):
        from .methods import linear_prob
        return linear_prob(c), dict(training_bags=0, fallback="no observed tile")
    scale = np.maximum(bags[..., :3].std((0, 1)), .05)
    x = np.c_[bags[..., :3].mean(1)/scale, np.ones(len(bags))]
    weight = balanced(y, c.k)
    theta = np.zeros((4, c.k))
    converged = False
    for iteration in range(70):
        value, gradient, hessian = mean_logit_objective(theta, x, y, weight, hessian=True)
        if np.max(np.abs(gradient)) < 1e-7:
            converged = True
            break
        direction = np.linalg.solve(hessian, gradient).reshape(theta.shape)
        step = 1.
        for _ in range(30):
            new = theta-step*direction
            if mean_logit_objective(new, x, y, weight)[0] <= value-1e-4*step*(gradient@direction.ravel()):
                break
            step *= .5
        theta = new
    residual = float(np.max(np.abs(mean_logit_objective(theta, x, y, weight)[1])))
    return softmax(native.logits(c.rgb, (theta[:3]/scale[:, None]).T, theta[3]), -1), dict(
        training_bags=len(y), observation="cross entropy of mean pixel logits",
        iterations=iteration+1, optimizer_success=converged, gradient_infinity_norm=residual,
        convex=True, ridge=.005)


def witness_objective(flat, x, y, weight, k):
    theta = flat.reshape(x.shape[-1], k)
    logits = x@theta
    pooled = logsumexp(logits, axis=1)-np.log(x.shape[1])
    p = softmax(pooled, -1)
    value = -np.sum(weight*np.log(np.maximum(p[np.arange(len(y)), y], 1e-300)))
    d = weight[:, None]*(p-np.eye(k)[y])
    derivative = softmax(logits, axis=1)*d[:, None, :]
    gradient = np.einsum("nsi,nsk->ik", x, derivative)+.005*theta
    return float(value+.0025*np.sum(theta**2)), gradient.ravel()


def witness_tiles(c, palette, he):
    bags, y = tile_bags(c)
    if not len(bags):
        from .methods import linear_prob
        return linear_prob(c), dict(training_bags=0, fallback="no observed tile")
    scale = np.maximum(bags[..., :3].std((0, 1)), .05)
    x = bags.copy()
    x[..., :3] /= scale
    fit = minimize(witness_objective, np.zeros(4*c.k), args=(x, y, balanced(y, c.k), c.k),
                   jac=True, method="L-BFGS-B", options=dict(maxiter=150, gtol=1e-6, ftol=1e-11))
    theta = fit.x.reshape(4, c.k)
    return softmax(native.logits(c.rgb, (theta[:3]/scale[:, None]).T, theta[3]), -1), dict(
        training_bags=len(y), observation="log-mean-exp class witness, then bag cross entropy",
        optimizer_success=bool(fit.success), termination=str(fit.message), iterations=int(fit.nit),
        gradient_infinity_norm=float(np.max(np.abs(fit.jac))), objective=float(fit.fun))


def kernel_bags(c, palette, he):
    bags, y = tile_bags(c)
    if not len(bags):
        from .methods import linear_prob
        return linear_prob(c), dict(training_bags=0, fallback="no observed tile")
    rng = np.random.default_rng(309)
    omega, phase = rng.normal(size=(3, 48))/.25, rng.uniform(0, 2*np.pi, 48)
    def lift(x):
        return np.concatenate([np.ones((*x.shape[:-1], 1)), np.sqrt(2/48)*np.cos(x@omega+phase)], -1)
    x = lift(bags[..., :3]).mean(1)
    weight = balanced(y, c.k)
    coefficient = np.linalg.solve((x.T*weight)@x+.01*np.eye(49), (x.T*weight)@np.eye(c.k)[y])
    flat = c.rgb.reshape(-1, 3)
    q = np.empty((len(flat), c.k), np.float32)
    for start in range(0, len(flat), 16384):
        values = lift(flat[start:start+16384])
        q[start:start+16384] = softmax(native.logits(values, (4*coefficient).T, np.zeros(c.k)), -1)
    return q.reshape(*c.shape, c.k), dict(training_bags=len(y), kernel="Gaussian random Fourier mean embedding",
        kernel_bandwidth=.25, features=48, seed=309, ridge=.01, objective="bag label squared error")


def bag_statistics(c):
    bags, labels = tile_bags(c)
    if not len(bags) or set(labels) != set(range(c.k)):
        return None
    means = bags[..., :3].mean(1)
    centroids = np.array([means[labels == g].mean(0) for g in range(c.k)])
    residual = bags[..., :3]-means[:, None]
    within = np.einsum("nsi,nsj->ij", residual, residual)/(len(bags)*25)
    return bags, labels, means, centroids, within


def bag_fallback(c):
    from .methods import linear_prob
    return linear_prob(c), dict(missing_known_bag_class=True, fallback="original pixel discriminant")


def increment_covariance(c, palette, he):
    stats = bag_statistics(c)
    if stats is None:
        return bag_fallback(c)
    _, _, _, means, _ = stats
    yy, xx = np.indices(c.shape)
    tile = (np.floor(yy/c.tile_y)*np.ceil(c.shape[1]/c.tile_x)+np.floor(xx/c.tile_x)).astype(int)
    covariance, count = np.zeros((3, 3)), 0
    for axis in (0, 1):
        a, b = ([slice(None), slice(None)] for _ in range(2))
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        take = c.known[a] & c.known[b] & (tile[a] == tile[b])
        delta = (c.rgb[a]-c.rgb[b])[take]
        covariance += delta.T.astype(float)@delta/2
        count += len(delta)
    covariance /= max(count, 1)
    covariance = .8*covariance+.2*np.diag(np.diag(covariance))+.0025*np.eye(3)
    coefficient = np.linalg.solve(covariance, means.T).T
    intercept = -.5*np.sum(coefficient*means, -1)
    return softmax(native.logits(c.rgb, coefficient, intercept), -1), dict(
        covariance="half second moment of adjacent within-tile colour increments", pairs=count,
        increment_spacing_xy_um=c.mpp.tolist(), model="independent equal-variance pixel noise")


def random_intercept(c, palette, he):
    stats = bag_statistics(c)
    if stats is None:
        return bag_fallback(c)
    _, labels, bag_mean, means, within = stats
    scores, rank = [], []
    for g in range(c.k):
        x = bag_mean[labels == g]-means[g]
        between = x.T@x/max(len(x)-1, 1)
        values, vectors = np.linalg.eigh(between-within/25)
        signal = (vectors*np.maximum(values, 0))@vectors.T
        covariance = within+signal
        covariance = .8*covariance+.2*np.diag(np.diag(covariance))+.0025*np.eye(3)
        delta = c.rgb-means[g]
        score = -.5*np.einsum("...i,ij,...j->...", delta, np.linalg.inv(covariance), delta)
        scores.append(score-.5*np.linalg.slogdet(covariance)[1])
        rank.append(int((values > 0).sum()))
    return softmax(np.stack(scores, -1), -1), dict(
        signal_covariance_rank=rank, noise_subtraction="within-tile covariance / 25",
        model="class-specific random tile intercept plus common pixel noise")


def margin_objective(flat, x, y, weight, k):
    theta = flat.reshape(x.shape[1], k)
    score = x@theta
    gap = np.maximum(0, 1+score-score[np.arange(len(y)), y, None])
    gap[np.arange(len(y)), y] = 0
    # A sum of squared rival hinges has a continuous gradient even when rival
    # scores tie. It is one joint multiclass classifier, not coupled binary fits.
    derivative = 2*weight[:, None]*gap/(k-1)
    derivative[np.arange(len(y)), y] = -derivative.sum(-1)
    return float(np.sum(weight[:, None]*gap**2)/(k-1)+.0025*np.sum(theta*theta)), (x.T@derivative+.005*theta).ravel()


def joint_margin(c, palette, he):
    stats = bag_statistics(c)
    if stats is None:
        return bag_fallback(c)
    _, labels, means, _, within = stats
    covariance = .8*within+.2*np.diag(np.diag(within))+.0025*np.eye(3)
    values, vectors = np.linalg.eigh(covariance)
    whitening = (vectors/np.sqrt(values))@vectors.T
    x = np.c_[means@whitening, np.ones(len(means))]
    fit = minimize(margin_objective, np.zeros(4*c.k), args=(x, labels, balanced(labels, c.k), c.k),
                   jac=True, method="L-BFGS-B", options=dict(maxiter=150, gtol=1e-6, ftol=1e-11))
    theta = fit.x.reshape(4, c.k)
    return softmax(native.logits(c.rgb, (whitening@theta[:3]).T, theta[3]), -1), dict(
        observation="joint multiclass mean squared all-rival hinge on whitened tile means",
        training_bags=len(labels), optimizer_success=bool(fit.success), iterations=int(fit.nit),
        termination=str(fit.message), objective=float(fit.fun),
        gradient_infinity_norm=float(np.max(np.abs(fit.jac))), convex=True)


def energy_potential(hist):
    """Expected Euclidean colour distance and half self-distance."""
    bins = hist.shape[0]
    z = np.arange(1-bins, bins)/bins
    kernel = np.sqrt(z[:, None, None]**2+z[None, :, None]**2+z[None, None, :]**2)
    potential = fftconvolve(hist, kernel, mode="same")
    return potential, .5*float(np.sum(hist*potential))


def distribution_energy(c, palette, he):
    bins, scores = 24, []
    coordinate = np.moveaxis(np.clip(c.rgb*bins-.5, 0, bins-1), -1, 0)
    for g in range(c.k):
        take = c.known & (c.labels == g)
        hist, _ = np.histogramdd(c.rgb[take], bins=[bins]*3, range=[(0, 1)]*3, weights=c.weight[take])
        hist = ndi.gaussian_filter(hist, .7, mode="constant")
        hist /= hist.sum()
        potential, self_distance = energy_potential(hist)
        scores.append((self_distance-ndi.map_coordinates(potential, coordinate, order=1, mode="nearest"))/.05)
    return softmax(np.stack(scores, -1), -1), dict(
        bins=24, score="negative energy score: half class self-distance minus expected query distance",
        distance="Euclidean fixed RGB", histogram_sigma_bins=.7)


def support_depth(query, points):
    unique = np.unique(points, axis=0)
    center = unique.mean(0)
    _, s, vh = np.linalg.svd(unique-center, full_matrices=False)
    rank = int((s > 1e-7).sum())
    if rank == 0:
        return -np.linalg.norm(query-center, axis=-1), dict(rank=0, facets=0)
    basis = vh[:rank].T
    projection = (unique-center)@basis
    x = (query-center)@basis
    residual = np.linalg.norm(query-center-x@basis.T, axis=-1)
    if rank == 1:
        depth = np.minimum(x[:, 0]-projection.min(), projection.max()-x[:, 0])
        facets = 2
    else:
        hull = ConvexHull(projection)
        planes = hull.equations.copy()
        planes /= np.linalg.norm(planes[:, :-1], axis=-1)[:, None]
        depth = native.hull_depth(x, planes)
        facets = len(planes)
    return depth-residual, dict(rank=rank, facets=facets)


def colour_support(c, palette, he):
    # Colour set geometry, not patch reconstruction, region snapping or a
    # probability density: choose the greatest signed hull depth.
    from .methods import training
    x, y, _ = training(c, c.rgb, limit=6000)
    flat = c.rgb.reshape(-1, 3)
    scores, geometry = [], []
    for g in range(c.k):
        depth, detail = support_depth(flat, x[y == g])
        scores.append(depth/.05)
        geometry.append(detail)
    return softmax(np.stack(scores, -1), -1).reshape(*c.shape, c.k), dict(
        geometry=geometry, score="signed distance to nearest supporting plane minus orthogonal residual",
        cpp_openmp_facets=True, training_limit=6000)


def copula_log_density(c, weight):
    bins, scores = 48, []
    grid = (np.arange(bins)+.5)/bins
    for g in range(c.k):
        take = weight*(c.labels == g) > 0
        ws = weight[take]
        z, score = [], np.zeros(c.shape)
        for j in range(3):
            hist, _ = np.histogram(c.rgb[..., j][take], bins=bins, range=(0, 1), weights=ws)
            hist = ndi.gaussian_filter1d(hist.astype(float), .7)
            hist += max(hist.sum(), 1)*1e-5
            hist /= hist.sum()
            cdf = np.cumsum(hist)-hist/2
            value = c.rgb[..., j]
            u = np.interp(value.ravel(), grid, cdf).reshape(c.shape)
            z.append(ndtri(np.clip(u, .001, .999)))
            density = np.interp(value.ravel(), grid, hist*bins).reshape(c.shape)
            score += np.log(np.maximum(density, 1e-12))
        z = np.stack(z, -1)
        train = z[take]
        centered = train-np.average(train, axis=0, weights=ws)
        cov = (centered.T*ws)@centered/ws.sum()
        std = np.maximum(np.sqrt(np.diag(cov)), .05)
        correlation = .7*cov/std[:, None]/std[None, :]+.3*np.eye(3)
        score -= .5*np.einsum("...i,ij,...j->...", z, np.linalg.inv(correlation)-np.eye(3), z)
        score -= .5*np.linalg.slogdet(correlation)[1]
        scores.append(score)
    return np.stack(scores, -1)


def typicality(c, palette, he):
    yy, xx = np.indices(c.shape)
    parity = (np.floor(yy/c.tile_y)+np.floor(xx/c.tile_x)).astype(int) % 2
    train = c.known & (parity == 0)
    calibration = c.known & ~train
    fallback = []
    for g in range(c.k):
        take = c.known & (c.labels == g)
        if not np.any(train & take) or not np.any(calibration & take):
            index = np.flatnonzero(take)
            train.ravel()[index] = False
            calibration.ravel()[index] = False
            if len(index) == 1:
                train.ravel()[index] = calibration.ravel()[index] = True
            else:
                train.ravel()[index[::2]] = True
                calibration.ravel()[index[1::2]] = True
            fallback.append(int(c.ids[g]))
    density = copula_log_density(c, c.weight*train)
    p, sizes = np.empty_like(density), []
    for g in range(c.k):
        values = np.sort(density[..., g][calibration & (c.labels == g)])
        p[..., g] = (1+np.searchsorted(values, density[..., g], side="right"))/(1+len(values))
        sizes.append(len(values))
    unusual = (p.max(-1) < .05) & c.support
    components, count = ndi.label(unusual, np.ones((3, 3)))
    areas = np.bincount(components.ravel())[1:]*c.mpp.prod()
    return normalize(p), dict(calibration_counts=sizes, pixel_split_fallback_class_ids=fallback,
        all_classes_atypical_pixels=int(unusual.sum()), atypical_component_areas_um2=areas.tolist(),
        atypical_components=count, semantic_new_classes_assigned=False,
        caveat="Spatially dependent, weakly labelled observations: descriptive typicality, no conformal coverage guarantee.")


IMPLEMENTATIONS = dict(F01=uncertain_prototype, F02=honest_prototype, F03=geometric_cleft,
    F04=blur_inverse, F05=calibrated_palette, F06=palette_mesh, F07=mean_logit,
    F08=witness_tiles, F09=kernel_bags, F10=increment_covariance, F11=random_intercept,
    F12=joint_margin, F13=distribution_energy, F14=colour_support, F15=typicality)


def predict(key, rgb, coarse, support, mpp_xy, palette, he=None):
    from .registry import CYCLE_THREE
    if key not in IMPLEMENTATIONS:
        raise ValueError("Unimplemented cycle-three child")
    c = Evidence(rgb, coarse, support, mpp_xy)
    if not c.support.any():
        raise ValueError("Empty tissue support")
    if c.k == 1 and key != "F03":
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
        parent=CYCLE_THREE[key].parent, mechanism=CYCLE_THREE[key].mechanism)
    return labels, detail
