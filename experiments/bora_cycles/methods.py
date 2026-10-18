"""Single-intervention candidates; no fine-reference I/O, no solver ensembles."""
import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import minimize, linprog, OptimizeWarning
import warnings
from scipy.special import softmax, log_ndtr, ndtri
from scipy import sparse
from scipy.sparse.linalg import spsolve
from bora.evidence import Evidence, amortized, normalize
from . import native
from .registry import CYCLE_ONE
from .diagnostics import white_regions


def shifted(x, dy, dx):
    h, w = x.shape[:2]
    return x[np.clip(np.arange(h)+dy, 0, h-1)[:, None],
             np.clip(np.arange(w)+dx, 0, w-1)[None, :]]


def training(c, features, weight=None, limit=16000):
    weight = c.weight if weight is None else weight
    x, y, ws = [], [], []
    for i in range(c.k):
        indices = np.flatnonzero(c.known.ravel() & (c.labels.ravel() == i)
                                 & (weight.ravel() > 0))
        if len(indices) > max(1, limit//c.k):
            indices = indices[np.linspace(0, len(indices)-1,
                                          max(1, limit//c.k), dtype=int)]
        if not len(indices):
            raise ValueError("Class has no training evidence")
        x.append(features.reshape(-1, features.shape[-1])[indices])
        y.append(np.full(len(indices), i))
        ws.append(weight.ravel()[indices])
    return np.concatenate(x), np.concatenate(y), np.concatenate(ws)


def fit_linear(c, features=None, weight=None):
    features = c.rgb if features is None else features
    x, labels, weights = training(c, features, weight)
    scale = np.maximum(np.std(x, axis=0), .05)
    x = x/scale
    means, covariance, mass = [], np.zeros((x.shape[1], x.shape[1])), 0.
    for i in range(c.k):
        take = labels == i
        mean = np.average(x[take], axis=0, weights=weights[take])
        means.append(mean)
        delta = x[take]-mean
        covariance += (delta.T*weights[take])@delta
        mass += weights[take].sum()
    covariance /= mass
    covariance = .8*covariance+.2*np.diag(np.diag(covariance))+.0025*np.eye(len(scale))
    means = np.asarray(means)
    coefficient_scaled = np.linalg.solve(covariance, means.T).T
    coefficient = coefficient_scaled/scale
    intercept = -.5*np.sum(coefficient_scaled*means, axis=1)
    return coefficient, intercept, means*scale, covariance*scale[:, None]*scale[None, :]


def linear_prob(c, features=None, weight=None):
    features = c.rgb if features is None else features
    coefficient, intercept, _, _ = fit_linear(c, features, weight)
    return softmax(native.logits(features, coefficient, intercept), axis=-1)


def plurality(c):
    coefficient, intercept, _, _ = fit_linear(c)
    initial = np.c_[coefficient, intercept].T
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
            bags.append(np.c_[c.rgb[yy[:, None], xx[None, :]].reshape(-1, 3),
                              np.ones(25)])
            labels.append(int(hist.argmax()))
    if not bags:
        return linear_prob(c), dict(training_bags=0)
    x, y = np.asarray(bags), np.asarray(labels)
    bag_weight = 1/np.maximum(np.bincount(y, minlength=c.k)[y], 1)
    bag_weight /= bag_weight.sum()

    def objective(flat):
        coef = flat.reshape(4, c.k)
        p = softmax(x@coef, axis=-1)
        derivative = np.zeros_like(p)
        loss = 0.
        for g in range(c.k):
            own = p[np.arange(len(y)), :, y]
            other = p[:, :, g]
            d = own-other
            mean = d.sum(1)
            variance = np.maximum((own+other-d*d).sum(1), .05)
            z = mean/np.sqrt(variance)
            active = y != g
            logcdf = log_ndtr(z)
            mills = np.exp(np.clip(-.5*z*z-.5*np.log(2*np.pi)-logcdf, -40, 15))
            dm = -mills/np.sqrt(variance)*bag_weight*active
            dv = .5*mills*mean/variance**1.5*bag_weight*active
            derivative[np.arange(len(y)), :, y] += dm[:, None]+dv[:, None]*(1-2*d)
            derivative[:, :, g] += -dm[:, None]+dv[:, None]*(1+2*d)
            loss -= float(np.sum(logcdf*bag_weight*active))
        logit_gradient = p*(derivative-(p*derivative).sum(-1, keepdims=True))
        penalty = coef-initial
        gradient = np.einsum("nsi,nsk->ik", x, logit_gradient)+.0004*penalty
        return loss+.0002*float(np.sum(penalty*penalty)), gradient.ravel()
    fit = minimize(objective, initial.ravel(), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=35, ftol=1e-8))
    coef = fit.x.reshape(4, c.k)
    return softmax(native.logits(c.rgb, coef[:3].T, coef[3]), -1), dict(
        training_bags=len(y), approximation="pairwise normal count likelihood",
        optimizer_success=bool(fit.success), iterations=int(fit.nit),
        objective=float(fit.fun), effective_pixels_per_bag=25)


def crossfit(c):
    yy, xx = np.indices(c.shape)
    folds = ((yy//max(1, round(3*c.tile_y))) % 2)*2 + (
        xx//max(1, round(3*c.tile_x))) % 2
    trust = np.ones(c.shape, np.float32)
    used = 0
    for fold in range(4):
        held = c.known & (folds == fold)
        weight = c.weight*(folds != fold)
        if not held.any() or any(not np.any(weight[c.labels == i] > 0) for i in range(c.k)):
            continue
        q = linear_prob(c, weight=weight)
        probability = np.take_along_axis(q, c.labels[..., None], -1)[..., 0]
        trust[held] = .2+.8*probability[held]
        used += 1
    return linear_prob(c, weight=c.weight*trust), dict(
        completed_spatial_folds=used, mean_training_trust=float(trust[c.known].mean()),
        missing_training_class_policy="skip fold, retain original reliability")


def local_bayes(c):
    _, _, means, covariance = fit_linear(c)
    inverse = np.linalg.inv(covariance)
    sigma = 48/c.um
    scores = []
    for i in range(c.k):
        weight = c.weight*(c.labels == i)
        mass = cv2.GaussianBlur(weight, (0, 0), sigma)
        local = np.stack([cv2.GaussianBlur(weight*c.rgb[..., j], (0, 0), sigma)
                          for j in range(3)], -1)
        local = (local+.015*means[i])/(mass[..., None]+.015)
        delta = c.rgb-local
        scores.append(-.5*np.einsum("...i,ij,...j->...", delta, inverse, delta))
    return softmax(np.stack(scores, -1), -1), dict(
        local_radius_um=48, global_shrinkage_mass=.015)


def pairwise(c):
    x, y, weight = training(c, c.rgb, limit=6000)
    x = np.c_[x, np.ones(len(x))]
    score = np.zeros((*c.shape, c.k), np.float32)
    pairs = 0
    for a in range(c.k):
        for b in range(a+1, c.k):
            take = (y == a) | (y == b)
            data, target, ws = x[take], (y[take] == a).astype(float), weight[take].copy()
            for value in (0, 1):
                ws[target == value] /= max(ws[target == value].sum(), 1e-8)
            coef = np.zeros(4)
            for _ in range(20):
                z = np.clip(data@coef, -25, 25)
                p = 1/(1+np.exp(-z))
                gradient = data.T@(ws*(p-target))+.002*coef
                hessian = (data.T*(ws*p*(1-p)))@data+.002*np.eye(4)
                step = np.linalg.solve(hessian, gradient)
                coef -= step
                if np.max(np.abs(step)) < 1e-5:
                    break
            field = native.logits(c.rgb, coef[None, :3], coef[3:])[..., 0]
            score[..., a] += -np.logaddexp(0, -field)/(c.k-1)
            score[..., b] += -np.logaddexp(0, field)/(c.k-1)
            pairs += 1
    return softmax(score, -1), dict(class_pairs=pairs, coupling="mean log pairwise win probability")


def cooccurrence(c):
    radius = max(1, round(6/c.um))
    mean = cv2.boxFilter(c.rgb, -1, (2*radius+1,)*2)
    features = [c.rgb]
    covariance = []
    for i in range(3):
        for j in range(i, 3):
            covariance.append(cv2.boxFilter(c.rgb[..., i]*c.rgb[..., j], -1,
                                           (2*radius+1,)*2)-mean[..., i]*mean[..., j])
    features.append(np.stack(covariance, -1))
    lag = max(1, round(4/c.um))
    product = (c.rgb*shifted(c.rgb, lag, 0)+c.rgb*shifted(c.rgb, 0, lag))/2
    features.append(cv2.boxFilter(product, -1, (2*radius+1,)*2))
    features = np.concatenate(features, -1)
    return linear_prob(c, features), dict(
        features=features.shape[-1], covariance_radius_um=6, lag_um=4)


def patch_samples(array, ys, xs, offset=2):
    h, w = array.shape[:2]
    return np.concatenate([
        array[np.clip(ys+dy, 0, h-1), np.clip(xs+dx, 0, w-1)]
        for dy in (-offset, 0, offset) for dx in (-offset, 0, offset)], -1)


def dictionary(c):
    offset = max(1, round(2/c.um))
    yy, xx = np.indices(c.shape)
    features = patch_samples(c.rgb, yy.ravel(), xx.ravel(), offset).reshape(*c.shape, -1)
    x, y, weight = training(c, features, limit=6000)
    xnorm = np.sum(features*features, -1)
    energies, ranks = [], []
    for i in range(c.k):
        sample, ws = x[y == i], weight[y == i]
        mean = np.average(sample, axis=0, weights=ws)
        centered = (sample-mean)*np.sqrt(ws/ws.mean())[:, None]
        _, _, v = np.linalg.svd(centered, full_matrices=False)
        basis = v[:min(6, len(v))]
        projection = native.logits(features, basis, -basis@mean)
        coefficient = np.sign(projection)*np.maximum(np.abs(projection)-.05, 0)
        norm = xnorm+native.logits(features, -2*mean[None], np.array([mean@mean]))[..., 0]
        energy = norm-2*np.sum(projection*coefficient, -1)+np.sum(
            coefficient**2+.1*np.abs(coefficient), -1)
        energies.append(energy/.25)
        ranks.append(len(basis))
    return softmax(-np.stack(energies, -1), -1), dict(
        dictionary_ranks=ranks, l1_penalty=.1, patch_offset_um=offset*c.um,
        classification="sparse orthogonal reconstruction energy, not neighbour votes")


def low_rank(c, palette):
    q = amortized(c, palette)
    flat = np.flatnonzero(c.support.ravel())
    take = flat[np.linspace(0, len(flat)-1, min(4096, len(flat)), dtype=int)]
    ys, xs = np.unravel_index(take, c.shape)
    offset = max(1, round(4/c.um))
    sample = patch_samples(q, ys, xs, offset)
    mean = sample.mean(0)
    _, _, basis = np.linalg.svd(sample-mean, full_matrices=False)
    basis = basis[:min(8, len(basis))]
    center = slice(4*c.k, 5*c.k)
    result = np.empty_like(q)
    for start in range(0, c.shape[0], 48):
        yy, xx = np.indices((min(48, c.shape[0]-start), c.shape[1]))
        yy += start
        data = patch_samples(q, yy.ravel(), xx.ravel(), offset)
        coefficient = (data-mean)@basis.T
        estimate = mean[center]+coefficient@basis[:, center]
        residual = q[start:start+len(yy)].reshape(-1, c.k)-estimate
        estimate += np.sign(residual)*np.maximum(np.abs(residual)-.035, 0)
        result[start:start+len(yy)] = estimate.reshape(len(yy), c.shape[1], c.k)
    return normalize(result), dict(
        rank=len(basis), fit_patches=len(sample), sparse_threshold=.035,
        approximation="one low-rank projection and residual soft threshold; not converged robust PCA")


def wavelet(c, palette):
    q = amortized(c, palette)
    h, w = c.shape
    data = np.pad(q, ((0, (-h) % 8), (0, (-w) % 8), (0, 0)), mode="edge")
    coefficients = []
    for level in range(3):
        a, b = data[::2, ::2], data[::2, 1::2]
        d, e = data[1::2, ::2], data[1::2, 1::2]
        low = (a+b+d+e)/2
        details = [(a-b+d-e)/2, (a+b-d-e)/2, (a-b-d+e)/2]
        threshold = .035/(2**level)
        details = [np.sign(z)*np.maximum(np.abs(z)-threshold, 0) for z in details]
        coefficients.append(details)
        data = low
    for horizontal, vertical, diagonal in coefficients[::-1]:
        out = np.empty((data.shape[0]*2, data.shape[1]*2, c.k), np.float32)
        out[::2, ::2] = (data+horizontal+vertical+diagonal)/2
        out[::2, 1::2] = (data-horizontal+vertical-diagonal)/2
        out[1::2, ::2] = (data+horizontal-vertical-diagonal)/2
        out[1::2, 1::2] = (data-horizontal-vertical+diagonal)/2
        data = out
    return normalize(data[:h, :w]), dict(levels=3, finest_threshold=.035,
                                       transform="orthonormal 2D Haar")


def minimum_description(c, palette):
    q = amortized(c, palette)
    costs = -np.log(np.maximum(q, 1e-7))*c.support[..., None]
    integral = np.pad(costs.cumsum(0, dtype=np.float64).cumsum(1),
                      ((1, 0), (1, 0), (0, 0)))
    labels = np.empty(c.shape, np.int32)
    leaves = 0
    penalty = np.log(max(int(c.support.sum()), 2))

    def partition(y0, y1, x0, x1):
        nonlocal leaves
        values = integral[y1, x1]-integral[y0, x1]-integral[y1, x0]+integral[y0, x0]
        winner = int(values.argmin())
        cost = float(values[winner])+penalty
        if y1-y0 <= 3 or x1-x0 <= 3:
            return cost, (y0, y1, x0, x1, winner)
        ym, xm = (y0+y1)//2, (x0+x1)//2
        children = [partition(a, b, d, e) for a, b, d, e in (
            (y0, ym, x0, xm), (ym, y1, x0, xm),
            (y0, ym, xm, x1), (ym, y1, xm, x1))]
        if sum(t[0] for t in children) < cost:
            return sum(t[0] for t in children), [t[1] for t in children]
        return cost, (y0, y1, x0, x1, winner)

    def assign(tree):
        nonlocal leaves
        if isinstance(tree, list):
            for child in tree:
                assign(child)
        else:
            y0, y1, x0, x1, label = tree
            labels[y0:y1, x0:x1] = label
            leaves += 1
    objective, tree = partition(0, c.shape[0], 0, c.shape[1])
    assign(tree)
    return np.eye(c.k, dtype=np.float32)[labels], dict(
        leaves=leaves, penalty_per_leaf=float(penalty),
        objective=objective, minimum_cell_side=3)


def tissue_runs(c, palette):
    q = amortized(c, palette)
    unary = -np.log(np.maximum(q, 1e-7))
    horizontal = native.semimarkov(unary)
    vertical = native.semimarkov(unary.transpose(1, 0, 2)).T
    logit = np.log(np.maximum(q, 1e-7))
    eye = np.eye(c.k, dtype=np.float32)
    logit += .6*(eye[horizontal]+eye[vertical])
    return softmax(logit, -1), dict(
        short_run_limit=24, long_run_tail="linear continuation; no hard maximum",
        run_penalty=.6, directions=2, implementation="C++ OpenMP dynamic programming")


def spline_design(ys, xs, shape, knot_shape, rgb):
    ny, nx = knot_shape
    v = ys/max(shape[0]-1, 1)*(ny-3)+1
    u = xs/max(shape[1]-1, 1)*(nx-3)+1
    iy, ix = np.floor(v).astype(int), np.floor(u).astype(int)
    def cubic(t):
        return np.stack([(1-t)**3, 3*t**3-6*t*t+4,
                         -3*t**3+3*t*t+3*t+1, t**3], -1)/6
    by, bx = cubic(v-iy), cubic(u-ix)
    rows, columns, values = [], [], []
    row = np.arange(len(ys))
    for a in range(4):
        for b in range(4):
            rows.append(row)
            columns.append(4+np.clip(iy+a-1, 0, ny-1)*nx+np.clip(ix+b-1, 0, nx-1))
            values.append(by[:, a]*bx[:, b])
    for a, data in enumerate((np.ones(len(row)), rgb[:, 0], rgb[:, 1], rgb[:, 2])):
        rows.append(row)
        columns.append(np.full(len(row), a))
        values.append(data)
    return sparse.coo_matrix((np.concatenate(values),
                              (np.concatenate(rows), np.concatenate(columns))),
                             shape=(len(ys), 4+ny*nx)).tocsr()


def spline_field(c):
    knot_shape = tuple(max(4, min(24, round(size*c.um/48)+3)) for size in c.shape)
    where = np.flatnonzero(c.known.ravel())
    if len(where) > 12000:
        where = where[np.linspace(0, len(where)-1, 12000, dtype=int)]
    yy, xx = np.unravel_index(where, c.shape)
    design = spline_design(yy, xx, c.shape, knot_shape, c.rgb[yy, xx])
    weights = c.weight[yy, xx].copy()
    target = c.labels[yy, xx]
    for label in range(c.k):
        take = target == label
        weights[take] /= max(float(weights[take].sum()), 1e-8)
    weights *= len(weights)/c.k
    ny, nx = knot_shape
    edges = []
    for y in range(ny):
        for x in range(nx):
            if y+1 < ny:
                edges.append((4+y*nx+x, 4+(y+1)*nx+x))
            if x+1 < nx:
                edges.append((4+y*nx+x, 4+y*nx+x+1))
    row = np.repeat(np.arange(len(edges)), 2)
    difference = sparse.coo_matrix(
        (np.tile([1., -1.], len(edges)), (row, np.array(edges).ravel())),
        shape=(len(edges), design.shape[1])).tocsr()
    laplacian = difference.T@difference
    regularizer = .05*sparse.eye(design.shape[1])+6*(laplacian.T@laplacian)
    weighted = design.multiply(weights[:, None])
    system = (design.T@weighted+regularizer).tocsc()
    rhs = weighted.T@np.eye(c.k)[target]
    coefficient = spsolve(system, rhs)
    result = np.empty((*c.shape, c.k), np.float32)
    for start in range(0, c.shape[0], 48):
        yy, xx = np.indices((min(48, c.shape[0]-start), c.shape[1]))
        yy += start
        design = spline_design(yy.ravel(), xx.ravel(), c.shape,
                               knot_shape, c.rgb[yy, xx].reshape(-1, 3))
        result[start:start+len(yy)] = (design@coefficient).reshape(len(yy), c.shape[1], c.k)
    return softmax(result*4, -1), dict(
        knot_shape=list(knot_shape), training_pixels=len(where),
        basis="RGB linear terms and tensor cubic B splines",
        spatial_curvature_penalty=6, unknown_labels_used_for_training=False)


def motifs(c, palette):
    q = amortized(c, palette)
    ys = np.clip(np.round(np.arange(c.tile_y/2, c.shape[0], c.tile_y)).astype(int),
                 0, c.shape[0]-1)
    xs = np.clip(np.round(np.arange(c.tile_x/2, c.shape[1], c.tile_x)).astype(int),
                 0, c.shape[1]-1)
    labels = c.labels[ys[:, None], xs[None, :]]
    known = c.known[ys[:, None], xs[None, :]]
    grammar = np.full((c.k, c.k, c.k), .5, np.float64)
    observations = 0
    for y in range(1, len(ys)):
        for x in range(1, len(xs)):
            if known[y, x] and known[y-1, x] and known[y, x-1]:
                grammar[labels[y, x-1], labels[y-1, x], labels[y, x]] += 1
                observations += 1
    grammar /= grammar.sum(-1, keepdims=True)
    left = shifted(q, 0, -max(1, round(c.tile_x/2)))
    up = shifted(q, -max(1, round(c.tile_y/2)), 0)
    context = np.empty_like(q)
    for label in range(c.k):
        context[..., label] = np.sum((left@grammar[:, :, label])*up, -1)
    return softmax(np.log(np.maximum(q, 1e-7))+.35*np.log(
        np.maximum(context, 1e-7)), -1), dict(
            training_triads=observations, pseudocount=.5, context_log_weight=.35)


def expected_iou(c, palette):
    q = amortized(c, palette)
    evidence = q[c.support]
    volume = evidence.sum(0, dtype=np.float64)
    current = evidence.argmax(-1)
    best, best_value, seen = current.copy(), -1., set()
    n = len(current)
    for iteration in range(30):
        key = hash(current.tobytes())
        if key in seen:
            break
        seen.add(key)
        predicted = np.bincount(current, minlength=c.k)
        intersection = np.bincount(current, weights=evidence[np.arange(n), current],
                                   minlength=c.k)
        union = np.maximum(predicted+volume-intersection, 1e-7)
        value = float(np.mean(intersection/union))
        if value > best_value:
            best, best_value = current.copy(), value
        marginal = evidence/union-intersection*(1-evidence)/union**2
        current = marginal.argmax(-1)
    labels = q.argmax(-1)
    labels[c.support] = best
    return np.eye(c.k, dtype=np.float32)[labels], dict(
        iterations=iteration+1, expected_ratio_objective=best_value,
        approximation="ratio of expected counts; not exact expectation of IoU",
        selection_support="all independently detected tissue, not fine annotations")


def copula(c):
    scores = []
    bins = 48
    grid = (np.arange(bins)+.5)/bins
    for label in range(c.k):
        take = c.known & (c.labels == label)
        weight = c.weight[take]
        if not take.any():
            scores.append(np.full(c.shape, -1e4))
            continue
        z, marginal = [], np.zeros(c.shape, np.float64)
        for channel in range(3):
            hist, _ = np.histogram(c.rgb[..., channel][take],
                                   bins=bins, range=(0, 1), weights=weight)
            hist = ndi.gaussian_filter1d(hist.astype(float), .7)
            hist += max(hist.sum(), 1)*1e-5
            hist /= hist.sum()
            cdf = np.cumsum(hist)-hist/2
            value = c.rgb[..., channel]
            uniform = np.interp(value.ravel(), grid, cdf).reshape(c.shape)
            z.append(ndtri(np.clip(uniform, .001, .999)))
            density = np.interp(value.ravel(), grid, hist*bins).reshape(c.shape)
            marginal += np.log(np.maximum(density, 1e-12))
        z = np.stack(z, -1)
        train = z[take]
        mean = np.average(train, axis=0, weights=weight)
        centered = train-mean
        covariance = (centered.T*weight)@centered/weight.sum()
        std = np.maximum(np.sqrt(np.diag(covariance)), .05)
        correlation = covariance/std[:, None]/std[None, :]
        correlation = .7*correlation+.3*np.eye(3)
        inv = np.linalg.inv(correlation)-np.eye(3)
        dependence = -.5*np.einsum("...i,ij,...j->...", z, inv, z)
        dependence -= .5*np.linalg.slogdet(correlation)[1]
        scores.append(marginal+dependence)
    return softmax(np.stack(scores, -1), -1), dict(
        marginal_bins=bins, copula="Gaussian dependence with empirical RGB marginals",
        correlation_identity_shrinkage=.3)


def influence(c):
    coefficient, intercept, means, covariance = fit_linear(c)
    inverse = np.linalg.inv(covariance)
    logit = native.logits(c.rgb, coefficient, intercept)
    yy, xx = np.indices(c.shape)
    cols = int(np.ceil(c.shape[1]/c.tile_x))
    tiles = (yy/c.tile_y).astype(int)*cols+(xx/c.tile_x).astype(int)
    groups = []
    for label in range(c.k):
        weight = c.weight*(c.labels == label)
        mass = np.bincount(tiles.ravel(), weights=weight.ravel())
        sums = np.stack([
            np.bincount(tiles.ravel(), weights=(weight*c.rgb[..., j]).ravel(),
                        minlength=len(mass)) for j in range(3)], -1)
        valid = (mass > 0) & (mass < mass.sum()*(1-1e-6))
        if valid.sum() < 2:
            groups.append(0)
            continue
        total = sums.sum(0)
        overall_mean = total/mass.sum()
        deletion = ((total-sums[valid])/(mass.sum()-mass[valid])[:, None]
                    -overall_mean)
        count = len(deletion)
        variance = deletion.T@deletion*(count-1)/count
        sensitivity = inverse@variance@inverse
        delta = c.rgb-means[label]
        score_variance = np.maximum(
            np.einsum("...i,ij,...j->...", delta, sensitivity, delta), 0)
        logit[..., label] -= .25*np.sqrt(score_variance)
        groups.append(count)
    return softmax(logit, -1), dict(
        influence_groups_per_class=groups, standard_error_penalty=.25,
        approximation="first-order delete-tile mean influence with covariance held fixed")


def correct_coarse(q, c):
    q = q.copy()
    sorted_q = np.sort(q, axis=-1)
    gap = sorted_q[..., -1]-sorted_q[..., -2]
    strength = .35*(1-gap)**2*c.weight
    for label in range(c.k):
        q[..., label] += strength*(c.labels == label)
    return normalize(q)


def palette_active_set(c, palette):
    colors = np.asarray(palette, np.float64)[c.ids-1]/255
    if np.linalg.matrix_rank(np.c_[colors, np.ones(c.k)].T) == c.k:
        return amortized(c, palette), dict(
            branch="same affine-identifiable branch as B22", active_set_required=False)
    prior = c.lattice()
    bins = 24
    cell = np.minimum((c.rgb*bins).astype(int), bins-1)
    index = (cell[..., 0]*bins+cell[..., 1])*bins+cell[..., 2]
    n = bins**3
    count = np.bincount(index.ravel(), minlength=n)
    occupied = count > 0
    color_mean = np.stack([
        np.bincount(index.ravel(), weights=c.rgb[..., j].ravel(), minlength=n)
        /np.maximum(count, 1) for j in range(3)], -1)[occupied]
    probability_mean = np.stack([
        np.bincount(index.ravel(), weights=prior[..., j].ravel(), minlength=n)
        /np.maximum(count, 1) for j in range(c.k)], -1)[occupied]
    solved, details = native.simplex_qp(color_mean, probability_mean, colors)
    table = np.zeros((n, c.k), np.float32)
    table[occupied] = solved
    details.update(occupied_colour_nodes=int(occupied.sum()), quantization_bins=bins,
                   active_set_required=True, convergence_tolerance=1e-8)
    return correct_coarse(table[index], c), details


def exterior_white(c, palette):
    # The actual exclusion is applied in predict, using H&E at the analysis grid.
    return amortized(c, palette), dict(background_policy="border-connected white exclusion")


def interval_palette(c, palette):
    q = amortized(c, palette)
    colors = np.asarray(palette, np.float64)[c.ids-1]/255
    bins = 12
    cells = np.minimum((c.rgb*bins).astype(int), bins-1)
    index = (cells[..., 0]*bins+cells[..., 1])*bins+cells[..., 2]
    counts = np.bincount(index.ravel(), minlength=bins**3)
    gap = np.sort(q, axis=-1)[..., -1]-np.sort(q, axis=-1)[..., -2]
    ambiguity = np.bincount(index.ravel(), weights=(gap < .15).ravel(),
                             minlength=len(counts))
    chosen = np.argsort(-ambiguity, kind="stable")[:32]
    chosen = chosen[ambiguity[chosen] > 0]
    solved, infeasible, changed = 0, 0, 0
    minimum_interval_width = []
    for cell in chosen:
        take = index == cell
        rgb = c.rgb[take]
        center = rgb.mean(0)
        epsilon = np.max(np.abs(rgb-center), axis=0)+2/255
        constraint = np.r_[colors.T, -colors.T]
        bound = np.r_[center+epsilon, -center+epsilon]
        lower, upper = [], []
        feasible = True
        for label in range(c.k):
            direction = np.eye(c.k)[label]
            with warnings.catch_warnings():
                # SciPy forwards this valid HiGHS resource option explicitly.
                warnings.filterwarnings("ignore", category=OptimizeWarning,
                                        message="Unrecognized options detected.*")
                options = dict(time_limit=.25, threads=2)
                lo = linprog(direction, A_ub=constraint, b_ub=bound,
                             A_eq=np.ones((1, c.k)), b_eq=[1], bounds=(0, 1),
                             method="highs-ds", options=options)
                hi = linprog(-direction, A_ub=constraint, b_ub=bound,
                             A_eq=np.ones((1, c.k)), b_eq=[1], bounds=(0, 1),
                             method="highs-ds", options=options)
            if not lo.success or not hi.success:
                feasible = False
                break
            lower.append(max(0., float(lo.fun)))
            upper.append(min(1., float(-hi.fun)))
        if not feasible:
            infeasible += 1
            continue
        solved += 1
        minimum_interval_width.append(float(np.mean(np.asarray(upper)-lower)))
        if max(lower) > .01:
            # Robust maximin class evidence. Prior breaks uninformative ties.
            robust = np.asarray(lower)+.02*q[take].mean(0)
            before = q[take].argmax(-1)
            replacement = normalize(np.broadcast_to(robust, q[take].shape).copy())
            changed += int((before != replacement.argmax(-1)).sum())
            q[take] = replacement
    return q, dict(
        bounded_colour_cells=solved, failed_or_infeasible_cells=infeasible,
        changed_pixels=changed, maximum_cells=32,
        mean_probability_interval_width=float(np.mean(minimum_interval_width))
        if minimum_interval_width else None,
        scope="32 most-populous ambiguous colour cells; other pixels retain palette evidence")


def adaptive_colors(c, palette):
    colors = np.asarray(palette, np.float64)[c.ids-1]/255
    if np.linalg.matrix_rank(np.c_[colors, np.ones(c.k)].T) == c.k:
        return amortized(c, palette), dict(branch="affine-identifiable", adaptive_nodes=0)
    prior = c.lattice()
    integers = np.rint(c.rgb.reshape(-1, 3)*255).astype(np.uint8)
    unique, first, inverse, count = np.unique(
        integers, axis=0, return_index=True, return_inverse=True, return_counts=True)
    x = c.rgb.reshape(-1, 3)[first]
    pi = prior.reshape(-1, c.k)[first]
    unresolved = np.arange(len(unique))
    table = np.empty((len(unique), c.k), np.float32)
    ridge = .012
    lipschitz_rgb = float(np.linalg.norm(colors, 2))/ridge
    history, residual = [], 0.
    for bins in (4, 8, 16, 32, 64, 128, 256):
        cell = np.minimum((x[unresolved]*bins).astype(int), bins-1)
        code = (cell[:, 0]*bins+cell[:, 1])*bins+cell[:, 2]
        _, group = np.unique(code, return_inverse=True)
        n = int(group.max())+1
        weight = np.bincount(group, weights=count[unresolved], minlength=n)
        mean_x = np.stack([np.bincount(group, weights=count[unresolved]*x[unresolved, j],
                                       minlength=n)/weight for j in range(3)], -1)
        mean_pi = np.stack([np.bincount(group, weights=count[unresolved]*pi[unresolved, j],
                                        minlength=n)/weight for j in range(c.k)], -1)
        # Use exactly the representatives received by the float32 native API.
        mean_x = mean_x.astype(np.float32)
        mean_pi = mean_pi.astype(np.float32)
        solved, info = native.simplex_qp(mean_x, mean_pi, colors)
        residual = max(residual, info["max_kkt_residual"])
        deviation = lipschitz_rgb*np.linalg.norm(x[unresolved]-mean_x[group], axis=1)
        deviation += np.linalg.norm(pi[unresolved]-mean_pi[group], axis=1)
        max_deviation = np.zeros(n)
        np.maximum.at(max_deviation, group, deviation)
        ordered = np.sort(solved, axis=1)
        # Worst-case change in a score difference, plus maximal coarse support.
        certified = ordered[:, -1]-ordered[:, -2] > 2*max_deviation+.35+1e-4
        if bins == 256:
            certified[:] = True  # exact occupied integer RGB colours
        accepted = certified[group]
        table[unresolved[accepted]] = solved[group[accepted]]
        history.append(dict(bins=bins, nodes=n, accepted_colours=int(accepted.sum())))
        unresolved = unresolved[~accepted]
        if not len(unresolved):
            break
    if len(unresolved):
        raise RuntimeError("Adaptive colour partition did not resolve every colour")
    q = table[inverse].reshape(*c.shape, c.k)
    return correct_coarse(q, c), dict(
        unique_rgb_colours=len(unique), refinement=history,
        max_kkt_residual=residual, strong_convexity=ridge,
        certificate="L2 solution sensitivity and bounded coarse correction; certification only for observed colours")


def adaptive_resolution(c):
    coefficient, intercept, _, _ = fit_linear(c)
    labels, details = native.adaptive_linear(c.rgb, coefficient, intercept)
    details.update(analysis_target_um=.5, certificate="affine decision lower bound over each observed RGB box",
                   densely_sampled_fallback=True)
    return np.eye(c.k, dtype=np.float32)[labels], details


IMPLEMENTED = {
    "C01": plurality, "C02": crossfit, "C03": local_bayes, "C04": pairwise,
    "C05": cooccurrence, "C06": dictionary, "C11": spline_field,
    "C15": copula, "C16": influence, "C19": adaptive_resolution,
}
PALETTE_IMPLEMENTED = {
    "C07": low_rank, "C08": wavelet, "C09": minimum_description,
    "C10": tissue_runs, "C12": motifs, "C13": expected_iou,
    "C14": interval_palette, "C17": palette_active_set,
    "C18": adaptive_colors, "C20": exterior_white}


def predict(key, rgb, coarse, support, mpp_xy, palette, he=None):
    if key.startswith("Q"):
        from .cycle_twelve import predict as predict_child
        return predict_child(key,rgb,coarse,support,mpp_xy,palette,he)
    if key.startswith("P"):
        from .cycle_eleven import predict as predict_child
        return predict_child(key,rgb,coarse,support,mpp_xy,palette,he)
    if key.startswith("N"):
        from .cycle_ten import predict as predict_child
        return predict_child(key, rgb, coarse, support, mpp_xy, palette, he)
    if key.startswith("M"):
        from .cycle_nine import predict as predict_child
        return predict_child(key, rgb, coarse, support, mpp_xy, palette, he)
    if key.startswith("L"):
        from .cycle_eight import predict as predict_child
        return predict_child(key, rgb, coarse, support, mpp_xy, palette, he)
    if key.startswith("K"):
        from .cycle_seven import predict as predict_child
        return predict_child(key, rgb, coarse, support, mpp_xy, palette, he)
    if key.startswith("J"):
        from .cycle_six import predict as predict_child
        return predict_child(key, rgb, coarse, support, mpp_xy, palette, he)
    if key.startswith("H"):
        from .cycle_five import predict as predict_child
        return predict_child(key, rgb, coarse, support, mpp_xy, palette, he)
    if key.startswith("G"):
        from .cycle_four import predict as predict_child
        return predict_child(key, rgb, coarse, support, mpp_xy, palette, he)
    if key.startswith("F"):
        from .cycle_three import predict as predict_child
        return predict_child(key, rgb, coarse, support, mpp_xy, palette, he)
    if key.startswith("D"):
        from .cycle_two import predict as predict_child
        return predict_child(key, rgb, coarse, support, mpp_xy, palette, he)
    if key not in CYCLE_ONE or not CYCLE_ONE[key].implemented:
        raise ValueError(f"Candidate is not implemented: {key}")
    c = Evidence(rgb, coarse, support, mpp_xy, target_um=.5 if key == "C19" else 1.)
    if not c.support.any():
        raise ValueError("Empty tissue support")
    if c.k == 1:
        q, detail = np.ones((*c.shape, 1), np.float32), dict(single_class=True)
    elif key in IMPLEMENTED:
        q, detail = IMPLEMENTED[key](c)
    elif key in PALETTE_IMPLEMENTED:
        q, detail = PALETTE_IMPLEMENTED[key](c, palette)
    else:
        raise NotImplementedError(key)
    if q.shape != (*c.shape, c.k) or not np.isfinite(q).all():
        raise ValueError(f"Invalid candidate probabilities: {key}")
    labels = c.ids[q.argmax(-1)].astype(np.uint8)
    if key == "C20":
        if he is None:
            raise ValueError("Exterior white-space candidate requires original H&E")
        resized = cv2.resize(he, c.shape[::-1], interpolation=cv2.INTER_AREA)
        _, exterior, policy = white_regions(resized)
        labels[exterior] = 0
        detail.update(proposed_glass_pixels=int((exterior & c.support).sum()),
                      white_policy=policy)
    detail.update(analysis_mpp_xy=c.mpp.tolist(), analysis_shape_yx=list(c.shape),
                  fine_reference_used=False, hard_anchors=False, fixed_margin=False,
                  candidate=key, mechanism=CYCLE_ONE[key].mechanism)
    return labels, detail
