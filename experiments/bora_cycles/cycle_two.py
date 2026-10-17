"""Fifteen review-derived interventions; fixed RGB and no fine-reference I/O.

Each entry is one statistical model, observation rule or experimental-design
intervention. Solvers are shared implementation machinery, not extra methods.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.special import softmax, log_ndtr, gammaincc, gammaln, logsumexp

from bora.evidence import Evidence, normalize
from . import native
from .diagnostics import white_regions
from .methods import fit_linear, linear_prob, local_bayes, training, correct_coarse
from .registry import CYCLE_TWO


def colour_nodes(c, exact=False):
    bins = 256 if exact else 24
    cell = np.minimum(np.rint(c.rgb*255).astype(int) if exact else
                      (c.rgb*bins).astype(int), bins-1)
    code = (cell[..., 0]*bins+cell[..., 1])*bins+cell[..., 2]
    _, inverse = np.unique(code.ravel(), return_inverse=True)
    count = np.bincount(inverse)
    prior = c.lattice().reshape(-1, c.k)
    x = np.stack([np.bincount(inverse, weights=c.rgb[..., j].ravel())/count
                  for j in range(3)], -1)
    pi = np.stack([np.bincount(inverse, weights=prior[:, j])/count
                   for j in range(c.k)], -1)
    return x, pi, inverse.reshape(c.shape)


def pair_inverse(c, palette, he):
    x, pi, index = colour_nodes(c, exact=True)
    colors = np.asarray(palette, float)[c.ids-1]/255
    q, detail = native.pair_palette(x, pi, colors)
    detail.update(observed_colours=len(x), max_active_fractions=2,
                  additive_coarse_correction=False)
    # Adding a third coarse fraction would violate this model's constraint.
    return q[index], detail


def conditional_prior(c, palette, he):
    if he is None:
        raise ValueError("Original aligned H&E is required")
    original = cv2.resize(he, c.shape[::-1], interpolation=cv2.INTER_AREA)
    od = -np.log((original.astype(np.float32)+1)/256)
    features = np.concatenate([c.rgb, od], -1)
    prior = linear_prob(c, features)
    colors = np.asarray(palette, float)[c.ids-1]/255
    q, detail = native.simplex_qp(c.rgb, prior, colors)
    detail.update(conditioning_variables="fixed RGB plus three original-H&E optical densities",
                  rgb_only_quantization=False, solver_pixels=int(np.prod(c.shape)))
    return correct_coarse(q, c), detail


def entropy_inverse(c, palette, he):
    x, pi, index = colour_nodes(c, exact=True)
    colors = np.asarray(palette, float)[c.ids-1]/255
    q, detail = native.entropy_palette(x, pi, colors)
    detail.update(observed_colours=len(x), prior_geometry="KL(q || pi)")
    return correct_coarse(q[index], c), detail


def huber_inverse(c, palette, he):
    x, pi, index = colour_nodes(c)
    colors = np.asarray(palette, float)[c.ids-1]/255
    q, detail = native.simplex_qp(x, pi, colors)
    history, difference = [], np.inf
    for iteration in range(30):
        residual = q@colors-x
        weight = np.minimum(1., .04/np.maximum(np.abs(residual), 1e-12))
        hessian = np.einsum("nc,ic,jc->nij", weight, colors, colors)
        hessian += .012*np.eye(c.k)
        linear = (weight*x)@colors.T+.012*pi
        changed, info = native.quadratic_simplex(hessian, linear, q)
        difference = float(np.max(np.abs(changed-q)))
        q = changed
        error = np.abs(q@colors-x)
        objective = np.minimum(error, .04)**2/2+.04*np.maximum(error-.04, 0)
        history.append(float(objective.sum()+.006*np.sum((q-pi)**2)))
        detail["max_kkt_residual"] = max(detail["max_kkt_residual"], info["max_kkt_residual"])
        if difference < 1e-5:
            break
    detail.update(irls_iterations=iteration+1, huber_transition=.04,
                  max_probability_change=difference, objective_history=history,
                  quantization_bins=24)
    return correct_coarse(q[index], c), detail


def nullspace_inverse(c, palette, he):
    x, pi, index = colour_nodes(c)
    colors = np.asarray(palette, float)[c.ids-1]/255
    affine = np.c_[colors, np.ones(c.k)].T
    _, _, vh = np.linalg.svd(affine, full_matrices=True)
    rank = np.linalg.matrix_rank(affine)
    invisible = vh[rank:].T
    projection = invisible@invisible.T
    hessian = colors@colors.T+.012*projection+1e-9*np.eye(c.k)
    linear = x@colors.T+.012*pi@projection+1e-9*pi
    q, detail = native.quadratic_simplex(hessian, linear, pi)
    detail.update(affine_palette_rank=int(rank), nullity=int(c.k-rank),
                  numerical_ridge=1e-9, quantization_bins=24)
    return correct_coarse(q[index], c), detail


def directional_inverse(c, palette, he):
    x, pi, index = colour_nodes(c)
    colors = np.asarray(palette, float)[c.ids-1]/255
    colors /= np.maximum(np.linalg.norm(colors, axis=1, keepdims=True), 1e-8)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    q, detail = native.simplex_qp(x, pi, colors)
    detail.update(observation_geometry="unit-direction RGB", quantization_bins=24)
    return correct_coarse(q[index], c), detail


def tile_bags(c):
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
            bags.append(np.c_[c.rgb[yy[:, None], xx[None, :]].reshape(-1, 3), np.ones(25)])
            labels.append(int(hist.argmax()))
    return np.asarray(bags), np.asarray(labels, int)


def poisson_winner(means, labels, maximum_count=96):
    """Joint count-order probability and derivative of negative log P.

    Lower class indices win count ties, making the events a partition.
    Counts above 96 have negligible mass when total Poisson mean is 25.
    """
    means = np.maximum(np.asarray(means, float), 1e-12)
    labels = np.asarray(labels, int)
    counts = np.arange(maximum_count+1)[None, :]
    own_mean = means[np.arange(len(labels)), labels]
    terms = counts*np.log(own_mean[:, None])-own_mean[:, None]-gammaln(counts+1)
    cdf_ratios = []
    for g in range(means.shape[1]):
        threshold = counts-(g < labels[:, None])
        cdf = np.where(threshold >= 0, gammaincc(np.maximum(threshold+1, 1), means[:, g, None]), 0.)
        log_cdf = np.log(np.maximum(cdf, 1e-300))
        pmf = np.where(threshold >= 0, np.exp(threshold*np.log(means[:, g, None])
                    -means[:, g, None]-gammaln(np.maximum(threshold+1, 1))), 0.)
        cdf_ratios.append(pmf/np.maximum(cdf, 1e-300))
        terms += np.where((labels != g)[:, None], log_cdf, 0.)
    log_probability = logsumexp(terms, axis=1)
    probability_weight = np.exp(terms-log_probability[:, None])
    gradient = np.stack([(probability_weight*r).sum(1) for r in cdf_ratios], -1)
    gradient[np.arange(len(labels)), labels] = -(
        probability_weight*(counts/own_mean[:, None]-1)).sum(1)
    return log_probability, gradient


def fit_winner(c, bags, labels, event="normal"):
    coefficient, intercept, _, _ = fit_linear(c)
    initial = np.c_[coefficient, intercept].T
    if not len(bags):
        return linear_prob(c), dict(training_bags=0)
    bag_weight = 1/np.maximum(np.bincount(labels, minlength=c.k)[labels], 1)
    bag_weight /= bag_weight.sum()

    def objective(flat):
        coef = flat.reshape(4, c.k)
        p = softmax(bags@coef, axis=-1)
        if event == "poisson":
            logp, dm = poisson_winner(p.sum(1), labels)
            loss = -float(logp@bag_weight)
            derivative = dm[:, None, :]*bag_weight[:, None, None]
        else:
            derivative = np.zeros_like(p)
            loss = 0.
            for g in range(c.k):
                own = p[np.arange(len(labels)), :, labels]
                other = p[:, :, g]
                d = own-other
                mean = d.sum(1)
                variance = np.maximum((own+other-d*d).sum(1), .05)
                z = mean/np.sqrt(variance)
                active = labels != g
                logcdf = log_ndtr(z)
                mills = np.exp(np.clip(-.5*z*z-.5*np.log(2*np.pi)-logcdf, -40, 15))
                dm = -mills/np.sqrt(variance)*bag_weight*active
                dv = .5*mills*mean/variance**1.5*bag_weight*active
                derivative[np.arange(len(labels)), :, labels] += dm[:, None]+dv[:, None]*(1-2*d)
                derivative[:, :, g] += -dm[:, None]+dv[:, None]*(1+2*d)
                loss -= float(np.sum(logcdf*bag_weight*active))
        logit_gradient = p*(derivative-(p*derivative).sum(-1, keepdims=True))
        penalty = coef-initial
        gradient = np.einsum("nsi,nsk->ik", bags, logit_gradient)+.0004*penalty
        return loss+.0002*float(np.sum(penalty*penalty)), gradient.ravel()

    fit = minimize(objective, initial.ravel(), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=35, ftol=1e-8))
    coef = fit.x.reshape(4, c.k)
    return softmax(native.logits(c.rgb, coef[:3].T, coef[3]), -1), dict(
        training_bags=len(labels), observation_model=event,
        optimizer_success=bool(fit.success), iterations=int(fit.nit),
        objective=float(fit.fun), effective_pixels_per_bag=25)


def poisson_tiles(c, palette, he):
    x, labels = tile_bags(c)
    return fit_winner(c, x, labels, "poisson")


def bag_covariance(c, palette, he):
    bags, labels = tile_bags(c)
    if not len(bags) or set(labels) != set(range(c.k)):
        return linear_prob(c), dict(missing_known_bag_class=True)
    means = bags[..., :3].mean(1)
    class_means = np.asarray([means[labels == g].mean(0) for g in range(c.k)])
    residual = bags[..., :3]-means[:, None, :]
    covariance = np.einsum("nsi,nsj->ij", residual, residual)/(len(bags)*25)
    covariance = .8*covariance+.2*np.diag(np.diag(covariance))+.0025*np.eye(3)
    coef = np.linalg.solve(covariance, class_means.T).T
    intercept = -.5*np.sum(coef*class_means, axis=1)
    return softmax(native.logits(c.rgb, coef, intercept), -1), dict(
        training_bags=len(bags), covariance="within-bag deviations", class_mean_unit="equally weighted tile")


def design_tiles(c, palette, he):
    bags, labels = tile_bags(c)
    if not len(bags):
        return linear_prob(c), dict(training_bags=0)
    means = bags[..., :3].mean(1)
    design = np.c_[(means-means.mean(0))/np.maximum(means.std(0), .05), np.ones(len(means))]
    selected = []
    gains = []
    for g in range(c.k):
        candidates = np.flatnonzero(labels == g).tolist()
        information = .01*np.eye(4)
        for _ in range(min(12, len(candidates))):
            inverse = np.linalg.inv(information)
            values = np.einsum("ni,ij,nj->n", design[candidates], inverse, design[candidates])
            at = int(values.argmax())
            chosen = candidates.pop(at)
            selected.append(chosen)
            gains.append(float(np.log1p(values[at])))
            information += np.outer(design[chosen], design[chosen])
    q, detail = fit_winner(c, bags[selected], labels[selected])
    detail.update(available_bags=len(bags), selected_bags=len(selected), design="greedy log-determinant",
                  information_gains=gains)
    return q, detail


def local_affine(c, palette, he):
    _, _, means, covariance = fit_linear(c)
    inverse = np.linalg.inv(covariance)
    radius = 48.
    yy, xx = np.indices(c.shape, dtype=np.float64)
    x, y = xx*c.mpp[0]/radius, yy*c.mpp[1]/radius
    size = (2*max(1, round(radius/c.mpp[0]))+1,
            2*max(1, round(radius/c.mpp[1]))+1)

    def box(a):
        return cv2.boxFilter(a, cv2.CV_64F, size, normalize=True, borderType=cv2.BORDER_REFLECT)

    scores = []
    for g in range(c.k):
        weight = c.weight*(c.labels == g)
        m, mx, my = box(weight), box(weight*x), box(weight*y)
        matrix = np.zeros((*c.shape, 3, 3), float)
        matrix[..., 0, 0] = m
        matrix[..., 0, 1] = matrix[..., 1, 0] = mx-x*m
        matrix[..., 0, 2] = matrix[..., 2, 0] = my-y*m
        matrix[..., 1, 1] = box(weight*x*x)-2*x*mx+x*x*m
        matrix[..., 2, 2] = box(weight*y*y)-2*y*my+y*y*m
        matrix[..., 1, 2] = matrix[..., 2, 1] = box(weight*x*y)-x*my-y*mx+x*y*m
        matrix += .015*np.eye(3)
        b0 = box(weight[..., None]*c.rgb)
        b1 = box((weight*x)[..., None]*c.rgb)-x[..., None]*b0
        b2 = box((weight*y)[..., None]*c.rgb)-y[..., None]*b0
        rhs = np.stack([b0+.015*means[g], b1, b2], -2)
        local = np.linalg.solve(matrix, rhs)[..., 0, :]
        delta = c.rgb-local
        scores.append(-.5*np.einsum("...i,ij,...j->...", delta, inverse, delta))
    return softmax(np.stack(scores, -1), -1), dict(
        window_radius_um=radius, local_model="spatial affine RGB prototype", shrinkage=.015)


def shared_drift(c, palette, he):
    _, _, means, _ = fit_linear(c)
    yy, xx = np.indices(c.shape, dtype=np.float32)
    x = (xx+.5)/c.shape[1]-.5
    y = (yy+.5)/c.shape[0]-.5
    basis = np.stack([np.ones(c.shape), x, y, x*x, x*y, y*y], -1)
    sample, labels, weight = training(c, np.concatenate([c.rgb, basis], -1))
    design, residual = sample[:, 3:], sample[:, :3]-means[labels]
    beta = np.linalg.solve((design.T*weight)@design+.001*np.eye(6), (design.T*weight)@residual)
    drift = basis@beta
    changed = c.rgb-drift
    return linear_prob(c, changed), dict(
        shared_drift_basis=["1", "x", "y", "x^2", "xy", "y^2"],
        maximum_absolute_drift=float(np.abs(drift).max()), coefficients=beta.tolist())


def glass_competition(c, palette, he):
    if he is None:
        raise ValueError("Original H&E is required for glass evidence")
    if c.k == 1:
        q, detail = np.ones((*c.shape, 1), np.float32), dict(single_known_class=True)
    else:
        q, detail = local_bayes(c)
    original = cv2.resize(he, c.shape[::-1], interpolation=cv2.INTER_AREA)
    bright, _, _ = white_regions(original)
    rates = []
    for g in range(c.k):
        weight = c.weight*(c.labels == g)*float(np.prod(c.mpp))
        # Prior corresponds to 100 um² of observations, 10% white. All actual
        # evidence is original H&E under known coarse labels, never references.
        rates.append(float((np.sum(weight*bright)+10)/(weight.sum()+100)))
    likelihood = np.where(bright[..., None], rates, 1-np.asarray(rates))
    tissue = .92*q*likelihood
    glass = .08*bright
    excluded = (glass > tissue.max(-1)) & bright & c.support
    detail.update(glass_prior=.08, tissue_white_rates=rates,
                  proposed_glass_pixels=int(excluded.sum()),
                  glass_exclusion=excluded)
    return normalize(tissue), detail


def tree_density(c, palette, he):
    bins = 24
    indices = np.minimum((c.rgb*bins).astype(int), bins-1)
    code = (indices[..., 0]*bins+indices[..., 1])*bins+indices[..., 2]
    coordinates = np.moveaxis(np.clip(c.rgb*bins-.5, 0, bins-1), -1, 0)
    edges = [(0, 1), (0, 2), (1, 2)]
    scores, chosen = [], []
    for g in range(c.k):
        hist = np.bincount(code.ravel(), weights=(c.weight*(c.labels == g)).ravel(),
                           minlength=bins**3).reshape((bins,)*3)
        hist = ndi.gaussian_filter(hist, .7, mode="nearest")
        hist += max(hist.sum(), 1)*1e-7
        hist /= hist.sum()
        marginals = [hist.sum(axis=tuple(j for j in range(3) if j != i)) for i in range(3)]
        pairs = [hist.sum(axis=next(j for j in range(3) if j not in edge)) for edge in edges]
        information = [float(np.sum(pair*np.log(pair/(marginals[i][:, None]*marginals[j][None, :]))))
                       for pair, (i, j) in zip(pairs, edges)]
        selected = np.argsort(information)[-2:]
        marginal_log = [np.log(np.maximum(ndi.map_coordinates(m, [coordinates[i]], order=1,
                         mode="nearest"), 1e-20)) for i, m in enumerate(marginals)]
        score = sum(marginal_log)
        for at in selected:
            i, j = edges[at]
            joint = ndi.map_coordinates(pairs[at], [coordinates[i], coordinates[j]], order=1, mode="nearest")
            score = score+np.log(np.maximum(joint, 1e-20))-marginal_log[i]-marginal_log[j]
        scores.append(score)
        chosen.append([list(edges[at]) for at in selected])
    return softmax(np.stack(scores, -1), -1), dict(class_channel_trees=chosen, histogram_bins=bins)


def polya_tree(c, palette, he):
    flat = c.rgb.reshape(-1, 3)
    support = c.support.ravel()
    weight, labels = c.weight.ravel(), c.labels.ravel()
    log_probability = np.empty((len(flat), c.k), float)
    leaves = 0

    def visit(indices, logp, depth):
        nonlocal leaves
        tissue = indices[support[indices]]
        if depth >= 9 or len(tissue) < 200:
            log_probability[indices] = logp
            leaves += 1
            return
        low, high = flat[tissue].min(0), flat[tissue].max(0)
        channel = int(np.argmax(high-low))
        if high[channel]-low[channel] < 1e-6:
            log_probability[indices] = logp
            leaves += 1
            return
        split = float(np.median(flat[tissue, channel]))
        if split <= low[channel] or split >= high[channel]:
            split = float((low[channel]+high[channel])/2)
        left = flat[indices, channel] <= split
        a, b = indices[left], indices[~left]
        count_a = np.bincount(labels[a], weights=weight[a], minlength=c.k)
        count_b = np.bincount(labels[b], weights=weight[b], minlength=c.k)
        probability = (count_a+1)/(count_a+count_b+2)
        visit(a, logp+np.log(probability), depth+1)
        visit(b, logp+np.log1p(-probability), depth+1)

    visit(np.arange(len(flat)), np.zeros(c.k), 0)
    return softmax(log_probability.reshape(*c.shape, c.k), -1), dict(
        leaves=leaves, maximum_depth=9, split_evidence="all detected tissue, no class labels",
        predictive_branch_prior="Beta(1,1)")


def fisher_density(c, palette, he):
    x, labels, weight = training(c, c.rgb)
    mean = np.average(x, axis=0, weights=weight)
    within, between = np.zeros((3, 3)), np.zeros((3, 3))
    for g in range(c.k):
        take = labels == g
        local = np.average(x[take], axis=0, weights=weight[take])
        residual = x[take]-local
        within += (residual.T*weight[take])@residual
        between += weight[take].sum()*np.outer(local-mean, local-mean)
    within /= weight.sum()
    between /= weight.sum()
    values, vectors = eigh(between, within+.0025*np.eye(3))
    projection = vectors[:, -2:]
    features = c.rgb@projection
    low = features[c.support].min(0)
    scale = np.maximum(features[c.support].max(0)-low, 1e-6)
    coordinates = np.clip((features-low)/scale, 0, 1)
    bins = 32
    cells = np.minimum((coordinates*bins).astype(int), bins-1)
    code = cells[..., 0]*bins+cells[..., 1]
    interpolation = np.moveaxis(np.clip(coordinates*bins-.5, 0, bins-1), -1, 0)
    scores = []
    for g in range(c.k):
        histogram = np.bincount(code.ravel(), weights=(c.weight*(c.labels == g)).ravel(),
                                minlength=bins*bins).reshape(bins, bins)
        histogram = ndi.gaussian_filter(histogram, .7, mode="nearest")
        histogram += max(histogram.sum(), 1)*1e-7
        histogram /= histogram.sum()
        likelihood = ndi.map_coordinates(histogram, interpolation, order=1, mode="nearest")
        scores.append(np.log(np.maximum(likelihood, 1e-20)))
    return softmax(np.stack(scores, -1), -1), dict(
        projection=projection.tolist(), contrast_eigenvalues=values.tolist(), joint_histogram_bins=bins)


IMPLEMENTATIONS = {
    "D01": pair_inverse, "D02": conditional_prior, "D03": entropy_inverse,
    "D04": huber_inverse, "D05": nullspace_inverse, "D06": directional_inverse,
    "D07": poisson_tiles, "D08": bag_covariance, "D09": design_tiles,
    "D10": local_affine, "D11": shared_drift, "D12": glass_competition,
    "D13": tree_density, "D14": polya_tree, "D15": fisher_density}


def predict(key, rgb, coarse, support, mpp_xy, palette, he=None):
    if key not in IMPLEMENTATIONS:
        raise ValueError("Unimplemented cycle-two child")
    c = Evidence(rgb, coarse, support, mpp_xy)
    if not c.support.any():
        raise ValueError("Empty tissue support")
    if c.k == 1 and key != "D12":
        q, detail = np.ones((*c.shape, 1), np.float32), dict(single_class=True)
    else:
        q, detail = IMPLEMENTATIONS[key](c, palette, he)
    excluded = detail.pop("glass_exclusion", None)
    if q.shape != (*c.shape, c.k) or not np.isfinite(q).all():
        raise ValueError("Invalid child posterior")
    labels = c.ids[q.argmax(-1)].astype(np.uint8)
    if excluded is not None:
        labels[excluded] = 0
    detail.update(analysis_mpp_xy=c.mpp.tolist(), analysis_shape_yx=list(c.shape),
                  fine_reference_used=False, hard_anchors=False, fixed_margin=False,
                  candidate=key, parent=CYCLE_TWO[key].parent,
                  mechanism=CYCLE_TWO[key].mechanism)
    return labels, detail
