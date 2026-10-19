"""Fit once and replay Bora's local, geometry-bound RGB9+19 evidence on CPU."""
from itertools import combinations
import numpy as np
from scipy import ndimage as ndi
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .appearance_state import (array_digest, finite, linear_state, linear_values,
    load_state, probabilities, probability_rgb, save_state, scale_values,
    scaler_state, validate_linear, validate_scaler, runtime_versions)
from .conversion import observations, select_dimensions, _align, _expand, _fit

FORMAT = "bora-local-rgb9plus19-v1"


def _inputs(tokens, shape):
    tokens = finite(tokens, "Spatial tokens", ndim=3)
    expected = tuple((np.asarray(shape) + 13) // 14)
    if tokens.shape != (*expected, 1280):
        raise ValueError("RGB9+19 needs full 1280-channel spatial tokens on the padded 14-pixel grid")
    return tokens


def _variant(spec, selected, grid, shape):
    values = finite(spec["values"], "Scale features", ndim=3).astype(np.float32)
    size = finite(spec["shape_yx"], "Scale image shape", ndim=1)
    if size.shape != (2,) or np.any(size < 1) or np.any(size != np.floor(size)):
        raise ValueError("Scale shape must have positive integer height and width")
    if values.shape[-1] == 1280:
        values = values[..., selected]
    elif values.shape[-1] != 64 or not np.array_equal(spec.get("selected_dimensions"), selected):
        raise ValueError("Scale-cache selection does not match this coarse mask")
    return _align(values, size, grid, shape)


def _neighbours(z):
    result = np.empty((*z.shape[:2], 192), np.float32)
    for j, k in enumerate((1, 2, 4)):
        padded = np.pad(z, ((k, k), (k, k), (0, 0)), mode="reflect")
        result[..., j * 64:(j + 1) * 64] = (padded[:-2*k, k:-k] + padded[2*k:, k:-k]
            + padded[k:-k, :-2*k] + padded[k:-k, 2*k:]) / 4
    return result


def _folds(shape):
    yy, xx = np.indices(shape)
    return (yy >= shape[0] // 2) * 2 + (xx >= shape[1] // 2)


def _scaled_fit(values, coarse):
    summaries, _ = observations(values, coarse)
    scaler = StandardScaler().fit(summaries)
    scaler.scale_ = np.maximum(scaler.scale_, .05)
    return scaler_state(scaler), scaler.transform(values.reshape(-1, values.shape[-1])).reshape(values.shape).astype(np.float32), scaler.transform(summaries)


class RGB9Plus19Recipe:
    """A same-field local artifact, deliberately not a cross-tissue model.

    Spatial #19 folds and anonymous groups belong to the fitted field. Replay
    checks token/scale hashes and shape. Use DiscriminativeAppearance shared
    mode for a separately trained transferable row-wise appearance mapping.
    """
    def __init__(self, state):
        self.state = state
        try:
            if state["format"] != FORMAT or state["label_kind"] != "anonymous":
                raise ValueError("Unsupported local RGB9+19 recipe")
            shape = finite(state["shape_yx"], "Local image shape", ndim=1)
            classes = finite(state["classes"], "Local groups", ndim=1)
            if (shape.dtype.kind not in "iu" or shape.shape != (2,) or np.any(shape < 224) or np.any(shape != np.floor(shape))
                    or classes.dtype.kind not in "iu" or len(classes) < 2 or np.any(classes < 1) or np.any(classes > 255)
                    or np.any(classes != np.floor(classes)) or len(np.unique(classes)) != len(classes)):
                raise ValueError("Invalid local shape or anonymous groups")
            selected = finite(state["selected_dimensions"], "Selected channels", ndim=1)
            keep = finite(state["persistent_positions"], "Persistent channels", ndim=1)
            for values, n, width in ((selected, 64, 1280), (keep, 32, 64)):
                if (values.dtype.kind not in "iu" or len(values) != n or len(np.unique(values)) != n or np.any(values != np.floor(values))
                        or np.any(values < 0) or np.any(values >= width)):
                    raise ValueError("Invalid local channel selection")
            for name in ("base", "scale_1", "scale_2"):
                validate_scaler(state["scalers"][name], 64)
            for name, width in (("p9", 32), ("p19", 64)):
                validate_linear(state[name], width, len(classes), logistic=True)
                if state[name]["classes"] != state["classes"]:
                    raise ValueError("Local probability class ordering changed")
            if len(state["ridge"]) != 4:
                raise ValueError("Local #19 needs exactly four fitted quadrant heads")
            for ridge in state["ridge"]:
                validate_linear(ridge, 192, 64)
            probability_rgb(np.ones((1, len(classes))), state["palette"])
            for name in ("base", "scale_1", "scale_2", "coarse"):
                digest = state["input_sha256"][name]
                if (not isinstance(digest, str) or len(digest) != 64
                        or any(c not in "0123456789abcdef" for c in digest)):
                    raise ValueError("Missing local source identity")
            for name in ("scale_1", "scale_2"):
                size = finite(state["variant_shapes"][name], "Scale shape", ndim=1)
                if size.dtype.kind not in "iu" or size.shape != (2,) or np.any(size < 1):
                    raise ValueError("Invalid fitted scale image shape")
        except (KeyError, TypeError) as exc:
            raise ValueError("Invalid fitted local RGB9+19 state") from exc

    @classmethod
    def fit(cls, tokens, coarse, palette, variants, *, provenance=None):
        coarse = finite(coarse, "Coarse groups", ndim=2)
        if (coarse.dtype.kind not in "iu" or np.any(coarse < 0) or np.any(coarse > 255)):
            raise ValueError("Local coarse groups must be integers in 0..255")
        tokens = _inputs(tokens, coarse.shape)
        summaries, labels = observations(tokens, coarse)
        classes = np.unique(labels)
        if not np.array_equal(classes, np.unique(coarse[coarse > 0])):
            raise ValueError("Every coarse group needs an eligible tile observation")
        if len(summaries) < 16 or len(classes) < 2:
            raise ValueError("RGB9+19 fit needs at least 16 eligible tiles and two groups")
        palette = finite(palette, "Palette", ndim=2)
        if palette.shape[1] != 3 or len(palette) < coarse.max() or np.any((palette < 0) | (palette > 255)):
            raise ValueError("Finite RGB palette indexed by coarse label minus one required")
        selected = select_dimensions(summaries, labels)
        scaler = StandardScaler().fit(summaries[:, selected])
        scaler.scale_ = np.maximum(scaler.scale_, .05)
        tile_z = scaler.transform(summaries[:, selected]).astype(np.float32)
        z = scaler.transform(tokens[..., selected].reshape(-1, 64)).reshape(*tokens.shape[:2], 64).astype(np.float32)
        scalers = {"base": scaler_state(scaler)}
        aligned = {name: _variant(variants[name], selected, z.shape[:2], coarse.shape)
                   for name in ("scale_1", "scale_2")}
        scalers["scale_1"], middle, mt = _scaled_fit(aligned["scale_1"], coarse)
        scalers["scale_2"], wide, wt = _scaled_fit(aligned["scale_2"], coarse)
        with np.errstate(invalid="ignore", divide="ignore"):
            correlations = np.stack([np.array([np.corrcoef(tile_z[:, i], other[:, i])[0, 1]
                                              for i in range(64)]) for other in (mt, wt)])
        persistence = np.maximum(np.nan_to_num(correlations.min(0)), 0)
        means = [tile_z[labels == g].mean(0) for g in classes]
        var = [tile_z[labels == g].var(0) for g in classes]
        rankings = []
        for a, b in combinations(range(len(classes)), 2):
            score = (means[a] - means[b]) ** 2 / (var[a] + var[b] + .05)
            rankings.append(np.argsort(-score * (.05 + persistence)))
        keep, positions = [], [0] * len(rankings)
        while len(keep) < 32:
            for i, ranking in enumerate(rankings):
                while positions[i] < len(ranking) and int(ranking[positions[i]]) in keep:
                    positions[i] += 1
                if positions[i] < len(ranking):
                    keep.append(int(ranking[positions[i]])); positions[i] += 1
                    if len(keep) == 32:
                        break
        consensus = (z[..., keep] + middle[..., keep] + wide[..., keep]) / 3
        p9_x, p9_y = observations(consensus, coarse)
        p9 = _fit(p9_x, p9_y)
        neighbours = _neighbours(z)
        folds = _folds(z.shape[:2])
        completed = np.empty_like(z)
        heads = []
        for fold in range(4):
            test = folds == fold
            train = ~ndi.binary_dilation(test, iterations=4)
            if not train.any() or not test.any():
                raise ValueError("Insufficient spatial token extent for the fixed #19 folds")
            model = Ridge(alpha=100).fit(neighbours[train], z[train])
            completed[test] = model.predict(neighbours[test])
            heads.append(linear_state(model))
        p19_x, p19_y = observations(completed, coarse)
        p19 = _fit(p19_x, p19_y)
        return cls(dict(format=FORMAT, label_kind="anonymous", shape_yx=list(coarse.shape),
            classes=classes.tolist(), palette=palette[classes - 1].tolist(),
            selected_dimensions=selected.tolist(), persistent_positions=keep, scalers=scalers,
            p9=linear_state(p9, logistic=True), p19=linear_state(p19, logistic=True), ridge=heads,
            input_sha256=dict(base=array_digest(tokens), coarse=array_digest(coarse),
                **{name: array_digest(variants[name]["values"]) for name in aligned}),
            variant_shapes={name: list(variants[name]["shape_yx"]) for name in aligned},
            producer=runtime_versions(), settings=dict(logistic_c=.3, class_weight="balanced",
                logistic_seed=430, ridge_alpha=100, fold_halo_tokens=4, scale_floor=.05,
                token_stride_pixels=14, tile_width_pixels=90, minimum_tiles=16),
            provenance=provenance or {}, fine_reference_used=False, geometry_bound=True))

    def predict_proba(self, tokens, variants):
        state = self.state
        shape = state["shape_yx"]
        tokens = _inputs(tokens, shape)
        if array_digest(tokens) != state["input_sha256"]["base"]:
            raise ValueError("Local RGB9+19 belongs to a different token field; fit local state explicitly")
        selected = np.asarray(state["selected_dimensions"])
        z = scale_values(state["scalers"]["base"], tokens[..., selected].reshape(-1, 64)).reshape(*tokens.shape[:2], 64).astype(np.float32)
        scaled = []
        for name in ("scale_1", "scale_2"):
            if (array_digest(variants[name]["values"]) != state["input_sha256"][name]
                    or list(variants[name]["shape_yx"]) != state["variant_shapes"][name]):
                raise ValueError("Local RGB9+19 scale cache or physical grid changed")
            aligned = _variant(variants[name], selected, z.shape[:2], shape)
            scaled.append(scale_values(state["scalers"][name], aligned.reshape(-1, 64)).reshape(aligned.shape).astype(np.float32))
        keep = state["persistent_positions"]
        consensus = (z[..., keep] + scaled[0][..., keep] + scaled[1][..., keep]) / 3
        p9 = probabilities(state["p9"], consensus.reshape(-1, 32)).reshape(*z.shape[:2], -1).astype(np.float32)
        neighbours = _neighbours(z)
        folds = _folds(z.shape[:2])
        completed = np.empty_like(z)
        for fold, head in enumerate(state["ridge"]):
            test = folds == fold
            completed[test] = linear_values(head, neighbours[test])
        p19 = probabilities(state["p19"], completed.reshape(-1, 64)).reshape(*z.shape[:2], -1).astype(np.float32)
        return np.mean(np.stack([_expand(p9, shape), _expand(p19, shape)]), axis=0, dtype=np.float32)

    def transform(self, tokens, variants, *, palette=None):
        p = self.predict_proba(tokens, variants)
        colors = self.state["palette"] if palette is None else np.asarray(palette)[np.asarray(self.state["classes"]) - 1]
        return probability_rgb(p, colors), dict(conversion="RGB9+19", components=[9, 19], weights=[.5, .5],
            selected_dimensions=self.state["selected_dimensions"], fine_reference_used=False,
            fitted_recipe=True, fit_repeated=False, geometry_bound=True)

    def save(self, path):
        save_state(path, self.state)

    @classmethod
    def load(cls, path):
        return cls(load_state(path, FORMAT))
