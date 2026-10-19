"""Frozen row-wise feature contrast; distinct from spatial RGB9+19 and PLS.

Local anonymous labels describe one project's groups. Shared semantic labels
must have the same independently annotated meaning across training tissues.
Neither mode infers a model, requests a GPU, or uses target labels on transform.
"""
from itertools import combinations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .appearance_state import (finite, linear_state, load_state, probabilities,
    probability_rgb, save_state, scale_values, scaler_state, validate_linear,
    validate_scaler, runtime_versions)

FORMAT = "bora-discriminative-appearance-v1"
FEATURE_WIDTHS = {"virchow2-spatial": 1280, "virchow2-pooled": 2560}


def feature_contract(kind, names, encoder):
    if kind not in FEATURE_WIDTHS:
        raise ValueError("feature_kind must be virchow2-spatial (1280) or virchow2-pooled (2560)")
    if (not isinstance(names, list) or len(names) != FEATURE_WIDTHS[kind]
            or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names)):
        raise ValueError("Declare unique feature_columns in the exact encoder feature order")
    if (not isinstance(encoder, dict) or not isinstance(encoder.get("revision"), str)
            or not encoder["revision"] or not isinstance(encoder.get("sha256"), str)
            or len(encoder["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in encoder["sha256"])):
        raise ValueError("encoder needs its revision and lowercase 64-character checkpoint sha256")
    return dict(feature_kind=kind, feature_columns=list(names), encoder=dict(encoder))


def select_pairwise(values, labels, count=64):
    """Round-robin pairwise F relevance, penalized by maximum redundancy."""
    values = finite(values, "Training features", ndim=2).astype(np.float64)
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= min(128, values.shape[1]):
        raise ValueError("selected_count must be an integer in 1..min(128, feature width)")
    z = StandardScaler().fit_transform(values)
    classes = np.unique(labels)
    means = np.stack([z[labels == label].mean(0) for label in classes])
    variances = np.stack([z[labels == label].var(0) for label in classes])
    relevance = [(means[a] - means[b]) ** 2 / (variances[a] + variances[b] + .01)
                 for a, b in combinations(range(len(classes)), 2)]
    pool = np.unique(np.concatenate([np.argsort(-r, kind="stable")[:128] for r in relevance]))
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.nan_to_num(np.corrcoef(z[:, pool], rowvar=False), nan=0.)
    correlation = np.atleast_2d(correlation)
    lookup = {int(index): i for i, index in enumerate(pool)}
    selected = []
    while len(selected) < count:
        for row in relevance:
            available = np.array([i for i in pool if int(i) not in selected])
            redundancy = (np.max(np.abs(correlation[np.ix_(
                [lookup[int(i)] for i in available], [lookup[i] for i in selected])]), axis=1)
                if selected else np.zeros(len(available)))
            selected.append(int(available[np.argmax(row[available] * (1 - .85 * redundancy))]))
            if len(selected) == count:
                break
    return np.asarray(selected, dtype=int)


class DiscriminativeAppearance:
    """Portable fitted selection/scaling/logistic state; load never refits."""
    def __init__(self, state):
        self.state = state
        try:
            contract = feature_contract(state["feature_kind"], state["feature_columns"], state["encoder"])
            if state["format"] != FORMAT or state["mode"] not in ("local", "shared"):
                raise ValueError("Unsupported discriminative appearance recipe")
            if state["label_kind"] not in ("anonymous", "semantic") or (
                    state["mode"] == "shared" and state["label_kind"] != "semantic"):
                raise ValueError("Shared appearance requires semantic training labels")
            classes = state["classes"]
            if (not isinstance(classes, list) or len(classes) < 2 or len(set(classes)) != len(classes)
                    or any(not isinstance(c, str) or not c for c in classes)
                    or set(state["class_names"]) != set(classes)
                    or any(not isinstance(v, str) or not v.strip() for v in state["class_names"].values())):
                raise ValueError("Invalid fitted class names or ordering")
            selected = finite(state["selected_dimensions"], "Selected dimensions", ndim=1)
            if (selected.dtype.kind not in "iu" or not len(selected) or np.any(selected != np.floor(selected))
                    or np.any(selected < 0) or np.any(selected >= len(contract["feature_columns"]))
                    or len(np.unique(selected)) != len(selected)):
                raise ValueError("Invalid selected feature dimensions")
            validate_scaler(state["scaler"], len(selected))
            validate_linear(state["logistic"], len(selected), len(classes), logistic=True)
            if state["logistic"]["classes"] != classes:
                raise ValueError("Fitted logistic class ordering changed")
            probability_rgb(np.ones((1, len(classes))), state["palette"])
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("Invalid fitted discriminative appearance state") from exc

    @classmethod
    def fit(cls, values, labels, *, mode, feature_kind, feature_columns, encoder,
            label_kind, classes, splits, selected_count=64, palette=None, provenance=None):
        contract = feature_contract(feature_kind, feature_columns, encoder)
        x = finite(values, "Training features", ndim=2).astype(np.float64)
        if x.shape[1] != FEATURE_WIDTHS[feature_kind]:
            raise ValueError("Feature width differs from the declared pooled/spatial contract")
        y = np.asarray(labels, dtype=str)
        if y.shape != (len(x),) or np.any(y == "") or len(np.unique(y)) < 2:
            raise ValueError("Fit needs aligned nonempty labels with at least two classes")
        roles = np.asarray(splits, dtype=str)
        if roles.shape != y.shape or not np.isin(roles, ["train", "reference"]).all():
            raise ValueError("Only explicit train/reference rows may enter fit; query/test rows are forbidden")
        if len(x) < 16 or any(np.sum(y == c) < 2 for c in np.unique(y)):
            raise ValueError("Fit needs at least 16 labelled rows and two rows per class")
        if not np.any(np.ptp(x, axis=0) > 0):
            raise ValueError("Training features have no variation; feature contrast cannot be fitted")
        if mode not in ("local", "shared") or label_kind not in ("anonymous", "semantic"):
            raise ValueError("Declare local/shared mode and anonymous/semantic label_kind")
        if mode == "shared" and label_kind != "semantic":
            raise ValueError("Shared transfer requires common semantic training labels, not cluster IDs")
        ordered = np.unique(y).tolist()
        names = {str(k): v for k, v in classes.items()}
        if set(names) != set(ordered) or any(not isinstance(v, str) or not v.strip() for v in names.values()):
            raise ValueError("classes must name exactly the represented training labels")
        selected = select_pairwise(x, y, selected_count)
        scaler = StandardScaler().fit(x[:, selected])
        scaler.scale_ = np.maximum(scaler.scale_, .05)
        logistic = LogisticRegression(C=.3, class_weight="balanced", max_iter=2000,
                                      random_state=430).fit(scaler.transform(x[:, selected]), y)
        if palette is None:
            # Display only; probabilities are the palette-independent evidence.
            colors = [[(47 + i * 89) % 256, (109 + i * 137) % 256, (191 + i * 61) % 256]
                      for i in range(len(ordered))]
        else:
            colors = [palette[c] for c in ordered]
        state = dict(format=FORMAT, mode=mode, label_kind=label_kind, **contract,
            classes=ordered, class_names=names, selected_dimensions=selected.tolist(),
            scaler=scaler_state(scaler), logistic=linear_state(logistic, logistic=True), palette=colors,
            training=dict(rows=len(x), class_counts={c: int(np.sum(y == c)) for c in ordered},
                          roles=sorted(set(roles)), provenance=provenance or {}),
            producer=runtime_versions(), settings=dict(selected_count=selected_count,
                pairwise_variance_floor=.01, redundancy_penalty=.85, scale_floor=.05,
                logistic_c=.3, class_weight="balanced", max_iter=2000, seed=430),
            method="pairwise-F redundancy selection + balanced logistic feature contrast",
            full_RGB9_plus_19=False, semantic_scope="frozen common vocabulary" if mode == "shared" else "local project groups")
        return cls(state)

    def predict_proba(self, values, *, feature_kind, feature_columns, encoder):
        contract = feature_contract(feature_kind, feature_columns, encoder)
        if any(contract[k] != self.state[k] for k in contract):
            raise ValueError("Query encoder, feature kind or feature order differs from fitted recipe")
        x = finite(values, "Query features", ndim=2)
        if not len(x) or x.shape[1] != FEATURE_WIDTHS[feature_kind]:
            raise ValueError("Query rows must match the fitted pooled/spatial feature width")
        z = scale_values(self.state["scaler"], x[:, self.state["selected_dimensions"]].astype(np.float64))
        return probabilities(self.state["logistic"], z)

    def rgb(self, probability, palette=None):
        probability = finite(probability, "Class probabilities", ndim=2)
        if probability.shape[1] != len(self.state["classes"]):
            raise ValueError("Probability columns differ from the fitted classes")
        colors = self.state["palette"] if palette is None else [palette[c] for c in self.state["classes"]]
        return probability_rgb(probability, colors)

    def save(self, path):
        save_state(path, self.state)

    @classmethod
    def load(cls, path):
        return cls(load_state(path, FORMAT))
