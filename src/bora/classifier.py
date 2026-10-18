"""The sole Bora classifier: Neighbor context, native CPU fastPLS SIMPLS-LDA."""
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter
import numpy as np

N_COMPONENTS = 24  # Fixed recipe for the 25-class benchmark.
FORMAT = "bora-neighbor-context-v1"
FASTPLS_REVISION = "fc0c011969b754ee196aacc220a217783afeb05d"


def class_names(value):
    result = {}
    for key, name in value.items():
        try:
            index = int(key)
        except (ValueError, TypeError) as exc:
            raise ValueError("Class IDs must be integers in 1..255") from exc
        if str(index) != str(key) or not 1 <= index <= 255 or not isinstance(name, str) or not name.strip():
            raise ValueError("Class IDs in 1..255 and nonempty names required")
        result[index] = name.strip()
    return result


def predictors(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2560 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Finite nonempty N x 2560 Neighbor context descriptors required")
    return np.asfortranarray(x)


class NeighborContextClassifier:
    """One native head; no alternative engine, decoder or feature selector.

    fastPLS 0.3 has no portable fitted-state serialization. A safe NPZ artifact
    therefore contains training descriptors/targets, not pickle or private C++
    pointers. Loading refits the same deterministic native head once. Keep the
    returned object alive for a batch; startup and per-image timing are separate.
    Artifacts retain training features and must be shared under their data terms.
    """
    def fit(self, x, y, names):
        from ._vendor.fastpls import PLS, __version__
        x = predictors(x)
        y = np.asarray(y)
        if (y.ndim != 1 or len(y) != len(x) or y.dtype.kind not in "iu"
                or np.any(y < 1) or np.any(y > 255) or len(np.unique(y)) < 2):
            raise ValueError("Aligned integer tissue targets in 1..255, with at least two classes required")
        names = class_names(names)
        classes = np.unique(y)
        if set(classes) != set(names):
            raise ValueError("Class names must match exactly the represented training targets")
        self.x_ = x.copy(order="F")
        self.y_ = y.astype(np.int64, copy=True)
        self.names_ = names
        self.classes_ = classes
        self.fastpls_version_ = __version__
        if self.fastpls_version_ != "0.3.0":
            raise ValueError("This audited Neighbor context artifact requires fastPLS 0.3.0")
        start = perf_counter()
        # The benchmark has 25 represented training classes, hence 24 axes.
        # Do not ask native LDA to invert spurious axes for a two-class cohort.
        self.model_ = PLS(n_components=min(N_COMPONENTS, len(classes) - 1, x.shape[1], len(x) - 1),
            method="simpls", classifier="lda", scaling="autoscaling", backend="cpu",
            seed=314159).fit(x, self.y_)
        self.fit_seconds_ = perf_counter() - start
        self.actual_components_ = int(self.model_.n_components_)
        return self

    def predict(self, x):
        if not hasattr(self, "model_"):
            raise RuntimeError("Fit or load the classifier before prediction")
        x = predictors(x)
        return np.asarray(self.model_.predict(x), dtype=np.uint8).reshape(-1)

    def save(self, path):
        if not hasattr(self, "model_"):
            raise RuntimeError("Fit the classifier before saving")
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(format=FORMAT, classifier="Neighbor context", components=N_COMPONENTS,
            actual_components=self.actual_components_, fastpls_version=self.fastpls_version_,
            fastpls_revision=FASTPLS_REVISION, backend="cpu", classes=self.names_,
            native_head_count=1, descriptor="1280 group means + 1280 physical boundary contrasts",
            load_refits_native_head=True, training_features_included=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as f:
            temp = Path(f.name)
        try:
            np.savez_compressed(temp, x=self.x_, y=self.y_, metadata=json.dumps(metadata))
            # Exclusive link prevents a concurrent writer being overwritten.
            os.link(temp, path)
        finally:
            temp.unlink(missing_ok=True)
        return metadata

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {"x", "y", "metadata"}:
                raise ValueError("Not a Bora Neighbor context artifact")
            info = json.loads(str(data["metadata"].item()))
            if (info.get("format") != FORMAT or info.get("components") != N_COMPONENTS
                    or info.get("fastpls_version") != "0.3.0"
                    or info.get("fastpls_revision") != FASTPLS_REVISION
                    or info.get("backend") != "cpu"):
                raise ValueError("Incompatible Neighbor context artifact")
            return cls().fit(data["x"], data["y"], info["classes"])
