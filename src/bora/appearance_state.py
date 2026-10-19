"""Small, non-executable fitted appearance artifacts (JSON, never pickle)."""
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.special import expit, softmax


def runtime_versions():
    return {name: version(name) for name in ("numpy", "scipy", "scikit-learn")}


def finite(value, name, *, ndim=None):
    try:
        result = np.asarray(value)
        if result.dtype.kind not in "fiu" or not np.isfinite(result).all():
            raise ValueError
        if ndim is not None and result.ndim != ndim:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite numeric values" ) from exc
    return result


def array_digest(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def scaler_state(scaler):
    return dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist())


def validate_scaler(state, width):
    mean = finite(state["mean"], "Scaler mean", ndim=1)
    scale = finite(state["scale"], "Scaler scale", ndim=1)
    if len(mean) != width or len(scale) != width or np.any(scale <= 0):
        raise ValueError("Scaler dimensions or positive scales are invalid")


def scale_values(state, values):
    # Match StandardScaler's in-place rounding for float16/32 input caches.
    result = np.array(values, copy=True)
    if result.dtype not in (np.dtype("float16"), np.dtype("float32"), np.dtype("float64")):
        result = result.astype(np.float64)
    result -= np.asarray(state["mean"], np.float64)
    result /= np.asarray(state["scale"], np.float64)
    return result


def linear_state(model, *, logistic=False):
    result = dict(coef=model.coef_.tolist(), intercept=np.asarray(model.intercept_).tolist(),
                  dtype=str(model.coef_.dtype))
    if logistic:
        result["classes"] = model.classes_.tolist()
    return result


def validate_linear(state, width, outputs, *, logistic=False):
    if state.get("dtype") not in ("float32", "float64"):
        raise ValueError("Invalid fitted coefficient dtype")
    coef = finite(state["coef"], "Fitted coefficients", ndim=2)
    intercept = finite(state["intercept"], "Fitted intercept", ndim=1)
    expected = 1 if logistic and outputs == 2 else outputs
    if coef.shape != (expected, width) or intercept.shape != (expected,):
        raise ValueError("Fitted coefficient dimensions are invalid")
    if logistic and len(state.get("classes", [])) != outputs:
        raise ValueError("Fitted class ordering is invalid")


def linear_values(state, values):
    dtype = np.dtype(state["dtype"])
    return values @ np.asarray(state["coef"], dtype).T + np.asarray(state["intercept"], dtype)


def probabilities(state, values):
    scores = linear_values(state, values)
    if scores.shape[-1] == 1:
        positive = expit(scores[..., 0])
        return np.stack([1 - positive, positive], axis=-1)
    return softmax(scores, axis=-1)


def probability_rgb(values, palette):
    # The published local formula quantizes a float32 palette product. Keeping
    # this dtype also preserves pixels exactly at half-integer RGB thresholds.
    palette = finite(palette, "Palette", ndim=2).astype(np.float32)
    if palette.shape != (values.shape[-1], 3) or np.any((palette < 0) | (palette > 255)):
        raise ValueError("Palette needs one RGB triplet in 0..255 for each fitted class")
    return np.rint(np.clip(values, 0, 1) @ palette).clip(0, 255).astype(np.uint8)


def _canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def save_state(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = dict(payload=payload, sha256=hashlib.sha256(_canonical(payload)).hexdigest())
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", suffix=".json", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(document, stream, allow_nan=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, path)  # Exclusive publication, never overwrite user work.
    finally:
        temporary.unlink(missing_ok=True)


def load_state(path, expected_format):
    path = Path(path)
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Appearance recipe exceeds the 32 MiB safety limit")
    try:
        document = json.loads(path.read_text())
        if not isinstance(document, dict) or not isinstance(document.get("payload"), dict):
            raise ValueError("Appearance recipe must contain a JSON object payload")
        payload = document["payload"]
        if document["sha256"] != hashlib.sha256(_canonical(payload)).hexdigest():
            raise ValueError("Appearance recipe checksum mismatch")
        if payload.get("format") != expected_format:
            raise ValueError("Unsupported appearance recipe format")
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid appearance recipe JSON") from exc
    return payload
