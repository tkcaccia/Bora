"""JSON/CSV CPU bridge: python -m bora.appearance_bridge request.json.

Only precomputed rows enter this command. It never reads a WSI, imports Torch
or fastPLS, downloads a model, starts a server, or launches a job.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from threadpoolctl import threadpool_limits

from .discriminative import DiscriminativeAppearance


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _column(rows, name, default=None):
    if name is None:
        return default
    if not rows or name not in rows[0]:
        raise ValueError("Missing CSV column: " + str(name))
    return [row[name] for row in rows]


def _read(request, base):
    columns = request.get("feature_columns")
    if not isinstance(columns, list) or not columns:
        raise ValueError("feature_columns must name the ordered encoder features")
    if ("features" in request) == ("feature_table" in request):
        raise ValueError("Supply exactly one features matrix or feature_table CSV")
    if "feature_table" in request:
        path = (base / request["feature_table"]).resolve()
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            header = reader.fieldnames or []
            if len(set(header)) != len(header):
                raise ValueError("Duplicate CSV column names are ambiguous")
            rows = list(reader)
        if not rows or any(column not in header for column in columns):
            raise ValueError("CSV has no observations or is missing feature_columns")
        try:
            x = np.asarray([[float(row[c]) for c in columns] for row in rows], np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("CSV features must be finite numeric values, without missing cells") from exc
        y = _column(rows, request.get("label_column"), request.get("labels"))
        roles = _column(rows, request.get("split_column"), request.get("splits"))
        ids = _column(rows, request.get("id_column", "id"), [str(i) for i in range(len(x))]) if "id_column" in request or "id" in header else [str(i) for i in range(len(x))]
        xs = _column(rows, request.get("x_column"), None)
        ys = _column(rows, request.get("y_column"), None)
        provenance = dict(feature_table_sha256=_hash(path))
    else:
        try:
            x = np.asarray(request["features"], np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("features must be a rectangular numeric row matrix") from exc
        y, roles = request.get("labels"), request.get("splits")
        ids = request.get("ids", [str(i) for i in range(len(x))])
        coordinates = request.get("coordinates", {})
        xs, ys = coordinates.get("x"), coordinates.get("y")
        provenance = dict(inline_features_sha256=hashlib.sha256(x.tobytes()).hexdigest())
    if x.ndim != 2 or not len(x) or x.shape[1] != len(columns) or not np.isfinite(x).all():
        raise ValueError("Feature rows must be nonempty, finite and match feature_columns")
    if len(ids) != len(x) or len(set(map(str, ids))) != len(x):
        raise ValueError("Observation IDs must be unique and aligned to feature rows")
    if (xs is None) != (ys is None):
        raise ValueError("Supply both x and y coordinates or neither")
    if xs is not None:
        try:
            positions = np.asarray([xs, ys], np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Coordinates must be numeric native-pixel centres") from exc
        if positions.shape != (2, len(x)) or not np.isfinite(positions).all() or np.any(positions < 0):
            raise ValueError("Coordinates must be finite nonnegative x/y native-pixel centres")
        if request.get("coordinate_space") != "native_pixel_centres":
            raise ValueError("Declare coordinate_space=native_pixel_centres; viewport coordinates are not accepted")
        xs, ys = positions.tolist()
    else:
        xs = ys = [""] * len(x)
    return x, y, roles, ids, xs, ys, provenance


def _write_csv(path, columns, rows):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def run_request(path):
    path = Path(path).resolve()
    request = json.loads(path.read_text())
    if request.get("format") != "bora-appearance-request-v1":
        raise ValueError("format must be bora-appearance-request-v1")
    mode = request.get("mode")
    if mode not in ("local", "shared"):
        raise ValueError("mode must be local or shared; row contrast is not full RGB9+19")
    base = path.parent
    x, y, roles, ids, xs, ys, provenance = _read(request, base)
    if "output_prefix" not in request:
        raise ValueError("output_prefix is required")
    prefix = (base / request["output_prefix"]).resolve()
    outputs = {name: Path(str(prefix) + suffix) for name, suffix in (
        ("recipe", ".recipe.json"), ("scores", ".scores.csv"), ("rgb", ".rgb.csv"), ("report", ".report.json"))}
    if any(p.exists() for p in outputs.values()):
        raise FileExistsError("Appearance outputs already exist; choose a new output_prefix")
    contract = {k: request.get(k) for k in ("feature_kind", "feature_columns", "encoder")}
    fitting = "recipe_path" not in request
    with threadpool_limits(limits=2):
        if fitting:
            if y is None or roles is None or len(y) != len(x) or len(roles) != len(x):
                raise ValueError("Fit needs labels/label_column and explicit splits/split_column")
            roles = np.asarray(roles, dtype=str)
            labels = np.asarray(y, dtype=str)
            if not np.isin(roles, ["train", "reference", "query", "validation", "test"]).all():
                raise ValueError("Unknown split role; use train/reference/query/validation/test")
            selected = np.isin(roles, ["train", "reference"])
            if not selected.any():
                raise ValueError("There are no explicit train/reference rows to fit")
            model = DiscriminativeAppearance.fit(x[selected], labels[selected], mode=mode, **contract,
                label_kind=request.get("label_kind"), classes=request.get("classes", {}),
                splits=roles[selected], selected_count=request.get("selected_count", 64),
                palette=request.get("palette"), provenance=dict(provenance, request_sha256=_hash(path),
                    user_provenance=request.get("provenance", {})))
            fit_rows = int(selected.sum())
        else:
            model = DiscriminativeAppearance.load(base / request["recipe_path"])
            if model.state["mode"] != mode:
                raise ValueError("Requested mode differs from the saved recipe")
            fit_rows = 0
        probability = model.predict_proba(x, **contract)
        rgb = model.rgb(probability, request.get("palette"))
    classes = model.state["classes"]
    predicted = np.asarray(classes)[probability.argmax(1)]
    prefix.parent.mkdir(parents=True, exist_ok=True)
    # Complete all files off-path; exclusive links avoid clobbering a concurrent writer.
    with tempfile.TemporaryDirectory(dir=prefix.parent, prefix=".bora-appearance-") as temp:
        staging = {name: Path(temp) / target.name for name, target in outputs.items()}
        model.save(staging["recipe"])
        _write_csv(staging["scores"], ["id", "x", "y", "predicted_label"] + ["probability:" + c for c in classes],
            ([ids[i], xs[i], ys[i], predicted[i], *probability[i]] for i in range(len(x))))
        _write_csv(staging["rgb"], ["id", "x", "y", "r", "g", "b"],
            ([ids[i], xs[i], ys[i], *map(int, rgb[i])] for i in range(len(x))))
        report = dict(format="bora-appearance-result-v1", method=model.state["method"],
            mode=mode, rows=len(x), fit_rows=fit_rows, loaded_without_refit=not fitting,
            feature_kind=model.state["feature_kind"], label_kind=model.state["label_kind"],
            classes=model.state["class_names"], probability_columns=["probability:" + c for c in classes],
            full_RGB9_plus_19=False, rgb_display_only=True, gpu_used=False, native_fastpls_used=False,
            coordinate_space=request.get("coordinate_space"),
            output_paths={name: str(target) for name, target in outputs.items()},
            file_sha256={name: _hash(staging[name]) for name in ("recipe", "scores", "rgb")})
        staging["report"].write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        published = []
        try:
            for name, target in outputs.items():
                os.link(staging[name], target)
                published.append(target)
        except BaseException:
            for target in published:
                target.unlink(missing_ok=True)
            raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit/reuse CPU feature contrast from cached JSON/CSV rows (no GPU or model download)")
    parser.add_argument("request", type=Path, help="bora-appearance-request-v1 JSON")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run_request(args.request), allow_nan=False))
    except (ValueError, OSError, KeyError, TypeError) as error:
        print("Feature contrast: " + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
