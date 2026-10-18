"""Command line interface for the sole supported Bora pipeline."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import cv2
import numpy as np
import tifffile
from threadpoolctl import threadpool_limits
from . import __version__
from .classifier import NeighborContextClassifier
from .annotations import read_annotation, group_targets, _class_map
from .pipeline import load_case, prepare, run, json_file, digest, _mask


def parser():
    p = argparse.ArgumentParser(prog="bora", description="Bora-AF + Neighbor context (native fastPLS CPU)")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prepare", help="prepare whole-group geometry/descriptors for training")
    a.add_argument("case", type=Path); a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("train", help="fit the single Neighbor context classifier on declared training groups")
    a.add_argument("manifest", type=Path); a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("refine", help="RGB9+19, Bora-AF, Neighbor context identity and compact native GeoJSON")
    a.add_argument("case", type=Path); a.add_argument("--model", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    return p


def prepare_file(case_path, output):
    if output.exists():
        raise FileExistsError(output)
    case = load_case(case_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bora-", dir=output.parent) as temp:
        folder = Path(temp) / "prepared"
        folder.mkdir()
        data = prepare(case)
        _mask(folder / "groups.ome.tif", data["regions"].astype(np.uint32), case.native_mpp_xy)
        meta = dict(format="bora-neighbor-context-descriptors-v1", provenance=case.provenance,
            mpp_xy=case.mpp_xy.tolist(), native_mpp_xy=case.native_mpp_xy.tolist(),
            prediction_unit="Whole anonymous group including disconnected pieces")
        np.savez_compressed(folder / "descriptors.npz", x=data["descriptors"],
            group_ids=data["group_ids"], area_pixels=data["area_pixels"], metadata=json.dumps(meta))
        json_file(folder / "report.json", dict(data["details"], **meta,
            descriptors_sha256=digest(folder / "descriptors.npz")))
        if output.exists():
            raise FileExistsError(output)
        os.rename(folder, output)
    return dict(output=str(output), groups=len(data["group_ids"]))


def train_file(manifest_path, output):
    if output.exists():
        raise FileExistsError(output)
    spec = json.loads(manifest_path.read_text())
    if spec.get("format") != "bora-training-v1" or not spec.get("cases"):
        raise ValueError("A nonempty bora-training-v1 manifest is required")
    names = {int(k): v for k, v in spec["classes"].items()}
    xs, ys, seen, records = [], [], set(), []
    for row in spec["cases"]:
        if row.get("split") != "train" or not row.get("source_group"):
            raise ValueError("Every descriptor file needs split=train and a source_group identity")
        path = (manifest_path.parent / row["descriptors"]).resolve()
        sha = digest(path)
        if row.get("sha256", sha) != sha or sha in seen:
            raise ValueError("Changed or duplicated training descriptors")
        seen.add(sha)
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"].item()))
            if meta.get("format") != "bora-neighbor-context-descriptors-v1":
                raise ValueError("Use descriptors produced by bora prepare")
            ids = data["group_ids"].copy()
            if ("labels" in row) == ("annotation" in row):
                raise ValueError("Supply either explicit labels or a training annotation, not both")
            if "labels" in row:
                labels = row["labels"]
                if set(labels) != set(map(str, ids.tolist())):
                    raise ValueError("Supply one tissue target for every whole anonymous group")
                target = np.asarray([labels[str(int(g))] for g in ids])
                keep = np.ones(len(ids), bool)
                excluded = []
                annotation_sha = None
                groups_sha = None
            else:
                groups_path = (manifest_path.parent / row["groups"]).resolve()
                annotation_path = (manifest_path.parent / row["annotation"]).resolve()
                groups_sha, annotation_sha = digest(groups_path), digest(annotation_path)
                if (row.get("groups_sha256", groups_sha) != groups_sha or
                        row.get("annotation_sha256", annotation_sha) != annotation_sha):
                    raise ValueError("Changed training groups or annotation")
                groups = tifffile.imread(groups_path)
                mapping = _class_map(row["annotation_values"], names)
                annotated = read_annotation(annotation_path, groups.shape, mapping,
                    field=row.get("annotation_field", "value"),
                    ignore_values=row.get("ignore_values", ["0"]),
                    coordinate_space=row.get("annotation_coordinate_space"))
                retained, target, excluded = group_targets(groups, ids, annotated,
                    min_coverage=float(row.get("min_coverage", .5)),
                    min_purity=float(row.get("min_purity", .8)))
                keep = np.isin(ids, retained)
                if not np.array_equal(ids[keep], retained):
                    raise ValueError("Training group IDs changed order")
            xs.append(data["x"][keep].copy()); ys.append(target)
            records.append(dict(source_group=row["source_group"], descriptor_sha256=sha,
                groups_sha256=groups_sha, annotation_sha256=annotation_sha,
                training_groups=int(np.count_nonzero(keep)),
                excluded_groups=excluded))
    if not any(len(x) for x in xs):
        raise ValueError("No sufficiently annotated training groups remain")
    model = NeighborContextClassifier().fit(np.concatenate(xs), np.concatenate(ys), spec["classes"])
    meta = model.save(output, training_provenance=dict(manifest_sha256=digest(manifest_path), cases=records))
    return dict(output=str(output), training_groups=len(model.y_), model=meta,
                fit_seconds=model.fit_seconds_, training_manifest_sha256=digest(manifest_path),
                excluded_groups=sum(len(r["excluded_groups"]) for r in records))


def main(argv=None):
    a = parser().parse_args(argv)
    cv2.setNumThreads(2)
    try:
        with threadpool_limits(limits=2):
            if a.command == "prepare":
                report = prepare_file(a.case, a.output)
            elif a.command == "train":
                report = train_file(a.manifest, a.output)
            else:
                case = load_case(a.case)
                model = NeighborContextClassifier.load(a.model)
                report = run(case, model, a.output)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        raise SystemExit(f"bora {a.command}: {exc}") from exc
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
